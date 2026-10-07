// Q2: one tensor-core GEMM family for sm_86 (A10G) with mixed-precision K segments, written with raw mma.sync / ldmatrix / cp.async.
//
// y[t, n] = sum over K segments of A[t, k] * B[n, k]  (A: activations [M, K] row-major, B: weights [N, K] row-major = "col" operand)
// A K tile is KB = 64 or 128 BYTES per row (64 B = 128 int4 = 64 int8 = 32 bf16), so all three precisions share one swizzled smem
// pipeline; only the mma instruction changes per tile:  s4 m16n8k64, s8 m16n8k32, bf16 m16n8k16 (fp32 accumulate). Each mma k-step
// consumes 32 bytes of K per row.
//
// Segment 0 (main): S4 | S8 | BF16, optionally group-scaled (g = 64 or 128 int4 columns per scale) or tile-sampled (B9 mask).
// Segment 1 (tail): NONE | BF16 (accumulates in place in fp32: the B1 low-rank correction Z * B_tail) | S8 (separate int32 accumulator:
//                   ResQ / B5 inner-dimension split).
// Epilogue (FORMATS.md section 2): f = fp32(fp32(float(acc) * sa[t]) * sb[n]) [* kscale], + tail, + bias[n], + T[idx[t], n], then
//   epi 0: bf16 store;  1: SwiGLU on interleaved (g, u) column pairs -> bf16 [M, N/2];  2: fp16(alpha * acc) (H2's convention);
//   3: raw int32;  4: raw fp32.   Output row = rowmap ? rowmap[t] : row0 + t (row-role / arbitrary row partitions).
// Two problems per launch (row-role grouped GEMM): CTAs [0, tiles0) run problem 0, the rest problem 1 (always S8, no tail).
// Batched launches (Strassen products) use gridDim.z with byte strides.
// Main loop follows CUTLASS's MmaMultistage order: fragments of k-step j+1 are loaded from smem while k-step j's mma issue, the
// global->smem copies of tile kt+ST-1 are issued at the start of tile kt, and one barrier per tile sits before its last k-step.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

enum { NONE = 0, S4 = 1, S8 = 2, BF16 = 3 };

struct Prob {
  const void* A0; const void* B0; long long lda0, ldb0, kt0;      // seg0: lda/ldb in bytes, kt0 = # of 64-byte K units
  const void* A1; const void* B1; long long lda1, ldb1, kt1;      // seg1 (tail), kt1 in 64-byte units
  const float* sa0; const float* sb0; const float* sa1; const float* sb1;
  const float* gsa; const float* gsb; long long ldgs;              // group scales [M, G], [N, G] (fp32), ldgs = G
  const float* bias; const void* tab; const int* tidx; long long ldt;   // B5 bias [N]; B3 table bf16 [Kc, ldt], idx [M]
  const int* rowmap; long long row0;
  void* out; long long ldo; long long epi;
  double alpha; double kscale; long long seed; long long nkeep; long long nkb;   // B9: keep nkeep of nkb 128-col K blocks per 128x128 tile
  long long M, N;
  long long bsA0, bsB0, bsO;                                       // batch strides in bytes (gridDim.z)
  long long mtiles, ntiles;
  float* stat; const float* statw;                                 // B6: stat[warp % 64] += sum_t w_t sum_n f_tn^2 (fp32 atomics)
  int* skws; int* sksem; long long splits;                         // split-K (plain s4/s8): int32 partials [splits][M][N], per-tile counters
};
struct Params { Prob p[2]; long long tiles0; };

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(uint32_t s, const void* g, int nb) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(nb));
}
__device__ __forceinline__ void cp8(uint32_t s, const void* g, int nb) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(g), "r"(nb));
}
__device__ __forceinline__ void cp4(uint32_t s, const void* g, int nb) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(g), "r"(nb));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
template <int T> __device__ __forceinline__ void mma(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1);
template <> __device__ __forceinline__ void mma<S4>(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <> __device__ __forceinline__ void mma<S8>(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <> __device__ __forceinline__ void mma<BF16>(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  float f0 = __uint_as_float(c[0]), f1 = __uint_as_float(c[1]), f2 = __uint_as_float(c[2]), f3 = __uint_as_float(c[3]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(f0), "+f"(f1), "+f"(f2), "+f"(f3) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  c[0] = __float_as_uint(f0); c[1] = __float_as_uint(f1); c[2] = __float_as_uint(f2); c[3] = __float_as_uint(f3);
}
// exact int32 -> fp32 for |x| < 2^22 without I2F (I2F is quarter rate on sm_86): used inside the group-scaled loop
__device__ __forceinline__ float i2f_small(uint32_t x) { return __fsub_rn(__int_as_float(int(x) + 0x4B400000), 12582912.0f); }
__device__ __forceinline__ uint32_t fmix32(uint32_t h) { h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16; return h; }

template <int KB> __device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (KB == 64) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
  else return row * 128 + ((chunk ^ (row & 7)) << 4);
}

template <int BM_, int BN_, int WGM_, int WGN_, int ST_, int KB_>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, ST = ST_, KB = KB_;
  static constexpr int NTH = WGM * WGN * 32;
  static constexpr int WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
  static constexpr int CH = KB / 16;                      // 16-byte chunks per row per tile
  static constexpr int KSTEPS = KB / 32;
  static constexpr int ABYTES = BM * KB, BBYTES = BN * KB;
  static constexpr int SCB = (BM + BN) * 16;              // group-scale bytes per stage (up to 4 fp32 per row)
  template <int GR> static constexpr int stage() { return ABYTES + BBYTES + (GR ? SCB : 0); }
  template <int GR> static constexpr int smem() { return ST * stage<GR>() + 64 * 4; }   // + K-tile list for B9
  static_assert(NT % 2 == 0, "NT even");
  static_assert((BM * CH) % NTH == 0 && (BN * CH) % NTH == 0, "loader divisibility");
};

// ------------------------------------------------------------------ one output tile
template <class C, int T0, int T1, int GR, bool SKIP>
__device__ __forceinline__ void run_tile(const Prob& P, int tm, int tn, int z, uint8_t* smem, int ks = 0) {
  constexpr int KB = C::KB, KS = C::KSTEPS, MT = C::MT, NT = C::NT;
  constexpr int U = (GR == 128) ? 2 : 1;          // k-steps per compute unit (one scale group for g128)
  constexpr int UNITS = KS / U;
  constexpr int UPT = KB / 64;                    // 64-byte units of the K index per tile
  static_assert(!(SKIP && KB != 64), "B9 skip uses 64-byte tiles (= 128 int4 columns)");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / C::WGN) * C::WM, wn0 = (warp % C::WGN) * C::WN;
  const int m0 = tm * C::BM, n0 = tn * C::BN;
  const int M = (int)P.M, N = (int)P.N;
  const uint8_t* A0 = (const uint8_t*)P.A0 + (long long)z * P.bsA0;
  const uint8_t* B0 = (const uint8_t*)P.B0 + (long long)z * P.bsB0;
  const uint8_t* A1 = (const uint8_t*)P.A1;
  const uint8_t* B1 = (const uint8_t*)P.B1;
  constexpr int STAGE = C::template stage<GR>();
  int* klist = reinterpret_cast<int*>(smem + C::ST * STAGE);

  int kt0 = (int)(P.kt0 / UPT);
  const int nsplit = (T1 == NONE && GR == 0 && !SKIP && P.splits > 1) ? (int)P.splits : 1;
  if (nsplit > 1) {                         // split-K: this CTA covers K tiles [ks*kps, min(kt0, (ks+1)*kps))
    const int kps = (kt0 + nsplit - 1) / nsplit;
    const int kb0 = ks * kps, kb1 = min(kt0, kb0 + kps);
    A0 += (long long)kb0 * KB; B0 += (long long)kb0 * KB;
    kt0 = max(0, kb1 - kb0);
  }
  if constexpr (SKIP) {
    const int nkb = (int)P.nkb, nkeep = (int)P.nkeep;
    const uint32_t mb = (uint32_t)(m0 >> 7), nb = (uint32_t)(n0 >> 7);
    if (tid < nkb) {
      uint32_t base = (uint32_t)P.seed ^ (mb * 0x9E3779B1u) ^ (nb * 0x85EBCA77u);
      uint32_t h = fmix32(base ^ ((uint32_t)tid * 0xC2B2AE35u));
      int rank = 0;
      for (int j = 0; j < nkb; ++j) {
        uint32_t hj = fmix32(base ^ ((uint32_t)j * 0xC2B2AE35u));
        rank += (hj < h) || (hj == h && j < tid);
      }
      if (rank < nkeep) klist[rank] = tid;
    }
    __syncthreads();
    kt0 = nkeep;
  }
  const int KT = kt0 + (int)(P.kt1 / UPT);

  // per-thread ldmatrix offsets (bytes within a stage)
  const int a_row = wm0 + (lane & 15);
  const int b_row = wn0 + (lane & 7) + ((lane >> 4) << 3);
  int a_off[KS], b_off[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    a_off[ks] = swz<KB>(a_row, ks * 2 + (lane >> 4));
    b_off[ks] = C::ABYTES + swz<KB>(b_row, ks * 2 + ((lane >> 3) & 1));
  }
  const uint32_t sbase = smem_u32(smem);

  // per-thread copy descriptors: smem offsets fixed, global pointers per segment (recomputed once at the segment switch)
  constexpr int NA = (C::BM * C::CH) / C::NTH, NB = (C::BN * C::CH) / C::NTH;
  int a_so[NA], b_so[NB], a_nb[NA], b_nb[NB];
  const uint8_t* a_g[NA]; const uint8_t* b_g[NB];
  auto setptr = [&](int seg) {
    const uint8_t* Ab = seg ? A1 : A0; const uint8_t* Bb = seg ? B1 : B0;
    const long long lda = seg ? P.lda1 : P.lda0, ldb = seg ? P.ldb1 : P.ldb0;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      int idx = tid + i * C::NTH, r = idx / C::CH, ch = idx % C::CH;
      int gr = m0 + r; bool ok = gr < M;
      a_so[i] = swz<KB>(r, ch); a_nb[i] = ok ? 16 : 0;
      a_g[i] = Ab + (long long)(ok ? gr : 0) * lda + ch * 16;
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      int idx = tid + i * C::NTH, r = idx / C::CH, ch = idx % C::CH;
      int gr = n0 + r; bool ok = gr < N;
      b_so[i] = C::ABYTES + swz<KB>(r, ch); b_nb[i] = ok ? 16 : 0;
      b_g[i] = Bb + (long long)(ok ? gr : 0) * ldb + ch * 16;
    }
  };
  setptr(0);
  int cur_seg = 0;
  // copy i of a tile (i < NA: A chunk, else B chunk) into stage address s at K byte offset ko
  auto load_one = [&](uint32_t s, int ko, int i) {
    if (i < NA) cp16(s + a_so[i], a_g[i] + ko, a_nb[i]);
    else cp16(s + b_so[i - NA], b_g[i - NA] + ko, b_nb[i - NA]);
  };
  auto load_ko = [&](int kt) {
    int kb;
    if (kt < kt0) kb = SKIP ? klist[kt] : kt;
    else { if (cur_seg == 0) { setptr(1); cur_seg = 1; } kb = kt - kt0; }
    return kb * KB;
  };
  auto load = [&](int stage, int kt) {
    const uint32_t s = sbase + stage * STAGE;
    const int ko = load_ko(kt);
#pragma unroll
    for (int i = 0; i < NA + NB; ++i) load_one(s, ko, i);
    if constexpr (GR != 0) {
      if (kt < kt0) {
        constexpr int GPT = (2 * KB) / GR;       // groups per tile (int4 columns per tile = 2 KB)
        uint8_t* sS = smem + stage * STAGE + C::ABYTES + C::BBYTES;            // [BM + BN][4] fp32
        for (int r = tid; r < C::BM + C::BN; r += C::NTH) {
          bool isA = r < C::BM; int gr = isA ? m0 + r : n0 + (r - C::BM);
          bool ok = gr < (isA ? M : N);
          const float* g = (isA ? P.gsa : P.gsb) + (long long)(ok ? gr : 0) * P.ldgs + kt * GPT;
          if constexpr (GPT == 4) cp16(smem_u32(sS + r * 16), g, ok ? 16 : 0);
          else if constexpr (GPT == 2) cp8(smem_u32(sS + r * 16), g, ok ? 8 : 0);
          else cp4(smem_u32(sS + r * 16), g, ok ? 4 : 0);
        }
      }
    }
  };

  uint32_t acc[MT][NT][4];
  uint32_t acc2[(T1 == S8) ? MT : 1][(T1 == S8) ? NT : 1][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0u;
  if constexpr (T1 == S8) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc2[i][j][q] = 0u;
  }

  auto rowof = [&](int mt, int hi) { return m0 + wm0 + mt * 16 + (lane >> 2) + hi * 8; };
  auto colof = [&](int nt) { return n0 + wn0 + nt * 8 + 2 * (lane & 3); };

  auto int_to_float = [&](const float* sa, const float* sb, float ks) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      int r0 = rowof(mt, 0), r1 = rowof(mt, 1);
      float a0 = r0 < M ? sa[r0] : 0.f, a1 = r1 < M ? sa[r1] : 0.f;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        int c = colof(nt);
        float b0 = c < N ? sb[c] : 0.f, b1 = c + 1 < N ? sb[c + 1] : 0.f;
        float v0 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][0], a0), b0);
        float v1 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][1], a0), b1);
        float v2 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][2], a1), b0);
        float v3 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][3], a1), b1);
        if (SKIP) { v0 = __fmul_rn(v0, ks); v1 = __fmul_rn(v1, ks); v2 = __fmul_rn(v2, ks); v3 = __fmul_rn(v3, ks); }
        acc[mt][nt][0] = __float_as_uint(v0); acc[mt][nt][1] = __float_as_uint(v1);
        acc[mt][nt][2] = __float_as_uint(v2); acc[mt][nt][3] = __float_as_uint(v3);
      }
    }
  };

  uint32_t af[2][U][MT][4], bf[2][U][NT / 2][4];
  // group scales of the unit, loaded with the fragments (the stage may be overwritten after the per-tile barrier)
  float gx[(GR != 0) ? 2 : 1][(GR != 0) ? MT : 1][2], gw[(GR != 0) ? 2 : 1][(GR != 0) ? NT : 1][2];
  auto ldfrag = [&](int buf, int stage, int unit) {
    const uint32_t s = sbase + stage * STAGE;
    if constexpr (GR != 0) {
      const float* sS = reinterpret_cast<const float*>(smem + stage * STAGE + C::ABYTES + C::BBYTES);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int lr0 = wm0 + mt * 16 + (lane >> 2);
        gx[buf][mt][0] = sS[lr0 * 4 + unit]; gx[buf][mt][1] = sS[(lr0 + 8) * 4 + unit];
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int lc = C::BM + wn0 + nt * 8 + 2 * (lane & 3);
        gw[buf][nt][0] = sS[lc * 4 + unit]; gw[buf][nt][1] = sS[(lc + 1) * 4 + unit];
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ks = unit * U + u;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) ldsm4(af[buf][u][mt], s + a_off[ks] + mt * 16 * KB);
#pragma unroll
      for (int p = 0; p < NT / 2; ++p) ldsm4(bf[buf][u][p], s + b_off[ks] + p * 16 * KB);
    }
  };

  // i-th ldmatrix of a unit (i < MT: A tile i; else B pair i - MT), U == 1 only
  auto ldsm_one = [&](int buf, uint32_t s, int unit, int i) {
    if (i < MT) ldsm4(af[buf][0][i], s + a_off[unit] + i * 16 * KB);
    else ldsm4(bf[buf][0][i - MT], s + b_off[unit] + (i - MT) * 16 * KB);
  };
  // mma of one unit with the next unit's ldmatrix (pf) and this tile's global copies (gm) interleaved between the mma, CUTLASS-style;
  // n-tiles in serpentine order. Non-group, U == 1 path.
  auto mma_il = [&](int buf, auto Tc, bool first_seg, bool pf, int pbuf, uint32_t ps, int punit, bool gm, uint32_t gs, int gko) {
    constexpr int T = decltype(Tc)::value;
    constexpr int NLD = MT + NT / 2, NCP = NA + NB, NMMA = MT * NT;
#pragma unroll
    for (int idx = 0; idx < NMMA; ++idx) {
      const int mt = idx / NT, ntr = idx % NT, nt = (mt & 1) ? (NT - 1 - ntr) : ntr;
      const uint32_t b0 = bf[buf][0][nt >> 1][(nt & 1) * 2], b1 = bf[buf][0][nt >> 1][(nt & 1) * 2 + 1];
      bool done = false;
      if constexpr (T1 == S8 && T == S8) { if (!first_seg) { mma<S8>(acc2[mt][nt], af[buf][0][mt], b0, b1); done = true; } }
      if (!done) mma<T>(acc[mt][nt], af[buf][0][mt], b0, b1);
      if (pf && idx < NLD) ldsm_one(pbuf, ps, punit, idx);
      if (gm && idx >= 1 && idx - 1 < NCP) load_one(gs, gko, idx - 1);
    }
    if (pf) {
#pragma unroll
      for (int i = NMMA; i < NLD; ++i) ldsm_one(pbuf, ps, punit, i);
    }
    if (gm) {
#pragma unroll
      for (int i = NMMA - 1; i < NCP; ++i) load_one(gs, gko, i);
    }
  };

  // mma of one unit (U k-steps) from fragment buffer buf, tile type T
  auto mma_unit = [&](int buf, auto Tc, bool first_seg, int stage, int unit) {
    constexpr int T = decltype(Tc)::value;
    if constexpr (GR != 0 && T == S4) {
      if (first_seg) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const float x0 = gx[buf][mt][0], x1 = gx[buf][mt][1];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            uint32_t d[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int u = 0; u < U; ++u) mma<S4>(d, af[buf][u][mt], bf[buf][u][nt >> 1][(nt & 1) * 2], bf[buf][u][nt >> 1][(nt & 1) * 2 + 1]);
            const float w0 = gw[buf][nt][0], w1 = gw[buf][nt][1];
            const float p00 = __fmul_rn(x0, w0), p01 = __fmul_rn(x0, w1), p10 = __fmul_rn(x1, w0), p11 = __fmul_rn(x1, w1);
            acc[mt][nt][0] = __float_as_uint(__fadd_rn(__uint_as_float(acc[mt][nt][0]), __fmul_rn(i2f_small(d[0]), p00)));
            acc[mt][nt][1] = __float_as_uint(__fadd_rn(__uint_as_float(acc[mt][nt][1]), __fmul_rn(i2f_small(d[1]), p01)));
            acc[mt][nt][2] = __float_as_uint(__fadd_rn(__uint_as_float(acc[mt][nt][2]), __fmul_rn(i2f_small(d[2]), p10)));
            acc[mt][nt][3] = __float_as_uint(__fadd_rn(__uint_as_float(acc[mt][nt][3]), __fmul_rn(i2f_small(d[3]), p11)));
          }
        }
        return;
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const uint32_t b0 = bf[buf][u][nt >> 1][(nt & 1) * 2], b1 = bf[buf][u][nt >> 1][(nt & 1) * 2 + 1];
          if constexpr (T1 == S8 && T == S8) {
            if (!first_seg) { mma<S8>(acc2[mt][nt], af[buf][u][mt], b0, b1); continue; }
          }
          mma<T>(acc[mt][nt], af[buf][u][mt], b0, b1);
        }
  };

  auto do_mma = [&](int buf, int kt, int stage, int unit) {
    if (kt < kt0) mma_unit(buf, std::integral_constant<int, T0>{}, true, stage, unit);
    else {
      if constexpr (T1 != NONE) mma_unit(buf, std::integral_constant<int, (T1 == NONE ? S8 : T1)>{}, false, stage, unit);
    }
  };

  // ---------------- main loop
#pragma unroll
  for (int s = 0; s < C::ST - 1; ++s) { if (s < KT) load(s, s); cp_commit(); }
  cp_wait<C::ST - 2>();
  __syncthreads();
  ldfrag(0, 0, 0);
  int rs = 0;                       // read stage (tile kt)
  int ws = C::ST - 1;               // write stage (tile kt + ST - 1)
  for (int kt = 0; kt < KT; ++kt) {
    if constexpr (T1 != NONE) { if (kt == kt0 && kt0 > 0 && T0 != BF16 && GR == 0) int_to_float(P.sa0, P.sb0, (float)P.kscale); }
    const int nk = kt + C::ST - 1;
#pragma unroll
    for (int j = 0; j < UNITS; ++j) {
      const int cur = j & 1, nxt = (j + 1) & 1;
      if (j == UNITS - 1) {
        if constexpr (UNITS == 1) {
          do_mma(0, kt, rs, j);                       // fragments of this tile are already in registers
          if (nk < KT) load(ws, nk);                  // gmem copies issue behind the mma
          cp_commit(); cp_wait<C::ST - 2>(); __syncthreads();
          rs = (rs + 1 == C::ST) ? 0 : rs + 1; ws = (ws + 1 == C::ST) ? 0 : ws + 1;
          ldfrag(0, rs, 0);                           // first unit of tile kt+1 (harmless garbage after the last tile)
        } else {
          if constexpr (GR == 0 && U == 1) {
            const uint32_t sr = sbase + rs * STAGE;
            if (kt < kt0) mma_il(cur, std::integral_constant<int, T0>{}, true, true, nxt, sr, 0, false, 0u, 0);
            else {
              if constexpr (T1 != NONE) mma_il(cur, std::integral_constant<int, (T1 == NONE ? S8 : T1)>{}, false, true, nxt, sr, 0, false, 0u, 0);
            }
          } else {
            ldfrag(nxt, rs, 0);                         // first unit of tile kt+1 (harmless garbage after the last tile)
            do_mma(cur, kt, rs, j);
          }
        }
      } else {
        if constexpr (GR == 0 && U == 1) {
          const bool gm = (j == 0) && (nk < KT);
          const int gko = gm ? load_ko(nk) : 0;
          const uint32_t sr = sbase + rs * STAGE, sw = sbase + ws * STAGE;
          if (kt < kt0) mma_il(cur, std::integral_constant<int, T0>{}, true, true, nxt, sr, j + 1, gm, sw, gko);
          else {
            if constexpr (T1 != NONE) mma_il(cur, std::integral_constant<int, (T1 == NONE ? S8 : T1)>{}, false, true, nxt, sr, j + 1, gm, sw, gko);
          }
        } else {
          ldfrag(nxt, rs, j + 1);
          do_mma(cur, kt, rs, j);
          if (j == 0) { if (nk < KT) load(ws, nk); }    // gmem copies issue behind the first unit's mma (CUTLASS order)
        }
        if (j == UNITS - 2) {
          cp_commit(); cp_wait<C::ST - 2>(); __syncthreads();
          rs = (rs + 1 == C::ST) ? 0 : rs + 1; ws = (ws + 1 == C::ST) ? 0 : ws + 1;
        }
      }
    }
  }
  cp_wait<0>();

  if (nsplit > 1) {   // write int32 partials, the last CTA of this tile sums them (fixed order -> deterministic) and runs the epilogue
    int* wsb = P.skws;
    const long long MN = (long long)M * N;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int r = rowof(mt, hi);
        if (r >= M) continue;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = colof(nt);
          if (c >= N) continue;
          *reinterpret_cast<int2*>(wsb + (long long)ks * MN + (long long)r * N + c) = make_int2((int)acc[mt][nt][hi * 2], (int)acc[mt][nt][hi * 2 + 1]);
        }
      }
    __threadfence();
    __syncthreads();
    const int tile = tm + tn * (int)P.mtiles;
    if (tid == 0) klist[0] = (atomicAdd(P.sksem + tile, 1) == nsplit - 1);
    __syncthreads();
    if (!klist[0]) return;
    __threadfence();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int r = rowof(mt, hi);
        if (r >= M) continue;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = colof(nt);
          if (c >= N) continue;
          int s0 = 0, s1 = 0;
          for (int q = 0; q < nsplit; ++q) {
            int2 v = __ldcg(reinterpret_cast<const int2*>(wsb + (long long)q * MN + (long long)r * N + c));
            s0 += v.x; s1 += v.y;
          }
          acc[mt][nt][hi * 2] = (uint32_t)s0; acc[mt][nt][hi * 2 + 1] = (uint32_t)s1;
        }
      }
    if (tid == 0) P.sksem[tile] = 0;          // ready for the next launch (graph replay)
  }

  // ---------------- epilogue
  const int epi = (int)P.epi;
  if (epi == 9) {   // benchmark only: no stores (keeps the accumulators live through a cheap reduction)
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) x ^= acc[i][j][0] ^ acc[i][j][1] ^ acc[i][j][2] ^ acc[i][j][3];
    if (x == 0x9E3779B9u) *(uint32_t*)P.out = x;
    return;
  }
  constexpr bool intacc = (T0 != BF16) && (GR == 0) && (T1 == NONE);
  if (intacc && epi != 2 && epi != 3 && epi != 6) int_to_float(P.sa0, P.sb0, (float)P.kscale);
  if constexpr (T1 != NONE) { if (KT == kt0 && T0 != BF16 && GR == 0) int_to_float(P.sa0, P.sb0, (float)P.kscale); }
  const long long ldo = P.ldo;
  uint8_t* outb = (uint8_t*)P.out + (long long)z * P.bsO;
  float statacc = 0.f;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
      int r = rowof(mt, hi);
      const bool rok = r < M;
      if (!rok && epi != 1) continue;
      if (!rok) r = M - 1;          // swiglu: keep the lane active for the shuffle, store suppressed below
      long long orow = P.rowmap ? (long long)P.rowmap[r] : P.row0 + r;
      float s1a = 0.f;
      if constexpr (T1 == S8) s1a = P.sa1[r];
      int tix = P.tab ? P.tidx[r] : 0;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        int c = colof(nt);
        if (c >= N && epi != 1) continue;
        if (epi == 1 && (!rok || c >= N)) {   // inactive lane in the swiglu pair-pack: participate in the shuffle only
          (void)__shfl_down_sync(0xffffffffu, 0u, 1);
          continue;
        }
        uint32_t u0 = acc[mt][nt][hi * 2], u1 = acc[mt][nt][hi * 2 + 1];
        if (epi == 2) {
          __half2 h = __floats2half2_rn((float)P.alpha * (float)(int)u0, (float)P.alpha * (float)(int)u1);
          *reinterpret_cast<__half2*>(outb + (orow * ldo + c) * 2) = h;
          continue;
        }
        if (epi == 3) {
          *reinterpret_cast<int2*>(outb + (orow * ldo + c) * 4) = make_int2((int)u0, (int)u1);
          continue;
        }
        if (epi == 6) {   // fp16(float(acc) * sb[col]): rows written in another GEMM's fp16 units (row-role with CUTLASS state rows)
          __half2 h = __floats2half2_rn(__fmul_rn((float)(int)u0, P.sb0[c]), __fmul_rn((float)(int)u1, P.sb0[c + 1]));
          *reinterpret_cast<__half2*>(outb + (orow * ldo + c) * 2) = h;
          continue;
        }
        float f0 = __uint_as_float(u0), f1 = __uint_as_float(u1);
        if constexpr (T1 == S8) {
          float w0 = P.sb1[c], w1 = P.sb1[c + 1];
          f0 = __fadd_rn(f0, __fmul_rn(__fmul_rn((float)(int)acc2[mt][nt][hi * 2], s1a), w0));
          f1 = __fadd_rn(f1, __fmul_rn(__fmul_rn((float)(int)acc2[mt][nt][hi * 2 + 1], s1a), w1));
        }
        if (P.bias) { f0 = __fadd_rn(f0, P.bias[c]); f1 = __fadd_rn(f1, P.bias[c + 1]); }
        if (P.tab) {
          __nv_bfloat162 t = *reinterpret_cast<const __nv_bfloat162*>((const __nv_bfloat16*)P.tab + (long long)tix * P.ldt + c);
          f0 = __fadd_rn(f0, __bfloat162float(t.x)); f1 = __fadd_rn(f1, __bfloat162float(t.y));
        }
        if (P.stat) statacc += (P.statw ? P.statw[r] : 1.0f) * (f0 * f0 + f1 * f1);
        if (epi == 0) {
          *reinterpret_cast<__nv_bfloat162*>(outb + (orow * ldo + c) * 2) = __floats2bfloat162_rn(f0, f1);
        } else if (epi == 1) {
          float g = __bfloat162float(__float2bfloat16_rn(f0)), u = __bfloat162float(__float2bfloat16_rn(f1));
          float sl = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g))));
          __nv_bfloat16 m = __float2bfloat16_rn(sl * u);
          // lanes 2j and 2j+1 hold adjacent outputs (c/2, c/2+1): pack them into one 4-byte store
          unsigned short mine = __bfloat16_as_ushort(m);
          unsigned short other = (unsigned short)__shfl_down_sync(0xffffffffu, (unsigned)mine, 1);
          if ((lane & 1) == 0) {
            if ((c >> 1) + 1 < (N >> 1))
              *reinterpret_cast<uint32_t*>(outb + (orow * ldo + (c >> 1)) * 2) = (uint32_t)mine | ((uint32_t)other << 16);
            else
              *reinterpret_cast<__nv_bfloat16*>(outb + (orow * ldo + (c >> 1)) * 2) = m;
          }
        } else {
          *reinterpret_cast<float2*>(outb + (orow * ldo + c) * 4) = make_float2(f0, f1);
        }
      }
    }
  }
  if (P.stat) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) statacc += __shfl_xor_sync(0xffffffffu, statacc, o);
    if (lane == 0) atomicAdd(P.stat + ((blockIdx.x * (C::NTH / 32) + warp) & 63), statacc);
  }
}

template <class C, int T0, int T1, int GR, bool SKIP>
__global__ void __launch_bounds__(C::NTH) k_one(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Prob& P = p.p[0];
  int b = blockIdx.x;
  run_tile<C, T0, T1, GR, SKIP>(P, b % (int)P.mtiles, b / (int)P.mtiles, blockIdx.z, smem, blockIdx.y);
}

template <class C, int T0, int T1, int GR, bool SKIP>
__global__ void __launch_bounds__(C::NTH) k_two(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  int b = blockIdx.x;
  const long long t1 = p.p[1].mtiles * p.p[1].ntiles, t0 = p.p[0].mtiles * p.p[0].ntiles, T = t0 + t1;
  // schedule: tiles0 == 0 -> int8 tiles first;  tiles0 == 1 -> int8 tiles spread evenly through the launch (slot i at floor(i*T/t1)),
  // so their weight streaming overlaps the int4 tiles' compute
  long long i8 = -1, i4 = 0;
  if (p.tiles0 == 0) { if (b < t1) i8 = b; else i4 = b - t1; }
  else {
    long long c = (b * t1 + T - 1) / T;                      // number of int8 slots at positions < b  (= ceil(b*t1/T))
    if (c < t1 && (c * T) / t1 == b) i8 = c; else i4 = b - c;
  }
  if (i8 >= 0) {
    const Prob& P = p.p[1];
    run_tile<C, S8, NONE, 0, false>(P, (int)(i8 % P.mtiles), (int)(i8 / P.mtiles), 0, smem);
  } else {
    const Prob& P = p.p[0];
    run_tile<C, T0, T1, GR, SKIP>(P, (int)(i4 % P.mtiles), (int)(i4 / P.mtiles), 0, smem);
  }
}

// ------------------------------------------------------------------ host dispatch
template <class C, int T0, int T1, int GR, bool SKIP, bool TWO>
static int launch(Params& p, int two, int batch, cudaStream_t st) {
  if constexpr (SKIP && C::KB != 64) { return 5; }
  else {
    if (two && !TWO) return 6;
    for (int i = 0; i < (two ? 2 : 1); ++i) {
      p.p[i].mtiles = (p.p[i].M + C::BM - 1) / C::BM;
      p.p[i].ntiles = (p.p[i].N + C::BN - 1) / C::BN;
      if ((p.p[i].kt0 % (C::KB / 64)) || (p.p[i].kt1 % (C::KB / 64))) return 7;
    }
    long long t0 = p.p[0].mtiles * p.p[0].ntiles;
    long long tot = t0 + (two ? p.p[1].mtiles * p.p[1].ntiles : 0);
    if (tot == 0) return 0;
    const unsigned spl = (!two && p.p[0].splits > 1) ? (unsigned)p.p[0].splits : 1u;
    dim3 grid((unsigned)tot, spl, (unsigned)batch);
    if constexpr (TWO) {
      if (two) {
        auto k = k_two<C, T0, T1, GR, SKIP>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::template smem<GR>());
        k<<<grid, C::NTH, C::template smem<GR>(), st>>>(p);
        cudaError_t e = cudaGetLastError();
        return e == cudaSuccess ? 0 : 1000 + (int)e;
      }
    }
    auto k = k_one<C, T0, T1, GR, SKIP>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::template smem<GR>());
    k<<<grid, C::NTH, C::template smem<GR>(), st>>>(p);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : 1000 + (int)e;
  }
}

//               BM   BN  WGM WGN ST  KB
using C0 = Cfg<64, 128, 2, 2, 4, 64>;    // warp 32x64, 128 thr
using C1 = Cfg<128, 128, 2, 4, 4, 64>;   // warp 64x32, 256 thr
using C2 = Cfg<128, 256, 2, 4, 3, 64>;   // warp 64x64, 256 thr
using C3 = Cfg<32, 128, 1, 4, 4, 64>;    // warp 32x32, 128 thr
using C4 = Cfg<64, 64, 2, 2, 4, 64>;     // warp 32x32, 128 thr
using C5 = Cfg<128, 64, 4, 2, 4, 64>;    // warp 32x32, 256 thr
using C6 = Cfg<64, 64, 2, 2, 6, 64>;     // warp 32x32, 128 thr, 6 stages
using C7 = Cfg<64, 128, 2, 4, 3, 64>;    // warp 32x32, 256 thr, 3 stages
using C8 = Cfg<64, 128, 2, 4, 4, 64>;    // warp 32x32, 256 thr
using C9 = Cfg<128, 64, 4, 2, 3, 64>;    // warp 32x32, 256 thr, 3 stages

#define DISPATCH_CFG(T0, T1, GR, SK, TW)                                                       \
  switch (cfg) {                                                                             \
    case 0: return launch<C0, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 1: return launch<C1, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 2: if (T1 != S8) return launch<C2, T0, T1, GR, SK, TW>(p, two, batch, st); return 2; \
    case 3: return launch<C3, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 4: return launch<C4, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 5: return launch<C5, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 6: if (T1 != S8) return launch<C6, T0, T1, GR, SK, TW>(p, two, batch, st); return 2; \
    case 7: return launch<C7, T0, T1, GR, SK, TW>(p, two, batch, st);                        \
    case 8: if (T1 != S8) return launch<C8, T0, T1, GR, SK, TW>(p, two, batch, st); return 2; \
    case 9: if (T1 != S8) return launch<C9, T0, T1, GR, SK, TW>(p, two, batch, st); return 2; \
    default: return 3;                                                                       \
  }

// variant: 0 S4 | 1 S8 | 2 BF16 | 3 S4+BF16 tail | 4 S4+S8 tail | 5 S4 g64 | 6 S4 g128 | 7 S4 skip | 8 S4 skip + BF16 tail
//          9 S8 + S8 tail | 10 S8 + BF16 tail.  two=1 (row-role, problem 1 = S8) for variants 0, 3, 4, 5, 7, 8.
extern "C" int q2_gemm(int variant, int cfg, Params* pp, int two, int batch, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  Params& p = *pp;
#ifdef Q2_FAST
  if (variant == 0) { DISPATCH_CFG(S4, NONE, 0, false, true) }
  if (variant == 1) { DISPATCH_CFG(S8, NONE, 0, false, false) }
  return 9;
#else
  switch (variant) {
    case 0: DISPATCH_CFG(S4, NONE, 0, false, true)
    case 1: DISPATCH_CFG(S8, NONE, 0, false, false)
    case 2: DISPATCH_CFG(BF16, NONE, 0, false, false)
    case 3: DISPATCH_CFG(S4, BF16, 0, false, true)
    case 4: DISPATCH_CFG(S4, S8, 0, false, true)
    case 5: DISPATCH_CFG(S4, NONE, 64, false, true)
    case 6: DISPATCH_CFG(S4, NONE, 128, false, false)
    case 7: DISPATCH_CFG(S4, NONE, 0, true, true)
    case 8: DISPATCH_CFG(S4, BF16, 0, true, true)
    case 9: DISPATCH_CFG(S8, S8, 0, false, false)
    case 10: DISPATCH_CFG(S8, BF16, 0, false, false)
    default: return 4;
  }
#endif
}
extern "C" int q2_sizeof_params() { return (int)sizeof(Params); }
