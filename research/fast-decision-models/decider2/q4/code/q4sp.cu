// Q4 B11: custom 2:4 structured-sparse int8 / int4 GEMMs for sm_86 (A10G) with raw mma.sp / ldmatrix / cp.async (no CUTLASS).
//
// Y[t, n] = sum_k X[t, k] * W[n, k], the weights W 2:4-sparse along K. mma.sp's sparse operand is A, so the kernel computes D = W X^T:
// mma rows = output channels n (A = compressed weights), mma columns = tokens t (B = activation codes, rows K-contiguous).
// Per mma k-step the byte geometry is the same for both types, so one smem pipeline serves both and only the instruction changes:
//   int8  mma.sp m16n8k64 : A = 16 channels x 32 kept bytes (64 logical K), B = 8 tokens x 64 bytes, E = one u32 per lane
//   int4  mma.sp m16n8k128: A = 16 channels x 32 kept bytes (128 logical K, kept in nibble PAIRS), B = 8 tokens x 64 bytes, E = one u32 per lane
// Global layouts (Kb = bytes of one activation row: K for int8, K/2 for int4 packed low nibble first):
//   X  [T, Kb] bytes, row stride ldx bytes
//   Wc [N, Kb/2] bytes: the kept values of each group in increasing index order (q4sp.py compress())
//   E  [ceil(N/16)][Kb/64][32] u32: metadata already in each lane's register order (q4sp.py)
//   sa [T] fp32 per-token scale, sb [N] fp32 per-channel scale
// Epilogues (output staged through smem so stores are 16-byte, row-major Y[t, n]):
//   0 bf16:   y = bf16( fp32( fp32(float(acc) * sa[t]) * sb[n] ) )            (Q2 FORMATS.md section 2)
//   1 SwiGLU: channels interleaved in blocks of 8 (rows 0-7 of every 16-row block = gate j..j+7, rows 8-15 = up j..j+7);
//             g = bf16(f_gate), u = bf16(f_up), m = bf16( bf16( g / (1 + expf(-g)) ) * u ) -> bf16 [T, N/2]
//   2 fp16a:  fp16( alpha * float(acc) )                                        (H2's CUTLASS convention, drop-in for QRT)
//   3 int32:  raw accumulators (tests)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

enum { SP8 = 0, SP4 = 1, D8 = 2, D4 = 3 };   // D8 / D4: dense control with the same main loop (A = full weights)

struct SPP {
  const uint8_t* W; const uint32_t* E; const uint8_t* X;
  const float* sa; const float* sb;
  void* out; long long ldo;            // output row stride in elements
  long long ldx;                       // activation row stride in bytes
  int N, T, Kb, epi; float alpha;
  int ntc, ntt;                        // channel tiles, token tiles
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(uint32_t s, const void* g, int nb) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(nb));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v;
}

template <int T> __device__ __forceinline__ void mma_sp(uint32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[4], uint32_t e);
template <> __device__ __forceinline__ void mma_sp<SP8>(uint32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[4], uint32_t e) {
  asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
}
template <> __device__ __forceinline__ void mma_sp<SP4>(uint32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[4], uint32_t e) {
  asm volatile("mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
}
// dense control: the same 64-byte B k-step as two dense mma (k32 int8 / k64 int4), A = 16 x 64 full bytes (8 regs)
template <int T> __device__ __forceinline__ void mma_dn(uint32_t (&c)[4], const uint32_t (&a)[8], const uint32_t (&b)[4]);
template <> __device__ __forceinline__ void mma_dn<D8>(uint32_t (&c)[4], const uint32_t (&a)[8], const uint32_t (&b)[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[2]), "r"(b[3]));
}
template <> __device__ __forceinline__ void mma_dn<D4>(uint32_t (&c)[4], const uint32_t (&a)[8], const uint32_t (&b)[4]) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[2]), "r"(b[3]));
}

// XOR swizzle of 16-byte chunks so that ldmatrix's 8 rows x 16 bytes hit 8 distinct bank groups; C = chunks per row
template <int C> __device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (C == 2) return row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4);
  else if constexpr (C == 4) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
  else return row * (C * 16) + ((chunk ^ (row & 7)) << 4);
}

template <int T_, int BM_, int BN_, int WGM_, int WGN_, int ST_, int KB_>
struct Cfg {
  static constexpr int TY = T_, BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, ST = ST_, KB = KB_;
  static constexpr bool SPARSE = (TY == SP8 || TY == SP4);
  static constexpr int NTH = WGM * WGN * 32;
  static constexpr int WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
  static constexpr int KS = KB / 64;                     // mma k-steps (64 bytes of B) per stage
  static constexpr int RA = SPARSE ? KB / 2 : KB;        // A smem row bytes
  static constexpr int CA = RA / 16, CB = KB / 16;       // 16-byte chunks per row
  static constexpr int ABYTES = BM * RA, BBYTES = BN * KB, EBYTES = SPARSE ? (BM / 16) * KS * 128 : 0;
  static constexpr int STAGE = ABYTES + BBYTES + EBYTES;
  static constexpr int SMEM_MAIN = ST * STAGE;
  static constexpr int OUTP = BM + 8;                    // staging row pitch (elements, 2-byte)
  static constexpr int SMEM_EPI = BN * OUTP * 2;
  static constexpr int SMEM = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
  static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
  static_assert(KS >= 1, "KB >= 64");
};

template <class C>
__global__ void __launch_bounds__(C::NTH) k_sp(const SPP P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int MT = C::MT, NT = C::NT, KS = C::KS, KB = C::KB, ST = C::ST;
  constexpr bool SPARSE = C::SPARSE;
  constexpr int AR = SPARSE ? 4 : 8;                     // A regs per 16-row block per k-step
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / C::WGN) * C::WM, wn0 = (warp % C::WGN) * C::WN;
  const int b = blockIdx.x;
  const int tt = b % P.ntt, tc = b / P.ntt;              // token tiles fastest: CTAs that share a weight tile run together
  const int n0 = tc * C::BM, t0 = tt * C::BN;
  const int N = P.N, T = P.T, Kb = P.Kb;
  const int KT = Kb / KB;
  const int lda = SPARSE ? Kb / 2 : Kb;                  // weight row bytes
  const int nmb = (N + 15) >> 4;                         // metadata 16-row blocks
  const int ksg = Kb / 64;                               // metadata k-steps per block row

  // ---- per-thread copy descriptors
  constexpr int NA = (C::BM * C::CA + C::NTH - 1) / C::NTH;
  constexpr int NBc = (C::BN * C::CB + C::NTH - 1) / C::NTH;
  constexpr int ECH = C::EBYTES / 16;
  constexpr int NE = SPARSE ? (ECH + C::NTH - 1) / C::NTH : 0;
  const uint8_t* a_g[NA]; int a_so[NA]; int a_nb[NA];
  const uint8_t* b_g[NBc]; int b_so[NBc]; int b_nb[NBc];
  const uint8_t* e_g[NE > 0 ? NE : 1]; int e_so[NE > 0 ? NE : 1]; int e_nb[NE > 0 ? NE : 1];
#pragma unroll
  for (int i = 0; i < NA; ++i) {
    int idx = tid + i * C::NTH; bool in = idx < C::BM * C::CA;
    int r = in ? idx / C::CA : 0, ch = in ? idx % C::CA : 0;
    int gr = n0 + r; bool ok = in && gr < N;
    a_so[i] = in ? swz<C::CA>(r, ch) : -1; a_nb[i] = ok ? 16 : 0;
    a_g[i] = P.W + (long long)(ok ? gr : 0) * lda + ch * 16;
  }
#pragma unroll
  for (int i = 0; i < NBc; ++i) {
    int idx = tid + i * C::NTH; bool in = idx < C::BN * C::CB;
    int r = in ? idx / C::CB : 0, ch = in ? idx % C::CB : 0;
    int gr = t0 + r; bool ok = in && gr < T;
    b_so[i] = in ? C::ABYTES + swz<C::CB>(r, ch) : -1; b_nb[i] = ok ? 16 : 0;
    b_g[i] = P.X + (long long)(ok ? gr : 0) * P.ldx + ch * 16;
  }
  if constexpr (SPARSE) {
#pragma unroll
    for (int i = 0; i < NE; ++i) {
      int idx = tid + i * C::NTH; bool in = idx < ECH;
      constexpr int CPB = KS * 128 / 16;                // chunks per metadata block per stage
      int mb = in ? idx / CPB : 0, c = in ? idx % CPB : 0;
      int gmb = (n0 >> 4) + mb; bool ok = in && gmb < nmb;
      e_so[i] = in ? C::ABYTES + C::BBYTES + idx * 16 : -1; e_nb[i] = ok ? 16 : 0;
      e_g[i] = reinterpret_cast<const uint8_t*>(P.E) + ((long long)(ok ? gmb : 0) * ksg) * 128 + c * 16;
    }
  }
  const uint32_t sbase = smem_u32(smem);
  auto load = [&](int stage, int kt) {
    const uint32_t s = sbase + stage * C::STAGE;
    const int ka = kt * C::RA, kb = kt * KB, ke = kt * KS * 128;
#pragma unroll
    for (int i = 0; i < NA; ++i) if (a_so[i] >= 0) cp16(s + a_so[i], a_g[i] + ka, a_nb[i]);
#pragma unroll
    for (int i = 0; i < NBc; ++i) if (b_so[i] >= 0) cp16(s + b_so[i], b_g[i] + kb, b_nb[i]);
    if constexpr (SPARSE) {
#pragma unroll
      for (int i = 0; i < NE; ++i) if (e_so[i] >= 0) cp16(s + e_so[i], e_g[i] + ke, e_nb[i]);
    }
  };

  // ---- ldmatrix offsets (bytes within a stage); +16 rows (A) / +8 rows (B) keep the swizzle phase
  int a_off[KS], b_off[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    if constexpr (SPARSE) a_off[ks] = swz<C::CA>(wm0 + (lane & 15), ks * 2 + (lane >> 4));
    else a_off[ks] = swz<C::CA>(wm0 + (lane & 15), ks * 4 + (lane >> 4));      // first of two 32-byte halves
    b_off[ks] = C::ABYTES + swz<C::CB>(wn0 + (lane & 7), ks * 4 + (lane >> 3));
  }
  const int e_off = C::ABYTES + C::BBYTES + (wm0 >> 4) * KS * 128 + lane * 4;

  uint32_t acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0u;

  uint32_t af[2][MT][AR], bfr[2][NT][4], ef[2][MT];
  auto ldfrag = [&](int buf, int stage, int ks) {
    const uint32_t s = sbase + stage * C::STAGE;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if constexpr (SPARSE) {
        ldsm4(*reinterpret_cast<uint32_t(*)[4]>(&af[buf][mt][0]), s + a_off[ks] + mt * 16 * C::RA);
        ef[buf][mt] = lds32(s + e_off + (mt * KS + ks) * 128);
      } else {
        ldsm4(*reinterpret_cast<uint32_t(*)[4]>(&af[buf][mt][0]), s + a_off[ks] + mt * 16 * C::RA);
        // second 32-byte half of the 64-byte k-step: chunks +2
        const int row = wm0 + (lane & 15);
        const int off2 = swz<C::CA>(row, ks * 4 + 2 + (lane >> 4));
        ldsm4(*reinterpret_cast<uint32_t(*)[4]>(&af[buf][mt][4]), s + off2 + mt * 16 * C::RA);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) ldsm4(bfr[buf][nt], s + b_off[ks] + nt * 8 * KB);
  };
  auto domma = [&](int buf) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (SPARSE) mma_sp<C::TY>(acc[mt][nt], *reinterpret_cast<const uint32_t(*)[4]>(&af[buf][mt][0]), bfr[buf][nt], ef[buf][mt]);
        else {
          // dense A frag order from two ldsm4: [a0 a1 a2 a3] = bytes 0-31 of the step, [a4..a7] = bytes 32-63; B regs b0,b1 = bytes 0-31, b2,b3 = 32-63
          mma_dn<C::TY>(acc[mt][nt], af[buf][mt], bfr[buf][nt]);
        }
      }
  };

  // ---- main loop (CUTLASS MmaMultistage order, as in q2gemm v2)
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < KT) load(s, s); cp_commit(); }
  cp_wait<ST - 2>();
  __syncthreads();
  ldfrag(0, 0, 0);
  int rs = 0, ws = ST - 1;
  for (int kt = 0; kt < KT; ++kt) {
    { int nk = kt + ST - 1; if (nk < KT) load(ws, nk); }
#pragma unroll
    for (int j = 0; j < KS; ++j) {
      const int cur = j & 1, nxt = cur ^ 1;
      if (j == KS - 1) {
        if constexpr (KS == 1) {
          cp_commit(); cp_wait<ST - 2>(); __syncthreads();
          rs = (rs + 1 == ST) ? 0 : rs + 1; ws = (ws + 1 == ST) ? 0 : ws + 1;
          domma(0);
          ldfrag(0, rs, 0);
        } else {
          ldfrag(nxt, rs, 0);
          domma(cur);
        }
      } else {
        ldfrag(nxt, rs, j + 1);
        domma(cur);
        if (j == KS - 2) {
          cp_commit(); cp_wait<ST - 2>(); __syncthreads();
          rs = (rs + 1 == ST) ? 0 : rs + 1; ws = (ws + 1 == ST) ? 0 : ws + 1;
        }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();

  // ---- epilogue. acc[mt][nt][q]: channel n = n0 + wm0 + mt*16 + (lane>>2) + 8*(q>>1), token t = t0 + wn0 + nt*8 + 2*(lane&3) + (q&1)
  const int epi = P.epi;
  const int g = lane >> 2, tq = lane & 3;
  if (epi == 3) {
    int32_t* o = reinterpret_cast<int32_t*>(P.out);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int n = n0 + wm0 + mt * 16 + g + 8 * (q >> 1), t = t0 + wn0 + nt * 8 + 2 * tq + (q & 1);
          if (n < N && t < T) o[(long long)t * P.ldo + n] = (int)acc[mt][nt][q];
        }
    return;
  }
  uint16_t* stg = reinterpret_cast<uint16_t*>(smem);
  constexpr int OP = C::OUTP;
  float sbv[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) { int n = n0 + wm0 + mt * 16 + g + 8 * h; sbv[mt][h] = (epi != 2 && n < N) ? P.sb[n] : 0.f; }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int tl = wn0 + nt * 8 + 2 * tq + e, t = t0 + tl;
      const float sav = (epi != 2 && t < T) ? P.sa[t] : 0.f;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int cl = wm0 + mt * 16 + g;
        if (epi == 2) {
          stg[tl * OP + cl] = __half_as_ushort(__float2half_rn(P.alpha * (float)(int)acc[mt][nt][e]));
          stg[tl * OP + cl + 8] = __half_as_ushort(__float2half_rn(P.alpha * (float)(int)acc[mt][nt][2 + e]));
        } else {
          const float f0 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][e], sav), sbv[mt][0]);
          const float f1 = __fmul_rn(__fmul_rn((float)(int)acc[mt][nt][2 + e], sav), sbv[mt][1]);
          if (epi == 0) {
            stg[tl * OP + cl] = __bfloat16_as_ushort(__float2bfloat16_rn(f0));
            stg[tl * OP + cl + 8] = __bfloat16_as_ushort(__float2bfloat16_rn(f1));
          } else {   // SwiGLU: f0 = gate (row g), f1 = up (row g+8) of output channel (wm0 + mt*16)/2 + g
            const float gg = __bfloat162float(__float2bfloat16_rn(f0)), uu = __bfloat162float(__float2bfloat16_rn(f1));
            const float sl = __bfloat162float(__float2bfloat16_rn(gg / (1.0f + expf(-gg))));
            stg[tl * OP + ((wm0 + mt * 16) >> 1) + g] = __bfloat16_as_ushort(__float2bfloat16_rn(sl * uu));
          }
        }
      }
    }
  }
  __syncthreads();
  // cooperative 16-byte stores: row tl of the staging tile -> out[t0 + tl][oc0 .. oc0 + W)
  const int W = (epi == 1) ? C::BM / 2 : C::BM;
  const int Nout = (epi == 1) ? N / 2 : N;
  const int oc0 = (epi == 1) ? n0 / 2 : n0;
  const int VPR = W / 8;
  uint16_t* out = reinterpret_cast<uint16_t*>(P.out);
  for (int i = tid; i < C::BN * VPR; i += C::NTH) {
    const int tl = i / VPR, v = i % VPR;
    const int t = t0 + tl, c = oc0 + v * 8;
    if (t < T && c < Nout) {
      const uint4 val = *reinterpret_cast<const uint4*>(stg + tl * OP + v * 8);
      *reinterpret_cast<uint4*>(out + (long long)t * P.ldo + c) = val;
    }
  }
}

template <class C>
static int launch(SPP p, cudaStream_t st) {
  if (p.Kb % C::KB) return 7;
  if (C::SMEM > 99 * 1024) return 8;
  p.ntc = (p.N + C::BM - 1) / C::BM; p.ntt = (p.T + C::BN - 1) / C::BN;
  long long tot = (long long)p.ntc * p.ntt;
  if (tot == 0) return 0;
  auto k = k_sp<C>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  k<<<(unsigned)tot, C::NTH, C::SMEM, st>>>(p);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}

//                      BM   BN  WGM WGN ST  KB
#define CFGS(X, TY) \
  X(0,  TY, 128, 64,  2, 2, 4, 128)   /* 4 warps 64x32 */ \
  X(1,  TY, 128, 128, 2, 4, 3, 128)   /* 8 warps 64x32 */ \
  X(2,  TY, 256, 128, 4, 2, 3, 64)    /* 8 warps 64x64 */ \
  X(3,  TY, 64,  64,  2, 2, 4, 128)   /* 4 warps 32x32 */ \
  X(4,  TY, 128, 32,  4, 1, 4, 128)   /* 4 warps 32x32 */ \
  X(5,  TY, 256, 64,  4, 2, 3, 128)   /* 8 warps 64x32 */ \
  X(6,  TY, 128, 128, 2, 2, 3, 128)   /* 4 warps 64x64 */ \
  X(7,  TY, 256, 128, 4, 4, 3, 64)    /* 16 warps 64x32 */ \
  X(8,  TY, 64,  128, 2, 2, 4, 128)   /* 4 warps 32x64 */ \
  X(9,  TY, 128, 64,  2, 2, 2, 256)   /* 4 warps 64x32, 2 x 256-byte stages */ \
  X(10, TY, 256, 64,  4, 2, 4, 64)    /* 8 warps 64x32, 4 x 64-byte stages */ \
  X(11, TY, 128, 128, 4, 2, 4, 64)    /* 8 warps 32x64 */ \
  X(12, TY, 256, 128, 2, 4, 3, 64)    /* 8 warps 128x32: B fragments reused by 8 channel blocks */ \
  X(13, TY, 256, 64,  2, 2, 4, 64)    /* 4 warps 128x32 */ \
  X(14, TY, 128, 128, 1, 4, 4, 64)    /* 4 warps 128x32 */ \
  X(15, TY, 128, 256, 2, 4, 2, 64)    /* 8 warps 64x64, 256 tokens per CTA */

#define CASE(id, TY, a, b, c, d, e, f) case id: return launch<Cfg<TY, a, b, c, d, e, f>>(p, st);

extern "C" int q4sp_gemm(int type, int cfg, const void* W, const void* E, const void* X, const float* sa, const float* sb, void* out,
                         long long ldo, long long ldx, int N, int T, int Kb, int epi, float alpha, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  SPP p{};
  p.W = (const uint8_t*)W; p.E = (const uint32_t*)E; p.X = (const uint8_t*)X; p.sa = sa; p.sb = sb;
  p.out = out; p.ldo = ldo; p.ldx = ldx; p.N = N; p.T = T; p.Kb = Kb; p.epi = epi; p.alpha = alpha;
  switch (type) {
    case SP8: switch (cfg) { CFGS(CASE, SP8) default: return 3; }
    case SP4: switch (cfg) { CFGS(CASE, SP4) default: return 3; }
    case D8:  switch (cfg) { CFGS(CASE, D8) default: return 3; }
    case D4:  switch (cfg) { CFGS(CASE, D4) default: return 3; }
    default: return 4;
  }
}

// ---- probe: one mma.sp with per-lane registers supplied by the host (to pin down the metadata layout empirically)
__global__ void k_probe(const uint32_t* A, const uint32_t* B, const uint32_t* E, int* D, int type) {
  int l = threadIdx.x;
  uint32_t a[4] = {A[l * 4], A[l * 4 + 1], A[l * 4 + 2], A[l * 4 + 3]};
  uint32_t b[4] = {B[l * 4], B[l * 4 + 1], B[l * 4 + 2], B[l * 4 + 3]};
  uint32_t c[4] = {0u, 0u, 0u, 0u};
  if (type == 0) mma_sp<SP8>(c, a, b, E[l]); else mma_sp<SP4>(c, a, b, E[l]);
  for (int q = 0; q < 4; ++q) D[l * 4 + q] = (int)c[q];
}
extern "C" int q4sp_probe(const void* A, const void* B, const void* E, void* D, int type) {
  k_probe<<<1, 32>>>((const uint32_t*)A, (const uint32_t*)B, (const uint32_t*)E, (int*)D, type);
  cudaError_t e = cudaDeviceSynchronize();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}
