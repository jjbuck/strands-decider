// Q4: one-level Strassen on sm_86 tensor cores, fused in one kernel (int8 on 7-bit codes, int4 on 3-bit codes, bf16), plus the same main
// loop run as a dense GEMM (mode 1) so the speedup is measured against an identical kernel.
//
// y = X W^T, X [M, K] (rows t), W [N, K] (channels n). Blocks: X = [X11 X12; X21 X22] (row halves Mh, K halves Kh), W likewise
// ([W11 W12; W21 W22]: channel halves Nh, K halves). In Strassen's notation C = A B with A = X, B = W^T, so B11 = W11^T, B12 = W21^T,
// B21 = W12^T, B22 = W22^T. Strassen's ORIGINAL form (every operand sum has two terms, so one extra bit suffices):
//   M1 = (A11 + A22)(B11 + B22)   M2 = (A21 + A22) B11   M3 = A11 (B12 - B22)   M4 = A22 (B21 - B11)
//   M5 = (A11 + A12) B22          M6 = (A21 - A11)(B11 + B12)                  M7 = (A12 - A22)(B21 + B22)
//   C11 = M1 + M4 - M5 + M7       C12 = M3 + M5       C21 = M2 + M4       C22 = M1 - M2 + M3 + M6
// (Winograd's variant has 4-term operand sums S4 = A11+A12-A21-A22, T4 likewise, i.e. two extra bits, so it is not used for integers.)
// Weight operands (B side) are formed offline; activation operands are formed by the producer (q4st_sums, or the quantize kernel).
// Each CTA owns one (Mh/BM x Nh/BN) tile position and runs the 7 products back to back through ONE continuous cp.async pipeline, in the
// order M4, M2, M1, M7, M5, M3, M6, which needs only three accumulator sets R0..R2:
//   M4 -> R0;  R1 = R0                    (R0 = C21, R1 = C11)
//   M2 -> R2;  R0 += R2 -> store C21; R0 = -R2              (R0 = C22)
//   M1 -> R2;  R1 += R2;  R0 += R2
//   M7 -> R1 directly
//   M5 -> R2;  R1 -= R2 -> store C11; R1 = R2               (R1 = C12)
//   M3 -> R2;  R1 += R2 -> store C12; R0 += R2
//   M6 -> R0 directly -> store C22
// Integer products accumulate exactly in int32, so integer Strassen equals the dense GEMM on the same codes bit for bit.
// Epilogue (FORMATS.md sec 2 / sec 9): f = fp32(fp32(float(acc) * sa[t mod Mh]) * sb[n mod Nh]) -> bf16 (sa/sb are the PAIR-shared scales);
// bf16 type: f = acc (fp32) -> bf16.  epi 1 = raw int32 / fp32 store (tests, two-level outer combine).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

enum { S4 = 1, S8 = 2, BF16 = 3 };

struct StP {
  const uint8_t* A[7]; long long lda[7];   // activation operands in SEQUENCE order (M4, M2, M1, M7, M5, M3, M6), rows Mh, row stride bytes
  const uint8_t* B[7]; long long ldb;      // weight operands in sequence order, rows Nh
  const float* sa; const float* sb;        // pair-shared scales [Mh], [Nh] (int types)
  void* out; long long ldo;                // output [M, N] (elements), C_ij at rows i*Mh, cols j*Nh
  int Mh, Nh, M, N; int kt;                // kt = K tiles per product
  int epi, mode;                           // mode 0 Strassen (7 products), 1 dense (one product A[0] x B[0], output block = whole [Mh, Nh])
  int mtiles, ntiles;
  long long bsA, bsB, bsO;                 // gridDim.z batch strides (bytes) for two-level outer products
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

template <int KB> __device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (KB == 64) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
  else return row * 128 + ((chunk ^ (row & 7)) << 4);
}

template <int T_, int BM_, int BN_, int WGM_, int WGN_, int ST_, int KB_>
struct Cfg {
  static constexpr int TY = T_, BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, ST = ST_, KB = KB_;
  static constexpr int NTH = WGM * WGN * 32, WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
  static constexpr int CH = KB / 16, KSTEPS = KB / 32;
  static constexpr int ABYTES = BM * KB, BBYTES = BN * KB, STAGE = ABYTES + BBYTES, SMEM = ST * STAGE;
  static_assert(NT % 2 == 0, "NT even");
};

template <class C>
__global__ void __launch_bounds__(C::NTH) k_st(const StP P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int MT = C::MT, NT = C::NT, KS = C::KSTEPS, KB = C::KB, ST = C::ST, TY = C::TY;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / C::WGN) * C::WM, wn0 = (warp % C::WGN) * C::WN;
  const int b = blockIdx.x, z = blockIdx.z;
  const int tm = b % P.mtiles, tn = b / P.mtiles;
  const int m0 = tm * C::BM, n0 = tn * C::BN;
  const int Mh = P.Mh, Nh = P.Nh;
  const int NP = P.mode == 0 ? 7 : 1;
  const int KT = P.kt, TOT = NP * KT;

  // copy descriptors (smem offsets fixed; global row offsets per product computed at load time)
  constexpr int NA = (C::BM * C::CH) / C::NTH, NB = (C::BN * C::CH) / C::NTH;
  static_assert((C::BM * C::CH) % C::NTH == 0 && (C::BN * C::CH) % C::NTH == 0, "loader");
  int a_so[NA], a_r[NA], a_c[NA], b_so[NB], b_r[NB], b_c[NB];
#pragma unroll
  for (int i = 0; i < NA; ++i) { int idx = tid + i * C::NTH; a_r[i] = idx / C::CH; a_c[i] = idx % C::CH; a_so[i] = swz<KB>(a_r[i], a_c[i]); }
#pragma unroll
  for (int i = 0; i < NB; ++i) { int idx = tid + i * C::NTH; b_r[i] = idx / C::CH; b_c[i] = idx % C::CH; b_so[i] = C::ABYTES + swz<KB>(b_r[i], b_c[i]); }
  const uint32_t sbase = smem_u32(smem);
  auto load = [&](int stage, int g) {
    const int p = g / KT, kt = g - p * KT;
    const uint8_t* Ab = P.A[p] + (long long)z * P.bsA; const long long lda = P.lda[p];
    const uint8_t* Bb = P.B[p] + (long long)z * P.bsB; const long long ldb = P.ldb;
    const uint32_t s = sbase + stage * C::STAGE;
    const int ko = kt * KB;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      int gr = m0 + a_r[i]; bool ok = gr < Mh;
      cp16(s + a_so[i], Ab + (long long)(ok ? gr : 0) * lda + ko + a_c[i] * 16, ok ? 16 : 0);
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      int gr = n0 + b_r[i]; bool ok = gr < Nh;
      cp16(s + b_so[i], Bb + (long long)(ok ? gr : 0) * ldb + ko + b_c[i] * 16, ok ? 16 : 0);
    }
  };
  const int a_row = wm0 + (lane & 15), b_row = wn0 + (lane & 7) + ((lane >> 4) << 3);
  int a_off[KS], b_off[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) { a_off[ks] = swz<KB>(a_row, ks * 2 + (lane >> 4)); b_off[ks] = C::ABYTES + swz<KB>(b_row, ks * 2 + ((lane >> 3) & 1)); }

  uint32_t R0[MT][NT][4], R1[MT][NT][4], R2[MT][NT][4];
  auto zero = [&](uint32_t (&R)[MT][NT][4]) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) R[i][j][q] = 0u;
  };
  // R op= S  (op: 0 copy, 1 add, 2 sub, 3 negate-copy); integer or fp32 per type
  auto comb = [&](uint32_t (&R)[MT][NT][4], const uint32_t (&S)[MT][NT][4], int op) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if constexpr (TY == BF16) {
            float r = __uint_as_float(R[i][j][q]), s = __uint_as_float(S[i][j][q]);
            float v = op == 0 ? s : op == 1 ? __fadd_rn(r, s) : op == 2 ? __fsub_rn(r, s) : -s;
            R[i][j][q] = __float_as_uint(v);
          } else {
            int r = (int)R[i][j][q], s = (int)S[i][j][q];
            int v = op == 0 ? s : op == 1 ? r + s : op == 2 ? r - s : -s;
            R[i][j][q] = (uint32_t)v;
          }
        }
  };
  // store block (bi, bj) of the output from R
  auto store = [&](const uint32_t (&R)[MT][NT][4], int bi, int bj) {
    const int epi = P.epi;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int r = m0 + wm0 + mt * 16 + (lane >> 2) + hi * 8;
        if (r >= Mh) continue;
        const long long orow = (long long)bi * Mh + r;
        if (orow >= P.M) continue;
        float sa = 1.f;
        if constexpr (TY != BF16) sa = P.sa[r];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int c = n0 + wn0 + nt * 8 + 2 * (lane & 3);
          if (c >= Nh) continue;
          const long long ocol = (long long)bj * Nh + c;
          uint32_t u0 = R[mt][nt][hi * 2], u1 = R[mt][nt][hi * 2 + 1];
          if (epi == 1) {
            *reinterpret_cast<uint2*>(reinterpret_cast<uint32_t*>(P.out) + (long long)z * (P.bsO / 4) + orow * P.ldo + ocol) = make_uint2(u0, u1);
            continue;
          }
          float f0, f1;
          if constexpr (TY == BF16) { f0 = __uint_as_float(u0); f1 = __uint_as_float(u1); }
          else {
            f0 = __fmul_rn(__fmul_rn((float)(int)u0, sa), P.sb[c]);
            f1 = __fmul_rn(__fmul_rn((float)(int)u1, sa), P.sb[c + 1]);
          }
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(P.out) + orow * P.ldo + ocol) = __floats2bfloat162_rn(f0, f1);
        }
      }
  };

  uint32_t af[2][MT][4], bfr[2][NT / 2][4];
  auto ldfrag = [&](int buf, int stage, int ks) {
    const uint32_t s = sbase + stage * C::STAGE;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) ldsm4(af[buf][mt], s + a_off[ks] + mt * 16 * KB);
#pragma unroll
    for (int p = 0; p < NT / 2; ++p) ldsm4(bfr[buf][p], s + b_off[ks] + p * 16 * KB);
  };
  auto domma = [&](uint32_t (&R)[MT][NT][4], int buf) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) mma<TY>(R[mt][nt], af[buf][mt], bfr[buf][nt >> 1][(nt & 1) * 2], bfr[buf][nt >> 1][(nt & 1) * 2 + 1]);
  };

  // ---- pipeline state shared across products
  int rs = 0, ws = ST - 1;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < TOT) load(s, s); cp_commit(); }
  cp_wait<ST - 2>();
  __syncthreads();
  ldfrag(0, 0, 0);
  int g = 0;   // global tile counter
  // one product: KT tiles accumulated into R
  auto product = [&](uint32_t (&R)[MT][NT][4]) {
    for (int kt = 0; kt < KT; ++kt, ++g) {
      { int nk = g + ST - 1; if (nk < TOT) load(ws, nk); }
#pragma unroll
      for (int j = 0; j < KS; ++j) {
        const int cur = j & 1, nxt = cur ^ 1;
        if (j == KS - 1) {
          ldfrag(nxt, rs, 0);
          domma(R, cur);
        } else {
          ldfrag(nxt, rs, j + 1);
          domma(R, cur);
          if (j == KS - 2) {
            cp_commit(); cp_wait<ST - 2>(); __syncthreads();
            rs = (rs + 1 == ST) ? 0 : rs + 1; ws = (ws + 1 == ST) ? 0 : ws + 1;
          }
        }
      }
    }
  };
  static_assert(KS % 2 == 0, "fragment buffer parity must restart at 0 for every tile");

  zero(R0);
  if (P.mode == 1) {
    product(R0);
    cp_wait<0>();
    store(R0, 0, 0);
    return;
  }
  product(R0);                                     // M4
  comb(R1, R0, 0);                                 // R1 = C11 = M4, R0 = C21 = M4
  zero(R2); product(R2);                           // M2
  comb(R0, R2, 1); store(R0, 1, 0); comb(R0, R2, 3);   // C21 = M2 + M4; R0 = C22 = -M2
  zero(R2); product(R2);                           // M1
  comb(R1, R2, 1); comb(R0, R2, 1);
  product(R1);                                     // M7 into C11
  zero(R2); product(R2);                           // M5
  comb(R1, R2, 2); store(R1, 0, 0); comb(R1, R2, 0);   // C11 = M1 + M4 + M7 - M5; R1 = C12 = M5
  zero(R2); product(R2);                           // M3
  comb(R1, R2, 1); store(R1, 0, 1); comb(R0, R2, 1);   // C12 = M3 + M5; C22 += M3
  product(R0);                                     // M6 into C22
  cp_wait<0>();
  store(R0, 1, 1);
}

template <class C>
static int launch(StP p, int batch, cudaStream_t st) {
  p.mtiles = (p.Mh + C::BM - 1) / C::BM; p.ntiles = (p.Nh + C::BN - 1) / C::BN;
  long long tot = (long long)p.mtiles * p.ntiles;
  if (tot == 0) return 0;
  auto k = k_st<C>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  k<<<dim3((unsigned)tot, 1, (unsigned)batch), C::NTH, C::SMEM, st>>>(p);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}

//                       BM   BN WGM WGN ST  KB      (all warp tiles 32 x 32: three accumulator sets = 96 registers)
#define SCFG(X, TY) \
  X(0, TY, 64, 64, 2, 2, 4, 128) X(1, TY, 64, 128, 2, 4, 4, 128) X(2, TY, 128, 64, 4, 2, 4, 128) X(3, TY, 32, 256, 1, 8, 3, 128) \
  X(4, TY, 32, 128, 1, 4, 4, 128) X(5, TY, 64, 64, 2, 2, 6, 128) X(6, TY, 64, 128, 2, 4, 3, 128) X(7, TY, 128, 64, 4, 2, 3, 128)
#define SCASE(id, TY, a, b, c, d, e, f) case id: return launch<Cfg<TY, a, b, c, d, e, f>>(p, batch, st);

extern "C" int q4st_run(int type, int cfg, StP* pp, int batch, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  StP& p = *pp;
  switch (type) {
    case S4: switch (cfg) { SCFG(SCASE, S4) default: return 3; }
    case S8: switch (cfg) { SCFG(SCASE, S8) default: return 3; }
    case BF16: switch (cfg) { SCFG(SCASE, BF16) default: return 3; }
    default: return 4;
  }
}
extern "C" int q4st_sizeof() { return (int)sizeof(StP); }

// ---- producer: the five activation sums of one level from the block views of X (int8 codes: byte-wise SIMD; int4: nibble SWAR; bf16: RN adds)
// S[0] = A11 + A22, S[1] = A21 + A22, S[2] = A11 + A12, S[3] = A21 - A11, S[4] = A12 - A22; each [Mh, Kbh] contiguous. Rows >= M are zero.
__device__ __forceinline__ uint32_t add4s(uint32_t a, uint32_t b) {   // packed int4 add, wraps mod 16 per nibble
  uint32_t s = (a & 0x77777777u) + (b & 0x77777777u);
  return s ^ ((a ^ b) & 0x88888888u);
}
__device__ __forceinline__ uint32_t neg4s(uint32_t a) { return add4s(~a, 0x11111111u); }
template <int TY>
__global__ void k_sums(const uint8_t* X, long long ldx, int M, int Mh, int Kbh, uint8_t* S) {
  const long long nw = (long long)Mh * (Kbh / 4);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < nw; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / (Kbh / 4)), c = (int)(i % (Kbh / 4)) * 4;
    const bool lo_ok = (r + Mh) < M;
    uint32_t a11 = *reinterpret_cast<const uint32_t*>(X + (long long)r * ldx + c);
    uint32_t a12 = *reinterpret_cast<const uint32_t*>(X + (long long)r * ldx + Kbh + c);
    uint32_t a21 = lo_ok ? *reinterpret_cast<const uint32_t*>(X + (long long)(r + Mh) * ldx + c) : 0u;
    uint32_t a22 = lo_ok ? *reinterpret_cast<const uint32_t*>(X + (long long)(r + Mh) * ldx + Kbh + c) : 0u;
    uint32_t s0, s1, s2, s3, s4;
    if constexpr (TY == S8) { s0 = __vadd4(a11, a22); s1 = __vadd4(a21, a22); s2 = __vadd4(a11, a12); s3 = __vsub4(a21, a11); s4 = __vsub4(a12, a22); }
    else if constexpr (TY == S4) { s0 = add4s(a11, a22); s1 = add4s(a21, a22); s2 = add4s(a11, a12); s3 = add4s(a21, neg4s(a11)); s4 = add4s(a12, neg4s(a22)); }
    else {
      auto h2 = [](uint32_t u) { return *reinterpret_cast<__nv_bfloat162*>(&u); };
      auto u2 = [](__nv_bfloat162 h) { return *reinterpret_cast<uint32_t*>(&h); };
      s0 = u2(__hadd2(h2(a11), h2(a22))); s1 = u2(__hadd2(h2(a21), h2(a22))); s2 = u2(__hadd2(h2(a11), h2(a12)));
      s3 = u2(__hsub2(h2(a21), h2(a11))); s4 = u2(__hsub2(h2(a12), h2(a22)));
    }
    const long long blk = (long long)Mh * Kbh, o = (long long)r * Kbh + c;
    *reinterpret_cast<uint32_t*>(S + o) = s0; *reinterpret_cast<uint32_t*>(S + blk + o) = s1; *reinterpret_cast<uint32_t*>(S + 2 * blk + o) = s2;
    *reinterpret_cast<uint32_t*>(S + 3 * blk + o) = s3; *reinterpret_cast<uint32_t*>(S + 4 * blk + o) = s4;
  }
}
extern "C" int q4st_sums(int type, const void* X, long long ldx, int M, int Mh, int Kbh, void* S, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  int nb = 80 * 8;
  if (type == S8) k_sums<S8><<<nb, 256, 0, st>>>((const uint8_t*)X, ldx, M, Mh, Kbh, (uint8_t*)S);
  else if (type == S4) k_sums<S4><<<nb, 256, 0, st>>>((const uint8_t*)X, ldx, M, Mh, Kbh, (uint8_t*)S);
  else k_sums<BF16><<<nb, 256, 0, st>>>((const uint8_t*)X, ldx, M, Mh, Kbh, (uint8_t*)S);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}

// ---- two-level outer combination: Y blocks from 7 outer products P[7][Mh][Nh] (int32 / fp32 raw), dequantized
template <int TY>
__global__ void k_comb(const uint32_t* Pp, int Mh, int Nh, const float* sa, const float* sb, __nv_bfloat16* Y, long long ldo) {
  const long long n = (long long)Mh * Nh;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / Nh), c = (int)(i % Nh);
    // sequence order of products in P: M4, M2, M1, M7, M5, M3, M6 -> indices 0..6
    float c11, c12, c21, c22;
    if constexpr (TY == BF16) {
      float v[7];
#pragma unroll
      for (int k = 0; k < 7; ++k) v[k] = __uint_as_float(Pp[k * n + i]);
      const float m4 = v[0], m2 = v[1], m1 = v[2], m7 = v[3], m5 = v[4], m3 = v[5], m6 = v[6];
      c11 = m1 + m4 - m5 + m7; c12 = m3 + m5; c21 = m2 + m4; c22 = m1 - m2 + m3 + m6;
    } else {
      int w[7];
#pragma unroll
      for (int k = 0; k < 7; ++k) w[k] = (int)Pp[k * n + i];
      const int m4 = w[0], m2 = w[1], m1 = w[2], m7 = w[3], m5 = w[4], m3 = w[5], m6 = w[6];
      const float s = sa[r], t = sb[c];
      c11 = __fmul_rn(__fmul_rn((float)(m1 + m4 - m5 + m7), s), t); c12 = __fmul_rn(__fmul_rn((float)(m3 + m5), s), t);
      c21 = __fmul_rn(__fmul_rn((float)(m2 + m4), s), t); c22 = __fmul_rn(__fmul_rn((float)(m1 - m2 + m3 + m6), s), t);
    }
    Y[(long long)r * ldo + c] = __float2bfloat16_rn(c11); Y[(long long)r * ldo + Nh + c] = __float2bfloat16_rn(c12);
    Y[(long long)(r + Mh) * ldo + c] = __float2bfloat16_rn(c21); Y[(long long)(r + Mh) * ldo + Nh + c] = __float2bfloat16_rn(c22);
  }
}
extern "C" int q4st_comb(int type, const void* Pp, int Mh, int Nh, const float* sa, const float* sb, void* Y, long long ldo, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (type == BF16) k_comb<BF16><<<80 * 8, 256, 0, st>>>((const uint32_t*)Pp, Mh, Nh, sa, sb, (__nv_bfloat16*)Y, ldo);
  else k_comb<S8><<<80 * 8, 256, 0, st>>>((const uint32_t*)Pp, Mh, Nh, sa, sb, (__nv_bfloat16*)Y, ldo);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}
