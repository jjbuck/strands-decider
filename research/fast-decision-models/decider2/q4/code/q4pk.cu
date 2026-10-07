// Q4 B14 prototype: a persistent W8A8 kernel that runs a whole forward pass's GEMM sequence (96 GEMMs) for a short decision (M <= 512 rows)
// in ONE launch. CTAs stay resident (grid = SMs x occupancy), take output tiles of GEMM g round-robin, then meet at a grid-wide barrier
// before GEMM g+1 (the barrier stands in for the data dependency through the glue). Before waiting at the barrier each CTA prefetches into L2
// the first K tiles of the weight slices it will read in GEMM g+1, so DRAM keeps streaming weights across the dependency.
// Activations (<= 512 x 6144 int8 = 3 MB) and outputs live in L2 (6 MB on the A10G): every weight byte is read from DRAM once per GEMM.
// Tile: BM x BN output tile, 128-byte K stages, 4 warps (2 x 2), cp.async multistage, int8 mma m16n8k32, fp16(alpha*acc) epilogue
// (H2's convention). Optional split-K (SK > 1): partial sums are reduced with int32 atomics into a zeroed int32 buffer.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct GDesc {
  const int8_t* A; const int8_t* W; void* out;
  int M, N, K, lda, ldw, ldo;
  float alpha; int sk;            // split-K factor (1 = none: fp16 output; > 1: int32 atomics into out)
  int tm, tn, items;              // filled on the host
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
__device__ __forceinline__ void mma8(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
__device__ __forceinline__ int swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void grid_barrier(unsigned* cnt, volatile unsigned* gen, unsigned nblk) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned my = *gen;
    __threadfence();
    if (atomicAdd(cnt, 1u) == nblk - 1) { atomicExch(cnt, 0u); __threadfence(); atomicAdd((unsigned*)gen, 1u); }
    else { while (*gen == my) { __nanosleep(32); } }
    __threadfence();
  }
  __syncthreads();
}

template <int BM, int BN, int ST>
struct PK {
  static constexpr int WGM = 2, WGN = 2, NTH = 128, WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8, KB = 128, CH = 8, KS = 4;
  static constexpr int ABYTES = BM * KB, BBYTES = BN * KB, STAGE = ABYTES + BBYTES, SMEM = ST * STAGE;
};

template <int BM, int BN, int ST>
__device__ void tile(const GDesc& G, int item, uint8_t* smem) {
  using C = PK<BM, BN, ST>;
  constexpr int MT = C::MT, NT = C::NT, KS = C::KS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / C::WGN) * C::WM, wn0 = (warp % C::WGN) * C::WN;
  // item -> (m tile fastest, then k split, then n tile): CTAs that share a weight slice run back to back
  const int tm = item % G.tm; const int r1 = item / G.tm; const int ks_id = r1 % G.sk; const int tn = r1 / G.sk;
  const int m0 = tm * BM, n0 = tn * BN;
  const int KT_all = G.K / C::KB, KT = KT_all / G.sk, kt0 = ks_id * KT;
  constexpr int NA = (BM * C::CH) / C::NTH, NB = (BN * C::CH) / C::NTH;
  const uint32_t sbase = smem_u32(smem);
  auto load = [&](int stage, int kt) {
    const uint32_t s = sbase + stage * C::STAGE; const int ko = (kt0 + kt) * C::KB;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      int idx = tid + i * C::NTH, r = idx / C::CH, ch = idx % C::CH, gr = m0 + r; bool ok = gr < G.M;
      cp16(s + swz(r, ch), G.A + (long long)(ok ? gr : 0) * G.lda + ko + ch * 16, ok ? 16 : 0);
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      int idx = tid + i * C::NTH, r = idx / C::CH, ch = idx % C::CH, gr = n0 + r; bool ok = gr < G.N;
      cp16(s + C::ABYTES + swz(r, ch), G.W + (long long)(ok ? gr : 0) * G.ldw + ko + ch * 16, ok ? 16 : 0);
    }
  };
  int a_off[KS], b_off[KS];
#pragma unroll
  for (int k = 0; k < KS; ++k) {
    a_off[k] = swz(wm0 + (lane & 15), k * 2 + (lane >> 4));
    b_off[k] = C::ABYTES + swz(wn0 + (lane & 7) + ((lane >> 4) << 3), k * 2 + ((lane >> 3) & 1));
  }
  uint32_t acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0u;
  uint32_t af[2][MT][4], bf[2][NT / 2][4];
  auto ldfrag = [&](int buf, int stage, int k) {
    const uint32_t s = sbase + stage * C::STAGE;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) ldsm4(af[buf][mt], s + a_off[k] + mt * 16 * C::KB);
#pragma unroll
    for (int p = 0; p < NT / 2; ++p) ldsm4(bf[buf][p], s + b_off[k] + p * 16 * C::KB);
  };
  auto domma = [&](int buf) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) mma8(acc[mt][nt], af[buf][mt], bf[buf][nt >> 1][(nt & 1) * 2], bf[buf][nt >> 1][(nt & 1) * 2 + 1]);
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < KT) load(s, s); cp_commit(); }
  cp_wait<ST - 2>(); __syncthreads();
  ldfrag(0, 0, 0);
  int rs = 0, ws = ST - 1;
  for (int kt = 0; kt < KT; ++kt) {
    { int nk = kt + ST - 1; if (nk < KT) load(ws, nk); }
#pragma unroll
    for (int j = 0; j < KS; ++j) {
      const int cur = j & 1, nxt = cur ^ 1;
      if (j == KS - 1) { ldfrag(nxt, rs, 0); domma(cur); }
      else {
        ldfrag(nxt, rs, j + 1); domma(cur);
        if (j == KS - 2) { cp_commit(); cp_wait<ST - 2>(); __syncthreads(); rs = (rs + 1 == ST) ? 0 : rs + 1; ws = (ws + 1 == ST) ? 0 : ws + 1; }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();      // the next tile reuses the stages
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
      const int r = m0 + wm0 + mt * 16 + (lane >> 2) + hi * 8;
      if (r >= G.M) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int c = n0 + wn0 + nt * 8 + 2 * (lane & 3);
        if (c >= G.N) continue;
        const int v0 = (int)acc[mt][nt][hi * 2], v1 = (int)acc[mt][nt][hi * 2 + 1];
        if (G.sk == 1) {
          *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(G.out) + (long long)r * G.ldo + c) = __floats2half2_rn(G.alpha * (float)v0, G.alpha * (float)v1);
        } else {
          int* o = reinterpret_cast<int*>(G.out) + (long long)r * G.ldo + c;
          atomicAdd(o, v0); atomicAdd(o + 1, v1);
        }
      }
    }
}

template <int BM, int BN, int ST>
__device__ void prefetch_next(const GDesc& G, int item, int ktiles) {
  // L2 prefetch of the first `ktiles` 128-byte K stages of this item's weight slice (BN rows)
  const int tm = item % G.tm; const int r1 = item / G.tm; const int ks_id = r1 % G.sk; const int tn = r1 / G.sk;
  const int n0 = tn * BN; const int KT = (G.K / 128) / G.sk; const int kt0 = ks_id * KT;
  const int nk = ktiles < KT ? ktiles : KT;
  for (int i = threadIdx.x; i < BN * nk; i += blockDim.x) {
    const int r = i % BN, k = i / BN, gr = n0 + r;
    if (gr < G.N) pf_l2(G.W + (long long)gr * G.ldw + (kt0 + k) * 128);
  }
  (void)tm;
}

template <int BM, int BN, int ST>
__global__ void __launch_bounds__(128) k_pk(const GDesc* __restrict__ prog, int ng, unsigned* bar, int pf) {
  extern __shared__ __align__(128) uint8_t smem[];
  const unsigned nblk = gridDim.x;
  for (int g = 0; g < ng; ++g) {
    const GDesc G = prog[g];
    for (int it = blockIdx.x; it < G.items; it += nblk) tile<BM, BN, ST>(G, it, smem);
    if (g + 1 < ng) {
      if (pf > 0) {
        const GDesc Gn = prog[g + 1];
        for (int it = blockIdx.x; it < Gn.items && it < (int)(blockIdx.x + 2 * nblk); it += nblk) prefetch_next<BM, BN, ST>(Gn, it, pf);
      }
      grid_barrier(bar, bar + 1, nblk);
    }
  }
}

template <int BM, int BN, int ST>
static int run_pk(GDesc* hprog, GDesc* dprog, int ng, unsigned* bar, int pf, int nblk, cudaStream_t st) {
  using C = PK<BM, BN, ST>;
  for (int g = 0; g < ng; ++g) {
    GDesc& G = hprog[g];
    if ((G.K / 128) % G.sk) return 7;
    G.tm = (G.M + BM - 1) / BM; G.tn = (G.N + BN - 1) / BN; G.items = G.tm * G.tn * G.sk;
  }
  cudaMemcpyAsync(dprog, hprog, sizeof(GDesc) * ng, cudaMemcpyHostToDevice, st);
  auto k = k_pk<BM, BN, ST>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, 128, C::SMEM);
  int dev; cudaGetDevice(&dev); int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int maxb = occ * sms;
  if (nblk <= 0 || nblk > maxb) nblk = maxb;          // all CTAs must be resident for the grid barrier
  cudaMemsetAsync(bar, 0, 2 * sizeof(unsigned), st);
  k<<<nblk, 128, C::SMEM, st>>>(dprog, ng, bar, pf);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + (int)e;
}

// cfg: 0 BM64 BN64 ST4 | 1 BM128 BN64 ST3 | 2 BM64 BN32 ST4 | 3 BM32 BN64 ST4 | 4 BM128 BN32 ST4 | 5 BM64 BN128 ST3
extern "C" int q4pk_run(int cfg, void* hprog, void* dprog, int ng, void* bar, int pf, int nblk, void* stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  GDesc* hp = (GDesc*)hprog; GDesc* dp = (GDesc*)dprog; unsigned* b = (unsigned*)bar;
  switch (cfg) {
    case 0: return run_pk<64, 64, 4>(hp, dp, ng, b, pf, nblk, st);
    case 1: return run_pk<128, 64, 3>(hp, dp, ng, b, pf, nblk, st);
    case 2: return run_pk<64, 32, 4>(hp, dp, ng, b, pf, nblk, st);
    case 3: return run_pk<32, 64, 4>(hp, dp, ng, b, pf, nblk, st);
    case 4: return run_pk<128, 32, 4>(hp, dp, ng, b, pf, nblk, st);
    case 5: return run_pk<64, 128, 3>(hp, dp, ng, b, pf, nblk, st);
    default: return 3;
  }
}
extern "C" int q4pk_sizeof() { return (int)sizeof(GDesc); }
