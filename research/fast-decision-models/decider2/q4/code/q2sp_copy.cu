// Q4 copy of Q2's q2sp.cu (+ configs 10-15 sized for the A10G's 99 KB of shared memory per block). Q2 B11: 2:4 structured-sparse int8 / int4 GEMMs on sm_86 tensor cores (mma.sp), via CUTLASS 2.x GemmSparseUniversal (v3.5.1).
// The sparse operand of mma.sp is A, so the weights are A: Y^T [N_out, T] = W [N_out, K] (2:4 along K, compressed to K/2) x X^T,
// with X [T, K] row-major (= ColumnMajor K x T). Output fp16 = alpha * acc, written as Y^T [N_out, T] row-major.
// Metadata E: CUTLASS's reordered layout; for timing (and the identity check) every group keeps positions 0 and 1 (0x4444...).
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template <typename EA, int TBM, int TBN, int TBK, int WM, int WN, int WK, int IK, int ST>
struct SG {
  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      EA, cutlass::layout::RowMajor, EA, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::RowMajor,
      int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, IK>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, ST>;
};

template <class GT>
int runsp(const void* A, const void* B, void* C, const void* E, int M, int N, int K, float alpha, cudaStream_t st) {
  using G = typename GT::Gemm;
  using LE = typename G::GemmKernel::LayoutE;
  constexpr int kS = G::kSparse, kEE = G::kElementsPerElementE;
  auto le = LE::packed(cutlass::make_Coord(M, K / kS / kEE));
  typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {alpha, 0.0f},
                             A, B, C, C, E, int64_t(), int64_t(), int64_t(), int64_t(), int64_t(),
                             int64_t(K / kS), int64_t(K), int64_t(N), int64_t(N), int64_t(le.stride(0)));
  G gemm;
  if (G::get_workspace_size(args) != 0) return 10;
  auto s = gemm.can_implement(args);
  if (s != cutlass::Status::kSuccess) return 20 + int(s);
  s = gemm.initialize(args, nullptr, st);
  if (s != cutlass::Status::kSuccess) return 40 + int(s);
  s = gemm(st);
  if (s != cutlass::Status::kSuccess) return 60 + int(s);
  return 0;
}

template <class GT> long long esize(int M, int K) {   // number of ElementE elements of the reordered metadata
  using G = typename GT::Gemm;
  using LE = typename G::GemmKernel::LayoutE;
  constexpr int kS = G::kSparse, kEE = G::kElementsPerElementE;
  auto ext = cutlass::make_Coord(M, K / kS / kEE);
  return (long long)LE::packed(ext).capacity(ext) * (long long)sizeof(typename G::ElementE);
}

#define S8C(X) X(0, 128, 128, 128, 64, 64, 128, 4) X(1, 256, 128, 128, 64, 64, 128, 3) X(2, 128, 256, 128, 64, 64, 128, 3) \
  X(3, 64, 128, 128, 32, 64, 128, 6) X(4, 128, 64, 128, 64, 32, 128, 6) X(5, 64, 64, 128, 32, 32, 128, 10) \
  X(6, 256, 64, 128, 64, 64, 128, 4) X(7, 128, 128, 256, 64, 64, 256, 3) X(8, 128, 32, 128, 32, 32, 128, 6) X(9, 256, 32, 128, 64, 32, 128, 6) \
  X(10, 128, 128, 128, 64, 64, 128, 3) X(11, 128, 64, 128, 64, 32, 128, 4) X(12, 256, 64, 128, 64, 32, 128, 3) \
  X(13, 64, 128, 128, 32, 64, 128, 3) X(14, 64, 64, 128, 32, 32, 128, 6) X(15, 256, 32, 128, 64, 32, 128, 4)

#define CASE8(id, a, b, c, d, e, f, st) case id: return runsp<SG<int8_t, a, b, c, d, e, f, 64, st>>(A, B, C, E, M, N, K, alpha, s);
#define CASE4(id, a, b, c, d, e, f, st) case id: return runsp<SG<cutlass::int4b_t, a, b, 2 * c, d, e, 2 * f, 128, st>>(A, B, C, E, M, N, K, alpha, s);
#define ES8(id, a, b, c, d, e, f, st) case id: return esize<SG<int8_t, a, b, c, d, e, f, 64, st>>(M, K);
#define ES4(id, a, b, c, d, e, f, st) case id: return esize<SG<cutlass::int4b_t, a, b, 2 * c, d, e, 2 * f, 128, st>>(M, K);

extern "C" int sp_s8(const void* A, const void* B, void* C, const void* E, int M, int N, int K, float alpha, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) { S8C(CASE8) default: return 1; }
}
extern "C" int sp_s4(const void* A, const void* B, void* C, const void* E, int M, int N, int K, float alpha, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) { S8C(CASE4) default: return 1; }
}
extern "C" long long sp_s8_ebytes(int M, int K, int cfg) { switch (cfg) { S8C(ES8) default: return -1; } }
extern "C" long long sp_s4_ebytes(int M, int K, int cfg) { switch (cfg) { S8C(ES4) default: return -1; } }
