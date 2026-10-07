// H2: W4A8 mixed-input GEMM on sm_86: A = int8 activations [M,K] row-major, B = int4 weights [N,K] row-major (== ColumnMajor KxN),
// upcast s4->s8 in registers (CUTLASS 4.8 OpMultiplyAddMixedInputUpcast), mma s8 m16n8k32, s32 accumulate, fp16 out = alpha * acc.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES>
struct GM {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      int8_t, cutlass::layout::RowMajor,
      cutlass::int4b_t, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::RowMajor,
      int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES, 16, 32,
      cutlass::arch::OpMultiplyAddMixedInputUpcast, cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;
};

template <typename GT>
int runm(const void* A, const void* B, void* C, int M, int N, int K, float alpha, cudaStream_t stream) {
  using G = typename GT::Gemm;
  typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {alpha, 0.0f},
      A, B, C, C, 0, 0, 0, 0, K, K, N, N);
  G gemm;
  if (G::get_workspace_size(args) != 0) return 10;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return 20 + int(st);
  st = gemm.initialize(args, nullptr, stream);
  if (st != cutlass::Status::kSuccess) return 40 + int(st);
  st = gemm(stream);
  if (st != cutlass::Status::kSuccess) return 60 + int(st);
  return 0;
}

extern "C" int s8s4_gemm(const void* A, const void* B, void* C, int M, int N, int K, float alpha, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) {
    case 0: return runm<GM<128, 256, 64, 64, 64, 64, 3>>(A, B, C, M, N, K, alpha, s);
    case 1: return runm<GM<128, 128, 64, 64, 64, 64, 4>>(A, B, C, M, N, K, alpha, s);
    case 2: return runm<GM<256, 128, 64, 64, 64, 64, 3>>(A, B, C, M, N, K, alpha, s);
    case 3: return runm<GM<64, 128, 64, 32, 64, 64, 4>>(A, B, C, M, N, K, alpha, s);
    case 4: return runm<GM<128, 128, 128, 64, 64, 128, 3>>(A, B, C, M, N, K, alpha, s);
    default: return 1;
  }
}
