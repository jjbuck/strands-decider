// G2: CUTLASS 2.x tensor-op GEMMs on sm_86 for int4 (s4 x s4 -> s32, mma m16n8k64) and int8 (s8 x s8 -> s32, mma m16n8k32).
// Plain C ABI (built with the box's nvcc 12.8, loaded with ctypes; torch is cu130 so a torch extension cannot be built).
// A [M, K] row-major (int4 packed 2 per byte, low nibble first), B [N, K] row-major (== column-major K x N), C [M, N] fp16 = alpha * (A @ B^T).
// Per-row / per-column dequant scales are applied by the caller (benchmark: scalar alpha only).
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template <typename EA, int TBM, int TBN, int TBK, int WM, int WN, int WK, int IK, int STAGES>
struct G {
  using Gemm = cutlass::gemm::device::Gemm<
      EA, cutlass::layout::RowMajor,
      EA, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::RowMajor,
      int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>,
      cutlass::gemm::GemmShape<WM, WN, WK>,
      cutlass::gemm::GemmShape<16, 8, IK>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      STAGES,
      128 / cutlass::sizeof_bits<EA>::value, 128 / cutlass::sizeof_bits<EA>::value,
      false, cutlass::arch::OpMultiplyAddSaturate>;
};

template <typename GT, typename EA>
int run(const void* A, const void* B, void* C, int M, int N, int K, float alpha, cudaStream_t stream) {
  typename GT::Gemm::Arguments args({M, N, K},
                                    {reinterpret_cast<EA*>(const_cast<void*>(A)), K},
                                    {reinterpret_cast<EA*>(const_cast<void*>(B)), K},
                                    {reinterpret_cast<cutlass::half_t*>(C), N},
                                    {reinterpret_cast<cutlass::half_t*>(C), N},
                                    {alpha, 0.0f});
  typename GT::Gemm gemm;
  if (GT::Gemm::get_workspace_size(args) != 0) return 10;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return 20 + int(st);
  st = gemm.initialize(args, nullptr, stream);
  if (st != cutlass::Status::kSuccess) return 40 + int(st);
  st = gemm(stream);
  if (st != cutlass::Status::kSuccess) return 60 + int(st);
  return 0;
}

extern "C" int s4_gemm(const void* A, const void* B, void* C, int M, int N, int K, float alpha, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) {
    case 0: return run<G<cutlass::int4b_t, 128, 256, 128, 64, 64, 128, 64, 3>, cutlass::int4b_t>(A, B, C, M, N, K, alpha, s);
    case 1: return run<G<cutlass::int4b_t, 128, 128, 128, 64, 64, 128, 64, 4>, cutlass::int4b_t>(A, B, C, M, N, K, alpha, s);
    case 2: return run<G<cutlass::int4b_t, 256, 128, 128, 64, 64, 128, 64, 3>, cutlass::int4b_t>(A, B, C, M, N, K, alpha, s);
    case 3: return run<G<cutlass::int4b_t, 64, 128, 128, 32, 64, 128, 64, 4>, cutlass::int4b_t>(A, B, C, M, N, K, alpha, s);
    case 4: return run<G<cutlass::int4b_t, 128, 128, 256, 64, 64, 256, 64, 3>, cutlass::int4b_t>(A, B, C, M, N, K, alpha, s);
    default: return 1;
  }
}

extern "C" int s8_gemm(const void* A, const void* B, void* C, int M, int N, int K, float alpha, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) {
    case 0: return run<G<int8_t, 128, 256, 64, 64, 64, 64, 32, 3>, int8_t>(A, B, C, M, N, K, alpha, s);
    case 1: return run<G<int8_t, 128, 128, 64, 64, 64, 64, 32, 4>, int8_t>(A, B, C, M, N, K, alpha, s);
    case 2: return run<G<int8_t, 256, 128, 64, 64, 64, 64, 32, 3>, int8_t>(A, B, C, M, N, K, alpha, s);
    case 3: return run<G<int8_t, 64, 128, 64, 32, 64, 64, 32, 4>, int8_t>(A, B, C, M, N, K, alpha, s);
    case 4: return run<G<int8_t, 128, 128, 128, 64, 64, 128, 32, 3>, int8_t>(A, B, C, M, N, K, alpha, s);
    default: return 1;
  }
}
