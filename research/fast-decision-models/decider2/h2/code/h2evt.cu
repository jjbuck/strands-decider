// H2: gate_up GEMM with a fused SwiGLU epilogue (CUTLASS 4.8 Sm80 epilogue visitor tree).
// A [M,K] int codes (activations), B [N,K] int codes (weights; rows interleaved g0,u0,g1,u1,...), N = 2*I.
// Epilogue per element: y = (acc * ra[row]) * cs[col]  (fp32), then per adjacent column pair (g,u):
//   m = bf16( bf16(silu(bf16(g))) * bf16(u) )   written to Out[row, col/2] (bf16, ld = ldo): half the bytes of the gate/up output.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

using namespace cute;
namespace tbk = cutlass::epilogue::threadblock;

template <class ThreadMap>
struct VisitorSwiGLUStore {
  struct Arguments { cutlass::bfloat16_t* ptr = nullptr; int64_t ld = 0; };
  using Params = Arguments;
  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  struct SharedStorage {};
  using Element = cutlass::bfloat16_t;
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE VisitorSwiGLUStore() {}
  CUTLASS_HOST_DEVICE VisitorSwiGLUStore(Params const& params, SharedStorage const&) : params_ptr(&params) {}
  Params const* params_ptr;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : tbk::detail::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(RTensor&& r, CTensor&& c, ProblemShape ps, Params const* pp)
        : tC_rAux(cute::forward<RTensor>(r)), tC_cAux(cute::forward<CTensor>(c)), problem_shape(ps), params_ptr(pp) {}
    RTensor tC_rAux; CTensor tC_cAux; ProblemShape problem_shape; Params const* params_ptr;

    CUTLASS_DEVICE void begin_step(int) { clear(tC_rAux); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
                              cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      Tensor frg = recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      cutlass::Array<Element, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < FragmentSize / 2; ++j) {
        float g = float(Element(float(frg_input[2 * j])));
        float u = float(Element(float(frg_input[2 * j + 1])));
        float s = float(Element(g / (1.0f + __expf(-g))));
        out[j] = Element(s * u);
        out[j + FragmentSize / 2] = Element(0.0f);
      }
      frg(frg_idx) = out;
      return frg_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        auto cd = coord_v(i);
        int m = get<0>(cd); int n = get<1>(cd);
        if (m < get<0>(problem_shape) && n < get<1>(problem_shape)) {
          uint64_t const* p = reinterpret_cast<uint64_t const*>(&src_v(i));
          uint64_t* dst = reinterpret_cast<uint64_t*>(params_ptr->ptr + int64_t(m) * params_ptr->ld + (n >> 1));
          *dst = p[0];
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    Tensor mAux = make_tensor(make_gmem_ptr(params_ptr->ptr), problem_shape, cute::Stride<int64_t, _1, int64_t>{int64_t(get<1>(problem_shape)), _1{}, int64_t(0)});
    Tensor tC_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)), Shape<Int<VecLength>>{}, (_0{}));
    return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr);
  }
};

template <class EA, class EB, int TBM, int TBN, int TBK, int WM, int WN, int WK, int IK, int STAGES, class Op>
struct GS {
  using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
  using WS = cutlass::gemm::GemmShape<WM, WN, WK>;
  using IS = cutlass::gemm::GemmShape<16, 8, IK>;
  using TM = tbk::OutputTileThreadLayout<TB, WS, cutlass::bfloat16_t, 8, 1>;
  using Accum = tbk::VisitorAccFetch;
  using RowScale = tbk::VisitorColBroadcast<TM, float, cute::Stride<_1, _0, int32_t>>;     // ra[m]
  using ColScale = tbk::VisitorRowBroadcast<TM, float, cute::Stride<_0, _1, int32_t>>;     // cs[n]
  using Mul = tbk::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = tbk::Sm80EVT<Mul, Accum, RowScale>;
  using EVT1 = tbk::Sm80EVT<Mul, EVT0, ColScale>;
  using Store = VisitorSwiGLUStore<TM>;
  using EVTD = tbk::Sm80EVT<Store, EVT1>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      EA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<EA>::value,
      EB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<EB>::value,
      cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
      int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS, EVTD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES, Op, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class GT>
int runs(const void* A, const void* B, void* Out, const float* ra, const float* cs, int M, int N, int K, int ldo, cudaStream_t stream) {
  using G = typename GT::Gemm;
  typename GT::EVTD::Arguments cb{
    {
      {
        {},
        {ra, 0.0f, {_1{}, _0{}, int32_t(M)}},
        {}
      },
      {cs, 0.0f, {_0{}, _1{}, int32_t(N)}},
      {}
    },
    {reinterpret_cast<cutlass::bfloat16_t*>(Out), int64_t(ldo)}
  };
  typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, cb, A, B, nullptr, nullptr,
                             0, 0, 0, 0, int64_t(K), int64_t(K), int64_t(0), int64_t(0));
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

using OpS = cutlass::arch::OpMultiplyAddSaturate;
extern "C" int s4_swiglu(const void* A, const void* B, void* Out, const float* ra, const float* cs, int M, int N, int K, int ldo, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) {
    case 0: return runs<GS<cutlass::int4b_t, cutlass::int4b_t, 128, 256, 128, 64, 64, 128, 64, 3, OpS>>(A, B, Out, ra, cs, M, N, K, ldo, s);
    case 3: return runs<GS<cutlass::int4b_t, cutlass::int4b_t, 64, 128, 128, 32, 64, 128, 64, 4, OpS>>(A, B, Out, ra, cs, M, N, K, ldo, s);
    default: return 1;
  }
}
extern "C" int s8_swiglu(const void* A, const void* B, void* Out, const float* ra, const float* cs, int M, int N, int K, int ldo, int cfg, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (cfg) {
    case 1: return runs<GS<int8_t, int8_t, 128, 128, 64, 64, 64, 64, 32, 4, OpS>>(A, B, Out, ra, cs, M, N, K, ldo, s);
    case 3: return runs<GS<int8_t, int8_t, 64, 128, 64, 32, 64, 64, 32, 4, OpS>>(A, B, Out, ra, cs, M, N, K, ldo, s);
    default: return 1;
  }
}
