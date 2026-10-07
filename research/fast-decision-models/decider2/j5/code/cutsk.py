"""ctypes wrapper for the split-K serial CUTLASS int8 GEMM in libg2s4.so (j5 g2s4x.cu): C fp16 = alpha * A @ B^T"""
import ctypes, os, torch
_L = None
def _lib():
    global _L
    if _L is None:
        _L = ctypes.CDLL(os.path.expanduser('~/work/h2/libg2s4.so'))
        f = _L.s8_gemm_sk; f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return _L
def gemm(A, B, alpha, cfg, slices, out=None):
    M, K = A.shape; N = B.shape[0]
    C = out if out is not None else torch.empty(M, N, device=A.device, dtype=torch.float16)
    rc = _lib().s8_gemm_sk(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, float(alpha), int(cfg), int(slices), torch.cuda.current_stream().cuda_stream)
    if rc != 0: raise RuntimeError(f'cutlass sk cfg{cfg} slices{slices} rc {rc}')
    return C
CANDS = [(c, s) for c in range(5) for s in (2, 3, 4, 6, 8)]
