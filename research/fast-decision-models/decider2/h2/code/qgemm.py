"""H2: ctypes wrappers for CUTLASS int GEMMs (G2's s4/s8 from g2s4.cu built against CUTLASS 3.5.1; H2's s8xs4 mixed-input from h2mix.cu, CUTLASS 4.8).
All: A [M,K] row-major codes, B [N,K] row-major codes, C [M,N] fp16 = alpha * (A @ B^T) (int32 accumulate)."""
import os, ctypes, torch
W = os.path.expanduser('~/work/h2')
_L = {}

def _lib(name):
    if name not in _L:
        lib = ctypes.CDLL(f'{W}/{name}')
        for fn in ('s4_gemm', 's8_gemm', 's8s4_gemm'):
            if hasattr(lib, fn):
                f = getattr(lib, fn); f.restype = ctypes.c_int
                f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        _L[name] = lib
    return _L[name]

def fn(kind):
    if kind == 's4': return _lib('libg2s4.so').s4_gemm
    if kind == 's8': return _lib('libg2s4.so').s8_gemm
    if kind == 's8s4': return _lib('libh2mix.so').s8s4_gemm
    raise ValueError(kind)

def kdim(kind, A):   # logical K from the A buffer
    return A.shape[1] * 2 if kind == 's4' else A.shape[1]

def gemm(kind, A, B, alpha, cfg, out=None):
    M = A.shape[0]; N = B.shape[0]; K = kdim(kind, A)
    C = out if out is not None else torch.empty(M, N, device=A.device, dtype=torch.float16)
    rc = fn(kind)(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, float(alpha), int(cfg), torch.cuda.current_stream().cuda_stream)
    if rc != 0: raise RuntimeError(f'cutlass {kind} cfg{cfg} rc {rc} M{M} N{N} K{K}')
    return C

def pack4(q):  # int in [-8,7] [..., K] -> uint8 [..., K/2], element 2j in the low nibble
    q = q.to(torch.int16)
    return ((q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()


_EVT = None
def swiglu_gemm(kind, A, B, ra, cs, cfg=3, out=None):
    """gate_up GEMM with fused dequant + SwiGLU epilogue (libh2evt.so): B rows interleaved (g0,u0,g1,u1,..), returns bf16 m [M, N/2]."""
    global _EVT
    if _EVT is None:
        _EVT = ctypes.CDLL(f'{W}/libh2evt.so')
        for f in (_EVT.s4_swiglu, _EVT.s8_swiglu):
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    M = A.shape[0]; N = B.shape[0]; K = kdim(kind, A)
    o = out if out is not None else torch.empty(M, N // 2, device=A.device, dtype=torch.bfloat16)
    f = _EVT.s4_swiglu if kind == 's4' else _EVT.s8_swiglu
    rc = f(A.data_ptr(), B.data_ptr(), o.data_ptr(), ra.data_ptr(), cs.data_ptr(), M, N, K, N // 2, int(cfg), torch.cuda.current_stream().cuda_stream)
    if rc != 0: raise RuntimeError(f'evt swiglu {kind} rc {rc}')
    return o
