"""Q2: ctypes wrapper for libq2gemm.so (q2gemm.cu) + torch references of FORMATS.md arithmetic.
All buffers are torch CUDA tensors. A operands [M, K] row-major, B (weights) [N, K] row-major; int4 packed 2/byte, low nibble = even k."""
import os, ctypes, torch
W = os.path.expanduser('~/work/q2')
V = dict(s4=0, s8=1, bf16=2, s4_tbf16=3, s4_ts8=4, s4g64=5, s4g128=6, s4skip=7, s4skip_tbf16=8, s8_ts8=9, s8_tbf16=10)
EPI = dict(bf16=0, swiglu=1, fp16a=2, i32=3, f32=4, fp16s=6, none=9)
L = ctypes.c_longlong; P_ = ctypes.c_void_p; D = ctypes.c_double


class Prob(ctypes.Structure):
    _fields_ = [('A0', P_), ('B0', P_), ('lda0', L), ('ldb0', L), ('kt0', L),
                ('A1', P_), ('B1', P_), ('lda1', L), ('ldb1', L), ('kt1', L),
                ('sa0', P_), ('sb0', P_), ('sa1', P_), ('sb1', P_),
                ('gsa', P_), ('gsb', P_), ('ldgs', L),
                ('bias', P_), ('tab', P_), ('tidx', P_), ('ldt', L),
                ('rowmap', P_), ('row0', L),
                ('out', P_), ('ldo', L), ('epi', L),
                ('alpha', D), ('kscale', D), ('seed', L), ('nkeep', L), ('nkb', L),
                ('M', L), ('N', L), ('bsA0', L), ('bsB0', L), ('bsO', L), ('mtiles', L), ('ntiles', L), ('stat', P_), ('statw', P_), ('skws', P_), ('sksem', P_), ('splits', L)]


class Params(ctypes.Structure):
    _fields_ = [('p', Prob * 2), ('tiles0', L)]


_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(f'{W}/libq2gemm.so')
        _lib.q2_gemm.restype = ctypes.c_int
        _lib.q2_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Params), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        assert _lib.q2_sizeof_params() == ctypes.sizeof(Params), (_lib.q2_sizeof_params(), ctypes.sizeof(Params))
    return _lib


def ptr(t): return 0 if t is None else t.data_ptr()
BYTES = dict(s4=0.5, s8=1, bf16=2)


def prob(A0, B0, kind0, out, sa0=None, sb0=None, A1=None, B1=None, kind1=None, sa1=None, sb1=None, gsa=None, gsb=None,
         bias=None, tab=None, tidx=None, rowmap=None, row0=0, epi='bf16', alpha=1.0, kscale=1.0, seed=0, nkeep=0, nkb=0,
         M=None, N=None, bsA0=0, bsB0=0, bsO=0, stat=None, statw=None, splits=1, skws=None, sksem=None):
    p = Prob()
    M = A0.shape[-2] if M is None else M; N = B0.shape[-2] if N is None else N
    p.A0, p.B0 = ptr(A0), ptr(B0)
    p.lda0 = A0.stride(-2) * A0.element_size(); p.ldb0 = B0.stride(-2) * B0.element_size()
    kb0 = A0.shape[-1] * A0.element_size(); assert kb0 % 64 == 0, kb0; p.kt0 = kb0 // 64
    if A1 is not None:
        p.A1, p.B1 = ptr(A1), ptr(B1)
        p.lda1 = A1.stride(0) * A1.element_size(); p.ldb1 = B1.stride(0) * B1.element_size()
        kb1 = A1.shape[1] * A1.element_size(); assert kb1 % 64 == 0, kb1; p.kt1 = kb1 // 64
    p.sa0, p.sb0, p.sa1, p.sb1 = ptr(sa0), ptr(sb0), ptr(sa1), ptr(sb1)
    p.gsa, p.gsb = ptr(gsa), ptr(gsb); p.ldgs = gsa.shape[1] if gsa is not None else 0
    p.bias, p.tab, p.tidx = ptr(bias), ptr(tab), ptr(tidx); p.ldt = tab.shape[1] if tab is not None else 0
    p.rowmap, p.row0 = ptr(rowmap), row0
    p.out = ptr(out); p.ldo = out.stride(-2); p.epi = EPI[epi]
    p.alpha, p.kscale, p.seed, p.nkeep, p.nkb = alpha, kscale, seed, nkeep, nkb
    p.M, p.N = M, N
    p.bsA0, p.bsB0, p.bsO = bsA0, bsB0, bsO
    p.stat, p.statw = ptr(stat), ptr(statw)
    p.splits = splits; p.skws, p.sksem = ptr(skws), ptr(sksem)
    p._keep = (A0, B0, A1, B1, out)
    return p


def run(variant, cfg, p0, p1=None, batch=1, stream=None, sched=0):
    prm = Params(); prm.p[0] = p0; prm.tiles0 = sched
    if p1 is not None: prm.p[1] = p1
    st = torch.cuda.current_stream().cuda_stream if stream is None else stream
    rc = lib().q2_gemm(V[variant] if isinstance(variant, str) else variant, cfg, ctypes.byref(prm), 1 if p1 is not None else 0, batch, st)
    if rc != 0: raise RuntimeError(f'q2_gemm {variant} cfg{cfg} rc {rc}')


# ------------------------------------------------------------------ packing / references (FORMATS.md)
def pack4(q):
    q = q.to(torch.int16)
    return ((q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()


def unpack4(p):
    p = p.to(torch.int16)
    lo = p & 0xF; hi = (p >> 4) & 0xF
    lo = torch.where(lo > 7, lo - 16, lo); hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(*p.shape[:-1], p.shape[-1] * 2).to(torch.int8)


def quant_rows(y, qmax, clip, mode='rtn', u=None):
    """FORMATS section 0. y fp32 [M, K] -> (codes int8 [M, K], scale fp32 [M])"""
    amax = y.abs().amax(1)
    s = (amax.clamp_min(1e-8) / torch.full_like(amax, qmax)) * clip   # tensor/tensor: IEEE division (a python-scalar divisor makes torch multiply by the reciprocal)
    v = y / s[:, None]
    q = torch.round(v) if mode == 'rtn' else torch.floor(v + u)
    return q.clamp(-qmax, qmax).to(torch.int8), s


def iacc(qa, qw):
    """exact int32 accumulate via fp64 matmul (codes small): qa [M, K] int8, qw [N, K] int8 -> int64 [M, N]"""
    return (qa.double() @ qw.double().t()).round().to(torch.int64)


def ref_int(qa, sa, qw, sw):
    acc = iacc(qa, qw)
    return (acc.float() * sa[:, None]) * sw[None, :]


def hash_u(seed, M, K, dev='cuda'):
    """FORMATS section 4 counter hash -> u in [0,1) fp32 [M, K]"""
    m = 0xFFFFFFFF
    t = torch.arange(M, device=dev, dtype=torch.int64)[:, None]; k = torch.arange(K, device=dev, dtype=torch.int64)[None, :]
    h = (seed ^ ((t * 0x9E3779B1) & m) ^ ((k * 0x85EBCA77) & m)) & m
    h = h ^ (h >> 16); h = (h * 0x85EBCA6B) & m; h = h ^ (h >> 13); h = (h * 0xC2B2AE35) & m; h = h ^ (h >> 16)
    return (h >> 8).float() * (2.0 ** -24)


def fmix32_t(h):
    m = 0xFFFFFFFF
    h = h ^ (h >> 16); h = (h * 0x85EBCA6B) & m; h = h ^ (h >> 13); h = (h * 0xC2B2AE35) & m; h = h ^ (h >> 16)
    return h


def skip_mask(seed, mtiles128, ntiles128, nkb, nkeep, dev='cuda'):
    """FORMATS B9: keep[mb, nb, kb] bool"""
    m = 0xFFFFFFFF
    mb = torch.arange(mtiles128, device=dev, dtype=torch.int64)[:, None, None]
    nb = torch.arange(ntiles128, device=dev, dtype=torch.int64)[None, :, None]
    kb = torch.arange(nkb, device=dev, dtype=torch.int64)[None, None, :]
    base = (seed ^ ((mb * 0x9E3779B1) & m) ^ ((nb * 0x85EBCA77) & m)) & m
    h = fmix32_t((base ^ ((kb * 0xC2B2AE35) & m)) & m)
    # rank by (h, kb)
    key = h * 64 + kb
    order = key.argsort(-1)
    rank = torch.empty_like(order); rank.scatter_(-1, order, torch.arange(nkb, device=dev).expand_as(order).contiguous())
    return rank < nkeep


_SK = {}
def splitk_ws(splits, M, N, dev='cuda'):
    """int32 partials [splits, M, N] and zeroed per-tile counters (enough for 16-row x 64-col tiles); cached per shape"""
    key = (splits, M, N)
    if key not in _SK:
        _SK[key] = (torch.empty(splits, M, N, device=dev, dtype=torch.int32), torch.zeros(((M + 15) // 16) * ((N + 63) // 64), device=dev, dtype=torch.int32))
    return _SK[key]
