"""G2: cost of the online rotate + quantize pass that W8A8 / W4A4 needs before a GEMM (one Triton kernel per GEMM input).
Per row: load K bf16 values, apply a 2048-point Hadamard per 2048-chunk as H32 (x) H64 with two tensor-core tl.dot's on a [32,64] tile,
per-row absmax, round to int8 (or int4 range), store int8 codes + fp32 scale. Compared with a plain copy of the same bytes and with d2's quant-only kernel.
python rq.py   (exclusive --timing)"""
import os, sys, json, statistics, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser('~/work/d2'))
W = os.path.expanduser('~/work/g2')


@triton.jit
def _rotq(X, H32, H64, Q, S, K: tl.constexpr, NCH: tl.constexpr, QMAX: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i32 = tl.arange(0, 32); i64 = tl.arange(0, 64)
    h32 = tl.load(H32 + i32[:, None] * 32 + i32[None, :])
    h64 = tl.load(H64 + i64[:, None] * 64 + i64[None, :])
    amax = 0.0
    for c in tl.static_range(NCH):
        x = tl.load(X + r * K + c * 2048 + i32[:, None] * 64 + i64[None, :])
        y = tl.dot(tl.dot(h32, x.to(tl.float16)).to(tl.float16), h64)
        amax = tl.maximum(amax, tl.max(tl.abs(y)))
    s = amax / QMAX + 1e-12
    for c in tl.static_range(NCH):      # recompute (cheaper than spilling) and store codes
        x = tl.load(X + r * K + c * 2048 + i32[:, None] * 64 + i64[None, :])
        y = tl.dot(tl.dot(h32, x.to(tl.float16)).to(tl.float16), h64)
        q = tl.extra.cuda.libdevice.rint(y / s).to(tl.int8)
        tl.store(Q + r * K + c * 2048 + i32[:, None] * 64 + i64[None, :], q)
    tl.store(S + r, s)


def rotq(x, h32, h64, qmax=127.0):
    T, K = x.shape
    q = torch.empty(T, K, device=x.device, dtype=torch.int8); s = torch.empty(T, device=x.device, dtype=torch.float32)
    _rotq[(T,)](x, h32, h64, q, s, K=K, NCH=K // 2048, QMAX=qmax, num_warps=4)
    return q, s


def tm(fn, reps=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return statistics.median(ts)


if __name__ == '__main__':
    import int8mm as I8
    def had(n):
        H = torch.ones(1, 1)
        while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        return (H / n ** 0.5).half().cuda().contiguous()
    h32, h64 = had(32), had(64)
    # correctness vs dense
    x = torch.randn(8, 2048, device='cuda').bfloat16()
    q, s = rotq(x, h32, h64)
    ref = x.float() @ torch.kron(h32.float(), h64.float())
    print('rotq max abs err (codes*scale vs fp32 rotation) / max:', ((q.float() * s[:, None] - ref).abs().max() / ref.abs().max()).item())
    out = []
    for T in (1000, 4000):
        for K in (2048, 6144):
            x = torch.randn(T, K, device='cuda').bfloat16(); y = torch.empty_like(x)
            r = dict(T=T, K=K, copy=tm(lambda: y.copy_(x)), quant_only=tm(lambda: I8.quant_rows(x)), rot_quant=tm(lambda: rotq(x, h32, h64)))
            print(r, flush=True); out.append(r)
    json.dump(out, open(W + '/res_rotq.json', 'w'), indent=1)
