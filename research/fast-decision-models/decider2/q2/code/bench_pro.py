"""Q2 prologue tests + timing. python bench_pro.py test | time"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q2')]
import q2k as Q, q2pro as PR
from bench_gemm import tm
dev = 'cuda'
OUT = os.path.expanduser('~/work/q2/res_pro.jsonl')


def test():
    torch.manual_seed(0)
    M, N = 300, 2048
    y = (torch.randn(M, N, device=dev) * torch.rand(M, 1, device=dev) * 3).to(torch.bfloat16)
    yf = y.float()
    b = PR.Bufs(M, N, K8=128, K4=1792, G=64, q0=200)
    # RTN
    PR.rowq(y, b, 0); q, s = Q.quant_rows(yf, 7, 0.9)
    print('rtn  codes eq', torch.equal(Q.unpack4(b.Q), q), 'scale eq', torch.equal(b.SA, s))
    # SR
    seed = 0xC0FFEE
    PR.rowq(y, b, 1, seed=seed); u = Q.hash_u(seed, M, N); q, s = Q.quant_rows(yf, 7, 0.9, 'sr', u)
    print('sr   codes eq', torch.equal(Q.unpack4(b.Q), q), 'mean(code - v)', (Q.unpack4(b.Q).float() - yf / s[:, None]).mean().item())
    # dX
    PR.rowq(y, b, 2); q, s = Q.quant_rows(yf, 7, 0.9); dx = (yf - s[:, None] * q.float()).to(torch.bfloat16)
    print('dX   eq', torch.equal(b.DX, dx))
    # group 64
    PR.rowq(y, b, 3); yg = yf.reshape(M, N // 64, 64); am = yg.abs().amax(-1); sg = (am.clamp_min(1e-8) / torch.full_like(am, 7.0)) * 0.9
    qg = torch.round(yg / sg[..., None]).clamp(-7, 7).reshape(M, N).to(torch.int8)
    print('g64  codes eq', torch.equal(Q.unpack4(b.Q), qg), 'scales eq', torch.equal(b.GS, sg))
    # split K8=128 int8, K4=1792 int4, rest (128) skipped
    PR.rowq(y, b, 4); q8, s8 = Q.quant_rows(yf[:, :128], 127, 1.0); q4, s4 = Q.quant_rows(yf[:, 128:1920], 7, 0.9)
    print('split eq', torch.equal(b.Q8.view(-1)[:M * 128].view(M, 128), q8),
          torch.equal(Q.unpack4(b.Q.view(-1)[:M * 896].view(M, 896)), q4), torch.equal(b.S8, s8), torch.equal(b.SA, s4))
    # row-role q0 = 200
    b2 = PR.Bufs(M, N, q0=200); PR.rowq(y, b2, 5)
    q4r, s4r = Q.quant_rows(yf[:200], 7, 0.9); q8r, s8r = Q.quant_rows(yf[200:], 127, 1.0)
    print('rowrole eq', torch.equal(Q.unpack4(b2.Q[:200]), q4r), torch.equal(b2.Q8[:100], q8r), torch.equal(b2.SA[:200], s4r), torch.equal(b2.S8[:100], s8r))
    # B6 stat
    b.STAT.zero_(); PR.rowq(y, b, 6); q, s = Q.quant_rows(yf, 7, 0.9); e = yf - s[:, None] * q.float()
    print('stat rel err', (b.STAT.sum() / (e * e).sum() - 1).abs().item())
    # B3 proto
    for KC in (64, 256):
        cb = (torch.randn(KC, N, device=dev) * 0.5).to(torch.bfloat16)
        sc = (cb.float().abs().amax(1).clamp_min(1e-8) / torch.full((KC,), 127.0, device=dev)); cq = torch.round(cb.float() / sc[:, None]).clamp(-127, 127).to(torch.int8)
        hn = 0.5 * cb.float().pow(2).sum(1)
        xb = (cb[torch.randint(0, KC, (M,), device=dev)].float() + 0.3 * torch.randn(M, N, device=dev)).to(torch.bfloat16)
        Qp = torch.empty(M, N // 2, device=dev, dtype=torch.uint8); SAp = torch.empty(M, device=dev); IDX = torch.empty(M, device=dev, dtype=torch.int32)
        PR.proto(xb, cq, sc, hn, cb, Qp, SAp, IDX)
        ri, rq, rs = PR.proto_ref(xb, cq, sc, hn, cb)
        print(f'proto Kc={KC}: idx eq {torch.equal(IDX, ri)} ({(IDX != ri).sum().item()} differ), codes eq {torch.equal(Q.unpack4(Qp), rq)}, scale eq {torch.equal(SAp, rs)}')


def emit(r):
    print(json.dumps(r), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')


def time_():
    for M in (140, 400, 1125, 4125):
        for N in (2048, 6144):
            ys = [torch.randn(M, N, device=dev).to(torch.bfloat16) for _ in range(3)]
            b = PR.Bufs(M, N, K8=128, K4=N - 256, G=64, q0=M - 125 if M > 125 else M)
            r = dict(kind='rowq', M=M, N=N)
            for mode, lab in [(0, 'rtn'), (1, 'sr'), (2, 'rtn_dx'), (3, 'g64'), (4, 'split'), (5, 'rowrole'), (6, 'stat'), (7, 'xb')]:
                r[lab] = tm(lambda i: PR.rowq(ys[i % 3], b, mode, seed=i))
            emit(r)
        # in-runtime residual-add + RMSNorm + quantize (K = 2048)
        x = torch.randn(M, 2048, device=dev).to(torch.bfloat16); d = torch.randn(M, 2048, device=dev).half()
        ra = torch.rand(M, device=dev); cs = torch.rand(2048, device=dev)
        b = PR.Bufs(M, 2048, K8=128, K4=1792, G=64, q0=M - 125 if M > 125 else M)
        r = dict(kind='addq', M=M, N=2048)
        for mode, lab in [(0, 'rtn'), (1, 'sr'), (2, 'rtn_dx'), (3, 'g64'), (4, 'split'), (5, 'rowrole'), (6, 'stat'), (7, 'xb')]:
            r[lab] = tm(lambda i: PR.addq(x, d, ra, cs, b, mode, seed=i))
        emit(r)
        # B3 search + residual quantization
        for K in (2048, 6144):
            xb = torch.randn(M, K, device=dev).to(torch.bfloat16)
            Qp = torch.empty(M, K // 2, device=dev, dtype=torch.uint8); SAp = torch.empty(M, device=dev); IDX = torch.empty(M, device=dev, dtype=torch.int32)
            r = dict(kind='proto', M=M, K=K)
            for KC in (64, 256, 1024):
                cb = torch.randn(KC, K, device=dev).to(torch.bfloat16); cq = torch.randint(-127, 128, (KC, K), device=dev, dtype=torch.int8)
                sc = torch.rand(KC, device=dev); hn = torch.rand(KC, device=dev)
                best = None
                for BM in (16, 32, 64):
                    for BC in (64, 128):
                        if BC > KC: continue
                        try:
                            t = tm(lambda i: PR.proto(xb, cq, sc, hn, cb, Qp, SAp, IDX, BM=BM, BC=BC))
                        except Exception as ex:
                            continue
                        if best is None or t[0] < best[1][0]: best = ((BM, BC), t)
                r[f'kc{KC}'] = best
            emit(r)


if __name__ == '__main__':
    if sys.argv[1] in ('test', 'time'): {'test': test, 'time': time_}[sys.argv[1]]()


def gtm(fn, n=20, reps=15):
    """time fn inside a CUDA graph of n back-to-back calls (no host launch cost); returns (median, p95) us per call"""
    import statistics
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3): fn(i)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): fn(i)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000 / n)
    ts.sort(); return round(statistics.median(ts), 2), round(ts[int(0.95 * (len(ts) - 1))], 2)


def gtime():
    global OUT
    OUT = os.path.expanduser('~/work/q2/res_pro_graph.jsonl')
    for M in (140, 400, 1125, 4125):
        for N in (2048, 6144):
            ys = [torch.randn(M, N, device=dev).to(torch.bfloat16) for _ in range(3)]
            b = PR.Bufs(M, N, K8=128, K4=N - 256, G=64, q0=M - 125 if M > 125 else M)
            r = dict(kind='rowq_graph', M=M, N=N)
            for mode, lab in [(0, 'rtn'), (1, 'sr'), (2, 'rtn_dx'), (3, 'g64'), (4, 'split'), (5, 'rowrole'), (6, 'stat'), (7, 'xb')]:
                r[lab] = gtm(lambda i: PR.rowq(ys[i % 3], b, mode, seed=i))
            emit(r)
        x = torch.randn(M, 2048, device=dev).to(torch.bfloat16); d = torch.randn(M, 2048, device=dev).half()
        ra = torch.rand(M, device=dev); cs = torch.rand(2048, device=dev)
        b = PR.Bufs(M, 2048, K8=128, K4=1792, G=64, q0=M - 125 if M > 125 else M)
        r = dict(kind='addq_graph', M=M, N=2048)
        for mode, lab in [(0, 'rtn'), (1, 'sr'), (2, 'rtn_dx'), (3, 'g64'), (4, 'split'), (5, 'rowrole'), (6, 'stat'), (7, 'xb')]:
            r[lab] = gtm(lambda i: PR.addq(x, d, ra, cs, b, mode, seed=i))
        emit(r)


if __name__ == '__main__' and sys.argv[1] == 'gtime':
    gtime()
