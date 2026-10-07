"""J5 GEMM microbench at short M on hobson's five shapes, DRAM-fresh weights (cycled copies > 48 MB), CUDA-event timing.
python gbench.py Ms [kinds]   kinds: cublas, tgemm (d1 fold Triton, BM=128), sk_bf16, sk_w8a16, sk_w4a16 (g128), sk_w8a8, cut_s8 (h2 CUTLASS), sk_w4a8
Writes ~/work/j5/res_gbench.jsonl rows {kind, shape, M, us, tflops, gbps} and the tuned sk configs to skcfg.json."""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import sk
SHAPES = {'gdn_in': (8224, 2048), 'attn_in': (5120, 2048), 'out': (2048, 2048), 'gate_up': (12288, 2048), 'down': (2048, 6144)}
EPI = {'gdn_in': 0, 'attn_in': 0, 'out': 3, 'gate_up': 1, 'down': 3}
dev = 'cuda'


def tcycle(fn, objs, reps=4):
    for o in objs[:2]: fn(o)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for r in range(reps):
        e0.record()
        for o in objs: fn(o)
        e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) / len(objs))
    ts.sort(); return ts[len(ts) // 2] * 1000


def ncopy(nbytes):
    return max(2, -(-(48 << 20) // nbytes))


def main():
    Ms = [int(x) for x in sys.argv[1].split(',')]
    kinds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['cublas', 'tgemm', 'sk_bf16', 'sk_w8a16', 'sk_w4a16', 'sk_w8a8', 'cut_s8']
    path = os.path.expanduser('~/work/j5/res_gbench.jsonl')
    done = set()
    if os.path.exists(path):
        for l in open(path): r = json.loads(l); done.add((r['kind'], r['shape'], r['M']))
    out = open(path, 'a')
    torch.manual_seed(0)
    for sh, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) * 0.02)
        Wb = W.to(torch.bfloat16).contiguous()
        s8 = W.abs().amax(1) / 127; q8 = torch.round(W / s8[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
        Wg = W.reshape(N, K // 128, 128); mx = Wg.amax(-1); mn = Wg.amin(-1); sc = ((mx - mn) / 15).contiguous(); z = torch.round(-mn / sc).clamp(0, 15)
        q4 = (torch.round(Wg / sc[..., None]) + z[..., None]).clamp(0, 15).reshape(N, K)
        objs = {}
        if any(k in kinds for k in ('cublas', 'tgemm', 'sk_bf16', 'mt_bf16')): objs['bf16'] = [Wb.clone() for _ in range(ncopy(Wb.numel() * 2))]
        if any(k in kinds for k in ('sk_w8a16', 'sk_w8a8', 'cut_s8', 'mt_w8a8', 'cut_s8sk')): objs['w8'] = [q8.clone() for _ in range(ncopy(q8.numel()))]
        if any(k in kinds for k in ('sk_w4a16', 'sk_w4a8')):
            p4 = sk.pack_blocked4(q4); objs['w4'] = [sk.QW('w4', p4.clone(), sc.clone(), z.to(torch.uint8).clone(), 128) for _ in range(ncopy(p4.numel()))]
        for M in Ms:
            A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); Ai = torch.randint(-60, 60, (M, K), device=dev, dtype=torch.int8)
            ra = torch.rand(M, device=dev); ss = torch.rand(M, device=dev) * K + 1
            R = torch.zeros(M, N, device=dev, dtype=torch.bfloat16); sso = torch.zeros(M, device=dev)
            e = EPI[sh]
            for kind in kinds:
                if (kind, sh, M) in done: continue
                try:
                    if kind == 'cublas':
                        fn = lambda w: torch.mm(A, w.t()); ob = objs['bf16']; wb = 2
                    elif kind == 'tgemm':
                        import lean2 as L2
                        fn = (lambda w: L2.tgemm(A, w, epi=e, res=R if e == 3 else None, ss=ss if e != 3 else None, ssout=sso if e == 3 else None)); ob = objs['bf16']; wb = 2
                    elif kind in ('sk_bf16', 'sk_w8a16', 'sk_w4a16', 'sk_w8a8', 'sk_w4a8'):
                        if kind == 'sk_bf16': ob = [sk.QW('bf16', w) for w in objs['bf16']]; wb = 2
                        elif kind in ('sk_w8a16', 'sk_w8a8'): ob = [sk.QW('w8', w, s8) for w in objs['w8']]; wb = 1
                        else: ob = objs['w4']; wb = 0.5 + 4 / 128
                        a_ = Ai if kind in ('sk_w8a8', 'sk_w4a8') else A
                        cfg = sk.pick(1 if a_ is Ai else 0, ob[0], M, e if a_ is A else (5 if e == 1 else 4))
                        if a_ is A:
                            fn = (lambda w, cfg=cfg: sk.launch(A, w, e, res=R if e == 3 else None, ss=ss if e != 3 else None, ssout=sso if e == 3 else None, cfg=cfg))
                        else:
                            ee = 5 if e == 1 else 4
                            fn = (lambda w, cfg=cfg, ee=ee: sk.launch(Ai, w, ee, ra=ra, alpha=1 / 4096, scale=(ee == 5), cfg=cfg))
                    elif kind in ('mt_bf16', 'mt_w8a8'):
                        if kind == 'mt_bf16': ob = [sk.QW('bf16', w) for w in objs['bf16']]; wb = 2; a_ = A; ee = e
                        else: ob = [sk.QW('w8', w, s8) for w in objs['w8']]; wb = 1; a_ = Ai; ee = 5 if e == 1 else 4
                        cfg = sk.pick_mt(1 if a_ is Ai else 0, ob[0], M, ee)
                        if a_ is A:
                            fn = (lambda w, cfg=cfg: sk.launch_mt(A, w, e, res=R if e == 3 else None, ss=ss if e != 3 else None, ssout=sso if e == 3 else None, cfg=cfg))
                        else:
                            fn = (lambda w, cfg=cfg, ee=ee: sk.launch_mt(Ai, w, ee, ra=ra, alpha=1 / 4096, scale=(ee == 5), cfg=cfg))
                    elif kind == 'cut_s8sk':
                        import cutsk
                        ob = objs['w8']; wb = 1; best = None
                        for (c, sl) in cutsk.CANDS:
                            if (K // sl) % 64: continue
                            try:
                                t = tcycle(lambda w, c=c, sl=sl: cutsk.gemm(Ai, w, 1 / 4096, c, sl), ob, reps=2)
                                if best is None or t < best[0]: best = (t, (c, sl))
                            except Exception: pass
                        cfg = best[1]
                        fn = (lambda w, c=cfg: cutsk.gemm(Ai, w, 1 / 4096, c[0], c[1]))
                    elif kind == 'cut_s8':
                        import qgemm as QG
                        ob = objs['w8']; wb = 1
                        best = None
                        for c in range(11):
                            try:
                                t = tcycle(lambda w, c=c: QG.gemm('s8', Ai, w, 1 / 4096, c), ob, reps=2)
                                if best is None or t < best[0]: best = (t, c)
                            except Exception: pass
                        fn = (lambda w, c=best[1]: QG.gemm('s8', Ai, w, 1 / 4096, c))
                    us = tcycle(fn, ob)
                    rec = dict(kind=kind, shape=sh, M=M, N=N, K=K, us=round(us, 2), tflops=round(2 * M * N * K / us / 1e6, 1), gbps=round(N * K * wb / us / 1e3, 1))
                    if kind.startswith('sk_') or kind.startswith('mt_') or kind == 'cut_s8sk': rec['cfg'] = cfg
                    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + '\n'); out.flush(); sk.save_tab()
                except Exception as ex:
                    print('ERR', kind, sh, M, repr(ex)[:200], flush=True)
        del objs; torch.cuda.empty_cache()
    sk.save_tab()


if __name__ == '__main__':
    main()
