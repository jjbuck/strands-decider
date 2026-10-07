"""Q4: per-config timing of q4sp sp8 (fp16a; SwiGLU for gate_up) at hobson's shapes, to see whether the tall-warp configs 12-15 help."""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q4')]
import q4sp as S, test_sp as TS
from bench_sp import tm, SH
TS.set_layout()
out = {}
for M in (140, 1125, 4125):
    for name, (N, K) in SH.items():
        sw = name == 'gate_up'
        w = torch.randint(-127, 128, (N, K), device='cuda', dtype=torch.int8)
        Wc, E, _ = S.compress(w, S.mask24_mag(w, 'sp8'), 'sp8')
        xs = [torch.randint(-127, 128, (M, K), device='cuda', dtype=torch.int8) for _ in range(3)]
        sa = torch.rand(M, device='cuda') + .5; sb = torch.rand(N, device='cuda') + .5
        o = torch.empty(M, N // 2 if sw else N, device='cuda', dtype=torch.bfloat16 if sw else torch.float16)
        r = {}
        for c in range(16):
            try:
                fn = lambda i, c=c: S.gemm('sp8', c, Wc, E, xs[i % 3], o, N, M, K, sa=sa, sb=sb, epi='swiglu' if sw else 'fp16a', alpha=2 ** -10)
                fn(0); torch.cuda.synchronize(); r[c] = tm(fn)[0]
            except Exception as ex:
                r[c] = None
        old = min(v for c, v in r.items() if v and c < 12); new = min((v for c, v in r.items() if v and c >= 12), default=None)
        out[f'{name}|{M}'] = dict(per_cfg=r, best_old=old, best_new=new)
        print(name, M, 'best 0-11', old, 'best 12-15', new, {c: v for c, v in r.items() if c >= 12}, flush=True)
json.dump(out, open(os.path.expanduser('~/work/q4/res_tune_sp.json'), 'w'), indent=1)
