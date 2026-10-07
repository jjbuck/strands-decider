"""split-K correctness (bit-exact, repeated launches reuse the counters) and timing for question-row GEMMs (125 rows)"""
import os, sys, torch, json
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q, qgemm as QG
from bench_gemm import best, SH
from bench_pro import gtm
dev = 'cuda'; torch.manual_seed(0)
ok = True
for (M, N, K) in [(125, 2048, 6144), (125, 2048, 2048), (77, 520, 1024), (300, 12288, 2048)]:
    a8 = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    qa = torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8); qw = torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8)
    sa = torch.rand(M, device=dev); sb = torch.rand(N, device=dev)
    r8 = Q.ref_int(a8, sa, w8, sb).to(torch.bfloat16); r4 = Q.ref_int(qa, sa, qw, sb).to(torch.bfloat16)
    o = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    for spl in (2, 3, 4):
        ws, sem = Q.splitk_ws(spl, M, N)
        for cfg in (0, 3, 4):
            for rep in range(3):
                o.zero_(); Q.run('s8', cfg, Q.prob(a8, w8, 's8', o, sa, sb, splits=spl, skws=ws, sksem=sem)); torch.cuda.synchronize()
                e8 = torch.equal(o, r8)
                o.zero_(); Q.run('s4', cfg, Q.prob(Q.pack4(qa), Q.pack4(qw), 's4', o, sa, sb, splits=spl, skws=ws, sksem=sem)); torch.cuda.synchronize()
                e4 = torch.equal(o, r4)
                ok &= e8 and e4
            print(M, N, K, 'splits', spl, 'cfg', cfg, 's8 eq', e8, 's4 eq', e4, 'sem clean', int(sem.abs().sum()) == 0, flush=True)
print('SPLITK', 'ALL OK' if ok else 'FAIL')
# timing: question rows (125) int8 at hobson shapes, splits 1/2/4, best config, in a graph
CNT = {'gdn_in': 18, 'attn_in': 6, 'out': 24, 'gate_up': 24, 'down': 24}
tot = {1: 0., 2: 0., 4: 0., 'best': 0., 'cut': 0.}
for name, (N, K) in SH.items():
    M = 125
    a8 = [torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for _ in range(3)]; w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    sa = torch.rand(M, device=dev); sb = torch.rand(N, device=dev)
    o = torch.empty(M, N // 2 if name == 'gate_up' else N, device=dev, dtype=torch.bfloat16); o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
    epi = 'swiglu' if name == 'gate_up' else 'bf16'
    r = {}
    for spl in (1, 2, 4):
        ws, sem = Q.splitk_ws(spl, M, N)
        bt = None
        for cfg in range(10):
            try:
                t = gtm(lambda i, cfg=cfg: Q.run('s8', cfg, Q.prob(a8[i % 3], w8, 's8', o, sa, sb, epi=epi, N=N, splits=spl, skws=ws, sksem=sem)))
            except Exception: continue
            if bt is None or t[0] < bt[1][0]: bt = (cfg, t)
        r[spl] = bt
    bc = None
    for cfg in range(5):
        try: t = gtm(lambda i, cfg=cfg: QG.gemm('s8', a8[i % 3], w8, 2 ** -10, cfg, out=o16))
        except Exception: continue
        if bc is None or t[0] < bc[1][0]: bc = (cfg, t)
    r['cut'] = bc
    for k in (1, 2, 4): tot[k] += CNT[name] * r[k][1][0]
    tot['best'] += CNT[name] * min(r[k][1][0] for k in (1, 2, 4)); tot['cut'] += CNT[name] * bc[1][0]
    print(name, json.dumps({str(k): v for k, v in r.items()}), flush=True)
print('question-row int8 GEMMs, 125 rows, sum over 96 GEMMs (ms, graph):', {k: round(v / 1000, 2) for k, v in tot.items()}, flush=True)
