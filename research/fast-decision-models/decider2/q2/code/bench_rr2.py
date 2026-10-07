"""question-row carriers with ONE int4 weight copy: state rows CUTLASS s4 + question rows CUTLASS s8xs4 (W4A8, H2 h2mix), back to back.
python bench_rr2.py T,T nq"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q, qgemm as QG
from bench_gemm import tm, best, SH
dev = 'cuda'
OUT = os.path.expanduser('~/work/q2/res_rr2.jsonl')
CNT = {'gdn_in': 18, 'attn_in': 6, 'out': 24, 'gate_up': 24, 'down': 24}
for T in [int(t) for t in sys.argv[1].split(',')]:
    nq = int(sys.argv[2]); M = T + nq; q0 = T
    tot = dict(s4_all=0., w4a8_q=0., seq=0., s8_q=0.)
    for name, (N, K) in SH.items():
        a4 = [Q.pack4(torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8)) for _ in range(3)]
        a8 = [torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for _ in range(3)]
        w4 = Q.pack4(torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8)); w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        r = dict(shape=name, T=T, nq=nq)
        r['s4_all'] = best([(c, (lambda i, c=c: QG.gemm('s4', a4[i % 3], w4, 0.5, c, out=o16))) for c in range(5)])
        b4 = best([(c, (lambda i, c=c: QG.gemm('s4', a4[i % 3][:q0], w4, 0.5, c, out=o16[:q0]))) for c in range(5)])
        bq = best([(c, (lambda i, c=c: QG.gemm('s8s4', a8[i % 3][q0:], w4, 2 ** -7, c, out=o16[q0:]))) for c in range(5)])
        b8 = best([(c, (lambda i, c=c: QG.gemm('s8', a8[i % 3][q0:], w8, 2 ** -10, c, out=o16[q0:]))) for c in range(5)])
        r['w4a8_q'] = bq; r['s8_q'] = b8; r['s4_state'] = b4
        r['seq'] = best([('seq', lambda i: (QG.gemm('s4', a4[i % 3][:q0], w4, 0.5, b4[0], out=o16[:q0]), QG.gemm('s8s4', a8[i % 3][q0:], w4, 2 ** -7, bq[0], out=o16[q0:])))])
        for k in tot: tot[k] += CNT[name] * r[k][1]
        print(json.dumps(r), flush=True)
        with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')
    print(T, nq, {k: round(v / 1000, 2) for k, v in tot.items()}, flush=True)
