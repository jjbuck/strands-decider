"""B1 fused prologue (codes + Z in one pass) vs RTN-only prologue and vs dX-write + separate Z GEMM; CUDA-graph timing.
python bench_qz.py"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q2')]
import q2k as Q, q2pro as PR
from bench_pro import gtm
dev = 'cuda'
OUT = os.path.expanduser('~/work/q2/res_qz.jsonl')
torch.manual_seed(0)
# correctness
M, K, R = 300, 2048, 64
y = torch.randn(M, K, device=dev).to(torch.bfloat16); A = (torch.randn(K, R, device=dev) * 0.05).to(torch.bfloat16)
Qb = torch.empty(M, K // 2, device=dev, dtype=torch.uint8); SA = torch.empty(M, device=dev); Z = torch.empty(M, R, device=dev, dtype=torch.bfloat16)
st = torch.zeros(64, device=dev); w = torch.ones(M, device=dev)
PR.qz(y, A, Qb, SA, Z, stat=st, w=w); torch.cuda.synchronize()
q, s = Q.quant_rows(y.float(), 7, 0.9)
dx = (y.float() - s[:, None] * q.float()).to(torch.bfloat16)
zr = dx.float() @ A.float()
print('codes eq', torch.equal(Q.unpack4(Qb), q), 'scales eq', torch.equal(SA, s), 'Z rel err', ((Z.float() - zr).abs().max() / zr.abs().max()).item(),
      'stat rel err', (st.sum() / (zr * zr).sum() - 1).item(), flush=True)
for M in (140, 400, 1125, 4125):
    for K in (2048, 6144):
        ys = [torch.randn(M, K, device=dev).to(torch.bfloat16) for _ in range(3)]
        b = PR.Bufs(M, K)
        r = dict(M=M, K=K, rtn=gtm(lambda i: PR.rowq(ys[i % 3], b, 0)), rtn_dx=gtm(lambda i: PR.rowq(ys[i % 3], b, 2)))
        for R in (32, 64, 128, 256):
            A = (torch.randn(K, R, device=dev) * 0.05).to(torch.bfloat16); Z = torch.empty(M, R, device=dev, dtype=torch.bfloat16)
            best = None
            for BM in (16, 32, 64):
                for BK in (64, 128):
                    try:
                        t = gtm(lambda i: PR.qz(ys[i % 3], A, b.Q, b.SA, Z, BM=BM, BK=BK))
                    except Exception as e:
                        continue
                    if best is None or t[0] < best[1][0]: best = ((BM, BK), t)
            r[f'qz_r{R}'] = best
            r[f'z_mm_r{R}'] = gtm(lambda i: torch.mm(b.DX, A, out=None))
        print(json.dumps(r), flush=True)
        with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')
