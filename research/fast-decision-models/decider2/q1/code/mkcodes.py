"""precompute GPTQ codes (H1 recipe) for every GEMM, 4 and 8 bits -> ~/work/q1/codes"""
import os, sys, time, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import q1lib as QL, q1fmt as QF
ba = int(sys.argv[1]) if len(sys.argv) > 1 else 0
g = QL.Q1(grad=False); fm = QF.Fmt(g, dict(ba16=bool(ba))); t0 = time.time()
for bits in (4, 8):
    for i in range(24):
        for k in QL.GEMMS:
            fm.wcodes(i, k, bits); fm.c = {}
        print(bits, i, f'{time.time()-t0:.0f}s', flush=True)
print('codes done', flush=True)
