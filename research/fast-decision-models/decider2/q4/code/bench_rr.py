"""Q4 B11 timing for Q5's most accurate structure so far: 2:4 on STATE rows only (question rows through the dense int8 weights; two weight
copies), as two launches per GEMM: q4sp sparse int8 on rows [0, q0) + the best deployed dense int8 kernel on rows [q0, M).
python bench_rr.py -> ~/work/q4/res_rr.jsonl  (best config by median of 30, us; dense comparator = the deployed dense kernel on all M rows)"""
import os, sys, json, ctypes, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q4sp as S
import test_sp as TS
import qgemm as QG
from bench_sp import tm, best, SH, j5lib
dev = 'cuda'
KIND = os.environ.get('SPK', 'sp8')
OUT = os.path.expanduser('~/work/q4/res_rr.jsonl' if KIND == 'sp8' else f'~/work/q4/res_rr_{KIND}.jsonl')


def dense(M, N, K, sw, x8, w8, sa, sb, o16, oh):
    c = [(f'cut{c}', (lambda i, c=c: QG.gemm('s8', x8[i % 3], w8, 2 ** -10, c, out=o16))) for c in range(5)]
    j5 = j5lib()
    def j5f(i, c):
        rc = j5.s8_gemm(x8[i % 3].data_ptr(), w8.data_ptr(), o16.data_ptr(), M, N, K, 2 ** -10, c, torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(rc)
    c += [(f'j5c{c}', (lambda i, c=c: j5f(i, c))) for c in range(11)]
    if sw: c += [(f'evt{c}', (lambda i, c=c: QG.swiglu_gemm('s8', x8[i % 3], w8, sa, sb, cfg=c, out=oh))) for c in (0, 1, 2, 3)]
    return best(c)


def main():
    TS.set_layout()
    for (ms, mq) in (((1000, 125), (4000, 125), (256, 125), (64, 125), (1000, 3720)) if KIND == 'sp8' else ((1000, 125), (4000, 125), (256, 125))):
        for name, (N, K) in SH.items():
            sw = name == 'gate_up'
            g = lambda *s: torch.randint(-127, 128, s, device=dev, dtype=torch.int8)
            w8 = g(N, K); Wc, E, _ = S.compress(w8, S.mask24_mag(w8, 'sp8'), 'sp8')
            if KIND == 'sp4':     # state rows as int4 codes through 2:4 int4 weights (pair-wise), question rows int8 dense
                w4 = torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8); Wc, E, _ = S.compress(w4, S.mask24_mag(w4, 'sp4'), 'sp4')
            M = ms + mq
            xs = [g(M, K) for _ in range(3)]
            sa = torch.rand(M, device=dev) + .5; sb = torch.rand(N, device=dev) + .5
            o16 = torch.empty(M, N, device=dev, dtype=torch.float16); oh = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
            r = dict(shape=name, M_state=ms, M_q=mq, N=N, K=K)
            r['dense_all'] = dense(M, N, K, sw, xs, w8, sa, sb, o16, oh)
            xq = [x[ms:] for x in xs]; xst = [x[:ms] for x in xs]
            r['dense_q'] = dense(mq, N, K, sw, xq, w8, sa[ms:], sb, o16[ms:], oh[ms:])
            epi = 'swiglu' if sw else 'fp16a'; ob = oh if sw else o16
            if KIND == 'sp4': xst = [S.pack4(torch.randint(-7, 8, (ms, K), device=dev, dtype=torch.int8)) for _ in range(3)]
            r['kind'] = KIND
            r['sparse_state'] = best([(f's{c}', (lambda i, c=c: S.gemm(KIND, c, Wc, E, xst[i % 3], ob, N, ms, K, sa=sa, sb=sb, epi=epi, alpha=2 ** -10 if KIND == 'sp8' else 0.5))) for c in range(16)])
            r['rowrole_sum'] = round(r['dense_q'][1] + r['sparse_state'][1], 2)
            r['speedup'] = round(r['dense_all'][1] / r['rowrole_sum'], 3)
            print(json.dumps(r), flush=True)
            with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
