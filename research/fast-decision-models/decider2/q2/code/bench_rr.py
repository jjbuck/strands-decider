"""Q2 row-role GEMM options at hobson's shapes: state rows int4 + question rows int8.
  q2_grouped: one q2gemm launch (int8 tiles scheduled first);  cut_seq: H2 CUTLASS s4 then s8 (EVT for gate_up);  cut_2s: same on two streams.
python bench_rr.py T,T,..  nq   (rows = T + nq, question rows = nq)"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q, qgemm as QG
from bench_gemm import tm, best, SH
dev = 'cuda'
OUT = os.path.expanduser('~/work/q2/res_rr.jsonl')
s2 = torch.cuda.Stream()


def run(Ts, nq):
    for T in Ts:
        M = T + nq; q0 = T
        for name, (N, K) in SH.items():
            sw = name == 'gate_up'
            a4 = [Q.pack4(torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8)) for _ in range(3)]
            a8 = [torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for _ in range(3)]
            w4 = Q.pack4(torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8)); w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
            sa = torch.rand(M, device=dev); sb = torch.rand(N, device=dev)
            ob = torch.empty(M, N // 2 if sw else N, device=dev, dtype=torch.bfloat16); o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
            epi = 'swiglu' if sw else 'bf16'
            r = dict(shape=name, T=T, nq=nq, M=M, N=N, K=K)
            r['all_s8_cut'] = best([(f'c{c}', (lambda i, c=c: (QG.swiglu_gemm('s8', a8[i % 3], w8, sa, sb, cfg=c, out=ob) if sw else QG.gemm('s8', a8[i % 3], w8, 2 ** -10, c, out=o16)))) for c in ((1, 3) if sw else range(5))])
            r['all_s4_cut'] = best([(f'c{c}', (lambda i, c=c: (QG.swiglu_gemm('s4', a4[i % 3], w4, sa, sb, cfg=c, out=ob) if sw else QG.gemm('s4', a4[i % 3], w4, 0.5, c, out=o16)))) for c in ((0, 3) if sw else range(5))])
            r['all_s4_q2'] = best([(f'q{c}', (lambda i, c=c: Q.run('s4', c, Q.prob(a4[i % 3], w4, 's4', ob, sa, sb, epi=epi, N=N)))) for c in range(10)])
            if T > 0:
                r['q2_grouped'] = best([(f'q{c}', (lambda i, c=c: Q.run('s4', c, Q.prob(a4[i % 3][:q0], w4, 's4', ob, sa, sb, epi=epi, N=N),
                                                                         Q.prob(a8[i % 3][q0:], w8, 's8', ob, sa[q0:], sb, epi=epi, N=N, row0=q0)))) for c in range(10)])
                # CUTLASS: best config of each part chosen separately, then timed together
                def cut4(i, c, out=None):
                    if sw: return QG.swiglu_gemm('s4', a4[i % 3][:q0], w4, sa, sb, cfg=c, out=ob[:q0])
                    return QG.gemm('s4', a4[i % 3][:q0], w4, 0.5, c, out=o16[:q0])
                def cut8(i, c):
                    if sw: return QG.swiglu_gemm('s8', a8[i % 3][q0:], w8, sa[q0:], sb, cfg=c, out=ob[q0:])
                    return QG.gemm('s8', a8[i % 3][q0:], w8, 2 ** -10, c, out=o16[q0:])
                b4 = best([(c, (lambda i, c=c: cut4(i, c))) for c in ((0, 3) if sw else range(5))])
                b8 = best([(c, (lambda i, c=c: cut8(i, c))) for c in ((1, 3) if sw else range(5))])
                c4, c8 = b4[0], b8[0]
                r['cut_s4_part'] = b4; r['cut_s8_part'] = b8
                r['cut_seq'] = best([('seq', lambda i: (cut4(i, c4), cut8(i, c8)))])
                def two(i):
                    ev = torch.cuda.Event(); ev.record(); s2.wait_event(ev)
                    with torch.cuda.stream(s2): cut8(i, c8)
                    cut4(i, c4)
                    ev2 = torch.cuda.Event(); ev2.record(s2); torch.cuda.current_stream().wait_event(ev2)
                r['cut_2s'] = best([('2s', two)])
            print(json.dumps(r), flush=True)
            with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    run([int(t) for t in sys.argv[1].split(',')], int(sys.argv[2]))
