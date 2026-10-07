"""Q4 (laptop): per-forward GEMM time at hobson's shapes from the measured per-shape kernel times (arithmetic over MEASURED).
W8A8-b8 = 88 int8 GEMMs (deployed kernels: best of H2 CUTLASS / J5 s8 with fp16 output; H2 EVT SwiGLU for gate_up) + 8 bf16 GEMMs (cuBLAS,
Q2's measurement at the same shapes: 0.Wo 7.Wo 9.Wo 10.Wo 11.Wo 12.Wo 23.Wo 23.Wd). Sparse variants swap the int8 kernel of chosen GEMMs for
q4sp sp8 (fp16a epilogue = drop-in for QRT; SwiGLU epilogue for gate_up)."""
import json, sys
R = {}
for l in open('~/decider2/q4/res/res_sparse.jsonl'):
    r = json.loads(l); R[(r['shape'], r['M'])] = r
Q2 = {}
for l in open('~/decider2/q2/res/res_gemm_base.jsonl'):
    r = json.loads(l); Q2[(r['shape'], r['M'])] = r
B8 = {(0, 'Wo'), (7, 'Wo'), (9, 'Wo'), (10, 'Wo'), (11, 'Wo'), (12, 'Wo'), (23, 'Wo'), (23, 'Wd')}
def shape(i, k):
    if k == 'Win': return 'attn_in' if i % 4 == 3 else 'gdn_in'
    return {'Wo': 'out', 'Wgu': 'gate_up', 'Wd': 'down'}[k]
def dense8(s, M):
    r = R[(s, M)]
    if s == 'gate_up': return r['h2_s8_evt'][1]
    return min(r['h2_s8'][1], r['j5_s8'][1])
def dense4(s, M):
    r = R[(s, M)]
    return r['h2_s4_evt'][1] if s == 'gate_up' else r['h2_s4'][1]
def sp8(s, M):
    r = R[(s, M)]
    return r['q4_sp8'][1] if s == 'gate_up' else r['q4_sp8_fp16a'][1]
def sp4(s, M):
    r = R[(s, M)]
    return r['q4_sp4'][1] if s == 'gate_up' else r['q4_sp4_fp16a'][1]
def bf16(s, M): return Q2[(s, M)]['bf16_cublas'][1]
def total(M, sparse_layers=(), kind='8', sparse_set=None):
    t = 0.0
    for i in range(24):
        for k in ('Win', 'Wo', 'Wgu', 'Wd'):
            s = shape(i, k)
            if (i, k) in B8: t += bf16(s, M); continue
            sp = (i in sparse_layers) if sparse_set is None else ((i, k) in sparse_set)
            if kind == '8': t += sp8(s, M) if sp else dense8(s, M)
            else: t += sp4(s, M) if sp else dense4(s, M)
    return t / 1000
out = {}
for M in (140, 400, 1125, 4125):
    b = total(M); w4 = total(M, kind='4')
    rows = dict(b8_dense=b, b8_sparse_all=total(M, range(24)), b8_sparse_12_22=total(M, range(12, 23)), b8_sparse_mlp_12_22=total(M, sparse_set={(i, k) for i in range(12, 23) for k in ('Wgu', 'Wd')}),
                w4a4_dense=w4, w4a4_sparse_all=total(M, range(24), kind='4'))
    out[M] = {k: round(v, 2) for k, v in rows.items()}
    out[M].update({f'{k}_x_b8': round(v / b, 3) for k, v in rows.items()})
    print(M, json.dumps(out[M]))
json.dump(out, open('~/decider2/q4/res/gemm_time_b11.json', 'w'), indent=1)
