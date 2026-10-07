"""J3: latency table from the measured A10G lat_dt.json + projections to 3090 / 4090 / 5090 (laptop, pure python).
Projection = measured A10G x (cost-model time on the card / cost-model time on the A10G) for the same configuration, length and question
tokens (costmodel.py: GEMM time / card GEMM ratio, memory-bound time / bandwidth ratio; FAST_DECISION_MODEL sec 9 method).
python3 proj_lat.py ../lat_dt.json ../lat_table.json"""
import sys, json, re
from costmodel import latency
lat = json.load(open(sys.argv[1]))
out = {}
rows = collections = {}
for k, v in lat.items():
    if k == 'meta': continue
    mq = re.match(r'(Q\d+)_T(\d+)_(\w+)', k); qn, T, cfg = mq.group(1), int(mq.group(2)), mq.group(3)
    nq = int(qn[1:]); qt = v['q_tokens'] / nq
    if cfg in ('hob1', 'hobB'): Ls = 24
    else: Ls = int(re.sub(r'\D', '', cfg))
    base = latency(T, qt, nq, Ls, 'A')
    pr = {c: round(v['median'] * latency(T, qt, nq, Ls, 'A', c) / base, 1) for c in ('3090', '4090', '5090')}
    out[k] = dict(A10G=v['median'], A10G_p95=v['p95'], **pr)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
Ts = sorted({int(re.match(r'Q\d+_T(\d+)_', k).group(1)) for k in out})
for qn in ('Q1', 'Q4'):
    cfgs = sorted({re.match(r'Q\d+_T\d+_(\w+)', k).group(1) for k in out if k.startswith(qn + '_')}, key=lambda c: (c[:3], len(c), c))
    for card in ('A10G', '3090', '4090', '5090'):
        print(f'\n{qn} {card} median ms' + ('' if card == 'A10G' else ' (projected)'))
        print('%-8s' % 'cfg' + ''.join('%9d' % T for T in Ts))
        for c in cfgs:
            print('%-8s' % c + ''.join('%9s' % (out.get(f'{qn}_T{T}_{c}', {}).get(card, '')) for T in Ts))
