"""J14 projections (laptop, arithmetic) from the measured A10G latency records (lat*.jsonl): J5's method.
GEMM: additive roofline split at the A10G (Tc = rows * 2.745 GFLOP / R, Tm = weight bytes / BW), card time = g * (f R_a/R_c + (1-f) BW_a/BW_c);
non-GEMM: divided by s = 1.2 (3090), 1.5 (4090), 1.8 (5090) [speculation, as J5]. Host time is inside the measured wall (H2D + D2H)."""
import json, sys, glob, os, collections
FL = 2.745e9; P_ALL = 1.3726e9
CARDS = {'A10G': dict(i8=140e12, f16=70e12, bw=510e9, s=1.0),
         '3090': dict(i8=284e12, f16=142e12, bw=0.85 * 936e9, s=1.2),
         '4090': dict(i8=660e12, f16=330e12, bw=0.85 * 1008e9, s=1.5),
         '5090': dict(i8=838e12, f16=419e12, bw=0.85 * 1792e9, s=1.8)}
BYTES = {'bf16': 2.0, 'w8a8b8': 1.0 + 0.06}
RATE = {'bf16': 'f16', 'w8a8b8': 'i8'}


def project(rec):
    fmt = rec['prec']; g = rec['kern'].get('gemm', 0.0); n = rec['wall']['median'] - g; rows = rec['rows']
    a = CARDS['A10G']; R = lambda c: CARDS[c][RATE[fmt]]
    Tc = rows * FL / R('A10G'); Tm = P_ALL * BYTES[fmt] / a['bw']; f = Tc / (Tc + Tm)
    return {c: round(g * (f * R('A10G') / R(c) + (1 - f) * a['bw'] / sp['bw']) + n / sp['s'], 2) for c, sp in CARDS.items()}


if __name__ == '__main__':
    recs = {}
    for f in (sys.argv[1:] or sorted(glob.glob(os.path.expanduser('~/decider2/j14/lat*.jsonl')))):
        for l in open(f):
            r = json.loads(l); tg = os.path.basename(f).replace('lat', '').replace('.jsonl', ''); r['mode'] = r['mode'] + ('' if r['mode'] == 'plain' else tg); r['key'] = r['key'] + tg; recs[r['key']] = r
    tab = collections.defaultdict(dict)
    for k, r in recs.items():
        tab[(r['prec'], r['set'], r['mode'])][r['T']] = r
    Ts = sorted({r['T'] for r in recs.values()})
    out = {}
    for (prec, sn, md), d in sorted(tab.items()):
        row = []
        for T in Ts:
            r = d.get(T)
            if r is None: row.append('   -  '); continue
            p = project(r); out[r['key']] = dict(rows=r['rows'], a10g=r['wall']['median'], p95=r['wall']['p95'], gemm=r['kern'].get('gemm'), **{c: v for c, v in p.items() if c != 'A10G'})
            row.append(f"{r['wall']['median']:6.2f}")
        print(f'{prec:7s} {sn:5s} {md:7s} ' + ' '.join(row))
    print('projections (3090 / 4090 / 5090):')
    for (prec, sn, md), d in sorted(tab.items()):
        print(f'{prec:7s} {sn:5s} {md:7s} ' + ' | '.join(f"T{T}: " + '/'.join(f"{v:.1f}" for c, v in project(d[T]).items() if c != 'A10G') for T in Ts if T in d))
    json.dump(out, open(os.path.expanduser('~/decider2/j14/lat_proj.json'), 'w'), indent=1)
