"""Laptop: which hobson-tracked CF / CF-probe pairs a config loses, with the dense (in-runtime bf16) margin of the failing item.
python3 q5cfloss.py dense.json cfg.json [tag]"""
import sys, os, json
sys.path[:0] = [os.path.expanduser('~/decider2/evalkit')]
import evalkit as EK
am = lambda d: max(d, key=d.get)
D = json.load(open(sys.argv[1]))['preds']['dense']; P = json.load(open(sys.argv[2]))['preds']
tag = sys.argv[3] if len(sys.argv) > 3 else next(iter(P))
p = P[tag]
for s in ('CF', 'CF-probe'):
    hp = EK._as_preds(s, 'hobson'); its = {it['id']: it for it in EK.load_suite(s)}
    for pr in EK._pairs(s):
        q = pr['q']
        try:
            ho = am(EK._norm(hp[pr['a']][q])) == pr['ea'] and am(EK._norm(hp[pr['b']][q])) == pr['eb']
            po = am(EK._norm(p[pr['a']][q])) == pr['ea'] and am(EK._norm(p[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho and not po:
            out = []
            for side, e in (('a', pr['ea']), ('b', pr['eb'])):
                dd = EK._norm(D[pr[side]][q]); pp = EK._norm(p[pr[side]][q])
                out.append(f"{side}: exp {e} dense P {dd.get(e, 0):.3f} cfg P {pp.get(e, 0):.3f}")
            print(s, its[pr['a']].get('domain'), pr.get('kind'), q, ' | '.join(out))
