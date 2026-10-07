"""Laptop: REAL-agree / LONG questions where the config disagrees with hobson but the in-runtime bf16 agrees (and the reverse), with margins.
python3 q5realloss.py dense.json cfg.json [tag]"""
import sys, os, json
sys.path[:0] = [os.path.expanduser('~/decider2/evalkit')]
import evalkit as EK
am = lambda d: max(d, key=d.get)
D = json.load(open(sys.argv[1]))['preds']['dense']; P = json.load(open(sys.argv[2]))['preds']
tag = sys.argv[3] if len(sys.argv) > 3 else next(iter(P)); p = P[tag]
for s in ('REAL-agree', 'LONG'):
    hp = EK._as_preds(s, 'hobson')
    for it in EK.load_suite(s):
        for q in it['questions']:
            h = hp.get(it['id'], {}).get(q); d = D.get(it['id'], {}).get(q); x = p.get(it['id'], {}).get(q)
            if None in (h, d, x): continue
            a = am(EK._norm(h)); da = am(EK._norm(d)) == a; xa = am(EK._norm(x)) == a
            if da != xa:
                print(s, 'LOST' if da else 'GAINED', it.get('domain'), q, f"hobson P({a}) {EK._norm(h)[a]:.3f} bf16 {EK._norm(d).get(a,0):.3f} cfg {EK._norm(x).get(a,0):.3f}")
