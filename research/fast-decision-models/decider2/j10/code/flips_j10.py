"""Decision flips of a variant against a reference tag (the A8 model), per suite, with mean TV.  python3 flips_j10.py REF TAG [TAG...]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_j10 as S
ref = S.load(sys.argv[1])
for t in sys.argv[2:]:
    p = S.load(t); out = {}
    for su in S.RUN:
        f = f'{S.R}/{su}.{t}.jsonl'
        if not os.path.exists(f): continue
        ids = [json.loads(l)['id'] for l in open(f)]
        n = fl = 0; tv = 0.0
        for i in ids:
            for q, d in p[i].items():
                r = ref.get(i, {}).get(q)
                if r is None: continue
                n += 1; fl += int(S._arg(d) != S._arg(r)); tv += 0.5 * sum(abs(d.get(k, 0) - r.get(k, 0)) for k in set(d) | set(r))
        out[su] = dict(n=n, flips=fl, flip_rate=round(fl / max(n, 1), 4), tv=round(tv / max(n, 1), 4))
    print(t, 'vs', sys.argv[1], json.dumps(out))
