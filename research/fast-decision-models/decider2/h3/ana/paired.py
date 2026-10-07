"""paired comparison of two configs on REAL-agree flips (vs hobson refs, or vs a given reference preds): exact McNemar on the per-question
flip indicator; plus TV difference with a bootstrap CI.  python3 paired.py A.json B.json [REF.json]"""
import sys, json, math, random
sys.path.insert(0, '~/decider2/evalkit'); import evalkit as EK
am = lambda d: max(d, key=d.get)
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
ref = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else EK._as_preds('REAL-agree', 'hobson')
ids = {it['id'] for it in EK.load_suite('REAL-agree')}
b = c = n = 0; dtv = []
for i in ids:
    for q in ref.get(i, {}):
        if q not in A.get(i, {}) or q not in B.get(i, {}): continue
        r = EK._norm(ref[i][q]); pa = EK._norm(A[i][q]); pb = EK._norm(B[i][q]); n += 1
        fa = am(pa) != am(r); fb = am(pb) != am(r); b += fa and not fb; c += fb and not fa
        dtv.append(EK._tv(pa, r) - EK._tv(pb, r))
k = min(b, c); m = b + c
p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
random.seed(0); bs = sorted(sum(random.choice(dtv) for _ in dtv) / len(dtv) for _ in range(2000))
print(f'n {n}: flips only-A {b}, only-B {c}, McNemar p {p:.3g}; mean TV(A)-TV(B) {sum(dtv)/len(dtv):+.4f} [95% {bs[50]:+.4f}, {bs[1950]:+.4f}]')
