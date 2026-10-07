"""paired comparison on CF / CF-probe pairs hobson tracks: per pair, does config A / B also track it? exact McNemar."""
import sys, json, math
sys.path.insert(0, '~/decider2/evalkit'); import evalkit as EK
am = lambda d: max(d, key=d.get)
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
for s in ('CF', 'CF-probe'):
    hp = EK._as_preds(s, 'hobson'); b = c = n = 0
    for pr in EK._pairs(s):
        q = pr['q']
        try:
            if not (am(EK._norm(hp[pr['a']][q])) == pr['ea'] and am(EK._norm(hp[pr['b']][q])) == pr['eb']): continue
            ta = am(EK._norm(A[pr['a']][q])) == pr['ea'] and am(EK._norm(A[pr['b']][q])) == pr['eb']
            tb = am(EK._norm(B[pr['a']][q])) == pr['ea'] and am(EK._norm(B[pr['b']][q])) == pr['eb']
        except KeyError: continue
        n += 1; b += ta and not tb; c += tb and not ta
    k = min(b, c); m = b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    print(f'{s}: n {n} tracked only-A {b}, only-B {c}, p {p:.3g}')
