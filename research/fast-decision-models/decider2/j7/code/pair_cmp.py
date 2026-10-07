"""paired comparison of two models (A vs B): McNemar on CF / CF-probe pairs (both right), JB-all, JB-hard, REAL-label, and agreement-with-hobson
on REAL-agree / LONG state-dependent questions.  python3 pair_cmp.py A.json B.json"""
import sys, json, math
sys.path.insert(0, '~/decider2/evalkit'); sys.path.insert(0, '~/decider2/j7/code')
import evalkit as EK
from evalkit import _norm, _arg
from score_j7 import mcnemar
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
out = {}
for s in ('CF', 'CF-probe'):
    a = []; b = []; kinds = {}
    for pr in EK._pairs(s):
        q = pr['q']
        try:
            x = _arg(_norm(A[pr['a']][q])) == pr['ea'] and _arg(_norm(A[pr['b']][q])) == pr['eb']
            y = _arg(_norm(B[pr['a']][q])) == pr['ea'] and _arg(_norm(B[pr['b']][q])) == pr['eb']
        except KeyError: continue
        a.append(x); b.append(y)
        k = 'identity' if pr['kind'].startswith('id_') else pr['kind'].replace('_distract', '')
        kinds.setdefault(k, ([], [])); kinds[k][0].append(x); kinds[k][1].append(y)
    out[s] = dict(all=mcnemar(a, b), **{k: mcnemar(*v) for k, v in kinds.items()})
for s in ('JB-all', 'JB-hard', 'REAL-label'):
    a = []; b = []
    for it in EK.load_suite(s):
        for q in it['questions']:
            e = (it.get('expected') or {}).get(q)
            if e is None or it['id'] not in A or it['id'] not in B: continue
            a.append(_arg(_norm(A[it['id']][q])) == str(e)); b.append(_arg(_norm(B[it['id']][q])) == str(e))
    out[s] = mcnemar(a, b)
for s in ('REAL-agree', 'LONG'):
    refs = EK.load_refs(s); a = []; b = []
    for it in EK.load_suite(s):
        r = refs[it['id']]
        for q in it['questions']:
            h = _arg(_norm(r['hobson'][q])); n0 = _arg(_norm(r['cfg']['nostate'][q]))
            if h == n0: continue
            a.append(_arg(_norm(A[it['id']][q])) == h); b.append(_arg(_norm(B[it['id']][q])) == h)
    out[s + '_agree_sd'] = mcnemar(a, b)
print(json.dumps(out, indent=None))
