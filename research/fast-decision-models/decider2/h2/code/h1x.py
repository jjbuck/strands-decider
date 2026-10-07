"""cross-check: H2 runtime (preds_<prec>.jsonl) vs H1's emulation (h1lib.H1, FORMAT v0, RTN) on the same REAL-agree questions."""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import h1lib as H, evalkit as EK
from kitrun import prep_question
N = int(sys.argv[1]); precs = sys.argv[2].split(',')
h = H.H1()
its = {it['id']: it for it in EK.load_suite('REAL-agree')}
mine = {}
for p in precs:
    mine[p] = {}
    for l in open(os.path.expanduser(f'~/work/h2/preds_{p}.jsonl')):
        r = json.loads(l)
        if r['suite'] == 'REAL-agree': mine[p][(r['id'], r['q'])] = r
keys = sorted(mine[precs[0]])[:: max(1, len(mine[precs[0]]) // N)][:N]
out = {}
for p in precs:
    h.set_prec(h.uniform(p)); h.drop_cache()
    agree = 0; dps = []; n = 0
    for iid, q in keys:
        pr = prep_question(h.p, its[iid], q)
        with torch.no_grad():
            hh, _ = h.fwd(pr['s'] + pr['q'])
            lg = h.logits(hh, pr)
        pe = torch.softmax(lg, -1).tolist(); pm = mine[p][(iid, q)]['probs']; labs = list(pm)
        pmv = [pm[l] for l in labs]
        agree += int(max(range(len(pe)), key=lambda j: pe[j]) == max(range(len(pmv)), key=lambda j: pmv[j]))
        dps.append(max(abs(a - b) for a, b in zip(pe, pmv))); n += 1
    dps.sort()
    out[p] = dict(n=n, decision_agree=agree / n, dp_median=dps[n // 2], dp_p90=dps[int(.9 * n)], dp_max=dps[-1])
    print(p, out[p], flush=True)
json.dump(out, open(os.path.expanduser('~/work/h2/h1x.json'), 'w'), indent=1)
