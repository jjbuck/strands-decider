"""J12 diagnostic: do the trained XR pointers find the right literals on CF-probe (template-OOD)?
Gold: in the keyed (non-distractor) record, the two compared fields (amount_vs_limit, date_order), or question literal vs record field
(status_equal: status; id_match: owner/user id).  For every XR instance and head, at the answer row: argmax pair hit and p_a*p_b on gold.
python diag_xr.py --ck CKPT --out diag.json"""
import os, sys, json, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from xrlib import HX, Prep

ap = argparse.ArgumentParser(); ap.add_argument('--ck', required=True); ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
FB = {'banking_knowledge': ('balance', 'daily_transfer_limit', 'statement_date', 'payment_due_date', 'owner_user_id'),
      'retail': ('total_paid', 'refund_limit', 'order_date', 'return_deadline', 'user_id'),
      'airline': ('total_paid', 'refund_limit', 'booking_date', 'cancellation_deadline', 'user_id')}
m = HX(); m.free_hf(); m.load_all(os.path.expanduser(a.ck)); prep = Prep(m.p.eng)
items = {it['id']: it for it in EK.load_suite('CF-probe')}
pairs = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/CF-probe.pairs.jsonl'))]
if a.limit: pairs = pairs[:a.limit]
res = collections.defaultdict(lambda: collections.defaultdict(list))
with torch.inference_mode():
    for p in pairs:
        kind = p['kind'].replace('probe_', ''); base = kind.replace('_distract', '')
        amt, lim, d1, d2, idf = FB[p['domain']]
        for side in ('a', 'b'):
            it = items[p[side]]; pr = prep(it['state'], it['questions']['probe'])
            ix = m.ix_for(pr); sl = ix['_sl']; lits = sl['lits']; isq = sl['isq']
            keyblk = {l.blk_in for l, q in zip(lits, isq) if not q and l.eq == p['detail']['key'].lower()}
            def fslot(fname):
                c = [j for j, (l, q) in enumerate(zip(lits, isq)) if not q and l.blk_in in keyblk and l.field == fname.replace('_', ' ')]
                return c[-1] if c else None
            if base == 'amount_vs_limit': g = (fslot(amt), fslot(lim))
            elif base == 'date_order': g = (fslot(d1), fslot(d2))
            elif base == 'status_equal':
                qv = [j for j, (l, q) in enumerate(zip(lits, isq)) if q and l.eq in ('open', 'frozen', 'closed', 'delivered', 'pending', 'cancelled', 'confirmed')]
                g = (qv[0] if qv else None, fslot('status'))
            else:
                qv = [j for j, (l, q) in enumerate(zip(lits, isq)) if q and l.kind in ('num', 'id') and l.text.isdigit()]
                g = (qv[0] if qv else None, fslot(idf))
            if None in g: res[kind]['located'].append(0); continue
            res[kind]['located'].append(1)
            m.fwd(pr, ix=ix, keep_last=True)
            best = 0.0; hit = 0
            for x in m.xr:
                pa, pb = x.last
                pp = torch.maximum(pa[:, g[0]] * pb[:, g[1]], pa[:, g[1]] * pb[:, g[0]])
                best = max(best, float(pp.max()))
                am, bm = pa.argmax(-1), pb.argmax(-1)
                hit |= int(any((int(am[h]), int(bm[h])) in ((g[0], g[1]), (g[1], g[0])) for h in range(pa.shape[0])))
            res[kind]['best_pab'].append(best); res[kind]['hit'].append(hit)
out = {k: {kk: (sum(v) / len(v) if v else None) for kk, v in d.items()} | {'n': len(d['located'])} for k, d in res.items()}
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))
