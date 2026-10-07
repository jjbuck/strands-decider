"""B13 exactness checks on evalkit questions (adapted from Q1's q1b13.py; Q1's file is not edited):
 skip23 : layer 23, STATE rows compute only their K/V (Win output columns 4096:5120, by a separate GEMM over those weight rows); their q/gate columns
          and their Wo, Wgu, Wd outputs are set to NaN (proves they are never read); question rows unchanged.
 table0 : layer 0's Win replaced by a per-token table: the GEMM run once per distinct token id (one representative row each), then gathered.
 both   : skip23 + table0.
Reports per config and mode: max |d logit|, argmax flips, NaN logits, against the unmodified run of the same format.
Also (--vocab): distinct token ids in banking train-split traffic and the share of banking eval tokens they cover (table size for a per-deployment table).
python q5b13.py --cfgs dense,b8 --modes skip23,table0,both [--sub all|real|bank] [--limit N]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF, q5lib as Q5
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--cfgs', default='dense,b8'); ap.add_argument('--modes', default='skip23,table0,both')
ap.add_argument('--sub', default='all'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--vocab', action='store_true')
ap.add_argument('--out', default=os.path.expanduser('~/work/q5/res_b13.json')); a = ap.parse_args()
CF = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))
g = QL.Q1(grad=False)
rows = list(EK.all_question_items())
dom = {}
for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe'):
    for it in EK.load_suite(s): dom[it['id']] = it.get('domain', '')
if a.sub == 'real': rows = [r for r in rows if r[0] == 'REAL-agree']
if a.sub == 'bank': rows = [r for r in rows if dom.get(r[1], '').startswith('banking')]
if a.limit: rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
out = json.load(open(a.out)) if os.path.exists(a.out) else {}

if a.vocab:
    tr = set(); ntr = 0
    for it in Q5.req_set('bank', 'cal', 100000, maxT=10 ** 9, minS=0, per_task=10 ** 6) + Q5.req_set('bank', 'dev', 100000, maxT=10 ** 9, minS=0, per_task=10 ** 6):
        pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']; tr.update(ids); ntr += len(ids)
    cov = tot = 0; ev = set()
    for (suite, iid, qn, state, qspec) in rows:
        if not dom.get(iid, '').startswith('banking'): continue
        pr = g.prep(state, qspec); ids = pr['s'] + pr['q']; ev.update(ids)
        tot += len(ids); cov += sum(1 for t in ids if t in tr)
    out['vocab'] = dict(train_bank_tokens=ntr, train_bank_distinct=len(tr), eval_bank_tokens=tot, eval_bank_distinct=len(ev),
                        eval_token_coverage=cov / max(tot, 1), eval_distinct_covered=len(ev & tr) / max(len(ev), 1),
                        table_MB_bf16=len(tr) * 8224 * 2 / 1e6, table_MB_int8=len(tr) * 8224 / 1e6, vocab_size=int(g.embed.shape[0]))
    print('vocab', out['vocab'], flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)

for tag in a.cfgs.split(','):
    spec = None if tag == 'dense' else CF[tag]
    fm = QF.Fmt(g, spec) if spec else None

    def plain(gg, i, k, x, xn):
        return fm(gg, i, k, x, xn) if fm is not None else x @ gg.L[i][k].t()

    def sub_rows(gg, i, k, x, xn, lo, hi, as_q):
        q0 = gg._q0; gg._q0 = 0 if as_q else (hi - lo + 1)
        y = plain(gg, i, k, x[lo:hi], None if xn is None else xn[lo:hi]); gg._q0 = q0
        return y

    def skip23(gg, i, k, x, xn):
        q0 = gg._q0; T = x.shape[0]
        N = gg.L[i][k].shape[0]
        y = torch.full((T, N), float('nan'), device=x.device, dtype=x.dtype)
        if q0 < T: y[q0:] = sub_rows(gg, i, k, x, xn, q0, T, True)
        if k == 'Win' and q0 > 0:                       # state rows: K/V columns only
            if fm is None:
                y[:q0, 4096:5120] = x[:q0] @ gg.L[i]['Win'][4096:5120].t()
            else:
                y[:q0, 4096:5120] = sub_rows(gg, i, k, x, xn, 0, q0, False)[:, 4096:5120]
        return y

    def table0(gg, i, k, x, xn):
        ids = gg._ids; q0 = gg._q0; T = len(ids)
        y = torch.empty(T, gg.L[i][k].shape[0], device=x.device, dtype=x.dtype)
        roles = ((0, q0, False), (q0, T, True)) if (fm is not None and spec.get('a_s') != spec.get('a_q')) else ((0, T, False),)
        for lo, hi, as_q in roles:
            if hi <= lo: continue
            u2, inv2 = torch.unique(ids[lo:hi], return_inverse=True)
            p2 = torch.full((len(u2),), hi, dtype=torch.long, device=ids.device).scatter_reduce(0, inv2, torch.arange(lo, hi, device=ids.device), reduce='amin')
            xs = x[p2]; xns = None if xn is None else xn[p2]
            q0s = gg._q0; gg._q0 = 0 if as_q else len(u2) + 1
            tb = plain(gg, i, k, xs, xns); gg._q0 = q0s
            y[lo:hi] = tb[inv2]
        return y

    modes = a.modes.split(',')
    R = {m: dict(max_dlogit=0.0, flips=0, n=0, nan_logits=0, by_suite={}) for m in modes}
    t0 = time.time()
    for qi, (suite, iid, qn, state, qspec) in enumerate(rows):
        pr = g.prep(state, qspec); ids = pr['s'] + pr['q']
        g._ids = torch.tensor(ids, device=g.dev)
        lgs = {}; x23 = None
        for m in ['base'] + modes:
            g.qfn = {(i, k): plain for i in range(24) for k in QL.GEMMS}
            if m in ('skip23', 'both'):
                for k in ('Win', 'Wo', 'Wgu', 'Wd'): g.qfn[(23, k)] = skip23
            if m in ('table0', 'both'): g.qfn[(0, 'Win')] = table0
            with torch.no_grad():
                if m == 'base':
                    h, caps = g.fwd(ids, keep_x=(22,), q0=pr['q0']); x23 = caps[22]
                elif m == 'skip23':                      # layers 0-22 are identical to base: rerun only layer 23 from base's residual
                    h, _ = Q5.fwd_from(g, ids, x23, 23, q0=pr['q0'])
                else:
                    h, _ = g.fwd(ids, q0=pr['q0'])
                lgs[m] = g.logits(h, pr)
        g.qfn = {}
        b = lgs['base']
        for m in modes:
            s_ = lgs[m]; res = R[m]
            res['n'] += 1; res['nan_logits'] += int(torch.isnan(s_).any())
            dl = float((s_ - b).abs().max()); fl = int(s_.argmax() != b.argmax())
            res['max_dlogit'] = max(res['max_dlogit'], dl); res['flips'] += fl
            bs = res['by_suite'].setdefault(suite, dict(n=0, flips=0, max_dlogit=0.0, exact=0))
            bs['n'] += 1; bs['flips'] += fl; bs['max_dlogit'] = max(bs['max_dlogit'], dl); bs['exact'] += int(dl == 0.0)
        if (qi + 1) % 200 == 0:
            print(tag, qi + 1, f'{time.time()-t0:.0f}s', {m: (R[m]['flips'], R[m]['max_dlogit']) for m in modes}, flush=True)
            for m in modes: out[f'{tag}::{m}::{a.sub}::{a.limit}'] = dict(R[m], partial=True)
            json.dump(out, open(a.out, 'w'), indent=1)
    for m in modes:
        res = R[m]; res['exact_share'] = sum(v['exact'] for v in res['by_suite'].values()) / max(res['n'], 1); res['secs'] = time.time() - t0
        out[f'{tag}::{m}::{a.sub}::{a.limit}'] = res; print(tag, m, {kk: v for kk, v in res.items() if kk != 'by_suite'}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
print('done', flush=True)
