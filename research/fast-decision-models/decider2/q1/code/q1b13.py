"""B13 exactness checks on evalkit questions, for the bf16 runtime and for quantized formats:
 (a) layer 23: Wo, Wgu, Wd computed only for question rows; state rows of those GEMM outputs set to NaN (proves they are never read);
 (b) layer 0: Win replaced by a per-token table (the GEMM run once per distinct token id of the request, then gathered).
Reports max |d logit| and argmax flips against the unmodified run of the same format.
python q1b13.py --n 80 --cfgs dense,w4q8"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=80); ap.add_argument('--cfgs', default='dense,w4q8')
ap.add_argument('--out', default=os.path.expanduser('~/work/q1/res_b13.json')); a = ap.parse_args()
CF = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))
g = QL.Q1(grad=False)
rows = list(EK.all_question_items())
step = max(1, len(rows) // a.n); rows = rows[::step][:a.n]
out = {}
for tag in a.cfgs.split(','):
    spec = CF.get(tag)
    fm = QF.Fmt(g, spec) if spec else None
    base_q = {}

    def plain(gg, i, k, x, xn):
        return fm(gg, i, k, x, xn) if fm is not None else x @ gg.L[i][k].t()

    def skip23(gg, i, k, x, xn):
        q0 = gg._q0; y = torch.full((x.shape[0], gg.L[i][k].shape[0]), float('nan'), device=x.device, dtype=x.dtype)
        sub_xn = None if xn is None else xn[q0:]
        gg._q0 = 0                                      # inside the sub-call all rows are question rows
        y[q0:] = plain(gg, i, k, x[q0:], sub_xn); gg._q0 = q0
        return y

    tab_cache = {}

    def table0(gg, i, k, x, xn):
        ids = gg._ids; uniq, inv = torch.unique(ids, return_inverse=True)
        # one representative row per distinct token (its xn row is a function of the id only)
        pos = torch.full((len(uniq),), len(ids), dtype=torch.long, device=ids.device).scatter_reduce(0, inv, torch.arange(len(ids), device=ids.device), reduce='amin')
        q0 = gg._q0
        if fm is not None and spec.get('a_s') != spec.get('a_q'):
            # row role differs: build one table per role (state / question rows)
            y = torch.empty(len(ids), gg.L[i][k].shape[0], device=x.device, dtype=x.dtype)
            for lo, hi, qq in ((0, q0, len(ids) + 1), (q0, len(ids), 0)):
                if hi <= lo: continue
                u2, inv2 = torch.unique(ids[lo:hi], return_inverse=True)
                p2 = torch.full((len(u2),), hi, dtype=torch.long, device=ids.device).scatter_reduce(0, inv2, torch.arange(lo, hi, device=ids.device), reduce='amin')
                gg._q0 = qq
                tb = plain(gg, i, k, x[p2], None if xn is None else xn[p2]); gg._q0 = q0
                y[lo:hi] = tb[inv2]
            return y
        gg._q0 = len(ids) + 1
        tb = plain(gg, i, k, x[pos], None if xn is None else xn[pos]); gg._q0 = q0
        return tb[inv]

    res = dict(max_dlogit_23=0.0, flips_23=0, max_dlogit_0=0.0, flips_0=0, n=0, nan_in_logits=0)
    for (suite, iid, qn, state, qspec) in rows:
        pr = g.prep(state, qspec); ids = pr['s'] + pr['q']
        g._ids = torch.tensor(ids, device=g.dev)
        lgs = []
        for mode in ('base', 'skip23', 'table0'):
            g.qfn = {(i, k): (lambda gg, i_, k_, x, xn: plain(gg, i_, k_, x, xn)) for i in range(24) for k in QL.GEMMS}
            if mode == 'skip23':
                for k in ('Wo', 'Wgu', 'Wd'): g.qfn[(23, k)] = skip23
            if mode == 'table0': g.qfn[(0, 'Win')] = table0
            with torch.no_grad():
                h, _ = g.fwd(ids, q0=pr['q0']); lgs.append(g.logits(h, pr))
        g.qfn = {}
        b, s23, t0_ = lgs
        res['n'] += 1; res['nan_in_logits'] += int(torch.isnan(s23).any())
        res['max_dlogit_23'] = max(res['max_dlogit_23'], float((s23 - b).abs().max())); res['flips_23'] += int(s23.argmax() != b.argmax())
        res['max_dlogit_0'] = max(res['max_dlogit_0'], float((t0_ - b).abs().max())); res['flips_0'] += int(t0_.argmax() != b.argmax())
    out[tag] = res; print(tag, res, flush=True)
json.dump(out, open(a.out, 'w'), indent=1)
