"""J15 main analysis (laptop): margins, DEV-fitted tau, eval scores of cascades, oracle bounds. -> res/*.json
python analyze.py DRAFT [DRAFT ...]   (draft names = preds/<name>_{eval,dev}.jsonl; verifier = b8)"""
import sys, json, collections, numpy as np
import casc2 as C
import evalkit as EK
R = '~/decider2/j15/res'


def margin_stats(D, V, H):
    rows = []
    for i, qs in D.items():
        for q, d in qs.items():
            if i not in V or q not in V[i]: continue
            hv = H.get(i, {}).get(q)
            rows.append((C.margin(d), C.am(d) != C.am(V[i][q]), (C.am(d) != C.am(EK._norm(hv))) if hv else None))
    m = np.array([r[0] for r in rows]); fv = np.array([r[1] for r in rows])
    qs = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    hist_edges = np.linspace(0, 1, 21)
    return dict(n=len(rows), flip_vs_V=float(fv.mean()),
                flipped_q={str(x): float(np.quantile(m[fv], x)) for x in qs} if fv.any() else {},
                nonflipped_q={str(x): float(np.quantile(m[~fv], x)) for x in qs},
                hist_flipped=np.histogram(m[fv], hist_edges)[0].tolist(), hist_nonflipped=np.histogram(m[~fv], hist_edges)[0].tolist(),
                edges=hist_edges.tolist())


def per_suite_defer(D, V, sig, tau):
    out = {}
    for s in C.SU:
        n = 0; nd = 0
        for it in EK.load_suite(s):
            for q in it['questions']:
                if it['id'] in D and q in D[it['id']]:
                    n += 1; nd += sig(it['id'], q, D[it['id']][q]) < tau
        out[s] = nd / max(n, 1)
    return out


if __name__ == '__main__':
    H = C.hob()
    V = C.load(f'{C.PD}/b8_eval.jsonl'); Vd = C.load(f'{C.PD}/b8_dev.jsonl')
    Fl = C.load(f'{C.PD}/bf16_eval.jsonl') or C.load('~/decider2/h6/preds/h2_bf16.jsonl')
    sigm = lambda i, q, d: C.margin(d)
    res = json.load(open(f'{R}/analysis.json')) if len(sys.argv) > 2 and sys.argv[-1] == '--append' else {}
    res['verifier_b8'] = C.full_score(V, floor=Fl)
    for name in [a for a in sys.argv[1:] if not a.startswith('--')]:
        D = C.load(f'{C.PD}/{name}_eval.jsonl'); Dd = C.load(f'{C.PD}/{name}_dev.jsonl')
        r = dict(margins=margin_stats(D, V, H), draft_alone=C.full_score(D, ref=V, floor=Fl))
        if Dd:
            fit, fdev, ndev = C.fit_tau(Dd, Vd, sigm)
            r['dev'] = dict(n=ndev, flip_vs_V=fdev, fit={str(k): v for k, v in fit.items()},
                            curve=C.curve(Dd, Vd, sigm, [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
            r['casc'] = {}
            for eps, (tau, fd, resd) in fit.items():
                P, f = C.cascade(D, V, sigm, tau)
                sc = C.full_score(P, ref=V, floor=Fl); sc['defer_all'] = f; sc['defer_suite'] = per_suite_defer(D, V, sigm, tau); sc['tau'] = tau; sc['dev_defer'] = fd
                r['casc'][str(eps)] = sc
        r['eval_curve'] = C.curve(D, V, sigm, [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        res[name] = r
        m = r['margins']
        print(f'\n## {name}: flips vs b8 {m["flip_vs_V"]:.4f}; REAL flips vs hobson {r["draft_alone"]["REAL-agree_flips"]:.4f}; CF {r["draft_alone"]["CF_fgh"]:.3f} '
              f'CF-probe {r["draft_alone"]["CF-probe_fgh"]:.3f}; flipped-margin median {m["flipped_q"].get("0.5", float("nan")):.3f} p95 {m["flipped_q"].get("0.95", float("nan")):.3f}')
        if 'dev' in r:
            print(f'  DEV n={r["dev"]["n"]} flip vs b8 {r["dev"]["flip_vs_V"]:.4f}')
            for eps, sc in r['casc'].items():
                print(f'  eps {eps}: tau {sc["tau"]:.3f} dev f {sc["dev_defer"]:.3f} | EVAL f {sc["defer_all"]:.3f}  REAL flips {sc["REAL-agree_flips"]:.4f} '
                      f'(paired vs b8 {sc["REAL-agree_paired_ref"]}, vs bf16 {sc["REAL-agree_paired_floor"]}) LONG {sc["LONG_flips"]:.4f} TV {sc["REAL-agree_tv"]:.4f} '
                      f'CF {sc["CF_fgh"]:.3f} CFp {sc["CF-probe_fgh"]:.3f} JBh {sc["JB-hard_acc"]:.3f} {sc["JB-hard_mcn_hob"]} lab {sc["REAL-label_acc"]:.3f} '
                      f'defer by suite {json.dumps({k: round(v, 3) for k, v in sc["defer_suite"].items()})}')
    json.dump(res, open(f'{R}/analysis.json', 'w'), indent=1)
