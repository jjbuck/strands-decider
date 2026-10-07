"""laptop: score_final.json -> markdown tables for DRAFT_REPORT.md"""
import json, sys
r = json.load(open(sys.argv[1]))
rows = [n for n in ('hobson', 'causal', 'causal_mntp', 'qag_all', 'qa_all', 'qag_early') if n in r]
f = lambda v: '-' if v is None or v != v else f'{v:.3f}'
print('| model | JB-all | JB-hard | REAL-label | REAL agree_sd | LONG agree / sd | CF acc / flip | CF-probe acc / flip | SHUF both | Brier / ECE |')
print('|---|---|---|---|---|---|---|---|---|---|')
for n in rows:
    s = r[n]; g = lambda a, b: s.get(a, {}).get(b)
    long_ = f"{f(g('LONG', 'agree'))} / {f(g('LONG', 'agree_sd'))}" if s.get('LONG', {}).get('coverage', 0) > 0.5 else 'not run'
    print(f"| {n} | {f(g('JB-all', 'acc'))} | {f(g('JB-hard', 'acc'))} | {f(g('REAL-label', 'acc'))} | {f(g('REAL-agree', 'agree_sd'))} | {long_} | "
          f"{f(g('CF', 'acc'))} / {f(g('CF', 'flip'))} | {f(g('CF-probe', 'acc'))} / {f(g('CF-probe', 'flip'))} | {f(g('SHUF', 'both_right'))} | "
          f"{f(g('calib', 'brier'))} / {f(g('calib', 'ece'))} |")
for k in [k for k in r if k.startswith('_paired')]:
    print(f'\n{k}: (model-only right, other-only right, exact McNemar p)')
    for n, d in r[k].items():
        print(f"- {n}: " + '; '.join(f"{s} {v[0]}/{v[1]} p {v[2]:.3g}" for s, v in d.items() if isinstance(v, (list, tuple))))
