"""Markdown baseline table for README (hobson + baselines on every suite). python3 make_table.py"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK

ROWS = ['hobson', 'merged_full', 'nostate', 'drop@7', 'drop@3', 'rand10@7', 'rand25@7', 'rand50@7', 'rand10@3', 'rand25@3', 'rand50@3', 'rand10@0', 'rand25@0', 'rand50@0', 'qattn10@7', 'qattn10@3']
COLS = [('JB-hard', 'acc'), ('JB-hard', 'agree'), ('JB-long', 'acc'), ('JB-long', 'agree'), ('REAL-agree', 'agree'), ('REAL-agree', 'agree_sd'), ('REAL-agree', 'tv'),
        ('LONG', 'agree'), ('LONG', 'agree_sd'), ('CF', 'acc'), ('CF', 'flip'), ('CF', 'flip_rel'), ('CF', 'dir'), ('CF', 'dmean_rel'), ('CF-probe', 'acc'), ('CF-probe', 'flip'), ('CF-probe', 'flip_rel'), ('CF-probe', 'dir'), ('CF-probe', 'dmean_rel'),
        ('SHUF', 'change'), ('SHUF', 'both_right'), ('REAL-label', 'acc')]


def main():
    res = {}
    for s in {c[0] for c in COLS}:
        if os.path.exists(f'{EK.KIT}/suites/{s}.jsonl'):
            try: res[s] = EK.score(s, {})
            except Exception as e: print('skip', s, e, file=sys.stderr)
    cols = [c for c in COLS if c[0] in res]
    ns = {}
    for s in res:
        h = res[s].get('hobson', {}); ns[s] = h.get('n_q') or h.get('n_pairs') or h.get('n_acc')
    print('| config | ' + ' | '.join(f'{s} {m}' for s, m in cols) + ' |')
    print('|---|' + '---|' * len(cols))
    print('| n | ' + ' | '.join(str(ns.get(s, '')) for s, m in cols) + ' |')
    for r in ROWS:
        vals = []
        for s, m in cols:
            v = res[s].get(r, {}).get(m)
            vals.append('–' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.3f}')
        print(f'| {r} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main()
