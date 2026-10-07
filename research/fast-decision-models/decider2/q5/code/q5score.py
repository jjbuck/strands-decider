"""Q5 laptop scorer: every config in preds files vs hobson refs and the in-runtime bf16 ('dense'), on the whole kit and on the banking subset,
checked against BRIEF10's bar (q1score.check). Reads Q1's / H1's scorers (imported, not edited).
python3 q5score.py --dense dense.json f1.json [f2.json ...]   -> prints + merges ~/decider2/q5/res/scores.json"""
import sys, os, json, math
sys.path[:0] = [os.path.expanduser('~/decider2/q1/code'), os.path.expanduser('~/decider2/h1/code'), os.path.expanduser('~/decider2/evalkit')]
import h1score as HS
import q1score as QS
import evalkit as EK

_dom = None


def bank_ids():
    global _dom
    if _dom is None:
        _dom = set()
        for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe'):
            for it in EK.load_suite(s):
                if it.get('domain', '').startswith('banking'): _dom.add(it['id'])
    return _dom


def restrict(p, ids):
    return {k: v for k, v in p.items() if k in ids}


def summarize(tag, pr, dense, meta=None):
    o = HS.row(pr, dense if tag != 'dense' else None)
    c = QS.check(o, meta)
    B = bank_ids()
    ob = HS.row(restrict(pr, B), restrict(dense, B) if (dense and tag != 'dense') else None)
    bank = dict(real_flips=ob.get('REAL-agree', {}).get('flips'), real_n=ob.get('REAL-agree', {}).get('n'), real_flip=ob.get('REAL-agree', {}).get('flip_rate'),
                real_paired=ob.get('REAL-agree', {}).get('paired'), long_flips=ob.get('LONG', {}).get('flips'), long_n=ob.get('LONG', {}).get('n'),
                cf_fgh=ob.get('CF', {}).get('fgh'), cf_n=ob.get('CF', {}).get('n_tracked'), cfp_fgh=ob.get('CF-probe', {}).get('fgh'),
                cfp_n=ob.get('CF-probe', {}).get('n_tracked'), real_label=ob.get('REAL-label', {}).get('acc'), tv=ob.get('REAL-agree', {}).get('tv'))
    return dict(full=o, bar=c, bank=bank)


if __name__ == '__main__':
    args = sys.argv[1:]; dense = None
    if '--dense' in args:
        j = args.index('--dense'); dense = json.load(open(args[j + 1]))['preds']['dense']; del args[j:j + 2]
    outp = os.path.expanduser('~/decider2/q5/res/scores.json')
    allres = json.load(open(outp)) if os.path.exists(outp) else {}
    for f in args:
        d = json.load(open(f)); P = d['preds']; M = d.get('meta', {})
        for t, pr in P.items():
            if not pr: continue
            r = summarize(t, pr, dense, M.get(t)); r['meta'] = M.get(t)
            allres[f'{os.path.basename(f)}::{t}'] = r
            c = r['bar']; b = r['bank']; w = (M.get(t) or {}).get('work', {})
            print(f"### {os.path.basename(f)}::{t}  n_items {len(pr)}  work {w}\n   "
                  f"KIT REAL flips {c['real_flips']}/{c['real_n']} = {100*(c['real_flip'] or float('nan')):.2f}% (bar <= 0.70%) | paired vs bf16 {c['paired_vs_bf16']} | TV {c['tv'] if c['tv'] is None else round(c['tv'],4)}\n   "
                  f"CF fgh {c['cf_fgh']} (>= .99) | CF-probe fgh {c['cfp_fgh']} (>= .95) | JB-hard {c['jb_acc']} McNemar p {c['jb_mcnemar']} | REAL-label {c['real_label']} (>= .78) | LONG flips {c['long_flip']}\n   "
                  f"PASS {c['pass']} {c['ok']}\n   "
                  f"BANK REAL flips {b['real_flips']}/{b['real_n']} paired {b['real_paired']} | LONG {b['long_flips']}/{b['long_n']} | CF fgh {b['cf_fgh']} ({b['cf_n']}) | CF-probe fgh {b['cfp_fgh']} ({b['cfp_n']}) | REAL-label {b['real_label']}",
                  flush=True)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(allres, open(outp, 'w'), indent=1)
