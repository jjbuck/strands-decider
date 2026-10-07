"""evalkit: capacity-sensitive evaluation for decider compressions (F0, round 5).  Pure python + numpy; runs anywhere.

    import sys; sys.path.insert(0, '/tmp/decider2/evalkit')      # on a box: ~/work/evalkit
    import evalkit as EK
    items = EK.load_suite('REAL-agree')                            # list of dicts: id, state, questions{name: spec}, expected{name: label}|None, ...
    preds = {it['id']: {qn: {label: p, ...} for qn in it['questions']} for it in items}
    print(EK.score('REAL-agree', preds))                          # dict of metrics, with hobson + baselines alongside
    EK.report(all_preds)                                          # every suite with baselines; all_preds = {suite: preds} or one flat {id: ...} dict

Prediction format: preds[item_id][question_name] = {label: prob}.  Labels: noul 'true'/'false' (a bare float = P(true) is accepted),
choice = option names, score = level indices '0','1',...  Decisions are the argmax label.
See README.md for the suites, how references were produced and what each metric means.
"""
import json, os, math, collections
import numpy as np

KIT = os.path.dirname(os.path.abspath(__file__))
SUITES = ['JB-hard', 'JB-long', 'JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
REF_FILE = {'JB-hard': 'JB-all', 'JB-long': 'JB-all', 'JB-all': 'JB-all', 'SHUF': 'CF', 'CF-probe': 'CF-probe', 'REAL-label': 'REAL-agree'}
# baselines shown next to every suite (all computed on hobson-v19 itself, merged-LoRA runtime, tokens/plib.py)
SHOW = ['merged_full', 'nostate', 'drop@7', 'rand10@7', 'rand25@7', 'rand50@7', 'rand10@3', 'rand25@3', 'rand50@3', 'rand10@0', 'rand25@0', 'rand50@0', 'qattn10@7', 'qattn10@3', 'drop@3', 'drop@0']
_cache = {}


# ------------------------------------------------------------------ loading
def _jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def load_suite(name):
    """Items of a suite. SHUF and REAL-label are views: their items carry 'pair' / 'label' fields and point at CF / REAL-agree states."""
    if name in _cache: return _cache[name]
    its = _jsonl(f'{KIT}/suites/{name}.jsonl')
    if name in ('SHUF', 'REAL-label'):
        base = {i['id']: i for i in load_suite('CF' if name == 'SHUF' else 'REAL-agree')}
        if name == 'SHUF':  # SHUF rows reference CF items; expose their states so a model can be run on them directly
            ids = {x for r in its for x in (r['a'], r['b'])}
            its = [base[x] for x in sorted(ids) if x in base]
        else:
            lab = {(r['id'], r['q']): r for r in its}
            out = []
            for (iid, q), r in sorted(lab.items()):
                b = base[iid]
                out.append(dict(b, id=iid, questions={q: b['questions'][q]}, expected={q: r['label']}, label_meta=r))
            its = out
    _cache[name] = its
    return its


def _pairs(name):
    return _jsonl(f'{KIT}/suites/{name}.pairs.jsonl') if name in ('CF', 'CF-probe') else _jsonl(f'{KIT}/suites/SHUF.jsonl')


def load_refs(name):
    """{item_id: {'hobson': {q: dist}, 'cfg': {baseline: {q: dist}}}}"""
    key = 'refs:' + REF_FILE.get(name, name)
    if key in _cache: return _cache[key]
    f = REF_FILE.get(name, name)
    out = {}
    for r in _jsonl(f'{KIT}/refs/{f}.hobson.jsonl'):
        out.setdefault(r['id'], {'cfg': {}})['hobson'] = r['hobson']
    for r in _jsonl(f'{KIT}/refs/{f}.base.jsonl'):
        d = out.setdefault(r['id'], {'cfg': {}})
        for q, v in r['q'].items():
            d['cfg'].setdefault('merged_full', {})[q] = v['merged_full']
            for c, dist in v['cfg'].items():
                d['cfg'].setdefault(c, {})[q] = dist
    _cache[key] = out
    return out


# ------------------------------------------------------------------ helpers
def _norm(p):
    if p is None: return None
    if isinstance(p, (int, float)): return {'true': float(p), 'false': 1.0 - float(p)}
    if isinstance(p, (list, tuple)): raise ValueError('give {label: prob}, not a list')
    s = sum(p.values())
    return {k: float(v) / s for k, v in p.items()} if s > 0 else dict(p)


def _arg(p):
    return max(p, key=p.get)


def _tv(p, q):
    ks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in ks)


def _kl(p, q):  # KL(p || q), p = reference
    return sum(pv * (math.log(max(pv, 1e-9)) - math.log(max(q.get(k, 0), 1e-9))) for k, pv in p.items() if pv > 0)


def _flat(preds, ids):
    """accept {suite: {id: ...}} or {id: ...}"""
    if preds and all(k in SUITES for k in preds):
        m = {}
        for v in preds.values(): m.update(v)
        return m
    return preds


# ------------------------------------------------------------------ metrics
def _qrows(name, preds):
    """per-question rows (item, q, pred, hobson, nostate, expected)"""
    its = load_suite(name); refs = load_refs(name)
    rows = []
    for it in its:
        ref = refs.get(it['id'], {})
        for q in it['questions']:
            h = ref.get('hobson', {}).get(q)
            p = preds.get(it['id'], {}).get(q) if preds is not None else None
            rows.append(dict(id=it['id'], q=q, pred=_norm(p), hob=_norm(h) if h else None,
                             nost=_norm(ref.get('cfg', {}).get('nostate', {}).get(q)),
                             exp=(it.get('expected') or {}).get(q), item=it))
    return rows


def _agree_metrics(rows):
    m = {}
    have = [r for r in rows if r['pred'] is not None]
    m['n_q'] = len(rows); m['coverage'] = len(have) / max(len(rows), 1)
    ex = [r for r in have if r['exp'] is not None]
    if ex: m['acc'] = float(np.mean([_arg(r['pred']) == r['exp'] for r in ex])); m['n_acc'] = len(ex)
    exc = [r for r in ex if (r['item'].get('label_meta') or {}).get('agree')]
    if exc: m['acc_consensus'] = float(np.mean([_arg(r['pred']) == r['exp'] for r in exc])); m['n_consensus'] = len(exc)  # REAL-label: Opus and Sonnet agree
    exh = [r for r in ex if r['hob'] is not None]
    if exh and len(exh) < len([r for r in rows if r['exp'] is not None]): m['acc_hobson_same'] = float(np.mean([_arg(r['hob']) == r['exp'] for r in exh]))  # hobson on the same covered questions
    hb = [r for r in have if r['hob'] is not None]
    if hb:
        m['agree'] = float(np.mean([_arg(r['pred']) == _arg(r['hob']) for r in hb]))
        m['tv'] = float(np.mean([_tv(r['pred'], r['hob']) for r in hb]))
        m['kl'] = float(np.mean([_kl(r['hob'], r['pred']) for r in hb]))
        sd = [r for r in hb if r['nost'] is not None and _arg(r['nost']) != _arg(r['hob'])]
        m['agree_sd'] = float(np.mean([_arg(r['pred']) == _arg(r['hob']) for r in sd])) if sd else float('nan'); m['n_sd'] = len(sd)
    return m


def _pair_metrics(name, preds, src_preds=None):
    """CF / CF-probe: each pair (a, b) has one question q and ground truth ea != eb."""
    pairs = _pairs(name); out = collections.Counter(); n = 0
    acc = []; flip_ok = []; flip_any = []; dirn = []; dlt = []; joint = []
    for pr in pairs:
        q = pr['q']; pa = preds.get(pr['a'], {}).get(q); pb = preds.get(pr['b'], {}).get(q)
        if pa is None or pb is None: continue
        pa, pb = _norm(pa), _norm(pb); n += 1
        acc += [_arg(pa) == pr['ea'], _arg(pb) == pr['eb']]
        flip_ok.append(_arg(pa) == pr['ea'] and _arg(pb) == pr['eb'])
        flip_any.append(_arg(pa) != _arg(pb))
        d = pb.get(pr['eb'], 0) - pa.get(pr['eb'], 0)  # P(correct-for-b) should rise from a to b
        dirn.append(d > 0); dlt.append(d)
    m = dict(n_pairs=n)
    if n:
        m.update(acc=float(np.mean(acc)), flip=float(np.mean(flip_ok)), flip_any=float(np.mean(flip_any)), dir=float(np.mean(dirn)), dmean=float(np.mean(dlt)))
    return m


def _shuf_metrics(preds):
    rows = _jsonl(f'{KIT}/suites/SHUF.jsonl'); n = 0; ch = []; ok = []
    for r in rows:
        q = r['q']; pa = preds.get(r['a'], {}).get(q); pb = preds.get(r['b'], {}).get(q)
        if pa is None or pb is None: continue
        pa, pb = _norm(pa), _norm(pb); n += 1
        ch.append(_arg(pa) != _arg(pb)); ok.append(_arg(pa) == r['ea'] and _arg(pb) == r['eb'])
    return dict(n_pairs=n, change=float(np.mean(ch)) if n else float('nan'), both_right=float(np.mean(ok)) if n else float('nan'))


def _as_preds(name, which):
    """reference outputs as a preds dict: which = 'hobson' or a baseline name"""
    refs = load_refs(name)
    out = {}
    for iid, r in refs.items():
        d = r.get('hobson') if which == 'hobson' else r.get('cfg', {}).get(which)
        if d: out[iid] = d
    return out


def _metrics(name, preds):
    if name in ('CF', 'CF-probe'):
        m = _pair_metrics(name, preds)
        # hobson on the SAME covered pairs (so partial coverage does not bias the ratios)
        hp_all = _as_preds(name, 'hobson')
        cov = {x for pr in _pairs(name) if pr['a'] in preds and pr['b'] in preds and pr['q'] in preds[pr['a']] and pr['q'] in preds[pr['b']] for x in (pr['a'], pr['b'])}
        hm = _pair_metrics(name, {k: v for k, v in hp_all.items() if k in cov})
        if 'flip' in m and hm.get('flip'): m['flip_rel'] = m['flip'] / hm['flip']
        if 'dmean' in m and hm.get('dmean'): m['dmean_rel'] = m['dmean'] / hm['dmean']
        # of the pairs hobson tracks correctly, how many does the model track
        hp = _as_preds(name, 'hobson'); tr = []
        for pr in _pairs(name):
            q = pr['q']
            try:
                ho = _arg(_norm(hp[pr['a']][q])) == pr['ea'] and _arg(_norm(hp[pr['b']][q])) == pr['eb']
                po = _arg(_norm(preds[pr['a']][q])) == pr['ea'] and _arg(_norm(preds[pr['b']][q])) == pr['eb']
            except KeyError: continue
            if ho: tr.append(po)
        if tr: m['flip_given_hobson'] = float(np.mean(tr))
        # agreement with hobson on the individual CF items
        m.update({k: v for k, v in _agree_metrics(_qrows(name, preds)).items() if k in ('agree', 'tv')})
        return m
    if name == 'SHUF':
        return _shuf_metrics(preds)
    return _agree_metrics(_qrows(name, preds))


def score(name, preds, baselines=True):
    """Metrics for one suite. preds = {item_id: {question: {label: p}}}.  Returns {'model': {...}, 'hobson': {...}, '<baseline>': {...}}."""
    preds = _flat(preds, None)
    res = {'model': _metrics(name, preds)}
    hp = _as_preds(name, 'hobson')
    if hp: res['hobson'] = _metrics(name, hp)
    if baselines:
        for b in SHOW:
            bp = _as_preds(name, b)
            if bp: res[b] = _metrics(name, bp)
    return res


COLS = {'agree': ['coverage', 'acc', 'acc_hobson_same', 'acc_consensus', 'agree', 'agree_sd', 'tv', 'kl'], 'pair': ['acc', 'flip', 'flip_rel', 'flip_given_hobson', 'dir', 'dmean', 'dmean_rel', 'agree'], 'shuf': ['change', 'both_right']}


def _fmt(v):
    return '   -  ' if v is None or (isinstance(v, float) and math.isnan(v)) else (f'{v:6.3f}' if isinstance(v, float) else f'{v:6d}')


def report(preds=None, suites=None, show=None, file=None):
    """Print every suite (or `suites`) with the model's numbers (if preds given), hobson's and the baselines."""
    import sys
    file = file or sys.stdout
    preds = _flat(preds or {}, None)
    for name in suites or [s for s in SUITES if s != 'JB-all']:
        if not os.path.exists(f'{KIT}/suites/{name}.jsonl'): continue
        res = score(name, preds)
        kind = 'pair' if name in ('CF', 'CF-probe') else 'shuf' if name == 'SHUF' else 'agree'
        cols = COLS[kind]
        n = res.get('hobson', res['model']).get('n_q') or res.get('hobson', res['model']).get('n_pairs')
        print(f'\n== {name}  (n={n})', file=file)
        print(f'{"":14s}' + ''.join(f'{c:>18s}' for c in cols), file=file)
        rows = (['model'] if preds else []) + ['hobson'] + [b for b in (show or SHOW) if b in res]
        for r in rows:
            m = res.get(r, {})
            print(f'{r:14s}' + ''.join(f'{_fmt(m.get(c)):>18s}' for c in cols), file=file)
        if kind == 'pair' and load_refs(name):
            print_breakdown(name, preds or None, 'kind', file=file)



def breakdown(name, preds=None, by='kind', rows=('hobson', 'nostate', 'drop@7', 'rand10@7', 'rand25@7', 'qattn10@7')):
    """CF / CF-probe flip rate per edit kind (by='kind'), per domain (by='domain') or per edit position (by='pos': first/second half of the state)."""
    pairs = _pairs(name)
    key = {'kind': lambda p: p['kind'], 'domain': lambda p: p['domain'], 'pos': lambda p: 'pos<0.5' if p.get('pos', 1) < 0.5 else 'pos>=0.5',
           'len': lambda p: 'state>=4k' if p.get('n_state_tok', 0) >= 4000 else 'state 2-4k' if p.get('n_state_tok', 0) >= 2000 else 'state<2k'}[by]
    groups = sorted({key(p) for p in pairs})
    srcs = ([('model', _flat(preds, None))] if preds else []) + [(r, _as_preds(name, r)) for r in rows]
    out = {}
    for g in groups:
        ps = [p for p in pairs if key(p) == g]; out[g] = {'n': len(ps)}
        for nm, pd in srcs:
            ok = [(_arg(_norm(pd[p['a']][p['q']])) == p['ea'] and _arg(_norm(pd[p['b']][p['q']])) == p['eb']) for p in ps if p['a'] in pd and p['b'] in pd and p['q'] in pd[p['a']] and p['q'] in pd[p['b']]]
            out[g][nm] = float(np.mean(ok)) if ok else None
    return out


def print_breakdown(name, preds=None, by='kind', file=None):
    import sys
    file = file or sys.stdout
    bd = breakdown(name, preds, by)
    cols = [c for c in next(iter(bd.values())) if c != 'n']
    print(f'\n-- {name} flip by {by}', file=file)
    print(f'{"":34s}{"n":>5s}' + ''.join(f'{c:>11s}' for c in cols), file=file)
    for g, d in bd.items():
        print(f'{g:34s}{d["n"]:5d}' + ''.join(f'{_fmt(d[c]):>11s}' for c in cols), file=file)


def kill_check(preds, budget=None, layer=7):
    """The brief's accuracy criteria: JB-hard within 1.5 points of hobson; CF (and CF-probe) flip >= 0.95 x hobson's.
    Also prints REAL-agree / LONG agree_sd next to random selection at `budget` (0.10 / 0.25 / 0.50) after `layer`. FLOP/latency criteria are yours."""
    preds = _flat(preds, None)
    jb = score('JB-hard', preds, baselines=False)
    a, h = jb['model'].get('acc'), jb['hobson'].get('acc')
    print(f"JB-hard acc {a:.3f} vs hobson {h:.3f}: diff {100 * (a - h):+.1f} points  (pass if >= -1.5; SE ~4.4 points at n=130)  {'PASS' if a - h >= -0.015 else 'FAIL'}")
    for s in ('CF', 'CF-probe'):
        if not load_refs(s): continue
        m = score(s, preds, baselines=False)['model']
        if 'flip_rel' in m:
            g = m.get('flip_given_hobson', float('nan'))
            print(f"{s}: flip {m['flip']:.3f}, flip_rel {m['flip_rel']:.3f} (brief: pass if >= 0.95) {'PASS' if m['flip_rel'] >= 0.95 else 'FAIL'}; "
                  f"flip_given_hobson {g:.3f} (retention of hobson's own correct flips; stricter, use it too), dir {m['dir']:.3f}, dmean_rel {m.get('dmean_rel', float('nan')):.3f}"
                  + ("  [CF-probe flip_rel is not monotone in reading: drop@7 scores 1.09]" if s == 'CF-probe' else ''))
    for s in ('REAL-agree', 'LONG'):
        if not load_refs(s): continue
        r = score(s, preds)
        b = f"rand{int(round(100 * budget))}@{layer}" if budget else None
        line = f"{s}: agree {r['model'].get('agree', float('nan')):.3f} agree_sd {r['model'].get('agree_sd', float('nan')):.3f} | nostate agree_sd {r.get('nostate', {}).get('agree_sd', float('nan')):.3f}"
        if b and b in r: line += f" | {b} agree {r[b]['agree']:.3f} agree_sd {r[b]['agree_sd']:.3f}"
        print(line)


def iter_questions(name):
    """(item_id, question_name, state, question_spec) for every question of a suite: what a model must answer."""
    for it in load_suite(name):
        for q, spec in it['questions'].items():
            yield it['id'], q, it['state'], spec


def all_question_items(suites=None):
    """every (suite, item_id, question_name, state, spec) to run, deduplicated across suites that share items (JB-*, SHUF->CF, REAL-label->REAL-agree)"""
    seen = set()
    for s in suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        if not os.path.exists(f'{KIT}/suites/{s}.jsonl'): continue
        for iid, q, st, spec in iter_questions(s):
            if (iid, q) in seen: continue
            seen.add((iid, q)); yield s, iid, q, st, spec


if __name__ == '__main__':
    report()
