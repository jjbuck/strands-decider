"""catalog.py: find every prediction file the agents wrote, normalise each into evalkit's preds format, and index it.

    python3 catalog.py            # writes norm/<run_id>.json and catalog.csv

Run ids are '<agent>/<file stem>[:<variant>]'. Normalised format: {item_id: {question: {label: prob}}}.

Accepted inputs:
  - JSON {item_id: {q: dist}}
  - JSON {variant: {item_id: {q: dist}}}, e.g. j6 preds_a.json {'student': ...} or j11 *_rot.json {'rot1': ...}
  - JSON {'preds': {...}} wrappers of either
  - JSONL rows with id + (q|qn) + (probs|p); 'p' may be {q: dist} when q is null; a 'v' field splits variants

Skipped: *.part.jsonl, *score*, *.meta.json, *.ntok.json, latency files, and dev-split sets with no evalkit items.
"""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK

ROOT = os.path.expanduser('~/decider2')
OUT = os.path.join(ROOT, 'analysis')
AGENTS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'f7', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15', 'q1', 'q2', 'q3', 'q4', 'q5', 'm1', 'm2', 'n1']
SKIP = re.compile(r'(\.part\.jsonl$|score|\.meta\.json$|\.ntok\.json$|lat[_\-.]|latency|/box_logs/|/code/|_dev\.jsonl$|_dev\.json$)', re.I)


def item_universe():
    ids = set()
    for s in EK.SUITES:
        try:
            for it in EK.load_suite(s): ids.add(it['id'])
        except Exception: pass
    for s in ('CF', 'CF-probe'):
        for p in EK._pairs(s): ids.update((p['a'], p['b']))
    return ids


IDS = item_universe()


def is_dist(v):
    return isinstance(v, (int, float)) or (isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()))


def is_preds(d):
    """{item: {q: dist}} with a meaningful share of known item ids"""
    if not isinstance(d, dict) or not d: return False
    ks = list(d)[:200]
    hit = sum(k in IDS for k in ks)
    if hit < max(3, 0.3 * len(ks)): return False
    v = d[next(k for k in ks if k in IDS)]
    return isinstance(v, dict) and any(is_dist(x) for x in v.values())


def from_json(path):
    d = json.load(open(path))
    if isinstance(d, dict) and 'preds' in d and isinstance(d['preds'], dict): d = d['preds']
    if is_preds(d): return {'': d}
    if isinstance(d, dict) and d and all(isinstance(v, dict) for v in d.values()):
        out = {k: v for k, v in d.items() if is_preds(v)}
        if out: return out
    return {}


def from_jsonl(path):
    out = {}
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        iid = r.get('id'); q = r.get('q', r.get('qn')); p = r.get('probs', r.get('p'))
        if iid is None or p is None: continue
        v = r.get('v', '')
        dst = out.setdefault(v, {})
        if q is None and isinstance(p, dict) and all(isinstance(x, dict) for x in p.values()):
            for qq, dist in p.items(): dst.setdefault(iid, {})[qq] = dist
        elif is_dist(p):
            dst.setdefault(iid, {})[q] = p
    return {k: v for k, v in out.items() if sum(i in IDS for i in v) >= 3}


def coverage(preds):
    cov = {}
    for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'REAL-label'):
        its = EK.load_suite(s); n = 0; h = 0
        for it in its:
            for q in it['questions']:
                n += 1; h += q in preds.get(it['id'], {})
        cov[s] = h / max(n, 1)
    return cov


def main():
    os.makedirs(f'{OUT}/norm', exist_ok=True)
    rows = []
    for a in AGENTS:
        base = os.path.join(ROOT, a)
        if not os.path.isdir(base): continue
        for path in sorted(glob.glob(f'{base}/**/*.json', recursive=True) + glob.glob(f'{base}/**/*.jsonl', recursive=True)):
            rel = os.path.relpath(path, ROOT)
            if SKIP.search(rel): continue
            if os.path.getsize(path) < 2000: continue
            try:
                sets = from_jsonl(path) if path.endswith('.jsonl') else from_json(path)
            except Exception as e:
                print('skip', rel, e); continue
            stem = os.path.relpath(path, base).rsplit('.', 1)[0].replace('.json', '').replace('preds/', '').replace('results/', '').replace('box/', '')
            for v, preds in sets.items():
                rid = f'{a}/{stem}' + (f':{v}' if v else '')
                safe = rid.replace('/', '__').replace(':', '~')
                json.dump(preds, open(f'{OUT}/norm/{safe}.json', 'w'))
                cov = coverage(preds)
                rows.append(dict(run=rid, file=rel, n_items=len(preds), **{f'cov_{k}': round(c, 3) for k, c in cov.items()}))
    with open(f'{OUT}/catalog.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} prediction sets -> {OUT}/catalog.csv')


if __name__ == '__main__':
    main()
