"""Build question lists (laptop, pure python): sens (step 1), dev (step 2 stage A), full (stage B)."""
import sys, json, random
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK
R = random.Random(13)
lab = {(r['id'], r['q']) for r in map(json.loads, open('~/decider2/evalkit/suites/REAL-label.jsonl'))}
def sdrows(s):
    rows = EK._qrows(s, {})
    return [r for r in rows if r['hob'] and r['nost'] and EK._arg(r['nost']) != EK._arg(r['hob'])], rows
def tracked(s):
    hp = EK._as_preds(s, 'hobson'); out = []
    for p in EK._pairs(s):
        try: ok = EK._arg(EK._norm(hp[p['a']][p['q']])) == p['ea'] and EK._arg(EK._norm(hp[p['b']][p['q']])) == p['eb']
        except KeyError: continue
        out.append((p, ok))
    return out
L = {}
sd_real, all_real = sdrows('REAL-agree'); sd_long, _ = sdrows('LONG')
# sens: 120 REAL sd (state <= 3000) + 50 CF pairs (state <= 3000)
c = [r for r in sd_real if r['item']['n_state_tok'] <= 3000]; R.shuffle(c)
sens = [["REAL-agree", r["id"], r["q"]] for r in c[:70]]
cfp = [p for p, ok in tracked('CF') if p['n_state_tok'] <= 3000]; R.shuffle(cfp)
for p in cfp[:15]: sens += [['CF', p['a'], p['q']], ['CF', p['b'], p['q']]]
L['sens'] = sens
# dev
c = list(sd_real); R.shuffle(c); dev = [['REAL-agree', r['id'], r['q']] for r in c[:100]]
c = list(sd_long); R.shuffle(c); dev += [['LONG', r['id'], r['q']] for r in c[:30]]
for s, nt, nu in (('CF', 40, 20), ('CF-probe', 35, 15)):
    tp = tracked(s); R.shuffle(tp)
    sel = [p for p, ok in tp if ok][:nt] + [p for p, ok in tp if not ok][:nu]
    for p in sel: dev += [[s, p['a'], p['q']], [s, p['b'], p['q']]]
for it in EK.load_suite('JB-hard'):
    for q in it['questions']: dev.append(['JB-all', it['id'], q])
L['dev'] = dev
# full
full = [['REAL-agree', r['id'], r['q']] for r in all_real if (r in sd_real) or (r['id'], r['q']) in lab]
full += [['LONG', r['id'], r['q']] for r in sd_long]
for s in ('CF', 'CF-probe', 'JB-all'):
    for it in EK.load_suite(s):
        for q in it['questions']: full.append([s, it['id'], q])
L['full'] = full
json.dump(L, open('lists.json', 'w'))
for k, v in L.items():
    import collections; print(k, len(v), dict(collections.Counter(x[0] for x in v)))
