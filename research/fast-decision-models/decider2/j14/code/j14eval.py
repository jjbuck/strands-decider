"""J14 eval: every evalkit question through the state-first compiled-question layout, several variants per state prefix.
python j14eval.py --variants 'all:replay,oa:replay,oa+sfx:replay' --out ~/work/j14/preds/untrained.jsonl [--ckpt ck.pt] [--suites ...] [--stride 1]
Output jsonl lines: {"v": variant, "id": iid, "q": qname, "p": {label: prob}}  (resumable)."""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j14lib import J14, live_positions

ap = argparse.ArgumentParser()
ap.add_argument('--variants', default='all:replay,oa:replay')
ap.add_argument('--out', required=True); ap.add_argument('--ckpt', default=None)
ap.add_argument('--suites', default='JB-all,REAL-agree,LONG,CF,CF-probe'); ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--offset', type=int, default=0)
a = ap.parse_args()
VAR = [tuple(v.split(':')) for v in a.variants.split(',')]   # live-set:gdn[:conv]

m = J14(); m.setup()
if a.ckpt:
    import j14train_util as TU
    TU.load_ckpt(m, a.ckpt)
m.free_hf() if m.p.tm is not None else None
dev = m.dev

done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: r = json.loads(l); done.add((r['v'], r['id'], r['q']))
        except Exception: pass
fo = open(a.out, 'a')

items = collections.OrderedDict()
for s, iid, q, st, spec in EK.all_question_items(a.suites.split(',')):
    items.setdefault(iid, (st, []))[1].append((q, spec))
keys = list(items)[a.offset::a.stride]
print('items', len(keys), 'variants', VAR, flush=True)
COMP = collections.OrderedDict(); CBUDGET = 3.0e9
def comp_bytes(c): return sum(sum(t.numel() * t.element_size() for t in d.values()) for d in c)
t0 = time.time(); nq = 0
for n, iid in enumerate(keys):
    st, qs = items[iid]
    todo = [(q, spec) for q, spec in qs if any((':'.join([lv, gm] + c_), iid, q) not in done for lv, gm, *c_ in VAR)]
    if not todo: continue
    prs = [(q, m.prep(st, spec)) for q, spec in todo]
    with torch.inference_mode():
        cache, T = m.prefix(prs[0][1]['s']); hcache = None
        for q, pr in prs:
            key = tuple(pr['q'])
            comp = COMP.get(key)
            for lv, gm, *cm in VAR:
                vn = ':'.join([lv, gm] + cm)
                if (vn, iid, q) in done: continue
                hyb = None
                if lv.startswith('H') and '/' in lv:          # 'H8/<live set>': questions with > 8 options run in hobson's own layout, hobson weights
                    thr, lvs = lv[1:].split('/', 1); hyb = int(thr)
                else: lvs = lv
                exact = hyb is not None and len(pr['opt']) > hyb
                lp = list(range(len(pr['q']))) if exact else live_positions(m.tok, pr, lvs)
                use = comp if len(lp) < len(pr['q']) else None
                if exact:
                    sv = (m.llora, m.live_ft); m.llora = None; m.live_ft = False
                    if sv[1]:
                        if hcache is None: hcache = m.prefix(prs[0][1]['s'])[0]
                        hl = m.suffix(hcache, T, pr['q'], lp)
                    else:
                        hl = m.suffix(cache, T, pr['q'], lp)
                    m.llora, m.live_ft = sv
                else:
                    hl = m.suffix(cache, T, pr['q'], lp, gdn_mode=gm, comp=use, conv_mode=cm[0] if cm else None)
                if use is None and len(lp) < len(pr['q']) and m.comp_out:
                    comp = m.comp_out; COMP[key] = comp
                    while sum(comp_bytes(c) for c in COMP.values()) > CBUDGET and len(COMP) > 1: COMP.popitem(last=False)
                lg = m.readout(hl, lp, pr['opt'], len(pr['q']), pr['rq'].kind, pr['rq'].n_slots)
                p = torch.softmax(lg.float(), -1).tolist()
                fo.write(json.dumps(dict(v=vn, id=iid, q=q, p={lab: p[j] for j, lab in enumerate(pr['rq'].slot_labels)}, nl=len(lp), Lq=len(pr['q']), T=T)) + '\n')
                nq += 1
        del cache
    fo.flush()
    if n % 50 == 0: print(n, len(keys), f'{time.time() - t0:.0f}s', nq, flush=True)
print('done', time.time() - t0, flush=True)
