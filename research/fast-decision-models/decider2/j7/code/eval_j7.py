"""J7 eval: every evalkit question (all suites, 3227) through the J7 forward, state computed once per item (H7 'seqs' branches).
--ckpt '' = hobson itself in this runtime (merged LoRA, hobson head).  --level 4k|16k|64k reads super-tokens (needs a T/TV ckpt or tests the
untrained transplant).  --chan 1 adds the channels (V/TV ckpts carry them).  Writes preds JSON {iid: {q: {label: p}}} + token counts.
python eval_j7.py OUT.json --ckpt ck_V/s400.pt [--level 64k] [--suites CF,CF-probe]"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j7lib import J7, Prep
from superbpe import Super
from strands_decider.prompting import render_state
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--level', default='')
ap.add_argument('--suites', default=''); ap.add_argument('--protect', type=int, default=0); ap.add_argument('--items', default=''); ap.add_argument('--ablate', default='', help='channel keys to zero at inference, e.g. val,place'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
W = os.path.expanduser('~/work/')
m = J7(); m.free_hf(); eng = m.p.eng
supers = {}
S64 = None
if a.level or a.ckpt:
    S64 = Super(W + 'j7/tok/sb64k.json', tok=eng.tok)
if a.level:
    S = Super(W + f'j7/tok/sb{a.level}.json', tok=eng.tok)
    assert S.toks == S64.toks[:len(S.toks)]; S.idx = {t: S64.V + i for i, t in enumerate(S.toks)}
    supers[a.level] = S
if a.ckpt:
    sd = m.load_j7(os.path.expanduser(a.ckpt), toks=S64.toks, V=S64.V)
else:
    m.head = m.head0
if a.level and m.sup is None:       # untrained transplant: mean-of-constituents init
    m.add_super(S64.toks, S64.V)
if m.sup is not None: m.sup.freeze_table()
CHAN = m.chans is not None
ABL = [x for x in a.ablate.split(',') if x]
P = Prep(eng, supers, protect=bool(a.protect))
suites = a.suites.split(',') if a.suites else None
byitem = collections.OrderedDict()
if a.items:
    for l in open(os.path.expanduser(a.items)):
        r = json.loads(l); byitem[r['id']] = dict(state=r['state'], qs=collections.OrderedDict(r['questions']))
else:
    for suite, iid, qn, st, spec in EK.all_question_items(suites):
        e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict()))
        e['qs'][qn] = spec
items = list(byitem.items())
si, sn = map(int, a.shard.split('/')); items = items[si::sn]
if a.limit: items = items[:a.limit]
side = os.path.expanduser(a.out) + '.part.jsonl'
done = {}; ntok = {}
if os.path.exists(side):
    for l in open(side):
        try: r = json.loads(l); done[r['id']] = r['p']; ntok[r['id']] = r['n']
        except Exception: pass
t0 = time.time(); n = 0
with open(side, 'a') as f, torch.inference_mode():
    for iid, e in items:
        if iid in done: continue
        st = render_state(e['state']); names = list(e['qs'])
        prs = [P.base(st, e['qs'][qn]) for qn in names]
        bs = [P.build(p, level=a.level or None, with_feats=CHAN) for p in prs]
        for b in bs:
            for k in ABL: b['feats'][k] = b['feats'][k] * 0
        if len(set(b['q0'] for b in bs)) > 1 or len(set(len(p['s']) for p in prs)) > 1:
            lgs = [m.logits1(b) for b in bs]
        else:
            lgs = m.logits_multi(bs)
        res = {}
        for qn, p, lg in zip(names, prs, lgs):
            pr_ = torch.softmax(lg.float(), -1).tolist()
            res[qn] = {lab: pr_[i] for i, lab in enumerate(p['rq'].slot_labels)}
        nt = dict(s_orig=len(prs[0]['s']), s=bs[0]['q0'], q_orig=[len(p['q']) for p in prs], q=[len(b['ids']) - b['q0'] for b in bs])
        done[iid] = res; ntok[iid] = nt
        f.write(json.dumps(dict(id=iid, p=res, n=nt)) + '\n'); n += 1
        if n % 100 == 0: f.flush(); print(n, len(items), f'{time.time() - t0:.0f}s', flush=True)
json.dump(done, open(os.path.expanduser(a.out), 'w'))
json.dump(ntok, open(os.path.expanduser(a.out) + '.ntok.json', 'w'))
print('done', len(done), f'{time.time() - t0:.0f}s', flush=True)
