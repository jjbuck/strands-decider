"""J7 checks: (1) the J7 forward with Qwen ids reproduces hobson's reference probabilities; (2) zero-init channels change nothing;
(3) untrained super-token transplant (mean-of-constituents init) agreement; (4) host-side costs (super-token merge, channel annotation)."""
import os, sys, json, time, collections
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch
import evalkit as EK, chan
from j7lib import J7, Prep
from superbpe import Super
from strands_decider.prompting import render_state
W = os.path.expanduser('~/work/')
m = J7(); m.free_hf(); m.head = m.head0; eng = m.p.eng
S64 = Super(W + 'j7/tok/sb64k.json', tok=eng.tok)
sup = {'64k': S64}
for lv in ('16k', '4k'):
    S = Super(W + f'j7/tok/sb{lv}.json', tok=eng.tok); S.idx = {t: S64.V + i for i, t in enumerate(S.toks)}; sup[lv] = S
P = Prep(eng, sup)
refs = EK.load_refs('REAL-agree')
its = EK.load_suite('REAL-agree')[:40]
print('emb rms', m.emb_rms(), flush=True)


def run(level=None, feats=False):
    out = {}
    with torch.inference_mode():
        for it in its:
            st = render_state(it['state']); names = list(it['questions'])
            prs = [P.base(st, it['questions'][q]) for q in names]
            bs = [P.build(p, level=level, with_feats=feats) for p in prs]
            lgs = m.logits_multi(bs)
            out[it['id']] = {q: torch.softmax(lg.float(), -1).tolist() for q, lg in zip(names, lgs)}
            out[it['id']]['_lab'] = {q: p['rq'].slot_labels for q, p in zip(names, prs)}
    return out


def cmp(out):
    dmax = []; agree = 0; n = 0
    for it in its:
        for q in it['questions']:
            ref = refs[it['id']]['hobson'][q]; labs = out[it['id']]['_lab'][q]; pr = out[it['id']][q]
            r = [ref[l] for l in labs]
            dmax.append(max(abs(x - y) for x, y in zip(r, pr))); agree += int(np.argmax(r) == np.argmax(pr)); n += 1
    return dict(n=n, agree=round(agree / n, 3), dmax_med=round(float(np.median(dmax)), 4), dmax_max=round(float(np.max(dmax)), 4))


t0 = time.time(); base = run(); print('qwen ids vs hobson refs', cmp(base), '%.1fs' % (time.time() - t0), flush=True)
m.add_channels()
ch = run(feats=True); print('zero-init channels vs refs', cmp(ch), flush=True)
m.chans = None
m.add_super(S64.toks, S64.V); m.sup.freeze_table()
for lv in ('4k', '16k', '64k'):
    o = run(level=lv); print('untrained transplant', lv, cmp(o), flush=True)
# host costs
texts = [render_state(it['state']) for it in its]
t0 = time.time(); ids = [eng.tok(t)['input_ids'] for t in texts]; t1 = time.time()
for x in ids: S64.merge_ids(x)
t2 = time.time()
for t in texts: chan.annotate(t)
t3 = time.time()
nt = sum(map(len, ids))
print('host per 1000 Qwen tokens: tokenize %.2f ms, super-merge %.2f ms, channel annotate %.2f ms' % ((t1 - t0) / nt * 1e6, (t2 - t1) / nt * 1e6, (t3 - t2) / nt * 1e6), flush=True)
