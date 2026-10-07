"""J1 eval: the encoder decider on every evalkit question (3227), rendered exactly as the references (render_state + render_question,
one question per sequence, no truncation), T5Gemma tokens = <bos> + state + question. Masked mode = state never sees the question.
Temperatures: one per kind fitted on rows_lceval_e.pt (held-out train-distribution rows, gold labels), like hobson's calibration.
usage: python ev_j1.py CKPT_DIR OUT_PREFIX [--mode masked] [--local 0,1,..] [--window W] [--limit N]
writes OUT_PREFIX.raw.json (T=1), OUT_PREFIX.json (calibrated), OUT_PREFIX.meta.json"""
import os, sys, json, time, argparse, math
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/work/evalkit')); sys.path.insert(0, os.path.expanduser('~/work/sd/src'))
import encj1 as E
import evalkit as EK
from transformers import AutoTokenizer
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax

ap = argparse.ArgumentParser(); ap.add_argument('ck'); ap.add_argument('out')
ap.add_argument('--mode', default='masked'); ap.add_argument('--local', default=''); ap.add_argument('--window', type=int, default=0)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--tokb', type=int, default=16384); ap.add_argument('--suites', default=''); ap.add_argument('--rot', type=int, default=1)
a = ap.parse_args()
dev = 'cuda'
ta = TypeAdapter(SC.Question)
tok = AutoTokenizer.from_pretrained(E.find_ckpt())
W, _ = E.load_weights()
local = [int(x) for x in a.local.split(',') if x != '']
torso = E.EncTorso(W, local_layers=local, window=a.window).to(dev); del W
torso.mode = a.mode
torso.lora.load_state_dict({k: v.to(dev) for k, v in torch.load(os.path.join(a.ck, 'lora.pt')).items()})
hc = StrandsDeciderConfig.from_json(os.path.join(a.ck, 'strands_decider_config.json'))
head = build_head(hc, E.D).to(dev); head.load_state_dict(torch.load(os.path.join(a.ck, 'slot_head.pt'))); head.eval(); torso.eval()


def prep(state, spec, order=None):
    rq = render_question(ta.validate_python(spec), option_order=order)
    s = [tok.bos_token_id] + tok(render_state(state), add_special_tokens=False)['input_ids']
    e = tok([rq.text], add_special_tokens=False, return_offsets_mapping=True)
    q = e['input_ids'][0]; opt = _option_token_index(e['offset_mapping'][0], rq.option_spans, 0)
    return dict(ids=torch.tensor(s + q), nstate=len(s), opt=[len(s) + o for o in opt], n=rq.n_slots, kind=rq.kind, rq=rq)


@torch.no_grad()
def run(rows, tokb):
    """rows: list of dicts (ids, nstate, opt, n) -> list of raw logits (n,)"""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['ids']))
    out = [None] * len(rows); k = 0
    while k < len(order):
        ch = []; tot = 0
        while k < len(order) and (not ch or tot + len(rows[order[k]]['ids']) <= tokb):
            ch.append(order[k]); tot += len(rows[order[k]]['ids']); k += 1
        rr = [rows[i] for i in ch]
        lens = [len(r['ids']) for r in rr]; nst = [r['nstate'] for r in rr]
        meta = E.pack_meta(lens, nst, dev)
        ids = torch.cat([r['ids'].long() for r in rr]).to(dev)
        hid = torso.forward_packed(ids, meta)
        starts = meta['starts'].tolist()
        Wd = max(r['n'] for r in rr)
        opt = torch.tensor([[st + o for o in r['opt']] + [st] * (Wd - r['n']) for st, r in zip(starts, rr)], device=dev)
        lg = head(hid[meta['last']].float(), hid[opt].float())
        for j, i in enumerate(ch): out[i] = lg[j, :rr[j]['n']].float().cpu()
    return out


def fit_temps(rows, logits):
    res = {}
    for kind in ('noul', 'choice', 'score'):
        idx = [i for i, r in enumerate(rows) if r['kind'] == kind]
        if not idx: continue
        best = (1e9, 1.0)
        for t in [0.3 + 0.02 * j for j in range(136)]:
            nll = 0.0
            for i in idx:
                nll -= float(F.log_softmax(logits[i] / t, -1)[rows[i]['label']])
            best = min(best, (nll / len(idx), t))
        res[kind] = dict(T=round(best[1], 3), nll=round(best[0], 4), n=len(idx), nll_T1=round(sum(-float(F.log_softmax(logits[i], -1)[rows[i]['label']]) for i in idx) / len(idx), 4))
    return res


E.use_compiled() if os.environ.get('NOCOMPILE') != '1' else None
t0 = time.time()
# 1) temperatures from held-out train-distribution rows
LC = torch.load(os.path.expanduser('~/work/j1/rows_lceval_e.pt'))['rows']
lcl = run(LC, a.tokb)
temps = fit_temps(LC, lcl)
lc_acc = sum(int(int(l.argmax()) == r['label']) for l, r in zip(lcl, LC)) / len(LC)
print('temps', temps, 'lc acc %.4f' % lc_acc, '%.0fs' % (time.time() - t0), flush=True)
# 2) every evalkit question
suites = a.suites.split(',') if a.suites else None
items = list(EK.all_question_items(suites))
if a.limit: items = items[:a.limit]
rows = []
for s, iid, q, st, spec in items:
    r = prep(st, spec); r.update(iid=iid, q=q, suite=s); rows.append(r)
print('questions', len(rows), 'tokens', sum(len(r['ids']) for r in rows), 'max', max(len(r['ids']) for r in rows), flush=True)
lg = run(rows, a.tokb)
raw, cal = {}, {}
for r, l in zip(rows, lg):
    T = temps.get(r['kind'], {}).get('T', 1.0)
    p1 = torch.softmax(l, -1).tolist(); pc = torch.softmax(l / T, -1).tolist()
    raw.setdefault(r['iid'], {})[r['q']] = {lab: p1[i] for i, lab in enumerate(r['rq'].slot_labels)}
    cal.setdefault(r['iid'], {})[r['q']] = {lab: pc[i] for i, lab in enumerate(r['rq'].slot_labels)}
json.dump(raw, open(a.out + '.raw.json', 'w')); json.dump(cal, open(a.out + '.json', 'w'))
json.dump(dict(ck=a.ck, mode=a.mode, local=local, window=a.window, temps=temps, lc_acc=lc_acc, n=len(rows), s=round(time.time() - t0)),
          open(a.out + '.meta.json', 'w'), indent=1)
print('done', len(rows), '%.0fs' % (time.time() - t0), flush=True)
if a.rot:   # option-order sensitivity on REAL-agree + JB-all: original + up to 3 rotations (rotlib), calibrated temps
    from rotlib import orders_for
    rr = []
    for s_, iid, q, st, spec in EK.all_question_items(['REAL-agree', 'JB-all']):
        base = render_question(ta.validate_python(spec))
        for j, o in enumerate([None] + orders_for(base.kind, base.n_slots)):
            r = prep(st, spec, o); r.update(iid=iid, q=q, name=f'r{j}'); rr.append(r)
    lg = run(rr, a.tokb); rot = {}
    for r, l in zip(rr, lg):
        T = temps.get(r['kind'], {}).get('T', 1.0); pc = torch.softmax(l / T, -1).tolist()
        rot.setdefault(r['iid'], {}).setdefault(r['q'], {})[r['name']] = {lab: pc[i] for i, lab in enumerate(r['rq'].slot_labels)}
    json.dump(rot, open(a.out + '.rot.json', 'w'))
    print('rot done', len(rr), '%.0fs' % (time.time() - t0), flush=True)
