"""J4 trainer.  Every arm is this script.
  --mode ft  : the fine-tune (F7 / v19-mix recipe): rows of ft.jsonl in their fixed order from --start; gold rows CE(gold, ordinal smoothing .1 on
               score) + w_kl*KL(hobson || student); real rows KL only.  Option order shuffled every row (choice/noul), score rubric reversed p .5,
               instruction variant p .3.  32 rows per update, packed varlen into micro-batches of <= --mtok tokens.
  --mode dp  : decision pretraining: root = state, children = its generated verification questions (gen_dp.py), CE on the construction label,
               options shuffled.  Updates of ~--step_tok tokens.
  --mode ntp : next-token prediction on the same dp states (same token budget) = the generative control.
Init: --stack ck1.pt[,ck2.pt]: LoRAs merged into the base weights in order (the intermediate phase), head from the last one with a head
(--fresh_head ignores it).  LoRA r16/a32 on the 4 fused weights of every layer, AdamW(.9,.95), wd .01, lr 1e-4 (head 1e-3), warmup 30 updates then
CONSTANT (so every checkpoint of one run is a valid point of a sample-efficiency curve and arm (c) = arm (a) continued).
Checkpoints: --save_rows 1000,2000,... (ft) or --save_tok (dp/ntp, millions).  Resumable from OUT/last.pt.
"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j4')]
import torch, torch.nn.functional as F
from j4lib import J4

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='ft'); ap.add_argument('--out', required=True); ap.add_argument('--data', required=True)
ap.add_argument('--teacher', default=''); ap.add_argument('--stack', default=''); ap.add_argument('--fresh_head', type=int, default=0)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--rows', type=int, default=16000)
ap.add_argument('--tok', type=float, default=6.0, help='dp/ntp budget, millions of tokens through the model')
ap.add_argument('--save_rows', default='1000,2000,4000,8000,16000'); ap.add_argument('--save_tok', default='')
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--hlr', type=float, default=1e-3); ap.add_argument('--r', type=int, default=16)
ap.add_argument('--alpha', type=float, default=32); ap.add_argument('--w_kl', type=float, default=1.0); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--mtok', type=int, default=8192); ap.add_argument('--step_tok', type=int, default=12000); ap.add_argument('--maxlen', type=int, default=4096)
ap.add_argument('--tok_stop', type=float, default=0, help='ft: also stop at this many total tokens (millions), for equal-compute arms'); ap.add_argument('--init_from', default='');
ap.add_argument('--mix_dp', default='', help='ft: also one DP state (<= qmax questions) per update from this file (arm e)'); ap.add_argument('--mix_w', type=float, default=1.0); ap.add_argument('--mix_start', type=int, default=995);
ap.add_argument('--warm', type=int, default=30); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--qmax', type=int, default=24)
a = ap.parse_args()
OUT = os.path.expanduser(a.out) + '/'; os.makedirs(OUT, exist_ok=True)
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
torch.manual_seed(a.seed)

m = J4('base'); dev = m.dev
m.eng.model.config.max_length = a.maxlen
if a.stack:
    m.load_stack([s for s in a.stack.split(',') if s])
    if m.lora is not None: m.merge_lora()            # every phase in the stack is merged; this run trains a fresh LoRA on top
    print('stack merged', a.stack, 'head carried' if m.head is not None and not a.fresh_head else 'fresh head', flush=True)
carried = m.head if (m.head is not None and not a.fresh_head) else None
p_lora = m.add_lora(r=a.r, alpha=a.alpha, seed=7 + a.seed)
if carried is not None:
    m.head = carried
    for p_ in m.head.parameters(): p_.requires_grad_(True)
    p_head = list(m.head.parameters())
else:
    p_head = m.set_head(dropout=0.05, seed=a.seed)
m.head.train()
params = p_lora + p_head
opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)], betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / a.warm))

# ---------------- data
rows = []
with open(os.path.expanduser(a.data)) as f:
    for l in f:
        r = json.loads(l)
        if r.get('task') in EV: raise SystemExit('eval task in training data: ' + r['task'])
        rows.append(r)
mixrows = []
if a.mix_dp:
    mixrows = [json.loads(l) for l in open(os.path.expanduser(a.mix_dp))][a.mix_start:]
    assert not any(r.get('task') in EV for r in mixrows)
teach = {}
if a.teacher:
    for l in open(os.path.expanduser(a.teacher)):
        t = json.loads(l); teach[t['i']] = t
print('rows', len(rows), 'teacher', len(teach), flush=True)
rng = random.Random(1000 + a.seed)


def nopt(qd):
    return 2 if qd['type'] == 'noul' else len(qd['criteria'])


def ft_example(r):
    qd = dict(r['q'])
    if r.get('variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['variants'])
    if qd['type'] == 'score':
        perm = list(range(nopt(qd)))[::-1] if rng.random() < 0.5 else None
    else:
        perm = list(range(nopt(qd))); rng.shuffle(perm)
    p = m.prep(r['state'], qd, perm=perm)
    return p


def dp_example(r):
    qs = r['qs'][:a.qmax]
    from strands_decider.prompting import render_question, render_state
    rqs = []
    for q in qs:
        qd = m._ta.validate_python(q['q']) if hasattr(m, '_ta') else None
        if qd is None:
            m.prep('x', q['q']); qd = m._ta.validate_python(q['q'])
        perm = list(range(nopt(q['q']))); rng.shuffle(perm)
        rqs.append(render_question(qd, option_order=perm))
    s, qtok = m.eng._fit(render_state(r['state']), [rq.text for rq in rqs])
    oi = m.eng._option_idx(rqs, 0)
    opts = [[int(x) for x in oi[j].tolist() if x >= 0] for j in range(len(rqs))]
    return s, qtok, opts, rqs, [q['label'] for q in qs], [q['fam'] for q in qs]


def smooth_target(n, gi, kind, eps=0.1):
    t = torch.zeros(n, device=dev)
    if kind != 'score' or n < 2 or eps <= 0:
        t[gi] = 1.0; return t
    nb = [j for j in (gi - 1, gi + 1) if 0 <= j < n]
    t[gi] = 1 - eps
    for j in nb: t[j] = eps / len(nb)
    return t


# ---------------- state / resume
st = dict(upd=0, pos=a.start, tok=0, rows_done=0, mpos=0)
LAST = OUT + 'last.pt' if os.path.exists(OUT + 'last.pt') else (os.path.expanduser(a.init_from) if a.init_from else '')
if LAST:
    ck = torch.load(LAST, map_location=dev, weights_only=False)
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(ck['sd'][f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(ck['sd'][f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in ck['sd'].items() if k.startswith('head.')})
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); st = ck['st']; rng.setstate(ck['rng'])
    print('resumed', st, flush=True)
save_rows = sorted(int(x) for x in a.save_rows.split(',') if x)
save_tok = sorted(float(x) for x in a.save_tok.split(',') if x)
log = open(OUT + 'train.log.jsonl', 'a'); json.dump(vars(a), open(OUT + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()


def save(tag):
    torch.save(m.lora_state(), OUT + f'{tag}.pt'); print('saved', tag, json.dumps(st), flush=True)


def checkpoint_state():
    torch.save(dict(sd=m.lora_state(), opt=opt.state_dict(), sched=sched.state_dict(), st=st, rng=rng.getstate()), OUT + 'last.tmp')
    os.replace(OUT + 'last.tmp', OUT + 'last.pt')


def run_micro(batch, nrm, mode=None, w=1.0):
    """batch: list of examples (mode-specific).  Backward of w*sum(loss)/nrm."""
    mode = mode or a.mode
    if mode == 'ft':
        roots = [p['s'] + p['q'] for p, _ in batch]
        h, rsp, _ = m.forward(roots, ckpt=True)
        loss = 0.0
        for (p, r), (r0, r1) in zip(batch, rsp):
            n = p['rq'].n_slots
            lg = m.pointer_logits(h, r1 - 1, [r0 + len(p['s']) + o for o in p['opt']])[:n].float()
            lp = F.log_softmax(lg, -1); sl = p['rq'].slot_labels
            t = teach.get(r['i'])
            kl = None
            if t is not None:
                tp = torch.tensor([t['p'][t['labels'].index(x)] for x in sl], device=dev)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
                lsum['kl_' + r['kind']] += float(kl); lcnt['kl_' + r['kind']] += 1
                lsum['agree_' + r['kind']] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + r['kind']] += 1
            if r['kind'] == 'gold':
                gi = sl.index(r['label'])
                tgt = smooth_target(n, gi, p['rq'].kind)
                ce = -(tgt * lp).sum()
                li = ce + (a.w_kl * kl if kl is not None else 0.0)
                lsum['ce'] += float(ce); lcnt['ce'] += 1; lsum['acc'] += float(int(lp.argmax()) == gi); lcnt['acc'] += 1
            else:
                if kl is None: continue
                li = a.w_kl * kl
            loss = loss + li
        ntok = sum(len(x) for x in roots)
    elif mode == 'dp':
        roots = []; ch = []; meta = []
        for j, (s, qtok, opts, rqs, labs, fams) in enumerate(batch):
            roots.append(s)
            for k in range(len(qtok)): ch.append((j, qtok[k])); meta.append((opts[k], rqs[k], labs[k], fams[k]))
        h, rsp, csp = m.forward(roots, ch, ckpt=True)
        loss = 0.0
        for (c0, c1), (op, rq, lab, fam) in zip(csp, meta):
            lg = m.pointer_logits(h, c1 - 1, [c0 + o for o in op])[:rq.n_slots].float()
            lp = F.log_softmax(lg, -1); gi = rq.slot_labels.index(lab)
            ce = -lp[gi]; loss = loss + ce
            pre = 'mix_' if mode != a.mode else ''
            lsum[pre + 'ce_' + fam] += float(ce); lcnt[pre + 'ce_' + fam] += 1; lsum[pre + 'acc_' + fam] += float(int(lp.argmax()) == gi); lcnt[pre + 'acc_' + fam] += 1
        ntok = sum(len(x) for x in roots) + sum(len(c) for _, c in ch)
    else:  # ntp
        roots = [b[0] for b in batch]
        h, rsp, _ = m.forward(roots, ckpt=True)
        rr = []; tt = []
        for s, (r0, r1) in zip(roots, rsp):
            rr += list(range(r0, r1 - 1)); tt += s[1:]
        rr = torch.tensor(rr, device=dev); tt = torch.tensor(tt, device=dev)
        lm = m.lm_loss(h, rr, tt)
        loss = lm * len(roots)
        lsum['nll'] += float(lm); lcnt['nll'] += 1
        ntok = sum(len(x) for x in roots)
    if torch.is_tensor(loss):
        (w * loss / nrm).backward()
    st['tok'] += ntok
    return ntok


def ft_update():
    exs = []
    while len(exs) < a.bs and st['pos'] < len(rows):
        r = rows[st['pos']]; st['pos'] += 1
        p = ft_example(r)
        if r['kind'] == 'real' and r['i'] not in teach: continue
        exs.append((p, r))
    if not exs: return False
    mb = []; cur = 0
    for p, r in exs:
        L = len(p['s']) + len(p['q'])
        if mb and cur + L > a.mtok:
            run_micro(mb, len(exs)); mb = []; cur = 0
        mb.append((p, r)); cur += L
    if mb: run_micro(mb, len(exs))
    if mixrows:
        r = mixrows[st.get('mpos', 0) % len(mixrows)]; st['mpos'] = st.get('mpos', 0) + 1
        e = dp_example(r)
        run_micro([e], len(e[1]), mode='dp', w=a.mix_w)
    st['rows_done'] += len(exs)
    return True


def dp_update():
    """~step_tok tokens of states(+questions); loss normalised per question (dp) / per state (ntp)"""
    exs = []; cur = 0
    while cur < a.step_tok and st['pos'] < len(rows):
        r = rows[st['pos']]; st['pos'] += 1
        e = dp_example(r)
        if a.mode == 'ntp':
            e = (e[0],)
            L = len(e[0])
        else:
            L = len(e[0]) + sum(len(q) for q in e[1])
        exs.append((e, L)); cur += L
    if not exs: return False
    nrm = sum(len(e[1]) for e, _ in exs) if a.mode == 'dp' else len(exs)
    mb = []; c = 0
    for e, L in exs:
        if mb and c + L > a.mtok:
            run_micro(mb, nrm); mb = []; c = 0
        mb.append(e); c += L
    if mb: run_micro(mb, nrm)
    st['rows_done'] += len(exs)
    return True


while True:
    if a.mode == 'ft':
        if st['rows_done'] >= a.rows: break
        if a.tok_stop and st['tok'] >= a.tok_stop * 1e6: break
        ok = ft_update()
    else:
        if st['tok'] >= a.tok * 1e6: break
        ok = dp_update()
    if not ok: print('data exhausted', flush=True); break
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    st['upd'] += 1
    if st['upd'] % 10 == 0 or st['upd'] == 1:
        rec = dict(upd=st['upd'], rows=st['rows_done'], mtok=round(st['tok'] / 1e6, 3), t=round(time.time() - t0),
                   tps=round(st['tok'] / max(1, time.time() - t0)), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in sorted(lsum)})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if a.mode == 'ft':
        while save_rows and st['rows_done'] >= save_rows[0]:
            save(f'r{save_rows.pop(0)}')
    else:
        while save_tok and st['tok'] >= save_tok[0] * 1e6:
            save(f't{save_tok.pop(0):g}')
    if st['upd'] % 25 == 0: checkpoint_state()
save('final'); checkpoint_state()
json.dump(st, open(OUT + 'done.json', 'w'))
print('done', json.dumps(st), round(time.time() - t0), flush=True)
