"""J9: train hobson-v19 for the compiled-deployment layout (F7 recipe + H7 lessons).
Student: hobson + LoRA r16/a32 on every projection of all 24 layers + pointer head, in layout --layout (R: constants first; S: in place),
         deployment-constant blocks compiled independently (U context) by the STUDENT itself (gradients flow through the compile).
Teacher: frozen bf16 hobson in its own layout on the native tokens (same process), calibrated logits (/ T_kind).
Loss per example (one question): KL(teacher || student)  [+ CE(gold) for cf / v5 rows]
      + w_hid * relative MSE of the student's rows vs the teacher's rows at layers --hid_layers, on the question's option-end rows and '<answer>'
        row plus --n_dyn sampled live state rows (they exist 1:1 in both layouts).
Data (train split only; eval tasks excluded):
  real : evalkit/train_pool.jsonl (one random question of the request)              p_real
  cf   : h7/data/cf_aug.jsonl (CF-style edits of train-split states; gold labels)   p_cf    loss CE + 0.3 KL
  v5   : training/data/train_v5.jsonl (gold, the v19 mix; JevBench-like, no constants => native layout)   rest   loss 0.3 CE + KL
Resumable: ck/last.pt.  python train_j9.py --updates 1500 --every 250 --layout R"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn.functional as F
import j9lib as J
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)

ap = argparse.ArgumentParser()
ap.add_argument('--updates', type=int, default=1500); ap.add_argument('--every', type=int, default=250); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1.5e-4); ap.add_argument('--hlr', type=float, default=5e-4); ap.add_argument('--maxtok', type=int, default=7000)
ap.add_argument('--p_real', type=float, default=0.75); ap.add_argument('--p_cf', type=float, default=0.0)
ap.add_argument('--w_cf', type=float, default=1.0); ap.add_argument('--w_cfkl', type=float, default=0.3); ap.add_argument('--w_v5', type=float, default=0.5)
ap.add_argument('--perm', type=float, default=0.3); ap.add_argument('--ck', default='~/work/j9/ck'); ap.add_argument('--layout', default='R')
ap.add_argument('--comp', default='affine'); ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--hid_layers', default='5,11,17,23')
ap.add_argument('--n_dyn', type=int, default=96); ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--adapter', default='all', help="all | compile (LoRA only on the compile rows; live path = hobson exactly)")
ap.add_argument('--focus', default=''); ap.add_argument('--p_focus', type=float, default=0.0)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
HL = tuple(int(x) for x in a.hid_layers.split(',')) if a.w_hid > 0 else ()

pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 64: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
cf = []
for l in open(os.path.expanduser('~/work/j9/data/cf_aug.jsonl')):
    r = json.loads(l); assert r['task'] not in EV
    cf.append(r)
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 8 == 0: v5.append(json.loads(l))
R = random.Random(7); R.shuffle(pool); R.shuffle(cf); R.shuffle(v5)
probe = pool[-96:]; pool = pool[:-96]          # fixed held-out probe set (train split) for a clean learning curve
_pr = random.Random(5); probe = [(r, _pr.choice(list(r['questions']))) for r in probe]
print('pool', len(pool), 'cf', len(cf), 'v5', len(v5), flush=True)

m = J.J9(); m.detach_inference(); dev = m.dev
m.compile_only = (a.adapter == 'compile')
eng = m.p.eng; tok = eng.tok; m.setup_u(tok)
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def prep(state_text, qd, perm=None):
    q = ta.validate_python(qd)
    rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    opt = eng._option_idx([rq], 0)[0].tolist()
    return dict(s=s, q=qs[0], opt=opt, rq=rq)


def nopt(qd):
    return 2 if qd['type'] == 'noul' else len(qd['criteria'])


class Teacher:
    def __enter__(self):
        self.s = (m.lora, m.head); m.lora = None; m.head = m.head0
    def __exit__(self, *e):
        m.lora, m.head = self.s


FOC = set(x for x in a.focus.split(',') if x)
fpool = [r for r in pool if FOC & set(r['questions'])]
fi = 0
print('focus pool', len(fpool), flush=True)


def sample(rng):
    global pi, ci, vi, fi
    u = rng.random()
    if fpool and rng.random() < a.p_focus:
        r = fpool[fi % len(fpool)]; fi += 1
        qn = rng.choice(sorted(FOC & set(r['questions'])))
        return 'real', r['state'], r['questions'][qn], None
    if u < a.p_real:
        r = pool[pi % len(pool)]; pi += 1
        qn = rng.choice(list(r['questions']))
        return 'real', r['state'], r['questions'][qn], None
    if u < a.p_real + a.p_cf:
        r = cf[ci % len(cf)]; ci += 1
        qn = list(r['questions'])[0]
        return 'cf', r['state'], r['questions'][qn], r['expected'][qn] if isinstance(r['expected'], dict) else r['expected']
    r = v5[vi % len(v5)]; vi += 1
    qd, gold = v5_q(r)
    if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
    return 'v5', r['state'], qd, gold


p_lora = m.add_lora(r=a.rank, alpha=2 * a.rank, seed=7)
p_head = m.set_head('std')
if m.compile_only:
    for p_ in p_head: p_.requires_grad_(False)
    params = p_lora
    opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr)], betas=(0.9, 0.95), weight_decay=0.0)
else:
    params = p_lora + p_head
    opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)], betas=(0.9, 0.95), weight_decay=0.0)
warm = max(20, int(0.05 * a.updates))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
pi = ci = vi = 0; upd0 = 0
srng = random.Random(11)
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(st['sd'][f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(st['sd'][f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in st['sd'].items() if k.startswith('head.')})
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi, ci, vi = st['pos']; srng.setstate(st['rng'])
    print('resumed at update', upd0, flush=True)

def run_probe():
    kls = []; ags = []
    with torch.no_grad():
        for r, qn in probe:
            qd = r['questions'][qn]
            st_txt = render_state(r['state']); pc = prep(st_txt, qd)
            if len(pc['s']) + len(pc['q']) > a.maxtok: continue
            req = J.tokenize_pieces(tok, J.pieces(st_txt, LL))
            if [t for _, ids, _ in req for t in ids] != pc['s']: req = [('dyn', pc['s'], None)]
            with Teacher():
                lt = m.native_logits(pc['s'], pc['q'], pc['opt'], pc['rq'].kind)
            P = J.build_plan(m, req, pc['q'], a.layout, dev)
            ls = m.logits_c(P, m.fwd_c(P, comp=a.comp), pc['opt'], pc['rq'].kind)
            n = pc['rq'].n_slots; tp = torch.softmax(lt[:n].float(), -1); lp = F.log_softmax(ls[:n].float(), -1)
            kls.append(float((tp * (tp.clamp_min(1e-8).log() - lp)).sum())); ags.append(int(lp.argmax()) == int(tp.argmax()))
    return float(np.mean(kls)), float(np.mean(ags))


log = open(CK + 'train.log.jsonl', 'a')
if upd0 == 0:
    pk, pa = run_probe(); print(json.dumps(dict(upd=0, probe_kl=round(pk, 4), probe_agree=round(pa, 4))), flush=True)
    log.write(json.dumps(dict(upd=0, probe_kl=pk, probe_agree=pa)) + '\n'); log.flush()
json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()
upd = upd0
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        kind, state, qd, gold = sample(srng)
        if isinstance(state, (dict, list)): state = json.dumps(state, indent=2, ensure_ascii=False)
        st_txt = render_state(state)
        pc = prep(st_txt, qd)
        if len(pc['s']) + len(pc['q']) > a.maxtok: continue
        ps = pc
        if qd['type'] in ('choice', 'noul') and srng.random() < a.perm:
            perm = list(range(nopt(qd))); srng.shuffle(perm); ps = prep(st_txt, qd, perm)
        req = J.tokenize_pieces(tok, J.pieces(st_txt, LL))
        if [t for _, ids, _ in req for t in ids] != pc['s']:
            req = [('dyn', pc['s'], None)]
        P = J.build_plan(m, req, ps['q'], a.layout, dev)
        if m.compile_only and not P.brow: continue          # no constants: the student IS hobson here (zero gradient)
        # rows for the dense term: question option rows + answer row (same tokens in both layouts only if unpermuted), sampled live state rows
        ns = len(pc['s']); s0 = P.s0
        st_rows = [r for r in range(P.n_live - P.nq)]                       # stream rows that are state tokens
        samp = st_rows[-32:] + (srng.sample(st_rows[:-32], min(a.n_dyn, max(0, len(st_rows) - 32))) if len(st_rows) > 32 else [])
        srows = [s0 + r for r in samp]; trows = [P.native_idx[r] for r in samp]
        if ps is pc:
            q0s = P.T - P.nq
            srows += [q0s + o for o in pc['opt']] + [P.T - 1]; trows += [ns + o for o in pc['opt']] + [ns + len(pc['q']) - 1]
        kr_s = torch.tensor(srows, device=dev); kr_t = torch.tensor(trows, device=dev)
        with torch.no_grad(), Teacher():
            lt = m.native_logits(pc['s'], pc['q'], pc['opt'], pc['rq'].kind, keep=HL, keep_rows=kr_t)
            kt = dict(m.kept)
        h = m.fwd_c(P, comp=a.comp, ckpt=True, keep=HL, keep_rows=kr_s); ks = dict(m.kept)
        ls = m.logits_c(P, h, ps['opt'], ps['rq'].kind)
        n = ps['rq'].n_slots
        tl = pc['rq'].slot_labels; sl = ps['rq'].slot_labels
        idx = torch.tensor([tl.index(x) for x in sl], device=dev)
        tp = torch.softmax(lt[:n].float(), -1)[idx]
        lp = F.log_softmax(ls[:n].float(), -1)
        kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
        loss = 0.0
        if HL:
            hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in HL) / len(HL)
            loss = loss + a.w_hid * hl
            lsum['hid'] += float(hl); lcnt['hid'] += 1
        if kind == 'real':
            loss = loss + kl
        else:
            gi = sl.index(str(gold)) if str(gold) in sl else None
            if gi is None: continue
            ce = -lp[gi]
            wce, wkl = (a.w_cf, a.w_cfkl) if kind == 'cf' else (a.w_v5, 1.0)
            loss = loss + wce * ce + wkl * kl
            lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
            lsum['acc_' + kind] += float(int(lp.argmax()) == gi); lcnt['acc_' + kind] += 1
        lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
        lsum['agree_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + kind] += 1
        lsum['live_frac'] += P.n_live / max(1, len(pc['s']) + len(pc['q'])); lcnt['live_frac'] += 1
        (loss / a.accum).backward()
        lsum['T'] += len(pc['s']) + len(pc['q']); lcnt['T'] += 1
        nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        if upd % 50 == 0:
            pk, pa = run_probe(); lsum['probe_kl'] = pk; lcnt['probe_kl'] = 1; lsum['probe_agree'] = pa; lcnt['probe_agree'] = 1
        rec = dict(upd=upd, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if upd % a.every == 0 or upd == a.updates or upd % 50 == 0:
        sd = m.trainable_state()
        if upd % a.every == 0 or upd == a.updates:
            torch.save(sd, CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(sd=sd, opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pos=(pi, ci, vi), rng=srng.getstate()), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
print('done', time.time() - t0, flush=True)
