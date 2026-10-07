"""J3: train the depth-split decision transformer (dtlib.DT) from hobson-v19, distilled from hobson itself (F7 recipe + CF augmentation + dense rows).
Student: hobson weights + LoRA r16/a32 on every projection of all 24 layers (shallow layers see state + question rows, deep layers question rows only)
         + memory adapters (LoRA r32 on the deep layers' memory projection rows) + hobson's pointer head (trainable).
Teacher: frozen hobson (LoRA off), state-first, same process, calibrated (logits / T_kind), same (possibly option-permuted) question as the student.
Data (train split only; eval tasks excluded):
  real : evalkit/train_pool.jsonl requests, all questions as branches (p .45) / 2-4 subset (p .30) / one (p .25): loss = KL(teacher || student)
  cf   : j3/data/cf_aug.jsonl (H7's CF-style edits of train-split states, labels by construction): W_CF * CE + W_CFKL * KL
  v5   : training/data/train_v5.jsonl (all rows, shuffled): W_V5 * CE(gold) + KL       (F7: CE + 1.0 KL on labelled rows)
  dense: W_HID * relMSE(student question rows, teacher question rows) at deep layer outputs HID_LAYERS (H7's necessary term)
Option order permuted with p .5 (choice + noul); score rubric reversed with p .5 (both seen by teacher and student identically).
Ls: one split (e.g. 8) or a comma list sampled per example (elastic). Resumable via CK/last.pt.
python train_dt.py --Ls 8 --bridge A --updates 1500"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from dtlib import DT
import dtset
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)

ap = argparse.ArgumentParser()
ap.add_argument('--Ls', default='8'); ap.add_argument('--bridge', default='A')
ap.add_argument('--updates', type=int, default=1500); ap.add_argument('--every', type=int, default=250); ap.add_argument('--accum', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--hlr', type=float, default=3e-4); ap.add_argument('--mlr', type=float, default=5e-4)
ap.add_argument('--maxtok', type=int, default=6144); ap.add_argument('--p_real', type=float, default=0.50); ap.add_argument('--p_cf', type=float, default=0.20)
ap.add_argument('--w_cf', type=float, default=1.0); ap.add_argument('--w_cfkl', type=float, default=0.3); ap.add_argument('--w_v5', type=float, default=1.0)
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--hid_layers', default='11,15,19,23')
ap.add_argument('--perm', type=float, default=0.5); ap.add_argument('--mem_r', type=int, default=32); ap.add_argument('--lora_r', type=int, default=16)
ap.add_argument('--shallow_lora', type=int, default=1); ap.add_argument('--ck', default='~/work/j3/ck'); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--max_hours', type=float, default=0.0); ap.add_argument('--layout', default='seqs'); ap.add_argument('--init', default=''); ap.add_argument('--v5_tasks', default='')
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
LS = [int(x) for x in a.Ls.split(',')]
HL = tuple(int(x) for x in a.hid_layers.split(',')) if a.w_hid > 0 else ()
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
assert a.shallow_lora or len(LS) == 1, 'a frozen shallow stack is shared with the teacher only for a single split'

pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 64: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
cf = collections.defaultdict(list)
for l in open(os.path.expanduser('~/work/j3/data/cf_aug.jsonl')):
    r = json.loads(l); assert r['task'] not in EV
    cf[r['pair']].append(r)
cf = list(cf.values())
v5 = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.jsonl'))]
if a.v5_tasks: v5 = [r for r in v5 if r['task'] in set(a.v5_tasks.split(','))]
R = random.Random(a.seed); R.shuffle(pool); R.shuffle(cf); R.shuffle(v5)
print('pool', len(pool), 'cf pairs', len(cf), 'v5', len(v5), 'Ls', LS, 'bridge', a.bridge, flush=True)

m = DT(); m.detach_inference(); dev = m.dev
eng = m.p.eng


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def prep(state_text, qd, perm=None):
    q = ta.validate_python(qd)
    rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)


def nopt(qd): return 2 if qd['type'] == 'noul' else len(qd['criteria'])


class Teacher:
    def __enter__(self): self.s = (m.lora, m.head); m.lora = None; m.head = m.head0
    def __exit__(self, *e): m.lora, m.head = self.s


def build(state, qdict, names, rng):
    st = render_state(state); ps = []
    for n in names:
        qd = qdict[n]; perm = None
        if qd['type'] in ('choice', 'noul') and rng.random() < a.perm:
            perm = list(range(nopt(qd))); rng.shuffle(perm)
        elif qd['type'] == 'score' and rng.random() < a.perm:
            perm = list(reversed(range(nopt(qd))))
        ps.append(prep(st, qd, perm))
    s = ps[0]['s']; bl = sum(len(p['q']) for p in ps); k = a.maxtok - bl
    if k < 64: return None
    if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
    return s, ps


def sample(rng):
    global pi, ci, vi
    u = rng.random()
    if u < a.p_real:
        r = pool[pi % len(pool)]; pi += 1; names = list(r['questions']); v = rng.random()
        if len(names) > 1 and v < 0.45: pass
        elif len(names) > 1 and v < 0.75: names = rng.sample(names, rng.randint(2, min(4, len(names))))
        else: names = [rng.choice(names)]
        return [('real', r['state'], r['questions'], names, None)]
    if u < a.p_real + a.p_cf:
        pr_ = cf[ci % len(cf)]; ci += 1
        return [('cf', it['state'], it['questions'], list(it['questions']), it['expected']) for it in pr_]
    r = v5[vi % len(v5)]; vi += 1
    qd, gold = v5_q(r)
    if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
    return [('v5', r['state'], {'q': qd}, ['q'], {'q': gold})]


# ---- student parameters
p_lora = m.add_lora(r=a.lora_r, alpha=2 * a.lora_r, seed=a.seed)
if not a.shallow_lora:
    for i in range(LS[0]):
        for k in ('Win', 'Wo', 'Wgu', 'Wd'):
            m.lora[i][k].A.requires_grad_(False); m.lora[i][k].B.requires_grad_(False)
    p_lora = [p for p in p_lora if p.requires_grad]
p_mem = m.add_mem(LS, a.bridge, r=a.mem_r, alpha=a.mem_r, seed=a.seed + 1)
p_head = m.set_head('std')
if a.init:      # continue from a trained J3 checkpoint (LoRA + memory adapters + head); optimizer state fresh
    sd0 = torch.load(os.path.expanduser(a.init), map_location=dev)
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(sd0[f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(sd0[f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in sd0.items() if k.startswith('head.')})
        for k, md in m.mem.items():
            if f'mem.{k}.A' in sd0: md.A.copy_(sd0[f'mem.{k}.A']); md.B.copy_(sd0[f'mem.{k}.B'])
    print('init from', a.init, flush=True)
params = p_lora + p_mem + p_head
print('trainable', sum(p.numel() for p in params) / 1e6, 'M', flush=True)
opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_mem, lr=a.mlr), dict(params=p_head, lr=a.hlr)], betas=(0.9, 0.95), weight_decay=0.0)
warm = max(10, int(0.03 * a.updates))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
pi = ci = vi = 0; upd0 = 0
srng = random.Random(a.seed + 4)
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(st['sd'][f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(st['sd'][f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in st['sd'].items() if k.startswith('head.')})
        for k, md in m.mem.items():
            md.A.copy_(st['sd'][f'mem.{k}.A']); md.B.copy_(st['sd'][f'mem.{k}.B'])
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi, ci, vi = st['pos']; srng.setstate(st['rng'])
    print('resumed at update', upd0, flush=True)

meta = dict(Ls=LS, bridge=a.bridge, layout=a.layout, args=vars(a))
log = open(CK + 'train.log.jsonl', 'a'); json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); tok = 0
upd = upd0
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        for kind, state, qdict, names, gold in sample(srng):
            b = build(state, qdict, names, srng)
            if b is None: continue
            s, ps = b
            Ls = srng.choice(LS)
            hl = tuple(i for i in HL if i >= Ls)
            qs = [p['q'] for p in ps]
            with torch.no_grad(), Teacher():
                if a.shallow_lora:
                    lt = m.teacher_logits(s, qs, ps, keep=hl); xs = None
                else:
                    lt, xs = m.teacher_logits(s, qs, ps, keep=hl, split=Ls)
                kt = dict(m.kept)
            if a.layout == 'set':
                ls = dtset.set_logits(m, s, ps, Ls, a.bridge, ckpt=True, keep=hl); ks = dict(m.kept)
                r0 = 0; ridx = []                       # teacher rows matching the set readout rows: option-end rows (same label order), '<answer>'
                for p in ps:
                    ridx += [r0 + o for o in p['opt']] + [r0 + len(p['q']) - 1]; r0 += len(p['q'])
                ridx = torch.tensor(ridx, device=dev); kt = {i: v[ridx] for i, v in kt.items()}
            else:
                ls = m.dt_logits(s, qs, ps, Ls, a.bridge, ckpt=True, keep=hl, x_shallow=xs)
                ks = dict(m.kept)
            loss = 0.0; nq = len(names)
            if hl:
                hd = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in hl) / len(hl)
                loss = loss + a.w_hid * hd; lsum[f'hid{Ls}'] += float(hd); lcnt[f'hid{Ls}'] += 1
            for j in range(nq):
                sl = ps[j]['rq'].slot_labels
                tp = torch.softmax(lt[j].float(), -1); lp = F.log_softmax(ls[j].float(), -1)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
                if kind == 'real':
                    loss = loss + kl / nq
                else:
                    gi = sl.index(gold[names[j]])
                    if ps[j]['rq'].kind == 'score':      # F7 ordinal smoothing 0.1 onto the adjacent levels (level space, mapped through the order)
                        lv = int(gold[names[j]]); nb = [str(x) for x in (lv - 1, lv + 1) if 0 <= x < len(sl)]
                        ce = -(0.9 * lp[gi] + sum(0.1 / len(nb) * lp[sl.index(x)] for x in nb)) if nb else -lp[gi]
                    else:
                        ce = -lp[gi]
                    wce, wkl = (a.w_cf, a.w_cfkl) if kind == 'cf' else (a.w_v5, 1.0)
                    loss = loss + (wce * ce + wkl * kl) / nq
                    lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                    lsum['acc_' + kind] += float(int(lp.argmax()) == sl.index(gold[names[j]])); lcnt['acc_' + kind] += 1
                lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
                lsum[f'agree{Ls}_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt[f'agree{Ls}_' + kind] += 1
            (loss / a.accum).backward()
            tok += len(s) + sum(len(q) for q in qs)
            nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        el = time.time() - t0
        rec = dict(upd=upd, t=round(el), tok_s=round(tok / max(el, 1)), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in sorted(lsum)})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    stop = a.max_hours > 0 and time.time() - t0 > a.max_hours * 3600
    if upd % 25 == 0 or upd % a.every == 0 or upd == a.updates or stop:
        sd = m.dt_state(meta)
        if upd % a.every == 0 or upd == a.updates or stop:
            torch.save(sd, CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(sd=sd, opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pos=(pi, ci, vi), rng=srng.getstate()), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
    if stop: print('max_hours reached', flush=True); break
print('done', time.time() - t0, flush=True)
