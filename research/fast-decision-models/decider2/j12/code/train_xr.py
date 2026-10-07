"""J12: fine-tune hobson-v19 (full 24 layers) with or without XR exact-relation modules. F7-style recipe from hobson init:
  LoRA r16/a32 on every projection of all 24 layers + trainable pointer head (+ XR modules at layers 11, 17 in arm 'xr').
  Teacher: frozen hobson-v19 (same weights, LoRA and XR off), hobson's own layout, calibrated (logits / T_kind), same process.
  Data (train split only; evalkit eval tasks excluded; never evalkit eval items):
    real : evalkit/train_pool.jsonl requests, one random question, KL(teacher || student)
    v5   : training/data/train_v5.jsonl gold rows, CE(gold) + KL
    syn  : j12 synthetic exact-reasoning pairs (gen_xr.py; templates disjoint from CF-probe), CE(gold) + W_SYNKL * KL (+ W_PTR * pointer aux, arm xr)
  Option-order permutation p .5 on the student (teacher canonical, matched by label).
python train_xr.py --arm xr|ctl --updates N --ck DIR   (resumable from DIR/last.pt)"""
import os, sys, json, time, random, argparse, collections, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from xrlib import HX, Prep, gold_slots, order_gold
from strands_decider.prompting import render_state

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='xr'); ap.add_argument('--updates', type=int, default=1000); ap.add_argument('--accum', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--hlr', type=float, default=2e-4); ap.add_argument('--xlr', type=float, default=1e-3)
ap.add_argument('--p_real', type=float, default=0.45); ap.add_argument('--p_v5', type=float, default=0.15)
ap.add_argument('--w_synkl', type=float, default=0.2); ap.add_argument('--w_ptr', type=float, default=0.3); ap.add_argument('--perm', type=float, default=0.5)
ap.add_argument('--maxtok', type=int, default=4096); ap.add_argument('--ck', default='~/work/j12/ck_xr'); ap.add_argument('--syn', default='~/work/j12/data/xr_train.jsonl')
ap.add_argument('--every', type=int, default=250); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--ptr_order', type=int, default=0)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])

pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 32: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
syn = collections.defaultdict(list)
for l in open(os.path.expanduser(a.syn)):
    r = json.loads(l); assert r['task'] not in EV
    syn[r['pair']].append(r)
syn = list(syn.values())
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 8 == 0: v5.append(json.loads(l))
R = random.Random(a.seed); R.shuffle(pool); R.shuffle(syn); R.shuffle(v5)
print('pool', len(pool), 'syn pairs', len(syn), 'v5', len(v5), flush=True)

m = HX(); m.detach_inference(); dev = m.dev
prep = Prep(m.p.eng)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(qd):
    return 2 if qd['type'] == 'noul' else len(qd['criteria'])


class Teacher:
    def __enter__(self):
        self.s = (m.lora, m.head, m.xr_on); m.lora = None; m.head = m.head0; m.xr_on = False
    def __exit__(self, *e):
        m.lora, m.head, m.xr_on = self.s


def sample(rng):
    global pi, si, vi
    u = rng.random()
    if u < a.p_real:
        r = pool[pi % len(pool)]; pi += 1
        qn = rng.choice(list(r['questions']))
        return [('real', r['state'], r['questions'][qn], None, None)]
    if u < a.p_real + a.p_v5:
        r = v5[vi % len(v5)]; vi += 1
        qd, gold = v5_q(r)
        if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
        return [('v5', r['state'], qd, gold, None)]
    pr_ = syn[si % len(syn)]; si += 1
    return [('syn', it['state'], it['questions']['x'], it['expected']['x'], it) for it in pr_]


# ---- student
if a.arm == 'xrf':      # frozen hobson torso and head: only the XR modules train (student = hobson + XR deltas on question rows)
    p_lora, p_head = [], []
    m.head = m.head0
    groups = []
else:
    p_lora = m.add_lora(r=16, alpha=32, seed=7)
    p_head = m.set_head('std')
    groups = [dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)]
p_xr = []
if a.arm in ('xr', 'xrf'):
    p_xr = m.add_xr(H=16, dk=64)
    groups.append(dict(params=p_xr, lr=a.xlr))
params = p_lora + p_head + p_xr
print('trainable', sum(p.numel() for p in params), 'xr', sum(p.numel() for p in p_xr), flush=True)
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(10, int(0.03 * a.updates))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
pi = si = vi = 0; upd0 = 0
srng = random.Random(11)
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    with torch.no_grad():
        if m.lora is not None:
            for i in range(24):
                for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                    m.lora[i][k].A.copy_(st['sd'][f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(st['sd'][f'lora.{i}.{k}.B'])
        if p_head: m.head.load_state_dict({k[5:]: v for k, v in st['sd'].items() if k.startswith('head.')})
        if m.xr is not None: m.xr.load_state_dict({k[3:]: v for k, v in st['sd'].items() if k.startswith('xr.')})
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi, si, vi = st['pos']; srng.setstate(st['rng'])
    print('resumed at update', upd0, flush=True)

log = open(CK + 'train.log.jsonl', 'a')
json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()
upd = upd0
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        for kind, state, qd, gold, it in sample(srng):
            try:
                pc = prep(state, qd, max_tok=a.maxtok)
            except Exception as e:
                print('prep fail', kind, repr(e)[:200], flush=True); continue
            if pc is None: continue
            ps = pc
            if qd['type'] in ('choice', 'noul') and srng.random() < a.perm:
                perm = list(range(nopt(qd))); srng.shuffle(perm)
                ps = prep(state, qd, perm=perm, max_tok=a.maxtok)
                if ps is None: continue
            with torch.no_grad(), Teacher():
                ht = m.fwd(pc); lt = m.logits_pr(ht, pc)
            ix = m.ix_for(ps) if m.xr is not None else None
            hs = m.fwd(ps, ix=ix, ckpt=True, keep_last=True)
            ls = m.logits_pr(hs, ps)
            tl = pc['rq'].slot_labels; sl = ps['rq'].slot_labels
            idx = torch.tensor([tl.index(x) for x in sl], device=dev)
            tp = torch.softmax(lt.float(), -1)[idx]
            lp = F.log_softmax(ls.float(), -1)
            kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
            if kind == 'real':
                loss = kl
            else:
                gi = sl.index(gold)
                ce = -lp[gi]
                loss = ce + (a.w_synkl if kind == 'syn' else 1.0) * kl
                lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                lsum['acc_' + kind] += float(int(lp.argmax()) == gi); lcnt['acc_' + kind] += 1
                lsum['tacc_' + kind] += float(int(tp.argmax()) == gi); lcnt['tacc_' + kind] += 1
                if kind == 'syn':
                    lsum['acc_' + it['kind']] += float(int(lp.argmax()) == gi); lcnt['acc_' + it['kind']] += 1
            if kind == 'syn' and m.xr is not None and a.w_ptr > 0 and ix['N'] > 0:
                gs = gold_slots(ix, it, ps)
                if gs is not None and a.ptr_order: gs = order_gold(gs, ix, ps)
                if gs is None:
                    lcnt['ptr_miss'] += 1
                else:
                    pl = 0.0
                    for x in m.xr:
                        pa, pb = x.last                     # [H, N+1] at the answer row
                        pa4, pb4 = pa[:4], pb[:4]
                        if gs['b'] is None:
                            pr_ = pa4[:, gs['a']]
                        else:
                            pr_ = (pa4[:, gs['a']] * pb4[:, gs['b']]) if a.ptr_order else 0.5 * (pa4[:, gs['a']] * pb4[:, gs['b']] + pa4[:, gs['b']] * pb4[:, gs['a']])
                        pl = pl - torch.log(pr_.mean() + 1e-6)
                        lsum['ptr_top'] += float(int(pa[0].argmax()) in (gs['a'], gs['b'] if gs['b'] is not None else -1)); lcnt['ptr_top'] += 1
                    loss = loss + a.w_ptr * pl
                    lsum['ptr'] += float(pl); lcnt['ptr'] += 1
            lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
            lsum['agree_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + kind] += 1
            (loss / a.accum).backward()
            lsum['T'] += ps['L']; lcnt['T'] += 1
            nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        rec = dict(upd=upd, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in lsum}, ptr_miss=lcnt['ptr_miss'])
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if upd % 50 == 0 or upd == a.updates:
        sd = m.trainable_state()
        if upd % a.every == 0 or upd == a.updates:
            torch.save(sd, CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(sd=sd, opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pos=(pi, si, vi), rng=srng.getstate()), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
print('done', time.time() - t0, flush=True)
