"""J7 training (F7 recipe, plain state-first layout, one question per sequence).  Arms:
  C  : LoRA r16/a32 (every projection, 24 layers) + pointer head.  Data: real (train_pool, KL to hobson), aug (j7 gen_aug pairs, CE gold + 0.3 KL),
       v5 (train_v5 1-in-8, CE gold 0.3 + KL).  Option-order permutation p .5 (student only).
  V  : C + zero-initialised side channels (chan.py: value identity, digit place, field key, record, role/section, turn).
  T  : super-token transplant: student reads super-tokens (level drawn per example from 4k/16k/64k), teacher reads Qwen tokens.  Loss: KL +
       dense token distillation (relative MSE of student rows vs hobson's rows at the aligned original positions, layers 3,7,11,15,19,23),
       real + v5 rows only, no permutation.  Trainable: LoRA + head + super-token embeddings (shared constituent gains A + per-token delta).
  TV : T checkpoint + channels, then the C/V recipe (aug included) on super-tokens.
The teacher is frozen hobson (LoRA off, hobson head, no channels, Qwen tokens) in the same process.  Resumable (ck/last.pt).
python train_j7.py --arm C --updates 400 --ck ~/work/j7/ck_C"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn.functional as F
from j7lib import J7, Prep, CH_KEYS
from superbpe import Super
from strands_decider.prompting import render_state

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True); ap.add_argument('--updates', type=int, default=400); ap.add_argument('--every', type=int, default=100)
ap.add_argument('--accum', type=int, default=8); ap.add_argument('--lr', type=float, default=1.5e-4); ap.add_argument('--hlr', type=float, default=5e-4)
ap.add_argument('--clr', type=float, default=2e-3); ap.add_argument('--slr', type=float, default=1e-2); ap.add_argument('--dlr', type=float, default=1.5e-4)
ap.add_argument('--maxtok', type=int, default=4096); ap.add_argument('--p_real', type=float, default=0.5); ap.add_argument('--p_aug', type=float, default=0.3)
ap.add_argument('--w_aug', type=float, default=1.0); ap.add_argument('--w_augkl', type=float, default=0.3); ap.add_argument('--w_v5', type=float, default=0.3)
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--hid_layers', default='3,7,11,15,19,23')
ap.add_argument('--perm', type=float, default=0.5); ap.add_argument('--ck', required=True); ap.add_argument('--init', default='')
ap.add_argument('--levels', default='4k:0.2,16k:0.3,64k:0.5'); ap.add_argument('--protect', type=int, default=0); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--ckpt_over', type=int, default=2600, help='activation checkpointing for student sequences longer than this')
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
W = os.path.expanduser('~/work/')
EV = set(json.load(open(W + 'evalkit/split.json'))['eval_tasks'])
SUPER = a.arm in ('T', 'TV'); CHAN = a.arm in ('V', 'TV'); DENSE_ON = a.arm == 'T'
if a.arm == 'T': a.p_aug = 0.0; a.perm = 0.0

pool = []
with open(W + 'evalkit/train_pool.jsonl') as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 64: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
aug = collections.defaultdict(list)
for l in open(W + 'j7/aug.jsonl'):
    r = json.loads(l); assert r['task'] not in EV
    aug[r['pair']].append(r)
aug = list(aug.values())
v5 = []
with open(W + 'training/data/train_v5.jsonl') as f:
    for li, l in enumerate(f):
        if li % 8 == 0: v5.append(json.loads(l))
R = random.Random(a.seed); R.shuffle(pool); R.shuffle(aug); R.shuffle(v5)
print('pool', len(pool), 'aug pairs', len(aug), 'v5', len(v5), flush=True)

m = J7(); m.detach_inference(); dev = m.dev
eng = m.p.eng
supers = {}
if SUPER:
    S64 = Super(W + 'j7/tok/sb64k.json', tok=eng.tok)
    supers = {'64k': S64, '16k': Super(W + 'j7/tok/sb16k.json', tok=eng.tok), '4k': Super(W + 'j7/tok/sb4k.json', tok=eng.tok)}
    for k, S in supers.items(): assert S.toks == S64.toks[:len(S.toks)], k; S.idx = {t: S64.V + i for i, t in enumerate(S.toks)}
levels = [(x.split(':')[0], float(x.split(':')[1])) for x in a.levels.split(',')]
P = Prep(eng, supers, protect=bool(a.protect))


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(qd): return 2 if qd['type'] == 'noul' else len(qd['criteria'])


class Teacher:
    def __enter__(self):
        self.s = (m.lora, m.head, m.chans, m.sup); m.lora = None; m.head = m.head0; m.chans = None; m.sup = None
    def __exit__(self, *e):
        m.lora, m.head, m.chans, m.sup = self.s


def trunc_state(st):
    """head 1/4 + tail 3/4 of the rendered state at the token level (keeps the newest turns), as text-free id surgery is not possible with
    offsets: truncate the TEXT by the token offsets instead."""
    enc = eng.tok(st, add_special_tokens=False, return_offsets_mapping=True)
    k = a.maxtok
    if len(enc['input_ids']) <= k: return st
    offs = enc['offset_mapping']; h = k // 4
    return st[:offs[h][0]] + '\n[...]\n' + st[offs[-(k - h)][0]:]


def sample(rng):
    global pi, ai, vi
    u = rng.random()
    if u < a.p_real:
        r = pool[pi % len(pool)]; pi += 1
        qn = rng.choice(list(r['questions']))
        return [('real', r['state'], r['questions'][qn], None)]
    if u < a.p_real + a.p_aug:
        pr_ = aug[ai % len(aug)]; ai += 1
        return [('aug', it['state'], it['questions']['detail'], it['expected']['detail']) for it in pr_]
    r = v5[vi % len(v5)]; vi += 1
    qd, gold = v5_q(r)
    if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
    return [('v5', r['state'], qd, gold)]


# ---- student
params_groups = []
p_lora = m.add_lora(r=16, alpha=32, seed=a.seed); p_head = m.set_head('std')
groups = [dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)]
if SUPER:
    m.add_super(S64.toks, S64.V); groups += [dict(params=[m.sup.A], lr=a.slr), dict(params=[m.sup.delta], lr=a.dlr)]
if a.init:
    sd = torch.load(os.path.expanduser(a.init), map_location=dev)
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(sd[f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(sd[f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
        if SUPER and 'sup.A' in sd: m.sup.A.copy_(sd['sup.A']); m.sup.delta.copy_(sd['sup.delta'].float())
    print('init from', a.init, flush=True)
if CHAN:
    p_ch = m.add_channels(); groups.append(dict(params=p_ch, lr=a.clr))
params = [p for g in groups for p in g['params']]
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(10, a.updates // 30)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
pi = ai = vi = 0; upd0 = 0
srng = random.Random(a.seed + 4)
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    with torch.no_grad():
        for n_, p_ in zip(st['names'], params): pass
        for p_, v_ in zip(params, st['params']): p_.copy_(v_)
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi, ai, vi = st['pos']; srng.setstate(st['rng'])
    print('resumed at update', upd0, flush=True)

HL = tuple(int(x) for x in a.hid_layers.split(',')) if DENSE_ON else ()
log = open(CK + 'train.log.jsonl', 'a'); json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()
upd = upd0
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        for kind, state, qd, gold in sample(srng):
            st_text = trunc_state(render_state(state))
            try:
                prc = P.base(st_text, qd)
            except Exception as e:
                print('skip', repr(e)[:100], flush=True); continue
            perm = None
            if qd['type'] in ('choice', 'noul') and srng.random() < a.perm:
                perm = list(range(nopt(qd))); srng.shuffle(perm)
            prs = P.base(st_text, qd, perm) if perm is not None else prc
            lev = None
            if SUPER:
                u = srng.random(); c = 0.0
                for nm_, pw in levels:
                    c += pw
                    if u <= c: lev = nm_; break
                lev = lev or levels[-1][0]
            bt = P.build(prc)
            bs = P.build(prs, level=lev, with_feats=CHAN)
            with torch.no_grad(), Teacher():
                lt = m.logits1(bt, keep=HL)
                kt = dict(m.kept)
            Ts = len(bs['ids'])
            ls = m.logits1(bs, ckpt=Ts > a.ckpt_over, keep=HL)
            ks = dict(m.kept)
            tl = prc['rq'].slot_labels; sl = prs['rq'].slot_labels
            idx = torch.tensor([tl.index(x) for x in sl], device=dev)
            tp = torch.softmax(lt.float(), -1)[idx]
            lp = F.log_softmax(ls.float(), -1)
            kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
            loss = 0.0
            if HL:
                rows_t = torch.tensor(bs['ends'], device=dev)[4:]
                hl = 0.0
                for i in HL:
                    s_ = ks[i][4:].float(); t_ = kt[i][rows_t].float()
                    hl = hl + (((s_ - t_) ** 2).sum(-1) / t_.pow(2).sum(-1).clamp_min(1e-6)).mean()
                hl = hl / len(HL)
                loss = loss + a.w_hid * hl
                lsum['hid'] += float(hl); lcnt['hid'] += 1
            if kind == 'real':
                loss = loss + kl
            else:
                gi = sl.index(gold)
                ce = -lp[gi]
                wce, wkl = (a.w_aug, a.w_augkl) if kind == 'aug' else (a.w_v5, 1.0)
                loss = loss + wce * ce + wkl * kl
                lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                lsum['acc_' + kind] += float(int(lp.argmax()) == gi); lcnt['acc_' + kind] += 1
                lsum['tacc_' + kind] += float(int(tp.argmax()) == gi); lcnt['tacc_' + kind] += 1
            lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
            lsum['agree_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + kind] += 1
            (loss / a.accum).backward()
            lsum['T'] += Ts; lcnt['T'] += 1; lsum['Torig'] += bs['n_orig']; lcnt['Torig'] += 1
            nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        rec = dict(upd=upd, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in lsum})
        if CHAN: rec['ch_norm'] = {k: round(float(v.norm()), 3) for k, v in list(m.chans.P.items()) + list(m.chans.E.items())}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if upd % a.every == 0 or upd == a.updates or upd % 25 == 0:
        sd = m.j7_state()
        if upd % a.every == 0 or upd == a.updates:
            torch.save(sd, CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(params=[p.detach() for p in params], names=[str(p.shape) for p in params], opt=opt.state_dict(), sched=sched.state_dict(),
                        upd=upd, pos=(pi, ai, vi), rng=srng.getstate()), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
print('done', time.time() - t0, flush=True)
