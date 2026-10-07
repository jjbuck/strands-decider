"""H7: fine-tune hobson-v19 for the compiled-schema layout (LoRA r16/a32 on every projection of all 24 layers + the pointer head).
Student layout: [Q1' .. Qn' (bundle, compiled once)][state][<answer>_1 .. <answer>_n], slot k sees its own question span + state (h7lib 'slots').
Teacher: frozen bf16 hobson, state-first (its own layout: one sequence per question), in the same process, calibrated (logits / T_kind).
Data (train split only; eval tasks excluded):
  real : evalkit/train_pool.jsonl requests, bundle = full question set (p .45) / random subset of 2-4 in random order (p .30) / one question (p .25)
         loss = KL(teacher || student) averaged over the bundle's questions
  cf   : h7/data/cf_aug.jsonl (CF-style edits of train-split states, labels by construction): loss = W_CF * CE(gold) + W_CFKL * KL
  v5   : training/data/train_v5.jsonl (1 in 8 rows): loss = W_V5 * CE(gold) + KL
  option-order permutation (choice + noul) with p .5 per student question; the teacher always sees canonical order (matched by label name).
Resumable: ck/last.pt holds LoRA + head + optimiser + step. Checkpoints ck/s{update}.pt (LoRA + head only) every --every updates.
python train_h7.py --updates 1500 --every 300"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn.functional as F
from h7lib import H7
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)

ap = argparse.ArgumentParser()
ap.add_argument('--updates', type=int, default=1500); ap.add_argument('--every', type=int, default=300); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1.5e-4); ap.add_argument('--hlr', type=float, default=5e-4); ap.add_argument('--maxtok', type=int, default=6144)
ap.add_argument('--p_real', type=float, default=0.70); ap.add_argument('--p_cf', type=float, default=0.20)
ap.add_argument('--w_cf', type=float, default=1.0); ap.add_argument('--w_cfkl', type=float, default=0.3); ap.add_argument('--w_v5', type=float, default=0.3)
ap.add_argument('--perm', type=float, default=0.5); ap.add_argument('--ck', default='~/work/h7/ck'); ap.add_argument('--layout', default='slots')
ap.add_argument('--w_hid', type=float, default=0.0); ap.add_argument('--hid_layers', default='5,11,17,23')
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])

pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 64: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
cf = collections.defaultdict(list)
for l in open(os.path.expanduser('~/work/h7/data/cf_aug.jsonl')):
    r = json.loads(l); assert r['task'] not in EV
    cf[r['pair']].append(r)
cf = list(cf.values())
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 8 == 0: v5.append(json.loads(l))
R = random.Random(7); R.shuffle(pool); R.shuffle(cf); R.shuffle(v5)
print('pool', len(pool), 'cf pairs', len(cf), 'v5', len(v5), flush=True)

m = H7(); m.detach_inference(); dev = m.dev
eng = m.p.eng


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


def build(state, qdict, names, rng):
    """-> s (shared, truncated), canonical preps, student (maybe permuted) preps"""
    st = render_state(state)
    pc = [prep(st, qdict[n]) for n in names]
    ps = []
    for n, p in zip(names, pc):
        qd = qdict[n]
        if qd['type'] in ('choice', 'noul') and rng.random() < a.perm:
            perm = list(range(nopt(qd))); rng.shuffle(perm); ps.append(prep(st, qd, perm))
        else: ps.append(p)
    s = pc[0]['s']
    bl = sum(len(p['q']) for p in ps)
    k = a.maxtok - bl
    if k < 64: return None
    if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
    return s, pc, ps


def sample(rng):
    global pi, ci, vi
    u = rng.random()
    if u < a.p_real:
        r = pool[pi % len(pool)]; pi += 1
        names = list(r['questions'])
        v = rng.random()
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


# ---- student
p_lora = m.add_lora(r=16, alpha=32, seed=7)
p_head = m.set_head('std')
params = p_lora + p_head
opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)], betas=(0.9, 0.95), weight_decay=0.0)
warm = 30
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

log = open(CK + 'train.log.jsonl', 'a')
json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()
upd = upd0
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        for kind, state, qdict, names, gold in sample(srng):
            b = build(state, qdict, names, srng)
            if b is None: continue
            s, pc, ps = b
            HL = tuple(int(x) for x in a.hid_layers.split(',')) if a.w_hid > 0 and a.layout == 'sets' else ()
            with torch.no_grad(), Teacher():
                lt = m.statefirst_logits(s, [p['q'] for p in pc], pc, keep=HL); kt = dict(m.kept)
            if a.layout == 'sets':
                ls = m.sets_logits(s, [p['q'] for p in ps], ps, ckpt=True, keep=HL); ks = dict(m.kept)
            else:
                ls = m.schema_logits(s, [p['q'] for p in ps], ps, ckpt=True)
            loss = 0.0; nq = len(names)
            if HL:   # dense signal: slot-set rows (option-end rows + '<answer>') vs hobson's own rows, unpermuted questions only
                r = 0; sel = []
                for j in range(nq):
                    L = len(ps[j]['opt']) + 1
                    if ps[j] is pc[j]: sel += list(range(r, r + L))
                    r += L
                if sel:
                    si = torch.tensor(sel, device=dev)
                    hl = sum(((ks[i][si].float() - kt[i][si].float()) ** 2).sum() / kt[i][si].float().pow(2).sum() for i in HL) / len(HL)
                    loss = loss + a.w_hid * hl
                    lsum['hid'] += float(hl); lcnt['hid'] += 1
            for j in range(nq):
                tl = pc[j]['rq'].slot_labels; sl = ps[j]['rq'].slot_labels
                idx = torch.tensor([tl.index(x) for x in sl], device=dev)
                tp = torch.softmax(lt[j].float(), -1)[idx]
                lp = F.log_softmax(ls[j].float(), -1)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
                if kind == 'real':
                    loss = loss + kl / nq
                else:
                    ce = -lp[sl.index(gold[names[j]])]
                    wce, wkl = (a.w_cf, a.w_cfkl) if kind == 'cf' else (a.w_v5, 1.0)
                    loss = loss + (wce * ce + wkl * kl) / nq
                    lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                    lsum['acc_' + kind] += float(int(lp.argmax()) == sl.index(gold[names[j]])); lcnt['acc_' + kind] += 1
                lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
                lsum['agree_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + kind] += 1
            (loss / a.accum).backward()
            lsum['T'] += len(s) + sum(len(p['q']) for p in ps); lcnt['T'] += 1
            nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
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
