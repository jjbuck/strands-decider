"""K1 stacktrain: the Brief-9 recipe, identical for every trained run; only --flags (and --seed for the s2 runs) differ.

Student: hobson-v19 + LoRA r32 (alpha 64) on Win/Wo/Wgu/Wd of all 24 layers + pointer head (from hobson's), plus the enabled components at their
         original sizes: D memory adapters r32; C compile adapter r16; V super-token rows (per-token delta) + per-position gains; Q shared r16 +
         per-question r8 LoRA + slot inputs. Every B factor zero-initialised (as in the source code).
Teacher: frozen hobson-v19 in its native layout (flags off, adapters off, hobson's head), same process, calibrated logits (/ T_kind).
Data (train split only; evalkit eval tasks never used). Per sequence draw: 60% real train_pool state (all questions as branches p .45, a 2-4
         subset p .30, one p .25; loss KL(teacher || student) averaged over its questions); 25% train_v5 gold row (CE + KL); 15% detail
         augmentation, half H7 cf_aug pairs, half J7 gen_aug pairs (both items of the pair; CE + 0.3 KL).
         Option order permuted with p .5 (choice / noul), score rubric reversed with p .5, seen identically by teacher and student; under Q a
         deployed question is never permuted (its compiled slots have a fixed order).
Dense row loss (weight 1.0): relative MSE between student and teacher hidden states after layers 5, 11, 17, 23 on the answer rows (each
         question's option-end rows and its <answer> row; slot rows under Q; under V the rows ending at the same Qwen offsets).
Budget: --updates updates of 8 sequences (a pair is never split, so an update may hold 9), sequences <= 6144 Qwen tokens (state truncated at
         line boundaries, head 1/4 + tail 3/4), AdamW (0.9, 0.95, wd 0), clip 1.0, 50 warm-up steps then cosine to 0.1x.
         The draw stream is a pure function of (--seed, draw index): identical across flags.
Resumable from CK/last.pt.
python stacktrain.py --flags DCVQ --updates 1000 --ck ~/work/k1/runs/DCVQ
"""
import os, sys, json, time, random, argparse, collections, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch, torch.nn.functional as F
import stacklib as SL

ap = argparse.ArgumentParser()
ap.add_argument('--flags', default=''); ap.add_argument('--updates', type=int, default=1000); ap.add_argument('--per_update', type=int, default=8)
ap.add_argument('--seed', type=int, default=7); ap.add_argument('--ck', required=True); ap.add_argument('--maxtok', type=int, default=6144)
ap.add_argument('--warm', type=int, default=50); ap.add_argument('--every', type=int, default=250); ap.add_argument('--max_hours', type=float, default=0)
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--p_real', type=float, default=0.60); ap.add_argument('--p_v5', type=float, default=0.25)
ap.add_argument('--perm', type=float, default=0.5)
# learning rates (the same for every run)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--hlr', type=float, default=2e-4); ap.add_argument('--mlr', type=float, default=5e-4)
ap.add_argument('--clr', type=float, default=1e-4); ap.add_argument('--slr_A', type=float, default=1e-2); ap.add_argument('--slr_d', type=float, default=1.5e-4)
ap.add_argument('--qlr', type=float, default=2e-4); ap.add_argument('--qslr', type=float, default=1e-3)
ap.add_argument('--log_every', type=int, default=10); ap.add_argument('--stop_after', type=int, default=0, help='pilot: stop after N updates of this process')
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
W = os.path.expanduser('~/work/')
FLAGS = SL.flagset(a.flags)
HL = SL.HID_LAYERS if a.w_hid > 0 else ()
EV = set(json.load(open(W + 'evalkit/split.json'))['eval_tasks'])

# ---------------------------------------------------------------- data (flag-independent)
pool = []
with open(W + 'evalkit/train_pool.jsonl') as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 64: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
v5 = [json.loads(l) for l in open(W + 'training/data/train_v5.jsonl')]
aug_h7 = collections.defaultdict(list)
for l in open(W + 'h7/data/cf_aug.jsonl'):
    r = json.loads(l); assert r['task'] not in EV; aug_h7[r['pair']].append(r)
aug_j7 = collections.defaultdict(list)
for l in open(W + 'k1/data/j7aug.jsonl'):
    r = json.loads(l); assert r['task'] not in EV; aug_j7[r['pair']].append(r)
aug_h7 = [aug_h7[k] for k in sorted(aug_h7)]; aug_j7 = [aug_j7[k] for k in sorted(aug_j7)]
R0 = random.Random(a.seed); R0.shuffle(pool); R0.shuffle(v5); R0.shuffle(aug_h7); R0.shuffle(aug_j7)
print('pool', len(pool), 'v5', len(v5), 'aug h7 pairs', len(aug_h7), 'aug j7 pairs', len(aug_j7), 'flags', ''.join(sorted(FLAGS)) or '-', flush=True)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(qd): return 2 if qd['type'] == 'noul' else len(qd['criteria'])


def draw(j):
    """draw j -> list of sequences (kind, state, {name: spec}, [(name, perm)], gold or None). Pure function of (seed, j)."""
    rng = random.Random(f'{a.seed}-{j}')
    u = rng.random()
    if u < a.p_real:
        r = pool[rng.randrange(len(pool))]; names = sorted(r['questions']); v = rng.random()
        if len(names) > 1 and v < 0.45: pass
        elif len(names) > 1 and v < 0.75: names = sorted(rng.sample(names, rng.randint(2, min(4, len(names)))))
        else: names = [rng.choice(names)]
        seqs = [('real', r['state'], r['questions'], names, None)]
    elif u < a.p_real + a.p_v5:
        r = v5[rng.randrange(len(v5))]; qd, gold = v5_q(r)
        if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
        seqs = [('v5', r['state'], {'q': qd}, ['q'], {'q': gold})]
    else:
        prs = aug_h7[rng.randrange(len(aug_h7))] if rng.random() < 0.5 else aug_j7[rng.randrange(len(aug_j7))]
        seqs = [('aug', it['state'], it['questions'], sorted(it['questions']), it['expected']) for it in prs]
    out = []
    for kind, state, qdict, names, gold in seqs:
        ql = []
        for n in names:
            qd = qdict[n]; perm = None
            if qd['type'] in ('choice', 'noul') and rng.random() < a.perm:
                perm = list(range(nopt(qd))); rng.shuffle(perm)
            elif qd['type'] == 'score' and rng.random() < a.perm:
                perm = list(reversed(range(nopt(qd))))
            ql.append((n, qd, perm))
        out.append((kind, state, ql, gold))
    return out


# ---------------------------------------------------------------- model
m = SL.Stack(); m.detach_inference(); dev = m.dev
from strands_decider.prompting import render_state
groups = m.add_student(FLAGS, seed=a.seed)
LR = dict(lora=a.lr, head=a.hlr, mem=a.mlr, compile=a.clr, sup_A=a.slr_A, sup_delta=a.slr_d, q_shared=a.qlr, q_perq=a.qlr, q_slots=a.qslr)
opt_groups = [dict(params=ps, lr=LR[k], name=k) for k, ps in groups.items()]
params = [p for g in opt_groups for p in g['params']]
print('trainable', {k: round(sum(p.numel() for p in ps) / 1e6, 2) for k, ps in groups.items()}, 'M', flush=True)
opt = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95), weight_decay=0.0)
U_ = a.updates
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / U_)))))
upd0 = 0; dj = 0
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    with torch.no_grad():
        for p_, v_ in zip(params, st['params']): p_.copy_(v_)
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); upd0 = st['upd']; dj = st['dj']
    print('resumed at update', upd0, 'draw', dj, flush=True)
json.dump(dict(vars(a), flags=''.join(sorted(FLAGS)), version=SL.VERSION), open(CK + 'config.json', 'w'), indent=1)
log = open(CK + 'train_log.jsonl', 'a')


def build(state, ql):
    """-> (teacher Base, student Base) or None. Same truncated state text for both; under Q deployed questions are unpermuted."""
    if isinstance(state, (dict, list)): state = json.dumps(state, indent=2, ensure_ascii=False)
    st = render_state(state)
    if 'Q' in FLAGS:
        ql = [(n, sp, None if (n in m.deployed and json.dumps(sp, sort_keys=True) == m.deployed_js[n]) else pm) for n, sp, pm in ql]
    b = m.prep(st, ql)
    qt = sum(len(q['q']) for q in b.qs)
    k = a.maxtok - qt
    if k < 64: return None
    if len(b.s) > k:
        st = SL.trunc_text(m.tok, st, k); b = m.prep(st, ql)
        if len(b.s) + qt > a.maxtok + 16: return None
    return b


lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); tlast = t0; ntok = 0
upd = upd0; n_this = 0
while upd < a.updates:
    nmic = 0
    while nmic < a.per_update:
        seqs = draw(dj); dj += 1
        for kind, state, ql, gold in seqs:
            b = build(state, ql)
            if b is None: lcnt['skipped'] += 1; continue
            with torch.no_grad(), m.teacher():
                ot = m.forward(m.view(b, ''), keep=HL)
            vs = m.view(b, FLAGS)
            os_ = m.forward(vs, ckpt=True, keep=HL)
            loss = 0.0; nq = len(b.qs)
            if HL:
                hd = sum(((os_['kept'][i].float() - ot['kept'][i].float()) ** 2).sum() / ot['kept'][i].float().pow(2).sum() for i in HL) / len(HL)
                loss = loss + a.w_hid * hd; lsum['hid'] += float(hd); lcnt['hid'] += 1
            for j, qd in enumerate(b.qs):
                lt, ls = ot['logits'][j], os_['logits'][j]
                tp = torch.softmax(lt.float(), -1); lp = F.log_softmax(ls.float(), -1)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
                if kind == 'real':
                    loss = loss + kl / nq
                else:
                    sl = qd['rq'].slot_labels; gi = sl.index(str(gold[qd['name']]))
                    ce = -lp[gi]
                    wce, wkl = (1.0, 0.3) if kind == 'aug' else (1.0, 1.0)
                    loss = loss + (wce * ce + wkl * kl) / nq
                    lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                    lsum['acc_' + kind] += float(int(lp.argmax()) == gi); lcnt['acc_' + kind] += 1
                lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
                lsum['agree_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt['agree_' + kind] += 1
            (loss / a.per_update).backward()
            P = os_['P']; ntok += P.T
            lsum['rows_student'] += P.T; lcnt['rows_student'] += 1; lsum['tok_teacher'] += len(b.s) + sum(len(q['q']) for q in b.qs); lcnt['tok_teacher'] += 1
            nmic += 1
            del os_, ot, loss
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1; n_this += 1
    if upd % a.log_every == 0 or upd == upd0 + 1:
        now = time.time()
        rec = dict(upd=upd, t=round(now - t0), s_per_upd=round((now - tlast) / (a.log_every if upd != upd0 + 1 else 1), 2), dj=dj,
                   mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in sorted(lsum)}, skipped=lcnt['skipped'])
        tlast = now
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    stop = (a.max_hours > 0 and time.time() - t0 > a.max_hours * 3600) or (a.stop_after and n_this >= a.stop_after)
    if upd % 25 == 0 or upd == a.updates or stop:
        if upd % a.every == 0 or upd == a.updates:
            torch.save(m.student_state(), CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(params=[p.detach() for p in params], opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, dj=dj), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
    if stop: print('stopping at', upd, flush=True); break
if upd >= a.updates:
    torch.save(m.student_state(), CK + 'final.pt'); open(CK + 'TRAIN_DONE', 'w').write(f'{upd} {time.time() - t0:.0f}s\n')
print('done', upd, f'{time.time() - t0:.0f}s', flush=True)
