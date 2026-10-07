"""M2: distil a decision-native layout (m2lib.Cfg) from frozen hobson-v19 (in process), train split only.
Student: hobson weights + LoRA r32/a64 on every projection of all 24 layers + hobson's pointer head (trainable)
         [+ new mixer modules, full rank, when the layout has them].
Teacher: hobson (LoRA off, head0), its own shared-prefix layout, on the same (possibly option-permuted, possibly truncated) inputs.
Data:  real  evalkit/train_pool.jsonl (eval tasks and J14's is_dev tasks excluded): KL(teacher || student), all/2-4/1 questions
       v5    training/data/train_v5.jsonl gold rows: CE + KL
       cf    h7/data/cf_aug.jsonl (CF-style edits of train-split states, labels by construction): CE + 0.3 KL   (<= 15%)
       dense relative MSE of the answer + option rows at layers 5, 11, 17, 23 (student vs teacher)
Dev:   J14's is_dev split of train_pool (fixed 150 requests): agree / agree_sd (vs hobson without state) / tv, every --dev_min minutes,
       with a checkpoint. Resumable via CK/last.pt.
python m2train.py --cfg 'gran=nat;iso=all;freeze=8' --ck ~/work/m2/ck_X --max_hours 5.5"""
import os, sys, json, time, random, argparse, collections, math, hashlib
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from m2lib import M2, Cfg
import m2train_util as TU
from strands_decider.prompting import render_state

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True); ap.add_argument('--updates', type=int, default=4000); ap.add_argument('--accum', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--hlr', type=float, default=3e-4); ap.add_argument('--nlr', type=float, default=5e-4)
ap.add_argument('--maxtok', type=int, default=6144); ap.add_argument('--p_real', type=float, default=0.55); ap.add_argument('--p_cf', type=float, default=0.15)
ap.add_argument('--w_cf', type=float, default=1.0); ap.add_argument('--w_cfkl', type=float, default=0.3); ap.add_argument('--w_v5', type=float, default=1.0)
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--hid_layers', default='5,11,17,23')
ap.add_argument('--perm', type=float, default=0.5); ap.add_argument('--lora_r', type=int, default=32)
ap.add_argument('--mem_r', type=int, default=32); ap.add_argument('--mlr', type=float, default=5e-4)
ap.add_argument('--ck', default='~/work/m2/ck'); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--max_hours', type=float, default=0.0); ap.add_argument('--dev_min', type=float, default=30.0); ap.add_argument('--ndev', type=int, default=150)
ap.add_argument('--slots', type=int, default=0); ap.add_argument('--reader', default='')
ap.add_argument('--elastic', default='', help="comma list of freeze depths sampled per example (e.g. '8,12'); '' = the cfg's own")
ap.add_argument('--dev_freeze', type=int, default=0, help='freeze depth used for the dev metric (default: the first elastic depth)')
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
CFG = Cfg(a.cfg)
EL = [int(x) for x in a.elastic.split(',')] if a.elastic else [CFG.freeze]
CFGS = {k: Cfg(';'.join([x for x in a.cfg.split(';') if x and not x.startswith('freeze=')] + [f'freeze={k}'])) for k in EL}
DEVK = a.dev_freeze or EL[0]
HL = tuple(int(x) for x in a.hid_layers.split(',')) if a.w_hid > 0 else ()
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])


def is_dev(task): return int(hashlib.sha1(task.encode()).hexdigest()[:8], 16) % 10 == 0


pool, dev = [], []
for l in open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')):
    r = json.loads(l)
    assert r['task'] not in EV
    if r['n_state_tok'] < 32: continue
    (dev if is_dev(r['task']) else pool).append(dict(state=r['state'], questions=r['questions'], task=r['task'], rid=r.get('rid', '')))
dev = [d for d in dev if d.get('rid') is not None]
Rd = random.Random(123); Rd.shuffle(dev); dev = [d for d in dev if len(d['state']) < 30000][:a.ndev]
cf = collections.defaultdict(list)
for l in open(os.path.expanduser('~/work/h7/data/cf_aug.jsonl')):
    r = json.loads(l); assert r['task'] not in EV
    if is_dev(r['task']): continue
    cf[r['pair']].append(r)
cf = list(cf.values())
v5 = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.jsonl'))]
R = random.Random(a.seed); R.shuffle(pool); R.shuffle(cf); R.shuffle(v5)
print('pool', len(pool), 'dev', len(dev), 'cf pairs', len(cf), 'v5', len(v5), 'cfg', CFG.spec, flush=True)

m = M2(); m.detach_inference(); m.setup(); dev_ = m.dev


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(qd): return 2 if qd['type'] == 'noul' else len(qd['criteria'])


class Teacher:
    def __enter__(self): self.s = (m.lora, m.head, getattr(m, 'extra', None), m.mem); m.lora = None; m.head = m.head0; m.extra = None; m.mem = None
    def __exit__(self, *e): m.lora, m.head, m.extra, m.mem = self.s


def build(state, qdict, names, rng, cfg=None):
    cfg = cfg or CFGS[EL[0]]
    st = render_state(state); ps = []
    for n in names:
        qd = qdict[n]; perm = None
        if qd['type'] in ('choice', 'noul') and rng.random() < a.perm:
            perm = list(range(nopt(qd))); rng.shuffle(perm)
        elif qd['type'] == 'score' and rng.random() < a.perm:
            perm = list(reversed(range(nopt(qd))))
        ps.append(m.prep_q(st, qd, perm))
    s = ps[0]['s']; bl = sum(len(p['q']) + 4 for p in ps); k = a.maxtok - bl
    if k < 64: return None
    ts = m.segs_for(st, s, CFG.gran)
    cut = (k // 4, k - k // 4) if len(s) > k else None
    it = m.build(s, ts, ps, cfg, cut=cut)
    end = s[len(s) - it.ne:] if it.ne else []
    s_t = list(it.ids_state) + list(end)                 # the teacher sees the same (possibly truncated) state
    return it, s_t, ps


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


# ---- parameters
p_lora = m.add_lora(r=a.lora_r, alpha=2 * a.lora_r, seed=a.seed)
p_head = m.set_head('std')
p_new = TU.add_extra(m, a, CFG)                  # new mixer modules (full rank), [] if none
p_mem = m.add_mem([k for k in EL if k < 24], r=a.mem_r, alpha=a.mem_r, seed=a.seed + 1) if any(k < 24 for k in EL) else []
params = p_lora + p_head + p_new + p_mem
print('trainable', round(sum(p.numel() for p in params) / 1e6, 2), 'M (new', round(sum(p.numel() for p in p_new) / 1e6, 2), 'M)', flush=True)
groups = [dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr)] + ([dict(params=p_new, lr=a.nlr)] if p_new else []) + \
    ([dict(params=p_mem, lr=a.mlr)] if p_mem else [])
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = 40
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
pi = ci = vi = 0; upd0 = 0; tok = 0; t_used = 0.0
srng = random.Random(a.seed + 4)
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev_, weights_only=False)
    TU.load_state(m, st['sd'])
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi, ci, vi = st['pos']; srng.setstate(st['rng']); tok = st.get('tok', 0); t_used = st.get('t_used', 0.0)
    print('resumed at update', upd0, 'tokens', tok, flush=True)

# ---- dev set (fixed): teacher decisions + no-state teacher decisions, cached
DEVREF = TU.DevRef(m, dev, CK + 'devref.pt', Teacher)


def dev_eval(upd):
    out = []
    for kd in EL:
        rows = []
        with torch.no_grad():
            for d in dev:
                st_ = render_state(d['state']); names = list(d['questions'])
                prs = [m.prep_q(st_, d['questions'][n]) for n in names]
                s = prs[0]['s']
                if len(s) > 12000: continue
                ts = m.segs_for(st_, s, CFG.gran)
                it = m.build(s, ts, prs, CFGS[kd])
                ls = m.logits_m2(it, m.fwd_m2(it))
                for n, l in zip(names, ls):
                    key = (d['task'], d.get('rid', ''), n)
                    if key not in DEVREF.ref: continue
                    pt, nsd = DEVREF.ref[key]
                    ps_ = torch.softmax(l.float(), -1).cpu()
                    rows.append((n, int(pt.argmax()), nsd, int(ps_.argmax()), float(0.5 * (pt - ps_).abs().sum())))
        ag = sum(r[1] == r[3] for r in rows) / max(1, len(rows))
        sd = [r for r in rows if r[1] != r[2]]
        rec = dict(upd=upd, k=kd, tok=tok, hours=round(t_used / 3600, 3), dev_n=len(rows), dev_agree=round(ag, 4),
                   dev_agree_sd=round(sum(r[1] == r[3] for r in sd) / max(1, len(sd)), 4), dev_n_sd=len(sd), dev_tv=round(sum(r[4] for r in rows) / max(1, len(rows)), 4))
        print('DEV', json.dumps(rec), flush=True)
        with open(CK + 'dev.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
        out.append(rec)
    return out


DEVREF.build()
log = open(CK + 'train.log.jsonl', 'a'); json.dump(vars(a), open(CK + 'args.json', 'w'))
lsum = collections.defaultdict(float); lcnt = collections.Counter()
t0 = time.time() - t_used; upd = upd0; last_dev = time.time()
if upd0 == 0 and not os.path.exists(CK + 'dev.jsonl'): dev_eval(0); last_dev = time.time(); t0 = time.time()
while upd < a.updates:
    nmic = 0
    while nmic < a.accum:
        for kind, state, qdict, names, gold in sample(srng):
            kk = srng.choice(EL)
            b = build(state, qdict, names, srng, CFGS[kk])
            if b is None: continue
            it, s_t, ps = b
            qs = [p['q'] for p in ps]
            with torch.no_grad(), Teacher():
                lt = m.statefirst_logits(s_t, qs, ps, keep=HL); kt = dict(m.kept)
            hh = m.fwd_m2(it, ckpt=True, keep=HL, keep_rows=m.qrows(it)); ks = dict(m.kept)
            ls = m.logits_m2(it, hh)
            loss = 0.0; nq = len(names)
            if HL:
                hd = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in HL) / len(HL)
                loss = loss + a.w_hid * hd; lsum[f'hid{kk}'] += float(hd); lcnt[f'hid{kk}'] += 1
            for j in range(nq):
                sl = ps[j]['rq'].slot_labels
                tp = torch.softmax(lt[j].float(), -1); lp = F.log_softmax(ls[j].float(), -1)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
                if kind == 'real':
                    loss = loss + kl / nq
                else:
                    gi = sl.index(gold[names[j]])
                    if ps[j]['rq'].kind == 'score':
                        lv = int(gold[names[j]]); nb = [str(x) for x in (lv - 1, lv + 1) if 0 <= x < len(sl)]
                        ce = -(0.9 * lp[gi] + sum(0.1 / len(nb) * lp[sl.index(x)] for x in nb)) if nb else -lp[gi]
                    else:
                        ce = -lp[gi]
                    wce, wkl = (a.w_cf, a.w_cfkl) if kind == 'cf' else (a.w_v5, 1.0)
                    loss = loss + (wce * ce + wkl * kl) / nq
                    lsum['ce_' + kind] += float(ce); lcnt['ce_' + kind] += 1
                    lsum['acc_' + kind] += float(int(lp.argmax()) == gi); lcnt['acc_' + kind] += 1
                lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1
                lsum[f'agree{kk}_' + kind] += float(int(lp.argmax()) == int(tp.argmax())); lcnt[f'agree{kk}_' + kind] += 1
            (loss / a.accum).backward()
            tok += len(it.ids)
            nmic += 1
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1; t_used = time.time() - t0
    if upd % 10 == 0 or upd == upd0 + 1:
        rec = dict(upd=upd, t=round(t_used), tok=tok, tok_s=round(tok / max(t_used, 1)), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 4) for k in sorted(lsum)})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    stop = a.max_hours > 0 and t_used > a.max_hours * 3600
    do_dev = (time.time() - last_dev) > a.dev_min * 60 or upd == a.updates or stop
    if upd % 20 == 0 or do_dev:
        sd = TU.state_dict(m, dict(cfg=CFG.spec, args=vars(a)))
        if do_dev:
            torch.save(sd, CK + f's{upd}.pt'); print('saved', upd, flush=True)
        torch.save(dict(sd=sd, opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pos=(pi, ci, vi), rng=srng.getstate(), tok=tok, t_used=t_used), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
    if do_dev:
        td = time.time(); dev_eval(upd); last_dev = time.time(); t0 += time.time() - td      # dev time is not training time
    if stop: print('max_hours reached', flush=True); break
print('done', time.time() - t0, flush=True)
