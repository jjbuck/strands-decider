"""J14 trainer: distill hobson's state-first decisions into the state-first compiled-question layout.
Live path (state rows + live question rows) = hobson weights, exactly, unless --live_r > 0 (LoRA on the live QUESTION rows only;
the state cache is never touched). Trainable compile path:
  --arm clora : LoRA (r, alpha) on the compile rows only (J9's compile adapter, generalised to the question bundle)
  --arm full  : full fine-tune of all 96 GEMM weights of the compile path (bf16 + stochastic rounding Adam), decoupled L2-SP pull
                toward hobson (--l2sp). Compile weights are used at compile time only -> no runtime cost.
  --arm none  : compile path = hobson (use with --live_r)
Loss per question: KL(hobson || student) on the option distribution + w_hid * relative MSE of the live rows against hobson's own rows
at the same positions (layers --hid_layers).
Data: evalkit/train_pool.jsonl (train-split tasks; 10% of tasks held out as DEV) + train_v5 rows (KL only). Never eval tasks.
Checkpoints (box only): CK/best.pt (best dev agree_sd), CK/last.pt (+ CK/opt.pt, CK/state.json) for resume.
"""
import os, sys, json, time, random, argparse, hashlib, math, collections
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from j14lib import J14, LoRA, live_positions, NAMES
import j14train_util as TU

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='clora'); ap.add_argument('--r', type=int, default=16); ap.add_argument('--alpha', type=float, default=32)
ap.add_argument('--live_r', type=int, default=0); ap.add_argument('--live_alpha', type=float, default=32); ap.add_argument('--live_lr', type=float, default=5e-5)
ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--updates', type=int, default=400); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--warm', type=int, default=20); ap.add_argument('--l2sp', type=float, default=100.0)
ap.add_argument('--lv', default='oa+sfx'); ap.add_argument('--gdn', default='replay')
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--hid_layers', default='5,11,17,23')
ap.add_argument('--p_v5', type=float, default=0.2); ap.add_argument('--maxtok', type=int, default=6000); ap.add_argument('--maxq', type=int, default=8)
ap.add_argument('--dev_every', type=int, default=50); ap.add_argument('--ndev', type=int, default=120); ap.add_argument('--ck', required=True)
ap.add_argument('--focus', default=''); ap.add_argument('--focus_w', type=float, default=1.0)
ap.add_argument('--conv', default='natural'); ap.add_argument('--factored', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ckpt_long', type=int, default=700)
a = ap.parse_args()
CK = os.path.expanduser(a.ck); os.makedirs(CK, exist_ok=True)
HL = tuple(int(x) for x in a.hid_layers.split(',')) if a.w_hid > 0 else ()
FOCUS = set(a.focus.split(',')) if a.focus else set()
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])


def is_dev(task): return int(hashlib.sha1(task.encode()).hexdigest()[:8], 16) % 10 == 0


pool, dev = [], []
for l in open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')):
    r = json.loads(l)
    assert r['task'] not in EV
    if r['n_state_tok'] > a.maxtok or r['n_state_tok'] < 32: continue
    (dev if is_dev(r['task']) else pool).append(dict(state=r['state'], questions=r['questions'], task=r['task'], rid=r['rid']))
R = random.Random(a.seed); R.shuffle(pool)
Rd = random.Random(123); Rd.shuffle(dev); dev = [d for d in dev if True][:a.ndev]
v5 = []
for li, l in enumerate(open(os.path.expanduser('~/work/training/data/train_v5.jsonl'))):
    if li % 8 == 0: v5.append(json.loads(l))
R.shuffle(v5)
print('pool', len(pool), 'dev', len(dev), 'dev tasks', len({d['task'] for d in dev}), 'v5', len(v5), flush=True)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}


m = J14(); m.setup(); m.detach_inference(); dev_ = m.dev; m.conv_mode = a.conv
torch.manual_seed(a.seed)
params = []; opts = []
if a.arm == 'clora':
    m.clora = LoRA(m.L, a.r, a.alpha, dev_, seed=a.seed)
    opts.append(torch.optim.AdamW(m.clora.parameters(), lr=a.lr, weight_decay=0.0))
elif a.arm in ('full', 'fullall'):
    ps = m.make_fullft()
    anchors = [m.L[i][nm] for i in range(24) for nm in NAMES]
    opts.append(TU.AdamSR(ps, anchors, lr=a.lr, l2sp=a.l2sp, factored=a.factored))
    m.live_ft = a.arm == 'fullall'
if a.live_r > 0:
    m.llora = LoRA(m.L, a.live_r, a.live_alpha, dev_, seed=a.seed + 1)
    opts.append(torch.optim.AdamW(m.llora.parameters(), lr=a.live_lr, weight_decay=0.0))
base_lr = [o.param_groups[0]['lr'] for o in opts]

st = dict(step=0, best=-1.0, hist=[])
if os.path.exists(f'{CK}/state.json') and os.path.exists(f'{CK}/last.pt'):
    st = json.load(open(f'{CK}/state.json'))
    meta = TU.load_ckpt(m, f'{CK}/last.pt')
    if a.arm == 'clora': m.clora.requires_grad_(True); opts[0] = torch.optim.AdamW(m.clora.parameters(), lr=a.lr, weight_decay=0.0)
    if a.arm in ('full', 'fullall'):
        ps = []
        for i in range(24):
            for nm in NAMES:
                p_ = torch.nn.Parameter(m.Lc[i][nm]); m.Lc[i][nm] = p_; ps.append(p_)
        opts[0] = TU.AdamSR(ps, [m.L[i][nm] for i in range(24) for nm in NAMES], lr=a.lr, l2sp=a.l2sp, factored=a.factored)
        m.live_ft = a.arm == 'fullall'
    if m.llora is not None:
        m.llora.requires_grad_(True); opts[-1] = torch.optim.AdamW(m.llora.parameters(), lr=a.live_lr, weight_decay=0.0)
    if os.path.exists(f'{CK}/opt.pt'):
        osd = torch.load(f'{CK}/opt.pt', map_location=dev_)
        for o, s_ in zip(opts, osd): o.load_state_dict(s_)
    print('resumed at', st['step'], flush=True)


def items_of(req, rng, maxq):
    names = list(req['questions']); rng.shuffle(names)
    return [(n, req['questions'][n]) for n in names[:maxq]]


class Hobson:
    """live-row LoRA off: pure hobson (teacher / references)"""
    def __enter__(self): self.s = (m.llora, m.live_ft); m.llora = None; m.live_ft = False
    def __exit__(self, *e): m.llora, m.live_ft = self.s


def teacher(cache, T, pr, lp):
    with torch.no_grad(), Hobson():
        Lq = len(pr['q']); allp = list(range(Lq))
        hl = m.suffix(cache, T, pr['q'], allp, keep=HL)
        lg = m.readout(hl, allp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots, head=m.head0)
        lpt = torch.as_tensor(lp, device=dev_)
        kept = {i: m.kept[i][lpt].float() for i in HL}
        return torch.log_softmax(lg.float(), -1), kept


def student(cache, T, pr, lp):
    Lq = len(pr['q'])
    hl = m.suffix(cache, T, pr['q'], lp, gdn_mode=a.gdn, keep=HL, ckpt=Lq > a.ckpt_long)
    lg = m.readout(hl, lp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots, head=m.head0)
    return torch.log_softmax(lg.float(), -1), {i: m.kept[i] for i in HL}


def loss_of(lt, kt, ls, ks, w=1.0):
    kl = (lt.exp() * (lt - ls)).sum()
    hid = torch.zeros((), device=dev_)
    for i in HL:
        xs = ks[i].float(); xt = kt[i]
        hid = hid + ((xs - xt).pow(2).sum(-1) / xt.pow(2).sum(-1).clamp_min(1e-6)).mean()
    if HL: hid = hid / len(HL)
    return w * (kl + a.w_hid * hid), float(kl), float(hid)


# ---------------------------------------------------------------- dev
DEVREF = {}
UC = None
_LF = [False]


def m_live_ft(): return a.arm == 'fullall'


def dev_eval():
    global UC
    with torch.no_grad():
        if UC is None: UC = m.prefix(list(m.U))
        rows = []
        for req in dev:
            prs = [(n, m.prep(req['state'], qd)) for n, qd in req['questions'].items()]
            cache, T = m.prefix(prs[0][1]['s'])
            hcache = None
            for n, pr in prs:
                key = (req['rid'], n); Lq = len(pr['q'])
                if key not in DEVREF:
                  with Hobson():
                    if hcache is None: hcache = m.prefix(prs[0][1]['s'])[0] if m_live_ft() else cache
                    hl = m.suffix(hcache, T, pr['q'], list(range(Lq)))
                    pt = torch.softmax(m.readout(hl, list(range(Lq)), pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots, head=m.head0).float(), -1)
                    prn = m.prep('', req['questions'][n])
                    hn = m.suffix(UC[0], UC[1], prn['q'], list(range(len(prn['q']))))
                    pn = torch.softmax(m.readout(hn, list(range(len(prn['q']))), prn['opt'], len(prn['q']), prn['rq'].kind, prn['rq'].n_slots, head=m.head0).float(), -1)
                    DEVREF[key] = (pt.cpu(), int(pn.argmax()))
                pt, nsd = DEVREF[key]
                lp = live_positions(m.tok, pr, a.lv)
                hs = m.suffix(cache, T, pr['q'], lp, gdn_mode=a.gdn)
                ps_ = torch.softmax(m.readout(hs, lp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots, head=m.head0).float(), -1).cpu()
                rows.append((n, int(pt.argmax()), nsd, int(ps_.argmax()), float(0.5 * (pt - ps_).abs().sum())))
            del cache
    ag = sum(r[1] == r[3] for r in rows) / len(rows)
    sd = [r for r in rows if r[1] != r[2]]
    agsd = sum(r[1] == r[3] for r in sd) / max(1, len(sd))
    tv = sum(r[4] for r in rows) / len(rows)
    proc = [r for r in sd if 'procedure' in r[0]]; oth = [r for r in sd if 'procedure' not in r[0]]
    res = dict(n=len(rows), n_sd=len(sd), agree=ag, agree_sd=agsd, tv=tv,
               sd_proc=sum(r[1] == r[3] for r in proc) / max(1, len(proc)), n_proc=len(proc),
               sd_other=sum(r[1] == r[3] for r in oth) / max(1, len(oth)))
    return res


def save(tag):
    meta = dict(arm=a.arm, lv=a.lv, gdn=a.gdn, conv=a.conv, step=st['step'], live_ft=bool(m.live_ft))
    TU.save_ckpt(m, f'{CK}/{tag}.pt', meta)


def set_lr(step):
    f = min(1.0, (step + 1) / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.updates))))
    for o, b in zip(opts, base_lr):
        for g in o.param_groups: g['lr'] = b * f


# ---------------------------------------------------------------- loop
if st['step'] == 0 and not st['hist']:
    t0 = time.time(); r0 = dev_eval(); r0['step'] = 0; r0['t'] = time.time() - t0
    print('DEV', json.dumps(r0), flush=True); st['hist'].append(r0); st['best'] = r0['agree_sd']
pi = st['step'] * a.accum * 3 % max(1, len(pool)); vi = st['step'] * a.accum % len(v5)
rng = random.Random(a.seed + st['step'])
t0 = time.time(); agg = collections.defaultdict(float); nagg = 0
while st['step'] < a.updates:
    set_lr(st['step'])
    for _ in range(a.accum):
        if rng.random() < a.p_v5:
            r = v5[vi % len(v5)]; vi += 1
            reqs = [(None, v5_q(r))]; state = r['state']
        else:
            req = pool[pi % len(pool)]; pi += 1
            reqs = items_of(req, rng, a.maxq); state = req['state']
        prs = []
        for n, qd in reqs:
            try: prs.append((n, m.prep(state, qd)))
            except Exception as e: print('prep fail', e, flush=True)
        if not prs: continue
        if m.live_ft:
            with torch.no_grad(), Hobson(): tcache, T = m.prefix(prs[0][1]['s'])
            cache, T = m.prefix(prs[0][1]['s'])
        else:
            cache, T = m.prefix(prs[0][1]['s']); tcache = cache
        tot = 0.0
        for n, pr in prs:
            lp = live_positions(m.tok, pr, a.lv)
            if len(lp) == len(pr['q']) and m.llora is None and not m.live_ft: continue   # fully live + live path frozen: student == teacher
            lt, kt = teacher(tcache, T, pr, lp)
            ls, ks = student(cache, T, pr, lp)
            w = (a.focus_w if n in FOCUS else 1.0) / len(prs)
            loss, kl, hid = loss_of(lt, kt, ls, ks, w)
            if m.live_ft: tot = tot + loss / a.accum
            elif loss.requires_grad: (loss / a.accum).backward()
            agg['kl'] += kl; agg['hid'] += hid; nagg += 1
        if m.live_ft and torch.is_tensor(tot) and tot.requires_grad: tot.backward()
        del cache, tcache
    for o in opts: o.step(); o.zero_grad(set_to_none=True)
    st['step'] += 1
    if st['step'] % 10 == 0:
        extra = f' drift {TU.rel_drift(m):.4f}' if a.arm in ('full', 'fullall') and st['step'] % 50 == 0 else ''
        print(f"step {st['step']} kl {agg['kl'] / max(1, nagg):.4f} hid {agg['hid'] / max(1, nagg):.4f} lr {opts[0].param_groups[0]['lr']:.2e} "
              f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB{extra}", flush=True)
        agg = collections.defaultdict(float); nagg = 0
    if st['step'] % a.dev_every == 0 or st['step'] == a.updates:
        te = time.time(); r0 = dev_eval(); r0['step'] = st['step']; r0['t'] = time.time() - te
        print('DEV', json.dumps(r0), flush=True); st['hist'].append(r0)
        save('last')
        if r0['agree_sd'] > st['best']:
            st['best'] = r0['agree_sd']; save('best'); print('new best', st['step'], flush=True)
        save(f"s{st['step']}") if a.arm not in ('full', 'fullall') else None
        torch.save([o.state_dict() for o in opts], f'{CK}/opt.pt.tmp'); os.replace(f'{CK}/opt.pt.tmp', f'{CK}/opt.pt')
        json.dump(st, open(f'{CK}/state.json', 'w'))
print('done', flush=True)
