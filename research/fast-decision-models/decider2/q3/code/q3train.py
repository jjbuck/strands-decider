"""Q3 trainer (B4): decision-level scales / rounding and full-weight QAD for plain rotated per-token W4A4 hobson-v19.
  --mode scale : GPTQ codes fixed; per-channel weight log-scales trained end to end (exact gradient; STE only through activation rounding)
  --mode round : AdaRound relaxation of every weight between floor/ceil of GPTQ's compensated weight (init = GPTQ's choice in the hard limit),
                 annealed regulariser, hard rounding at the end
  --mode qad   : full-weight QAD: bf16 latent weights (init = GPTQ's compensated weights, so step 0 = the GPTQ codes exactly), STE through the
                 exact integer W4A4 arithmetic, stochastic-rounding Adam with factored second moment, decoupled L2 pull toward the init
Loss per question: KL(bf16 hobson || student) on the decision + w_hid * relative MSE of the residual at layers 5/11/17/23 on the answer and
option rows. Batches: one train-split request = [state][question 1]...[question k] (hobson's shared-prefix layout), k <= maxq; train_v5 rows
with probability p_v5. Dev = 100 fixed requests from the 10% of TRAIN tasks held out by hash (never trained on). Never eval tasks.
Checkpoints (box only) in --ck: last.pt/opt.pt/state.json (resume), best.pt (lowest dev TV), t{N}M.pt snapshots at every dev eval."""
import os, sys, json, time, random, argparse, math, collections
sys.path[:0] = [os.path.expanduser('~/work/q3')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import q3lib as QL

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['scale', 'round', 'qad', 'none'])
ap.add_argument('--init', default='~/work/q3/gptq4.pt'); ap.add_argument('--seed_ck', default='')
ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--tokens', type=float, default=50e6); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--warm', type=int, default=30); ap.add_argument('--l2sp', type=float, default=100.0); ap.add_argument('--factored', type=int, default=1)
ap.add_argument('--w_hid', type=float, default=0.5); ap.add_argument('--maxtok', type=int, default=5000); ap.add_argument('--maxq', type=int, default=8)
ap.add_argument('--max_rows', type=int, default=7000); ap.add_argument('--p_v5', type=float, default=0.2)
ap.add_argument('--dev_every', type=float, default=10e6); ap.add_argument('--ndev', type=int, default=100); ap.add_argument('--ck', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--reg', type=float, default=1.0); ap.add_argument('--reg_warm', type=float, default=0.2)
ap.add_argument('--round_h0', type=float, default=0.0, help='0: AdaRound init from the fractional part of the compensated weight; else h0 toward the GPTQ choice (e.g. .95)')
ap.add_argument('--beta0', type=float, default=20.0); ap.add_argument('--beta1', type=float, default=2.0)
ap.add_argument('--qrow_ab', type=int, default=0, help='activation bits for question rows (0 = same as state rows)')
ap.add_argument('--pmap', default='', help='json {"i.k": "w4a4"|"w8a8"|"bf16"} (default all w4a4)')
ap.add_argument('--mem_gb', type=float, default=21.0)
ap.add_argument('--soft_dev', type=int, default=0)
ap.add_argument('--lat_init', default='center', choices=['center', 'comp'], help='qad latent start: grid points of the seed codes (center) or GPTQ-compensated weights (comp)')
ap.add_argument('--curve_fast', type=int, default=1, help='curve scoring with the compiled glue (rounding ties may differ from eager)')
ap.add_argument('--ckpt', type=int, default=1, help='activation checkpointing per layer (0 on big GPUs)')
ap.add_argument('--q8', default='', help='gptq8.pt: row-role w4q8 (question rows W8A8 with these fixed codes)')
ap.add_argument('--curve', default='', help='evalkit suites scored at every dev eval (reporting only), e.g. CF-probe,REAL-agree')
ap.add_argument('--run', default='', help='run name for curve preds files')
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
W = os.path.expanduser('~/work/q3'); CK = os.path.expanduser(a.ck); os.makedirs(CK, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
LOG = open(f'{CK}/log.txt', 'a')


def log(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()


m = QL.Q3(); dev = m.dev
QL.FAST = True
if a.qrow_ab: m.qrow_ab = a.qrow_ab
pool, devset, v5 = QL.load_data(a.maxtok, a.ndev, a.seed)
log('args', json.dumps(vars(a)))
log('pool', len(pool), 'dev', len(devset), 'dev tasks', len({d['task'] for d in devset}), 'v5', len(v5))

# ---------------------------------------------------------------- student init
init = torch.load(os.path.expanduser(a.init), map_location='cpu')
pmap = None
if a.pmap:
    pmap = {(int(k.split('.')[0]), k.split('.')[1]): v for k, v in json.load(open(os.path.expanduser(a.pmap))).items()}


def load_student_ck(path):
    sd = torch.load(path, map_location='cpu'); return sd


src = {key: dict(q=e['q'], s=e['s'], Wc=e['Wc'] if a.lat_init == 'comp' else (e['q'].float() * e['s'][:, None]).to(torch.bfloat16)) for key, e in init.items()}
if a.seed_ck:
    sd = load_student_ck(os.path.expanduser(a.seed_ck))
    nchg = ntot = 0
    for key, e in sd['gemms'].items():
        if e.get('mode') == 'lat':
            src[key]['Wc'] = e['A']; src[key]['q'] = torch.round(e['A'].float() / e['s'][:, None]).clamp(-7, 7).to(torch.int8); src[key]['s'] = e['s']
        elif 'q' in e:
            # hardened codes (+ scales): where the code equals GPTQ's, keep GPTQ's compensated latent at its position within the cell
            # (Wc * s_new / s_gptq); elsewhere put the latent at the new code's grid point
            s0 = src[key]['s']; q0 = src[key]['q']; s1 = e['s'].float(); q1 = e['q']
            lat = (src[key]['Wc'].float() * (s1 / s0)[:, None])
            same = q1 == q0
            lat = torch.where(same & (a.lat_init == 'comp'), lat, q1.float() * s1[:, None]).to(torch.bfloat16)
            bad = torch.round(lat.float() / s1[:, None]).clamp(-7, 7) != q1.float()
            if bad.any(): lat[bad] = (q1[bad].float() * s1[:, None].expand_as(lat)[bad]).to(torch.bfloat16)
            src[key]['q'] = q1; src[key]['s'] = s1; src[key]['Wc'] = lat
            nchg += int((~same).sum()); ntot += same.numel()
    log('seeded from', a.seed_ck, 'codes changed vs GPTQ', nchg / max(1, ntot))
if a.mode == 'round':
    for key, e in src.items():
        u = e['Wc'].float() / e['s'][:, None]
        cf = torch.floor(u).clamp(-7, 6); rest = (u - cf).clamp(0, 1)
        if a.round_h0 > 0:      # soft model ~ hard GPTQ model from the start: h = h0 on the side GPTQ rounded to, 1 - h0 otherwise
            up = (e['q'].float() - cf) >= 0.5
            rest = torch.where(up, torch.full_like(rest, a.round_h0), torch.full_like(rest, 1 - a.round_h0))
        p_ = ((rest - QL.GAMMA) / (QL.ZETA - QL.GAMMA)).clamp(1e-4, 1 - 1e-4)
        e['cf'] = cf.to(torch.int8); e['V'] = torch.log(p_ / (1 - p_)).to(torch.bfloat16)
        hard = (cf + (QL.hsoft(e['V']) >= 0.5).float()).clamp(-7, 7)
        e['init_mismatch'] = float((hard != e['q'].float()).float().mean())
    log('round init: hard(V0) != GPTQ code fraction', sum(e['init_mismatch'] for e in src.values()) / len(src))
MODE = {'scale': 'code', 'round': 'soft', 'qad': 'lat', 'none': 'lat'}[a.mode]
q8src = torch.load(os.path.expanduser(a.q8), map_location='cpu') if a.q8 else None
params = m.set_student(MODE, src, pmap=pmap, trainable=a.mode != 'none', q8src=q8src); del q8src
q_init = {key: e['q'] for key, e in src.items()}
del init
if a.mode == 'qad':
    anchors = [p_.detach().clone() for p_ in params]
    opt = QL.AdamSR(params, anchors, lr=a.lr, l2sp=a.l2sp, factored=bool(a.factored))
elif a.mode == 'round':
    opt = QL.AdamSR(params, None, lr=a.lr, l2sp=0.0, factored=bool(a.factored))
elif a.mode == 'scale':
    opt = torch.optim.Adam(params, lr=a.lr)
else:
    opt = None
del src
import gc; gc.collect(); torch.cuda.empty_cache()
log('student ready', MODE, f'params {sum(p_.numel() for p_ in params) / 1e6:.1f}M', f'mem {torch.cuda.memory_allocated() / 1e9:.1f}GB')


def save(path):
    sd = {'mode': a.mode, 'gemms': {}}
    for key, st in m.S.items():
        if st['mode'] == 'lat': sd['gemms'][key] = dict(mode='lat', A=st['A'].detach().cpu(), s=st['s'].cpu())
        elif st['mode'] == 'code':
            q, s = m.codes(*key); sd['gemms'][key] = dict(mode='code', q=q.cpu(), s=s.cpu())
        elif st['mode'] == 'soft': sd['gemms'][key] = dict(mode='soft', V=st['A'].detach().cpu(), cf=st['cf'].cpu(), s=st['s'].cpu())
    tmp = path + '.tmp'; torch.save(sd, tmp); os.replace(tmp, path)


def save_hard(path):
    sd = {'mode': 'code', 'gemms': {}}
    for key in m.S:
        if m.S[key]['mode'] == 'bf16': continue
        q, s = m.codes(*key); sd['gemms'][key] = dict(mode='code', q=q.cpu(), s=s.cpu())
    tmp = path + '.tmp'; torch.save(sd, tmp); os.replace(tmp, path)


def code_change():
    """fraction of weights whose exported code differs from the GPTQ init code"""
    n = d_ = 0
    for key in m.S:
        if m.S[key]['mode'] == 'bf16': continue
        q, _ = m.codes(*key); d_ += int((q.cpu() != q_init[key]).sum()); n += q.numel()
    return d_ / n


def drift():
    if a.mode != 'qad': return 0.0
    num = den = 0.0
    for p_, a_ in zip(params, anchors):
        num += float((p_.detach().float() - a_.float()).pow(2).sum()); den += float(a_.float().pow(2).sum())
    return math.sqrt(num / den)


# ---------------------------------------------------------------- dev set + teacher references
def dev_rq(r):
    names = sorted(r['questions'])[:a.maxq]
    return m.prep(r['state'], [r['questions'][n] for n in names]), names


DEVP = f'{W}/devref_{a.ndev}_{a.maxtok}_{a.maxq}.pt'
if os.path.exists(DEVP):
    DEVREF = torch.load(DEVP)
else:
    DEVREF = {}
    with torch.no_grad():
        for r in devset:
            rq, names = dev_rq(r)
            (lt, _), _ = m.run(rq, student=False, keep=())
            rq0 = m.prep('', [r['questions'][n] for n in names])
            (l0, _), _ = m.run(rq0, student=False, keep=())
            DEVREF[r['rid']] = dict(lt=[x.cpu() for x in lt], ns=[int(x.argmax()) for x in l0])
    torch.save(DEVREF, DEVP)
log('dev refs', len(DEVREF), 'questions', sum(len(v['lt']) for v in DEVREF.values()))


def dev_eval():
    rows = []; QL.FAST = False
    with torch.no_grad():
        for r in devset:
            rq, names = dev_rq(r)
            (ls, _), _ = m.run(rq, student=True, keep=())
            ref = DEVREF[r['rid']]
            for lt, l_, ns in zip(ref['lt'], ls, ref['ns']):
                pt = lt.exp(); ps = l_.float().cpu().exp()
                rows.append(dict(t=int(pt.argmax()), s=int(ps.argmax()), ns=ns, tv=float(0.5 * (pt - ps).abs().sum()), kl=float((pt * (lt - l_.float().cpu())).sum()),
                                 mt=float(pt.max())))
    QL.FAST = True
    n = len(rows); sd = [x for x in rows if x['t'] != x['ns']]
    return dict(n=n, flips=sum(x['t'] != x['s'] for x in rows) / n, tv=sum(x['tv'] for x in rows) / n, kl=sum(x['kl'] for x in rows) / n,
                n_sd=len(sd), agree_sd=sum(x['t'] == x['s'] for x in sd) / max(1, len(sd)), nflip=sum(x['t'] != x['s'] for x in rows))


# ---------------------------------------------------------------- state / resume
st = dict(step=0, tokens=0, pi=0, vi=0, hist=[], best=1e9, next_dev=0.0)
if os.path.exists(f'{CK}/state.json') and os.path.exists(f'{CK}/last.pt'):
    st = json.load(open(f'{CK}/state.json'))
    sd = torch.load(f'{CK}/last.pt', map_location='cpu')
    with torch.no_grad():
        for key, e in sd['gemms'].items():
            s_ = m.S[key]
            if e['mode'] == 'lat': s_['A'].copy_(e['A'].to(dev))
            elif e['mode'] == 'soft': s_['A'].copy_(e['V'].to(dev))
            elif e['mode'] == 'code' and s_.get('rho') is not None: s_['rho'].copy_(torch.log(e['s'].to(dev) / s_['s']))
    if opt is not None and os.path.exists(f'{CK}/opt.pt'):
        opt.load_state_dict(torch.load(f'{CK}/opt.pt', map_location=dev))
    m.refresh_codes()
    log('resumed at step', st['step'], 'tokens', st['tokens'])
base_lr = a.lr


def set_lr():
    prog = min(1.0, st['tokens'] / a.tokens)
    f = min(1.0, (st['step'] + 1) / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    if opt is not None:
        for g in opt.param_groups: g['lr'] = base_lr * f
    return f


def checkpoint_and_dev(final=False):
    te = time.time()
    if a.mode == 'round':
        # evaluate the HARD rounding (what would be deployed); keep training in the soft relaxation
        saved = {key: (s_['mode'], s_['A'], s_['cf']) for key, s_ in m.S.items() if s_['mode'] == 'soft'}
        m.harden(); r0 = dev_eval(); r0['code_change'] = code_change()
        for key, (md, A, cf) in saved.items(): m.S[key].update(mode=md, A=A, cf=cf, rho=None)
        if a.soft_dev:
            rs = dev_eval(); r0['soft_tv'] = rs['tv']; r0['soft_flips'] = rs['flips']
        r0['reg'] = float(sum(QL.adaround_reg(s_['A'], 2.0) for s_ in m.S.values() if s_['mode'] == 'soft') / 96)
    else:
        r0 = dev_eval(); r0['code_change'] = code_change()
    r0.update(step=st['step'], tokens=st['tokens'], t=round(time.time() - te, 1), drift=drift())
    log('DEV', json.dumps(r0)); st['hist'].append(r0)
    tag = f"t{st['tokens'] / 1e6:.0f}M"
    if a.mode != 'none':
        save(f'{CK}/last.pt')
        if opt is not None:
            torch.save(opt.state_dict(), f'{CK}/opt.pt.tmp'); os.replace(f'{CK}/opt.pt.tmp', f'{CK}/opt.pt')
        save_hard(f'{CK}/{tag}.pt')
        if r0['tv'] < st['best']:
            st['best'] = r0['tv']; save_hard(f'{CK}/best.pt'); log('new best', tag)
    if a.curve:
        if a.mode == 'round':
            saved = {key: (s_['mode'], s_['A'], s_['cf']) for key, s_ in m.S.items() if s_['mode'] == 'soft'}; m.harden()
        os.makedirs(f'{W}/preds', exist_ok=True)
        QL.eval_suites(m, a.curve.split(','), f"{W}/preds/{a.run or os.path.basename(CK)}_{tag}.jsonl", student=True, log=log, fast=bool(a.curve_fast))
        if a.mode == 'round':
            for key, (md, A, cf) in saved.items(): m.S[key].update(mode=md, A=A, cf=cf, rho=None)
    json.dump(st, open(f'{CK}/state.json.tmp', 'w')); os.replace(f'{CK}/state.json.tmp', f'{CK}/state.json')


def sample(rng):
    if rng.random() < a.p_v5:
        r = v5[st['vi'] % len(v5)]; st['vi'] += 1
        try: return m.prep(r['state'], [QL.v5_q(r)])
        except Exception as e: return None
    r = pool[st['pi'] % len(pool)]; st['pi'] += 1
    names = list(r['questions']); rng.shuffle(names); names = names[:a.maxq]
    try: rq = m.prep(r['state'], [r['questions'][n] for n in names])
    except Exception as e: return None
    while len(rq['qs']) > 1 and len(rq['s']) + sum(len(x['q']) for x in rq['qs']) > a.max_rows: rq['qs'].pop()
    if len(rq['s']) + sum(len(x['q']) for x in rq['qs']) > a.max_rows: return None
    return rq


# ---------------------------------------------------------------- loop
if not st['hist']:
    checkpoint_and_dev(); st['next_dev'] = a.dev_every
if a.mode == 'none': sys.exit(0)
rng = random.Random(a.seed * 1000 + st['step'])
t0 = time.time(); tok0 = st['tokens']; agg = collections.defaultdict(float); nagg = 0
while st['tokens'] < a.tokens:
    f = set_lr()
    for _ in range(a.accum):
        rq = None
        while rq is None: rq = sample(rng)
        with torch.no_grad():
            (lt, kt), _ = m.run(rq, student=False)
        (ls, ks), lay = m.run(rq, student=True, ckpt=bool(a.ckpt))
        loss, kl, hid = QL.loss_fn(lt, kt, ls, ks, a.w_hid, None)
        (loss / a.accum).backward()
        st['tokens'] += lay['T']; agg['kl'] += kl; agg['hid'] += hid; agg['nq'] += len(rq['qs']); nagg += 1
        del lt, kt, ls, ks, loss, lay
    if a.mode == 'round':
        prog = min(1.0, st['tokens'] / a.tokens); w0 = a.reg_warm
        lam = 0.0 if prog < w0 else a.reg * (prog - w0) / (1 - w0)
        beta = a.beta0 + (a.beta1 - a.beta0) * max(0.0, (prog - w0) / (1 - w0))
        if lam > 0:
            with torch.no_grad():
                for p_ in params:
                    if p_.grad is None: continue
                    rg = QL.adaround_reg_grad(p_, beta)
                    gr = p_.grad.float(); sc = gr.pow(2).mean().sqrt() / rg.pow(2).mean().sqrt().clamp_min(1e-20)
                    p_.grad.copy_((gr + lam * sc * rg).to(p_.grad.dtype)); del rg, gr
    opt.step(); opt.zero_grad(set_to_none=True); m.refresh_codes()
    st['step'] += 1
    if st['step'] % 10 == 0:
        el = time.time() - t0
        log(f"step {st['step']} tok {st['tokens'] / 1e6:.2f}M kl {agg['kl'] / max(1, nagg):.4f} hid {agg['hid'] / max(1, nagg):.4f} q/mb {agg['nq'] / max(1, nagg):.1f} "
            f"lr {base_lr * f:.2e} {(st['tokens'] - tok0) / el:.0f} tok/s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            + (f' drift {drift():.4f}' if st['step'] % 100 == 0 else ''))
        agg = collections.defaultdict(float); nagg = 0
    if st['tokens'] >= st['next_dev'] or st['tokens'] >= a.tokens:
        checkpoint_and_dev(); st['next_dev'] = st['tokens'] - (st['tokens'] % a.dev_every) + a.dev_every
        json.dump(st, open(f'{CK}/state.json', 'w'))
if a.mode == 'round':
    m.harden(); save_hard(f'{CK}/final_hard.pt')
log('done', st['tokens'])
