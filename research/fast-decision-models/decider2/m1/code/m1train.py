"""M1 trainer. Train split only (evalkit train_pool minus the 10% dev tasks; train_v5; h7 cf_aug train-task pairs). Never evalkit eval items.
  --stage transfer : teacher-forced local fit of every converted mixer to its GDN layer's residual contribution (relMSE over all rows),
                     all layers in one teacher pass (each layer's graph is local). Full-rank on the converted mixers only.
  --stage e2e      : end-to-end distillation from the frozen in-process hobson: KL(teacher || student) on every question's decision
                     + w_hid * relMSE of the residual at layers 5/11/17/23 on the answer and option rows; train_v5 gold rows CE + KL;
                     cf_aug pairs (<= 15%) CE + 0.3 KL. Full-rank on the converted mixers, LoRA r32 on everything else (pointer head frozen).
                     --schedule all | 'a,b,c|d,e,f|...' (groups converted cumulatively, one distillation stage per group, equal time split).
Time-budgeted (--hours of training time), resumable (CK/last.pt + CK/state.json), dev eval + bf16 snapshot every --every_min minutes."""
import os, sys, json, time, random, argparse, math, collections
sys.path[:0] = [os.path.expanduser('~/work/m1')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn.functional as F
import m1lib as ML

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['transfer', 'e2e'])
ap.add_argument('--ck', required=True); ap.add_argument('--init', default=''); ap.add_argument('--hours', type=float, default=1.0)
ap.add_argument('--variant', default='full'); ap.add_argument('--untrained', default='~/work/m1/untrained.json')
ap.add_argument('--schedule', default='all'); ap.add_argument('--every_min', type=float, default=30.0)
ap.add_argument('--lr_big', type=float, default=3e-5); ap.add_argument('--lr_small', type=float, default=3e-4); ap.add_argument('--lr_lora', type=float, default=2e-4)
ap.add_argument('--warm', type=int, default=20); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--w_hid', type=float, default=1.0); ap.add_argument('--p_v5', type=float, default=0.25); ap.add_argument('--p_cf', type=float, default=0.10)
ap.add_argument('--maxq', type=int, default=8); ap.add_argument('--maxtok', type=int, default=5000); ap.add_argument('--max_rows', type=int, default=6000)
ap.add_argument('--lora_r', type=int, default=32); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--mem_gb', type=float, default=21.5)
ap.add_argument('--ndev', type=int, default=100); ap.add_argument('--ckpt_rows', type=int, default=2500)
ap.add_argument('--w_ce', type=float, default=1.0); ap.add_argument('--probe_n', type=int, default=16)
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
WM = os.path.expanduser('~/work/m1'); CK = os.path.expanduser(a.ck); os.makedirs(CK, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
LOG = open(f'{CK}/log.txt', 'a')


def log(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()


VARIANTS = {
    'plain': dict(conv=True, rope=True, beta=False, decay=False), 'beta': dict(conv=True, rope=True, beta=True, decay=False),
    'decay': dict(conv=True, rope=True, beta=False, decay=True), 'full': dict(conv=True, rope=True, beta=True, decay=True), 'full136': dict(conv=True, rope=True, beta=True, decay=True, slots=0),
    'full_norope': dict(conv=True, rope=False, beta=True, decay=True), 'full_noconv': dict(conv=False, rope=True, beta=True, decay=True)}
m = ML.M1(cfg=VARIANTS[a.variant]); dev = m.dev
pool, devset, v5, cf = ML.load_data(a.maxtok, a.ndev, a.seed, cf_path=f'{WM}/data/cf_aug.jsonl')
log('args', json.dumps(vars(a)))
log('pool', len(pool), 'dev', len(devset), 'v5', len(v5), 'cf pairs', len(cf))

# ---------------------------------------------------------------- model init
if a.init:
    meta = m.load_trainable(os.path.expanduser(a.init), lora=False); m.cfg = VARIANTS[a.variant]
    log('init mixers from', a.init, 'layers', sorted(int(k) for k in m.C.keys()))
else:
    taus = 1.0
    up = os.path.expanduser(a.untrained)
    if os.path.exists(up):
        loc = json.load(open(up))['local']['best'][a.variant]['layers']; taus = {int(i): v['tau'] for i, v in loc.items()}
    m.convert(ML.GDN, tau=taus); log('converted all 18 GDN layers, tau', taus)
if a.schedule == 'all':
    GROUPS = [list(ML.GDN)]
else:
    GROUPS = [[int(x) for x in g.split(',')] for g in a.schedule.split('|')]
    assert sorted(sum(GROUPS, [])) == list(ML.GDN), GROUPS
NST = len(GROUPS)

if a.stage == 'transfer':
    m.active = set(ML.GDN)
    groups = [dict(params=m.mixer_params(True), lr=a.lr_big), dict(params=m.mixer_params(False), lr=a.lr_small)]
else:
    p_lora = m.add_lora(r=a.lora_r, alpha=2 * a.lora_r, seed=a.seed)
    groups = [dict(params=m.mixer_params(True), lr=a.lr_big), dict(params=m.mixer_params(False), lr=a.lr_small), dict(params=p_lora, lr=a.lr_lora)]
    groups = [g_ for g_ in groups if g_['lr'] > 0]
    for g_ in (dict(params=m.mixer_params(True), lr=a.lr_big), dict(params=m.mixer_params(False), lr=a.lr_small), dict(params=p_lora, lr=a.lr_lora)):
        if g_['lr'] <= 0:
            for p_ in g_['params']: p_.requires_grad_(False)
for g_ in groups: g_['base_lr'] = g_['lr']
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
allp = [p_ for g_ in groups for p_ in g_['params']]
log('trainable', {i: round(sum(p_.numel() for p_ in g_['params']) / 1e6, 1) for i, g_ in enumerate(groups)}, 'M', f'mem {torch.cuda.memory_allocated() / 1e9:.1f}GB')

# ---------------------------------------------------------------- state / resume
st = dict(step=0, tokens=0, elapsed=0.0, pi=0, vi=0, ci=0, hist=[], next_ck=0.0, stage=0, stage_step=0)
if os.path.exists(f'{CK}/state.json') and os.path.exists(f'{CK}/last.pt'):
    st = json.load(open(f'{CK}/state.json'))
    sd = torch.load(f'{CK}/last.pt', map_location='cpu')
    with torch.no_grad():
        for key, mm in m.C.items():
            for n_, p_ in mm.named_parameters(): p_.copy_(sd[f'C.{key}.{n_}'].to(dev).float())
        if m.lora is not None:
            for key, mm in m.lora.items(): mm.A.copy_(sd[f'lora.{key}.A'].to(dev)); mm.B.copy_(sd[f'lora.{key}.B'].to(dev))
    if os.path.exists(f'{CK}/opt.pt'): opt.load_state_dict(torch.load(f'{CK}/opt.pt', map_location=dev))
    log('resumed at step', st['step'], 'tokens', st['tokens'], 'elapsed', round(st['elapsed']))
    del sd


def set_active():
    if a.stage == 'transfer': m.active = set(ML.GDN); return
    m.active = set(sum(GROUPS[:st['stage'] + 1], []))


set_active()
STAGE_SEC = a.hours * 3600 / (NST if a.stage == 'e2e' else 1)


def set_lr():
    prog = min(1.0, (st['elapsed'] - st['stage'] * STAGE_SEC) / STAGE_SEC) if a.stage == 'e2e' else min(1.0, st['elapsed'] / STAGE_SEC)
    f = min(1.0, (st['stage_step'] + 1) / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    for g_ in opt.param_groups: g_['lr'] = g_['base_lr'] * f
    return f


ref = ML.make_devref(m, devset, f'{WM}/devref.pt')


def checkpoint_and_dev(final=False):
    te = time.time()
    r0 = ML.dev_eval(m, devset, ref)
    r0.update(step=st['step'], tokens=st['tokens'], elapsed_min=round(st['elapsed'] / 60, 1), stage=st['stage'], active=len(m.active), t_eval=round(time.time() - te))
    tag = f"m{int(round(st['elapsed'] / 60)):03d}"; r0['tag'] = tag
    log('DEV', json.dumps(r0)); st['hist'].append(r0)
    torch.save(m.trainable_state(half=True), f'{CK}/{tag}.pt.tmp'); os.replace(f'{CK}/{tag}.pt.tmp', f'{CK}/{tag}.pt')
    torch.save(m.trainable_state(half=False), f'{CK}/last.pt.tmp'); os.replace(f'{CK}/last.pt.tmp', f'{CK}/last.pt')
    torch.save(opt.state_dict(), f'{CK}/opt.pt.tmp'); os.replace(f'{CK}/opt.pt.tmp', f'{CK}/opt.pt')
    json.dump(st, open(f'{CK}/state.json.tmp', 'w')); os.replace(f'{CK}/state.json.tmp', f'{CK}/state.json')
    json.dump(st['hist'], open(f'{CK}/curve.json', 'w'), indent=1)
    log('saved', tag, f'{time.time() - te:.0f}s')


def fits(rq):
    while len(rq['qs']) > 1 and len(rq['s']) + sum(len(x['q']) for x in rq['qs']) > a.max_rows: rq['qs'].pop()
    return len(rq['s']) + sum(len(x['q']) for x in rq['qs']) <= a.max_rows


def sample(rng):
    """-> list of (kind, rq, gold index list or None)"""
    u = rng.random()
    if u < a.p_v5:
        r = v5[st['vi'] % len(v5)]; st['vi'] += 1
        qd, gold = ML.v5_q(r)
        if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
        try: rq = m.prep(r['state'], [qd])
        except Exception: return []
        if not fits(rq): return []
        return [('v5', rq, [rq['qs'][0]['labels'].index(gold)])]
    if u < a.p_v5 + a.p_cf and cf:
        pr = cf[st['ci'] % len(cf)]; st['ci'] += 1; out = []
        for it in pr:
            names = list(it['questions'])
            try: rq = m.prep(it['state'], [it['questions'][n] for n in names])
            except Exception: return []
            if len(rq['qs']) != len(names) or not fits(rq) or len(rq['qs']) != len(names): return []
            out.append(('cf', rq, [x['labels'].index(it['expected'][n]) for x, n in zip(rq['qs'], names)]))
        return out
    r = pool[st['pi'] % len(pool)]; st['pi'] += 1
    names = list(r['questions']); rng.shuffle(names); names = names[:a.maxq]
    try: rq = m.prep(r['state'], [r['questions'][n] for n in names])
    except Exception: return []
    if not fits(rq): return []
    return [('real', rq, None)]


# ---------------------------------------------------------------- loop
def probe(n=16):
    """no-update baseline on training-pool requests (the last n of the shuffled pool): KL / agreement / hidden relMSE vs the teacher"""
    kl_ = ag = hd = nq = 0.0
    with torch.no_grad():
        for r in pool[-n:]:
            names = sorted(r['questions'])[:a.maxq]
            rq = m.prep(r['state'], [r['questions'][x] for x in names])
            if not fits(rq): continue
            (lt, kt), _ = m.run(rq, student=False, keep=ML.KEEP); (ls, ks), _ = m.run(rq, student=True, keep=ML.KEEP)
            for x, y in zip(lt, ls): kl_ += float((x.exp() * (x - y)).sum()); ag += int(x.argmax() == y.argmax()); nq += 1
            hd += float(sum(((ks[i] - kt[i]).pow(2).sum(-1) / kt[i].pow(2).sum(-1).clamp_min(1e-6)).mean() for i in ML.KEEP) / len(ML.KEEP))
    return dict(nq=nq, kl=kl_ / max(1, nq), agree=ag / max(1, nq), hid=hd / n)


if not st['hist'] and a.stage == 'e2e':
    log('PROBE train-pool step 0', json.dumps(probe(a.probe_n)))
    checkpoint_and_dev(); st['next_ck'] = a.every_min * 60
    json.dump(st, open(f'{CK}/state.json', 'w'))
rng = random.Random(a.seed * 1000 + st['step'])
agg = collections.defaultdict(float); cnt = collections.Counter(); tok0 = st['tokens']; tw = time.time()
TOTAL = a.hours * 3600
while st['elapsed'] < TOTAL:
    t_step = time.time()
    if a.stage == 'e2e':
        want = min(NST - 1, int(st['elapsed'] // STAGE_SEC))
        if want != st['stage']:
            st['stage'] = want; st['stage_step'] = 0; set_active(); log('STAGE', want, 'active', sorted(m.active))
    f = set_lr()
    nmic = 0
    while nmic < a.accum:
        batch = sample(rng)
        for kind, rq, gold in batch:
            if a.stage == 'transfer':
                lay = m.layout(rq)
                ls_ = m.local(lay, ML.GDN, backward=True)
                for i, v in ls_.items(): agg[f'L{i}'] += v; cnt[f'L{i}'] += 1
                agg['mean'] += sum(ls_.values()) / len(ls_); cnt['mean'] += 1
                st['tokens'] += lay['T']; nmic += 1
                continue
            with torch.no_grad():
                (lt, kt), _ = m.run(rq, student=False, keep=ML.KEEP)
            Tn = len(rq['s']) + sum(len(x['q']) for x in rq['qs'])
            (ls, ks), lay = m.run(rq, student=True, keep=ML.KEEP, ckpt=Tn > a.ckpt_rows)
            nq = len(ls)
            kl = sum((x.exp() * (x - y)).sum() for x, y in zip(lt, ls)) / nq
            hid = sum(((ks[i] - kt[i]).pow(2).sum(-1) / kt[i].pow(2).sum(-1).clamp_min(1e-6)).mean() for i in ML.KEEP) / len(ML.KEEP)
            loss = a.w_hid * hid
            if kind == 'real':
                loss = loss + kl
            else:
                ce = -sum(y[gi] for y, gi in zip(ls, gold)) / nq
                wce, wkl = (a.w_ce, 0.3) if kind == 'cf' else (a.w_ce, 1.0)
                loss = loss + wce * ce + wkl * kl
                agg['ce_' + kind] += float(ce); cnt['ce_' + kind] += 1
                agg['acc_' + kind] += sum(int(y.argmax()) == gi for y, gi in zip(ls, gold)) / nq; cnt['acc_' + kind] += 1
            (loss / a.accum).backward()
            agg['kl_' + kind] += float(kl.detach()); cnt['kl_' + kind] += 1; agg['hid'] += float(hid.detach()); cnt['hid'] += 1
            agg['agree_' + kind] += sum(int(x.argmax()) == int(y.argmax()) for x, y in zip(lt, ls)) / nq; cnt['agree_' + kind] += 1
            st['tokens'] += lay['T']; nmic += 1
            del lt, kt, ls, ks, loss, lay
    gn = torch.nn.utils.clip_grad_norm_([p_ for p_ in allp if p_.grad is not None], 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    st['step'] += 1; st['stage_step'] += 1
    st['elapsed'] += time.time() - t_step
    agg['gn'] += float(gn); cnt['gn'] += 1
    if st['step'] % 10 == 0:
        el = time.time() - tw
        rec = {k: round(agg[k] / max(1, cnt[k]), 4) for k in sorted(agg)}
        log(f"step {st['step']} stage {st['stage']} tok {st['tokens'] / 1e6:.2f}M min {st['elapsed'] / 60:.1f} lrf {f:.3f} {(st['tokens'] - tok0) / el:.0f} tok/s "
            f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB " + json.dumps(rec))
        agg.clear(); cnt.clear(); tok0 = st['tokens']; tw = time.time()
    if a.stage == 'e2e' and st['elapsed'] >= st['next_ck']:
        st['next_ck'] = st['elapsed'] + a.every_min * 60
        log('PROBE train-pool', json.dumps(probe())); checkpoint_and_dev()
        json.dump(st, open(f'{CK}/state.json', 'w'))
if a.stage == 'transfer':
    torch.save(m.trainable_state(half=False), f'{CK}/transfer.pt')
    r0 = ML.dev_eval(m, devset, ref); r0.update(step=st['step'], tokens=st['tokens'], elapsed_min=round(st['elapsed'] / 60, 1))
    log('DEV after transfer', json.dumps(r0)); json.dump(r0, open(f'{CK}/dev_after_transfer.json', 'w'), indent=1)
else:
    checkpoint_and_dev(final=True)
log('done', st['tokens'])
