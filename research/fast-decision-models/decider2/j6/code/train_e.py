"""J6 Brooker test: does reading the state WITH the question in mind help?  Both arms start from the trained (a) checkpoint (question in the weights
of the slot rows, state read question-blind) and train on the same samples for the same number of updates:
  late  : continue (a) as is (state rows = base hobson, question-blind).
  early : add a per-question LoRA (r8, Win/Wo/Wd, all layers, zero-init) on the STATE rows, so every state token is read by a network that knows the
          question from layer 0 (variant A); slot rows keep (a)'s adapters. At init early == late exactly (up to runtime numerics).
Loss for both = (a)'s loss (KL to hobson-in-context + dense row term). One question per sample (A needs one state pass per question).
python train_e.py CKPT_A --updates N   (MINUTES env deadline)"""
import os, sys, json, time, random, argparse, collections, math, copy
sys.path[:0] = [os.path.expanduser('~/work/j6')]
import torch, torch.nn as nn, torch.nn.functional as F
from j6lib import J6, Seg, QAdapters, KEEP, FastState, MODS, lora_pair, QStateAd
from h3lib import StdHead
import qtab

ap = argparse.ArgumentParser(); ap.add_argument('ckpt')
ap.add_argument('--updates', type=int, default=600); ap.add_argument('--accum', type=int, default=4); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--slr', type=float, default=1e-4); ap.add_argument('--maxtok', type=int, default=3072); ap.add_argument('--w_hid', type=float, default=1.0)
ap.add_argument('--ck', default='~/work/j6/ck_e'); ap.add_argument('--r_e', type=int, default=8); ap.add_argument('--p_cf', type=float, default=0.5)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)


pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
m = J6(); fs = FastState(m); m.detach_inference(); dev = m.dev; eng = m.p.eng
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
ck = torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)
qn = ck['qnames']; ar = ck['args']
arms = {}
for name in ('late', 'early'):
    ad = QAdapters(m, qn, {q: torch.zeros(PQ[q]['K'] + 1, 2048) for q in qn}, r_s=ar['r_s'], r_q=ar['r_q'])
    ad.load_state_dict(ck['ad'])
    hd = StdHead(m.head0).to(dev); hd.load_state_dict(ck['head'])
    for p_ in hd.parameters(): p_.requires_grad_(True)
    params = [dict(params=list(ad.parameters()), lr=a.slr), dict(params=list(hd.parameters()), lr=a.slr)]
    if name == 'early':
        ad.state_ad = QStateAd(m, len(qn), r=a.r_e)
        params.append(dict(params=list(ad.state_ad.parameters()), lr=a.lr))
    opt = torch.optim.AdamW(params, betas=(0.9, 0.95), weight_decay=0.0)
    arms[name] = dict(ad=ad, head=hd, opt=opt, params=params)
R0 = random.Random(5); R0.shuffle(pool)
# counterfactually edited TRAIN-split states (h7/gen_cf.py: human / amount / wrap-up edits; probe items excluded). Labels are NOT used:
# both arms distil hobson's own answers on these states (KL + dense), exactly like the real states.
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
cfa = [json.loads(l) for l in open(os.path.expanduser('~/work/j6/cf_aug.jsonl'))]
cfa = [r for r in cfa if not r['kind'].startswith('probe') and r['task'] not in EV and all(q in qn for q in r['questions'])]
R0.shuffle(cfa); print('cf_aug items', len(cfa), flush=True)
ci = 0
log = open(CK + 'train.log.jsonl', 'a')


def teacher(s, q):
    cache = fs(s); p = PQ[q]
    with torch.no_grad():
        xt = F.embedding(torch.tensor(p['q'], device=dev), m.embed)
        sg = Seg(len(s), [len(p['q'])], dev)
        kr = torch.tensor([o for o in p['opt']] + [len(p['q']) - 1], device=dev)
        ht, kt = m.branch(xt, sg, cache, None, keep=KEEP, krows=kr)
        hk = ht[kr]; K = p['K']
        tl = m.head_logits(m.head0, hk[K], hk[:K], p['kind'])
    return cache, tl, kt


def student(arm, s, q, cache):
    ad = arms[arm]['ad']; hd = arms[arm]['head']; p = PQ[q]; K = p['K']; qi = ad.qi[q]
    xs = ad.slot_inputs([q]).to(torch.bfloat16)
    if arm == 'late':
        ss = Seg(len(s), [K + 1], dev); ad.seg = ss; ad.cur = [qi]
        hs, ks = m.branch(xs, ss, cache, ad, keep=KEEP, krows=torch.arange(K + 1, device=dev))
    else:
        ad.cur = [qi]; ad.state_ad.cur = qi; ad.Lslot = len(s)
        hs, ks = full_keep(s, xs, ad)
    return m.head_logits(hd, hs[K], hs[:K], p['kind']), ks


def full_keep(s, xslot, ad):
    """m.full with the layer outputs of the slot rows kept at KEEP (checkpointed per layer)"""
    from torch.utils.checkpoint import checkpoint
    Ls = len(s); L = xslot.shape[0]; T = Ls + L
    x = torch.cat([F.embedding(torch.as_tensor(s, device=dev), m.embed), xslot], 0)
    cos, sin = m.rope_tab(torch.arange(T, device=dev, dtype=torch.float32))
    kept = {}
    for i in range(24):
        x = checkpoint(m._layer_full, i, x, cos, sin, ad, use_reentrant=False)
        if i in KEEP: kept[i] = x[Ls:]
    return H_norm(x[Ls:]), kept


def H_norm(x):
    from h3lib import rms_zc
    return rms_zc(x, m.norm_w, m.eps)


def loss_of(ls, tl, ks, kt):
    tp = torch.softmax(tl.float(), -1); lp = F.log_softmax(ls.float(), -1)
    kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
    hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in KEEP) / len(KEEP)
    return kl + a.w_hid * hl, float(kl), float(hl), int(lp.argmax()) == int(tp.argmax())


# parity check at init: early (zero state adapter) == late
with torch.no_grad():
    dd = []
    for r in pool[-6:]:
        q = [x for x in r['questions'] if x in qn][0]
        s = qtab.state_ids(eng, r['state'])[:a.maxtok]
        cache, tl, kt = teacher(s, q)
        l1, _ = student('late', s, q, cache); l2, _ = student('early', s, q, cache)
        dd.append(float((torch.softmax(l1.float(), -1) - torch.softmax(l2.float(), -1)).abs().max()))
print('init parity early vs late max|dp|', max(dd), dd, flush=True)

pi = 0; upd = 0; t0 = time.time(); acc = collections.defaultdict(float); cnt = collections.Counter()
deadline = time.time() + float(os.environ.get('MINUTES', '1e9')) * 60
while upd < a.updates and time.time() < deadline:
    for _ in range(a.accum):
        if random.random() < a.p_cf:
            r = cfa[ci % len(cfa)]; ci += 1
        else:
            r = pool[pi % len(pool)]; pi += 1
        names = [x for x in r['questions'] if x in qn]
        q = names[(pi + ci) % len(names)]
        s = qtab.state_ids(eng, r['state'])
        if len(s) > a.maxtok: s = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]
        cache, tl, kt = teacher(s, q)
        for arm in ('late', 'early'):
            ls, ks = student(arm, s, q, cache)
            L, kl, hl, ag = loss_of(ls, tl, ks, kt)
            (L / a.accum).backward()
            acc[arm + '_kl'] += kl; acc[arm + '_hid'] += hl; acc[arm + '_agree'] += ag; cnt[arm] += 1
    for arm in ('late', 'early'):
        A = arms[arm]
        torch.nn.utils.clip_grad_norm_([p_ for g_ in A['params'] for p_ in g_['params'] if p_.grad is not None], 1.0)
        A['opt'].step(); A['opt'].zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0:
        rec = dict(upd=upd, pi=pi, ci=ci, t=round(time.time() - t0), **{k_: round(v_ / cnt[k_.split('_')[0]], 4) for k_, v_ in acc.items()})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); acc.clear(); cnt.clear()
for arm in ('late', 'early'):
    A = arms[arm]
    sd = dict(ad={k_: v_.detach().cpu() for k_, v_ in A['ad'].state_dict().items() if not k_.startswith('state_ad')},
              head={k_: v_.cpu() for k_, v_ in A['head'].state_dict().items()}, qnames=qn, args=ar)
    if arm == 'early': sd['state_ad'] = {k_: v_.detach().cpu() for k_, v_ in A['ad'].state_ad.state_dict().items()}; sd['r_e'] = a.r_e
    torch.save(sd, CK + f'{arm}_s{upd}.pt')
print('done', upd, time.time() - t0, flush=True)
