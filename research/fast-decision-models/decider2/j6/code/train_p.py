"""J6 capacity probe for the many-way procedure questions (procedure 23-way, needed_procedure 48-way): from (a) s4723, two arms on the SAME samples
(train-pool requests holding these questions; KL + dense, as (a)); shared adapter, head and other questions frozen:
  r8   : continue the two questions' r8 adapters + slots (control)
  r40  : same + an extra rank-32 per-question delta (zero-init B) on Win/Wo/Wd
python train_p.py CKPT_A --minutes M ; writes ck_p/{r8,r40}.pt and eval preds for REAL-agree + LONG items holding these questions."""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn as nn, torch.nn.functional as F
from j6lib import J6, Seg, QAdapters, KEEP, FastState, MODS, lora_pair, merge_caches
from h3lib import StdHead
import qtab, evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--minutes', type=float, default=30); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--maxtok', type=int, default=6144)
a = ap.parse_args()
CK = os.path.expanduser('~/work/j6/ck_p/'); os.makedirs(CK, exist_ok=True)
PROC = ['procedure', 'needed_procedure']
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
pool = [r for r in pool if any(q in r['questions'] for q in PROC)]
m = J6(); fs = FastState(m); m.detach_inference(); dev = m.dev; eng = m.p.eng
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
ck = torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)
qn = ck['qnames']; ar = ck['args']


class Extra(nn.Module):
    def __init__(self, r=32, seed=3):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.A = nn.ModuleDict(); self.B = nn.ModuleDict()
        for q in PROC:
            pa = nn.ParameterDict(); pb = nn.ParameterDict()
            for i in range(24):
                for nm in MODS:
                    out, inp = m.L[i][nm].shape
                    A, B = lora_pair(out, inp, r, g, dev); pa[f'{i}_{nm}'] = A; pb[f'{i}_{nm}'] = B
            self.A[q] = pa; self.B[q] = pb


arms = {}
for name in ('r8', 'r40'):
    ad = QAdapters(m, qn, {q: torch.zeros(PQ[q]['K'] + 1, 2048) for q in qn}, r_s=ar['r_s'], r_q=ar['r_q'])
    ad.load_state_dict(ck['ad'])
    for p_ in ad.parameters(): p_.requires_grad_(False)
    params = []
    for q in PROC:
        j = ad.qi[q]
        for p_ in ad.q_params(j): p_.requires_grad_(True); params.append(p_)
    hd = StdHead(m.head0).to(dev); hd.load_state_dict(ck['head'])
    for p_ in hd.parameters(): p_.requires_grad_(False)
    ex = None
    if name == 'r40':
        ex = Extra(); params += list(ex.parameters())
    arms[name] = dict(ad=ad, head=hd, ex=ex, opt=torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0), params=params)


def call(arm):
    ad = arms[arm]['ad']; ex = arms[arm]['ex']

    def f(i, nm, h, part):
        y = ad(i, nm, h, part)
        if ex is not None:
            seg = ad.seg; k = f'{i}_{nm}'; parts = []
            for j in range(seg.n):
                s0, L = seg.r0[j], seg.lens[j]; q = qn[ad.cur[j]]
                parts.append(((h[s0:s0 + L] @ ex.A[q][k].t().to(h.dtype)) @ ex.B[q][k].t().to(h.dtype)) * 2.0 if q in PROC else h.new_zeros(L, y.shape[1]))
            y = y + torch.cat(parts, 0)
        return y
    return f


def run(arm, s, names, cache, keep=KEEP):
    ad = arms[arm]['ad']
    ss = Seg(len(s), [PQ[q]['K'] + 1 for q in names], dev); ad.seg = ss; ad.cur = [ad.qi[q] for q in names]
    hs, ks = m.branch(ad.slot_inputs(names).to(torch.bfloat16), ss, cache, call(arm), keep=keep, krows=torch.arange(ss.R, device=dev) if keep else None)
    out = []
    for j, q in enumerate(names):
        K = PQ[q]['K']; r0 = ss.r0[j]
        out.append(m.head_logits(arms[arm]['head'], hs[r0 + K], hs[r0:r0 + K], PQ[q]['kind']))
    return out, ks


R0 = random.Random(9); R0.shuffle(pool); pi = 0; upd = 0; t0 = time.time(); acc = collections.defaultdict(float); cnt = collections.Counter()
log = open(CK + 'train.log.jsonl', 'w')
while time.time() - t0 < a.minutes * 60:
    for _ in range(a.accum):
        r = pool[pi % len(pool)]; pi += 1
        names = [q for q in PROC if q in r['questions']]
        s = qtab.state_ids(eng, r['state']); k = a.maxtok - max(len(PQ[q]['q']) for q in names)
        if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
        cache = fs(s)
        with torch.no_grad():
            qt = [PQ[q] for q in names]
            xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
            sg = Seg(len(s), [len(p['q']) for p in qt], dev)
            kr = []
            for j, p in enumerate(qt): kr += [sg.r0[j] + o for o in p['opt']] + [sg.r0[j] + len(p['q']) - 1]
            kr = torch.tensor(kr, device=dev)
            ht, kt = m.branch(xt, sg, cache, None, keep=KEEP, krows=kr); hk = ht[kr]; tl = []; rr = 0
            for p in qt: K = p['K']; tl.append(m.head_logits(m.head0, hk[rr + K], hk[rr:rr + K], p['kind'])); rr += K + 1
        for arm in arms:
            ls, ks = run(arm, s, names, cache)
            loss = 0.0
            for j in range(len(names)):
                tp = torch.softmax(tl[j].float(), -1); lp = F.log_softmax(ls[j].float(), -1)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum(); loss = loss + kl / len(names)
                acc[arm + '_kl'] += float(kl.detach()); acc[arm + '_agree'] += float(int(lp.argmax()) == int(tp.argmax())); cnt[arm] += 1
            hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in KEEP) / len(KEEP)
            ((loss + hl) / a.accum).backward()
    for arm in arms:
        A_ = arms[arm]; torch.nn.utils.clip_grad_norm_(A_['params'], 1.0); A_['opt'].step(); A_['opt'].zero_grad(set_to_none=True)
    upd += 1
    if upd % 20 == 0:
        rec = dict(upd=upd, pi=pi, t=round(time.time() - t0), **{k_: round(v_ / cnt[k_.split('_')[0]], 4) for k_, v_ in acc.items()})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); acc.clear(); cnt.clear()
# eval: REAL-agree + LONG items holding these questions
preds = {arm: {} for arm in arms}
with torch.no_grad():
    for sname in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(sname):
            names = [q for q in PROC if q in it['questions']]
            if not names: continue
            s = qtab.state_ids(eng, it['state']); cache = fs(s)
            for arm in arms:
                ls, _ = run(arm, s, names, cache, keep=())
                preds[arm][it['id']] = {q: {lab: float(p_) for lab, p_ in zip(PQ[q]['rq'].slot_labels, torch.softmax(ls[j].float(), -1).tolist())} for j, q in enumerate(names)}
json.dump(preds, open(os.path.expanduser('~/work/j6/preds_p.json'), 'w'))
print('done', upd, time.time() - t0, flush=True)
