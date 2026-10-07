"""J6 (b): hypernetwork question spec -> low-rank weight delta (+ slot inputs), question-blind state rows (variant B layout), same student as (a).
Data per update: P pool requests (deployed questions minus HOLD; KL + dense) + V train_v5 rows (each its own question; CE(gold) + KL + dense,
option order permuted with p .5, rendered identically for teacher and student). Teacher = hobson in context, same process and cached state.
python train_h.py --updates N  (MINUTES env = wall-clock deadline). Resumable."""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j6')]
import torch, torch.nn.functional as F
from j6lib import J6, Seg, HyperAdapters, KEEP, FastState, merge_caches
from h3lib import StdHead
import qtab

HOLD_DEFAULT = 'states_amount,WrapsUp,cc_insists,details_match,failure_cause,cc_refuses,UserAskedForHuman'
ap = argparse.ArgumentParser()
ap.add_argument('--updates', type=int, default=3000); ap.add_argument('--np', type=int, default=2); ap.add_argument('--nv', type=int, default=3)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--hlr', type=float, default=3e-4); ap.add_argument('--glr', type=float, default=5e-4)
ap.add_argument('--maxtok', type=int, default=6144); ap.add_argument('--v5max', type=int, default=3072); ap.add_argument('--w_hid', type=float, default=1.0)
ap.add_argument('--w_ce', type=float, default=1.0); ap.add_argument('--r_h', type=int, default=24)
ap.add_argument('--ck', default='~/work/j6/ck_h'); ap.add_argument('--every', type=int, default=500); ap.add_argument('--holdout', default=HOLD_DEFAULT)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
HOLD = set(x for x in a.holdout.split(',') if x)

pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
pool = [r for r in pool if any(q not in HOLD for q in r['questions'])]
v5 = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.jsonl'))]
R0 = random.Random(7); R0.shuffle(pool); R0.shuffle(v5)
print('pool', len(pool), 'v5', len(v5), 'hold', sorted(HOLD), flush=True)

m = J6(); fs = FastState(m); m.detach_inference(); dev = m.dev; eng = m.p.eng
S0 = qtab.state_ids(eng, '')
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs if q not in HOLD}
ha = HyperAdapters(m, r_h=a.r_h)
FEAT = {q: ha.features(m, PQ[q], S0) for q in PQ}
SINIT = {q: qtab.slot_init(m, eng, PQ[q]) for q in PQ}
head = StdHead(m.head0).to(dev)
for p_ in head.parameters(): p_.requires_grad_(True)
g_lora = list(ha.sA.parameters()) + list(ha.sB.parameters()) + list(ha.U.parameters()) + list(ha.V.parameters())
g_gen = list(ha.enc.parameters()) + list(ha.gen.parameters()) + list(ha.ln_o.parameters()) + list(ha.Wo.parameters()) + list(ha.ln_a.parameters()) + list(ha.Wa.parameters())
groups = [dict(params=g_lora, lr=a.lr), dict(params=g_gen, lr=a.glr), dict(params=list(head.parameters()), lr=a.hlr)]
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = 30
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
upd0 = 0; pi = 0; vi = 0
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    ha.load_state_dict(st['ha']); head.load_state_dict(st['head']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi = st['pi']; vi = st['vi']
    print('resumed at', upd0, flush=True)
json.dump(vars(a), open(CK + 'args.json', 'w'))
log = open(CK + 'train.log.jsonl', 'a')
srng = random.Random(11 + upd0)


def v5_spec(r):
    ins = r['instructions']
    if r.get('instruction_variants') and srng.random() < 0.3: ins = srng.choice(r['instruction_variants'])
    if r['kind'] in ('choice', 'noul'):
        return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(sp):
    return 2 if sp['type'] == 'noul' else len(sp['criteria'])


def get_batch():
    """-> list of requests: (s, [(pq, feat, sinit, gold_label_or_None)])"""
    global pi, vi
    out = []
    while len(out) < a.np:
        r = pool[pi % len(pool)]; pi += 1
        names = [q for q in r['questions'] if q not in HOLD]
        if len(names) > 6: names = srng.sample(names, 6)
        s = qtab.state_ids(eng, r['state'])
        k = a.maxtok - max(len(PQ[q]['q']) for q in names)
        if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
        out.append((s, [(PQ[q], FEAT[q], SINIT[q], None) for q in names]))
    nv = 0
    while nv < a.nv:
        r = v5[vi % len(v5)]; vi += 1
        try:
            sp, gold = v5_spec(r)
            perm = None
            if sp['type'] in ('choice', 'noul') and srng.random() < 0.5:
                perm = list(range(nopt(sp))); srng.shuffle(perm)
            pq = qtab.prep_q(eng, sp, perm)
        except Exception as e:
            continue
        s = qtab.state_ids(eng, r['state'])
        k = a.v5max - len(pq['q'])
        if k < 32: continue
        if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
        out.append((s, [(pq, ha.features(m, pq, S0), qtab.slot_init(m, eng, pq), gold)]))
        nv += 1
    return out


def step(batch):
    caches = [fs(s) for s, _ in batch]
    cache = merge_caches(caches)
    Ls = [len(s) for s, _ in batch]
    allq = [(r, x) for r, (_, qs) in enumerate(batch) for x in qs]
    with torch.no_grad():
        qt = [x[0] for _, x in allq]
        xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
        sg = Seg(Ls, [len(p['q']) for p in qt], dev, req=[r for r, _ in allq])
        kr = []
        for j, p in enumerate(qt): kr += [sg.r0[j] + o for o in p['opt']] + [sg.r0[j] + len(p['q']) - 1]
        kr = torch.tensor(kr, device=dev)
        ht, kt = m.branch(xt, sg, cache, None, keep=KEEP, krows=kr)
        hk = ht[kr]; tl = []; r = 0
        for p in qt:
            K = p['K']; tl.append(m.head_logits(m.head0, hk[r + K], hk[r:r + K], p['kind'])); r += K + 1
    codes = [ha.generate(x[1], x[2]) for _, x in allq]
    ss = Seg(Ls, [x[0]['K'] + 1 for _, x in allq], dev, req=[r for r, _ in allq])
    ha.seg = ss; ha.cur = codes
    xs = torch.cat([c['slots'] for c in codes], 0).to(torch.bfloat16)
    hs, ks = m.branch(xs, ss, cache, ha, keep=KEEP, krows=torch.arange(ss.R, device=dev))
    loss = 0.0; stats = collections.defaultdict(float); cnt = collections.Counter()
    nper = collections.Counter(r for r, _ in allq)
    for j, (rr, (p, _, _, gold)) in enumerate(allq):
        K = p['K']; r0 = ss.r0[j]
        ls = m.head_logits(head, hs[r0 + K], hs[r0:r0 + K], p['kind'])
        tp = torch.softmax(tl[j].float(), -1); lp = F.log_softmax(ls.float(), -1)
        kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
        wq = 1.0 / (nper[rr] * len(batch))
        src = 'pool' if gold is None else 'v5'
        if gold is None:
            loss = loss + kl * wq
        else:
            gi = p['rq'].slot_labels.index(gold); ce = -lp[gi]
            loss = loss + (a.w_ce * ce + kl) * wq
            stats['ce_v5'] += float(ce); cnt['ce_v5'] += 1
            stats['acc_v5'] += float(int(lp.argmax()) == gi); cnt['acc_v5'] += 1
            stats['tacc_v5'] += float(int(tp.argmax()) == gi); cnt['tacc_v5'] += 1
        stats['kl_' + src] += float(kl); cnt['kl_' + src] += 1
        stats['agree_' + src] += float(int(lp.argmax()) == int(tp.argmax())); cnt['agree_' + src] += 1
    if a.w_hid > 0:
        hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in KEEP) / len(KEEP)
        loss = loss + a.w_hid * hl; stats['hid'] += float(hl); cnt['hid'] += 1
    return loss, {k_: v_ / cnt[k_] for k_, v_ in stats.items()}, sum(Ls)


lsum = collections.defaultdict(float); lc = collections.Counter(); t0 = time.time(); upd = upd0; ntok = 0
deadline = time.time() + float(os.environ.get('MINUTES', '1e9')) * 60
while upd < a.updates and time.time() < deadline:
    loss, stats, nt = step(get_batch())
    loss.backward()
    for k_, v_ in stats.items(): lsum[k_] += v_; lc[k_] += 1
    ntok += nt
    torch.nn.utils.clip_grad_norm_([p_ for g_ in groups for p_ in g_['params'] if p_.grad is not None], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        rec = dict(upd=upd, pi=pi, vi=vi, t=round(time.time() - t0), tok_s=round(ntok / (time.time() - t0)), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                   lr=sched.get_last_lr()[0], **{k_: round(lsum[k_] / max(1, lc[k_]), 4) for k_ in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lc.clear()
    if upd % a.every == 0:
        torch.save(dict(ha=ha.state_dict(), head=head.state_dict(), args=vars(a)), CK + f's{upd}.pt')
    if upd % 100 == 0:
        torch.save(dict(ha=ha.state_dict(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pi=pi, vi=vi), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
torch.save(dict(ha=ha.state_dict(), head=head.state_dict(), args=vars(a)), CK + f's{upd}.pt')
torch.save(dict(ha=ha.state_dict(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pi=pi, vi=vi), CK + 'last.tmp')
os.replace(CK + 'last.tmp', CK + 'last.pt')
print('done', upd, time.time() - t0, flush=True)
