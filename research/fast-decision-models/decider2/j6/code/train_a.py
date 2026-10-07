"""J6 (a): per-question adapters, question in the weights, state read question-blind (variant B layout).
Student: [state rows: base hobson, shared across questions] + per question a slot set [K option slots, answer] whose rows run with
shared LoRA + that question's LoRA (Win/Wo/Wd, all 24 layers) and per-question learned slot inputs. Pointer head shared, trainable.
Teacher: hobson-v19 (merged) with the question in context, same process, same cached state rows. Loss: KL(teacher || student) on the calibrated
distributions + w_hid * relative MSE between student slot rows and hobson's own option-end / answer rows at layers 5, 11, 17, 23.
Data: evalkit/train_pool.jsonl only (train-split tasks; eval tasks excluded), every question of the request.
Resumable (ck/last.pt).  python train_a.py --updates N --minutes M"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j6')]
import torch, torch.nn.functional as F
from j6lib import J6, Seg, QAdapters, KEEP, FastState, merge_caches
from h3lib import StdHead
import qtab

ap = argparse.ArgumentParser()
ap.add_argument('--updates', type=int, default=3000); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--slr', type=float, default=1e-3); ap.add_argument('--hlr', type=float, default=3e-4)
ap.add_argument('--maxtok', type=int, default=6144); ap.add_argument('--w_hid', type=float, default=1.0)
ap.add_argument('--r_s', type=int, default=16); ap.add_argument('--r_q', type=int, default=8)
ap.add_argument('--ck', default='~/work/j6/ck_a'); ap.add_argument('--every', type=int, default=250)
ap.add_argument('--holdout', default='')          # comma list of question names never trained (for the hypernet's held-out test)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)

pool = qtab.load_pool()
specs = qtab.deployed_specs(pool)
HOLD = set(x for x in a.holdout.split(',') if x)
qnames = [q for q in specs if q not in HOLD]
R0 = random.Random(7); R0.shuffle(pool)
print('pool', len(pool), 'questions', len(qnames), flush=True)

m = J6(); fs = FastState(m); m.detach_inference(); dev = m.dev; eng = m.p.eng
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
for q in qnames: print(q, PQ[q]['kind'], 'K', PQ[q]['K'], 'qtok', len(PQ[q]['q']), flush=True)
sinit = {q: qtab.slot_init(m, eng, PQ[q]) for q in qnames}
ad = QAdapters(m, qnames, sinit, r_s=a.r_s, r_q=a.r_q)
head = StdHead(m.head0).to(dev)
for p_ in head.parameters(): p_.requires_grad_(True)
groups = [dict(params=ad.shared_params(), lr=a.lr), dict(params=list(head.parameters()), lr=a.hlr)]
for j in range(len(qnames)):
    groups.append(dict(params=list(ad.qA[j].parameters()) + list(ad.qB[j].parameters()), lr=a.lr))
    groups.append(dict(params=[ad.slots[j]], lr=a.slr))
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = 30
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.updates)))))
upd0 = 0; pi = 0
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    ad.load_state_dict(st['ad']); head.load_state_dict(st['head']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    upd0 = st['upd']; pi = st['pi']
    print('resumed at', upd0, flush=True)
json.dump(dict(vars(a), qnames=qnames), open(CK + 'args.json', 'w'))
log = open(CK + 'train.log.jsonl', 'a')


def request(r):
    names = [q for q in r['questions'] if q in ad.qi]
    if not names: return None
    s = qtab.state_ids(eng, r['state'])
    qmax = max(len(PQ[q]['q']) for q in names)
    k = a.maxtok - qmax
    if len(s) > k: s = s[:k // 4] + s[-(k - k // 4):]
    return s, names


def step(batch):
    """batch: list of (s, names) requests -> one teacher branch call + one student branch call over all of them"""
    caches = [fs(s) for s, _ in batch]
    cache = merge_caches(caches) if len(caches) > 1 else caches[0]
    Ls = [len(s) for s, _ in batch]
    allq = [(r, q) for r, (_, names) in enumerate(batch) for q in names]
    with torch.no_grad():
        qt = [PQ[q] for _, q in allq]
        xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
        sg = Seg(Ls, [len(p['q']) for p in qt], dev, req=[r for r, _ in allq])
        kr = []
        for j, p in enumerate(qt): kr += [sg.r0[j] + o for o in p['opt']] + [sg.r0[j] + len(p['q']) - 1]
        kr = torch.tensor(kr, device=dev)
        ht, kt = m.branch(xt, sg, cache, None, keep=KEEP, krows=kr)
        hk = ht[kr]
        tl = []; r = 0
        for p in qt:
            K = p['K']; tl.append(m.head_logits(m.head0, hk[r + K], hk[r:r + K], p['kind'])); r += K + 1
    ss = Seg(Ls, [PQ[q]['K'] + 1 for _, q in allq], dev, req=[r for r, _ in allq])
    ad.seg = ss; ad.cur = [ad.qi[q] for _, q in allq]
    xs = ad.slot_inputs([q for _, q in allq]).to(torch.bfloat16)
    hs, ks = m.branch(xs, ss, cache, ad, keep=KEEP, krows=torch.arange(ss.R, device=dev))
    loss = 0.0; stats = collections.defaultdict(float)
    nper = collections.Counter(r for r, _ in allq)
    for j, (rr, q) in enumerate(allq):
        p = PQ[q]; K = p['K']; r0 = ss.r0[j]
        ls = m.head_logits(head, hs[r0 + K], hs[r0:r0 + K], p['kind'])
        tp = torch.softmax(tl[j].float(), -1); lp = F.log_softmax(ls.float(), -1)
        kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
        wq = 1.0 / (nper[rr] * len(batch))
        loss = loss + kl * wq
        stats['kl'] += float(kl) * wq; stats['agree'] += float(int(lp.argmax()) == int(tp.argmax())) * wq
    if a.w_hid > 0:
        hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in KEEP) / len(KEEP)
        loss = loss + a.w_hid * hl; stats['hid'] = float(hl)
    return loss, stats, sum(Ls)


lsum = collections.defaultdict(float); lc = collections.Counter(); t0 = time.time(); upd = upd0; ntok = 0
deadline = time.time() + float(os.environ.get('MINUTES', '1e9')) * 60
while upd < a.updates and time.time() < deadline:
    batch = []
    while len(batch) < a.accum:
        r = pool[pi % len(pool)]; pi += 1
        b = request(r)
        if b is not None: batch.append(b)
    loss, stats, nt = step(batch)
    loss.backward()
    for k_, v_ in stats.items(): lsum[k_] += v_; lc[k_] += 1
    ntok += nt
    torch.nn.utils.clip_grad_norm_([p_ for g_ in groups for p_ in g_['params'] if p_.grad is not None], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    upd += 1
    if upd % 10 == 0 or upd == upd0 + 1:
        rec = dict(upd=upd, pi=pi, t=round(time.time() - t0), tok_s=round(ntok / (time.time() - t0)), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                   lr=sched.get_last_lr()[0], **{k_: round(lsum[k_] / max(1, lc[k_]), 4) for k_ in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lc.clear()
    if upd % a.every == 0 or upd == a.updates:
        torch.save(dict(ad=ad.state(), head={k_: v_.cpu() for k_, v_ in head.state_dict().items()}, qnames=qnames, args=vars(a)), CK + f's{upd}.pt')
    if upd % 100 == 0 or upd == a.updates:
        torch.save(dict(ad=ad.state_dict(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pi=pi), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
torch.save(dict(ad=ad.state(), head={k_: v_.cpu() for k_, v_ in head.state_dict().items()}, qnames=qnames, args=vars(a)), CK + f's{upd}.pt')
torch.save(dict(ad=ad.state_dict(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), upd=upd, pi=pi), CK + 'last.tmp')
os.replace(CK + 'last.tmp', CK + 'last.pt')
print('done', upd, time.time() - t0, flush=True)
