"""J10 trainer = F7's recipe (as reproduced by G3's train_g3.py) on the BitNet b1.58 2B4T torso with LoRA-in-latent QAT.
pointer head dim 256 (fp32), LoRA r16 / alpha 32 on all 7 projections of all 30 layers (in the LATENT weight: deployed weights stay ternary),
RMSNorm gains trainable, lr 1e-4 (head 1e-3, norms --norm_lr), AdamW(0.9, 0.95), 3% warmup + cosine, clip 1.0, 32 rows/step, length-grouped
megabatches, CE(gold, ordinal smoothing 0.1) + 1.0 KL(hobson || student) on labelled rows + 1.0 KL on real-state KL-only rows, hobson's calibrated
temperatures.  Same rows and teacher logits as F7 (prep_g3 twin tokenisation of the same rendered text).
Resumable: OUT/last.pt.   python train_j10.py --out ck_a8 [--max_steps N] [--aq4 qkv,gu]"""
import os, sys, json, math, time, random, argparse
import torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/sd/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
import bitnet_j10 as BJ

ap = argparse.ArgumentParser()
ap.add_argument('--rows', nargs='+', default=['rows_corpus_s.pt', 'rows_real_s.pt'])
ap.add_argument('--out', required=True); ap.add_argument('--max_steps', type=int, default=0)
ap.add_argument('--tw', type=float, default=1.0); ap.add_argument('--rw', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--head_lr', type=float, default=1e-3); ap.add_argument('--norm_lr', type=float, default=1e-4)
ap.add_argument('--rank', type=int, default=16); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--tokb', type=int, default=4096); ap.add_argument('--ckpt_above', type=int, default=2048); ap.add_argument('--mega', type=int, default=50)
ap.add_argument('--save_at', default='0.25,0.5,0.75,1.0'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--aq4', default='', help='GEMM classes trained with 4-bit activations, e.g. qkv,gu')
ap.add_argument('--blk4', type=int, default=0)
ap.add_argument('--frac_real', type=float, default=1.0, help='fraction of real KL-only rows kept')
ap.add_argument('--init', default='', help='start from a saved checkpoint dir (torso_trainable.pt + slot_head.pt), e.g. QAT continuation')
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)

cfg, W = BJ.load()
torso = BJ.BitNetTorso(cfg, W, r=a.rank, alpha=2 * a.rank).cuda()
del W
for c in [c for c in a.aq4.split(',') if c]:
    torso.aq[c] = (4, a.blk4)
print('activation quant', torso.aq, flush=True)
hcfg = StrandsDeciderConfig(base_model=BJ.REPO, head_type='pointer', pointer_dim=256, head_dropout=0.05, max_length=4096, use_lora=True,
                            lora_r=a.rank, lora_alpha=2 * a.rank, ordinal_smoothing=0.1, lora_dropout=0.0)
head = build_head(hcfg, torso.d).cuda()
if a.init:
    torso.load_trainable(torch.load(os.path.join(a.init, 'torso_trainable.pt'))); head.load_state_dict(torch.load(os.path.join(a.init, 'slot_head.pt')))
    print('initialised from', a.init, flush=True)
torso.train(); head.train()
lora = list(torso.lora.parameters()); norms = list(torso.norms.parameters())
print(f'lora params {sum(p.numel() for p in lora):,} norm params {sum(p.numel() for p in norms):,} head {sum(p.numel() for p in head.parameters()):,}', flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BJ.snap())
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def fwd(ids, am, ns, opt):
    hid = torso(ids, am)
    pooled = pool_last_token(hid, am).float()
    logits = head(pooled, gather_options(hid, opt).float())
    return masked_log_softmax(logits, ns)


def kl_rows(t_logits, s_lp, ns):
    Wd = s_lp.shape[-1]
    valid = torch.arange(Wd, device=s_lp.device)[None, :] < ns[:, None]
    t_lp = F.log_softmax(t_logits.masked_fill(~valid, -1e4), -1)
    s_safe = s_lp.masked_fill(~valid, 0.0)
    p = t_lp.exp() * valid
    return (p * (t_lp - s_safe)).sum(-1)


TT = {}


def batch_tensors(rows):
    ids = pad_sequence([r['ids'].long() for r in rows], batch_first=True, padding_value=pad)
    am = pad_sequence([torch.ones(len(r['ids']), dtype=torch.long) for r in rows], batch_first=True)
    Wd = max(r['n'] for r in rows)
    opt = torch.tensor([r['opt'] + [-1] * (Wd - r['n']) for r in rows])
    ns = torch.tensor([r['n'] for r in rows]); lab = torch.tensor([r['label'] for r in rows])
    tl = torch.zeros((len(rows), Wd)); has_t = torch.zeros(len(rows), dtype=torch.bool)
    dist = torch.zeros(len(rows), Wd); has_d = torch.zeros(len(rows), dtype=torch.bool)
    for j, r in enumerate(rows):
        if r.get('t_logits') is not None:
            tl[j, :r['n']] = torch.tensor(r['t_logits']) / TT.get(r['kind'], 1.0); has_t[j] = True
        if r.get('dist') is not None:
            dist[j, :r['n']] = torch.tensor(r['dist']); has_d[j] = True
    w = torch.tensor([r['w'] for r in rows])
    return ids, am, opt, ns, lab, tl, has_t, dist, has_d, w


@torch.no_grad()
def lc_eval(rows, tokb=16384):
    torso.eval(); head.eval(); res = {}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['ids']))
    k = 0
    while k < len(order):
        L = len(rows[order[min(len(order) - 1, k + 15)]]['ids'])
        nb = max(1, min(64, tokb // max(L, 1)))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids, am, opt, ns, lab, tl, has_t, *_ = batch_tensors(ch)
        lp = fwd(ids.cuda(), am.cuda(), ns.cuda(), opt.cuda()).float().cpu()
        for j, r in enumerate(ch):
            d = res.setdefault(r['src'], dict(n=0, acc=0, agree=0, nll=0.0))
            d['n'] += 1; pred = int(lp[j, :r['n']].argmax())
            d['acc'] += int(pred == r['label']); d['nll'] += -float(lp[j, r['label']])
            if r.get('t_logits') is not None:
                d['agree'] += int(pred == int(torch.tensor(r['t_logits']).argmax()))
    torso.train(); head.train()
    return {s: dict(n=d['n'], acc=round(d['acc'] / d['n'], 4), agree_hobson=round(d['agree'] / d['n'], 4), nll=round(d['nll'] / d['n'], 4)) for s, d in res.items()}


def save(path, extra):
    os.makedirs(path, exist_ok=True)
    torch.save(torso.trainable_state(), os.path.join(path, 'torso_trainable.pt'))
    torch.save(head.state_dict(), os.path.join(path, 'slot_head.pt'))
    with open(os.path.join(path, 'strands_decider_config.json'), 'w') as f: f.write(hcfg.to_json())
    json.dump(dict(extra, aq=torso.aq), open(os.path.join(path, 'j10_meta.json'), 'w'))


rows = []
for p in a.rows:
    D = torch.load(p, weights_only=False); rr = D['rows']
    if 'teacher_temps' in D: TT = dict(D['teacher_temps']['by_kind'])
    if a.frac_real < 1.0 and rr and rr[0].get('src') == 'real':
        random.Random(4).shuffle(rr); rr = rr[:int(len(rr) * a.frac_real)]
    rows += rr; print(p, len(rr), 'rows', flush=True)
lc_rows = torch.load('rows_lceval_s.pt', weights_only=False)['rows'] if os.path.exists('rows_lceval_s.pt') else []
lc_rows = lc_rows[::2]
print('teacher temps', TT, flush=True)
print('train rows', len(rows), 'teacher-covered', sum(r.get('t_logits') is not None for r in rows), 'KL-only', sum(r['w'] == 0 for r in rows),
      'tokens', sum(len(r['ids']) for r in rows), 'lc-eval rows', len(lc_rows), flush=True)

rng = random.Random(a.seed)
idx = list(range(len(rows))); rng.shuffle(idx)
MB = a.mega * a.bs; steps = []
for s in range(0, len(idx), MB):
    mb = sorted(idx[s:s + MB], key=lambda i: len(rows[i]['ids']))
    steps += [mb[j:j + a.bs] for j in range(0, len(mb) - a.bs + 1, a.bs)]
rng.shuffle(steps)
total = a.max_steps or len(steps); steps = steps[:total]
warm = max(1, int(0.03 * total))
save_at = sorted({max(1, int(round(float(f) * total))) for f in a.save_at.split(',')})
print(f'{total} steps, warmup {warm}, save at {save_at}, tokens in schedule {sum(len(rows[i]["ids"]) for st in steps for i in st)}', flush=True)

hd, hnd = [], []
for n, p in head.named_parameters(): (hnd if p.ndim <= 1 else hd).append(p)
opt = torch.optim.AdamW([dict(params=hd, lr=a.head_lr, weight_decay=0.01), dict(params=hnd, lr=a.head_lr, weight_decay=0.0),
                         dict(params=lora, lr=a.lr, weight_decay=0.0), dict(params=norms, lr=a.norm_lr, weight_decay=0.0)], betas=(0.9, 0.95), eps=1e-8)
lam = lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
step0 = 0; ntok = 0
LAST = os.path.join(a.out, 'last.pt')
if os.path.exists(LAST):
    st = torch.load(LAST, weights_only=False)
    torso.load_trainable(st['torso']); head.load_state_dict(st['head'])
    opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); step0 = st['step']; ntok = st['ntok']
    print('resumed at step', step0, flush=True)
logf = open(os.path.join(a.out, 'train_log.jsonl'), 'a')
t0 = time.time(); tok0 = ntok; acc = dict(loss=0., ce=0., kl=0., rkl=0., n=0, nl=0, nt=0, nr=0, agree=0, na=0)
flip = 0.0
for step in range(step0 + 1, total + 1):
    srows = [rows[i] for i in steps[step - 1]]
    micro, cur = [], []
    for r in srows:
        if cur and (len(cur) + 1) * max(len(x['ids']) for x in cur + [r]) > a.tokb:
            micro.append(cur); cur = []
        cur.append(r)
    micro.append(cur)
    for mb in micro:
        ids, am, optx, ns, lab, tl, has_t, dist, has_d, w = [t.cuda(non_blocking=True) for t in batch_tensors(mb)]
        torso.ckpt = ids.numel() > a.ckpt_above
        lp = fwd(ids, am, ns, optx)
        safe = lp.masked_fill(torch.arange(lp.shape[-1], device=lp.device)[None, :] >= ns[:, None], 0.0)
        ce_hard = -safe.gather(1, lab[:, None]).squeeze(1)
        ce_soft = -(dist * safe).sum(-1)
        ce = torch.where(has_d, ce_soft, ce_hard)
        kl = kl_rows(tl, lp, ns)
        lab_rows = (w > 0).float(); kl_only = (w == 0).float()
        per = lab_rows * ce + has_t.float() * kl * (lab_rows * a.tw + kl_only * a.rw)
        loss = per.sum() / len(srows)
        loss.backward()
        with torch.no_grad():
            acc['loss'] += float(per.detach().sum()); acc['n'] += len(mb)
            acc['ce'] += float((lab_rows * ce).sum()); acc['nl'] += int(lab_rows.sum())
            acc['kl'] += float((lab_rows * has_t.float() * kl).sum()); acc['nt'] += int((lab_rows * has_t.float()).sum())
            acc['rkl'] += float((kl_only * has_t.float() * kl).sum()); acc['nr'] += int((kl_only * has_t.float()).sum())
            tam = tl.masked_fill(torch.arange(tl.shape[-1], device=tl.device)[None, :] >= ns[:, None], -1e4).argmax(-1)
            acc['agree'] += int(((lp.argmax(-1) == tam) & has_t).sum()); acc['na'] += int(has_t.sum())
        ntok += int(am.sum())
    torch.nn.utils.clip_grad_norm_(hd + hnd + lora + norms, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    flip = torso.refresh()
    if step % 20 == 0 or step == total or step in (1, 5):
        el = time.time() - t0
        e = dict(step=step, loss=acc['loss'] / max(1, acc['n']), ce=acc['ce'] / max(1, acc['nl']), kl=acc['kl'] / max(1, acc['nt']),
                 rkl=acc['rkl'] / max(1, acc['nr']), agree=acc['agree'] / max(1, acc['na']), flips=round(flip, 5), lr=sched.get_last_lr()[2], el=round(el),
                 tok_s=round((ntok - tok0) / el), eta_min=round(el / (step - step0) * (total - step) / 60, 1), mem=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        print(json.dumps(e), flush=True); logf.write(json.dumps(e) + '\n'); logf.flush()
        acc = dict(loss=0., ce=0., kl=0., rkl=0., n=0, nl=0, nt=0, nr=0, agree=0, na=0)
    if step % 100 == 0 or step in save_at:
        torch.save(dict(torso=torso.trainable_state(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, ntok=ntok), LAST + '.tmp')
        os.replace(LAST + '.tmp', LAST)
    if step in save_at:
        ck = os.path.join(a.out, f'step{step}')
        save(ck, dict(step=step, total=total, rows_seen=step * a.bs, tokens_seen=ntok, flips=flip))
        if lc_rows:
            te = time.time(); r = lc_eval(lc_rows)
            e = dict(step=step, lc=r, eval_s=round(time.time() - te))
            print('LC', json.dumps(e), flush=True); logf.write(json.dumps(e) + '\n'); logf.flush()
print('done', round(time.time() - t0), 's', flush=True)
