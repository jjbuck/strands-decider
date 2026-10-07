"""J2 training: (1) optional MNTP adaptation (LLM2Vec: 20% of state tokens -> '_', the token is predicted from position i-1 with the base LM head
= tied embedding), (2) decider fine-tuning, F7 recipe: LoRA r16/a32 on every projection of all 24 layers, hobson's pointer head (trainable),
loss CE(gold) + 1.0 KL(hobson || student) on labelled rows (ordinal smoothing 0.1 on score), 1.0 KL on real rows; AdamW(.9,.95), lr 1e-4,
head lr 1e-3, gate lr --glr, 3% warmup, cosine, clip 1.0, 32 rows / step, length-grouped packs. Every arm sees the same rows in the same order.
Resumable (ck/last.pt).
python j2train.py --mode qag --rev all --mntp 150 --steps 600 --ck ~/work/j2/ck/qag_all"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn.functional as F
from j2lib import J2, Pack, GDN_LAYERS
from h3lib import StdHead

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='qag'); ap.add_argument('--rev', default='all'); ap.add_argument('--attn_nc', type=int, default=1)
ap.add_argument('--mntp', type=int, default=0); ap.add_argument('--mntp_tok', type=int, default=8192); ap.add_argument('--mntp_glr', type=float, default=3e-3)
ap.add_argument('--steps', type=int, default=600); ap.add_argument('--rows', type=int, default=32); ap.add_argument('--maxpack', type=int, default=8192)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--hlr', type=float, default=1e-3); ap.add_argument('--glr', type=float, default=2e-3)
ap.add_argument('--wkl', type=float, default=1.0); ap.add_argument('--maxlen', type=int, default=4096); ap.add_argument('--skip', type=int, default=0)
ap.add_argument('--init', default=''); ap.add_argument('--cf', default=''); ap.add_argument('--cf_per_step', type=int, default=8)
ap.add_argument('--ck', required=True); ap.add_argument('--data', default='~/work/j2/data/ft.jsonl'); ap.add_argument('--every', type=int, default=200)
a = ap.parse_args()
CK = os.path.expanduser(a.ck) + '/'; os.makedirs(CK, exist_ok=True)
json.dump(vars(a), open(CK + 'args.json', 'w'))
rev = {'all': GDN_LAYERS, 'early': tuple(i for i in GDN_LAYERS if i <= 10), 'none': ()}[a.rev]

m = J2(); m.detach_inference(); dev = m.dev
for d in m.L:
    for k, v in d.items():
        if torch.is_tensor(v): v.requires_grad_(False)
m.embed.requires_grad_(False)
p_lora = m.add_lora(r=16, alpha=32, seed=7)
p_head = m.set_head('std')
p_gate = m.setup_bidir(a.mode, rev_layers=rev, attn_nc=bool(a.attn_nc))
log = open(CK + 'train.log.jsonl', 'a')
print('mode', a.mode, 'rev', sorted(m.rev_layers), 'attn_nc', m.attn_nc, flush=True)


def save(path, extra=None):
    sd = m.trainable_state(); sd.update(m.gates_state()); sd['_j2'] = dict(mode=a.mode, rev=sorted(m.rev_layers), attn_nc=m.attn_nc)
    if extra: sd.update(extra)
    torch.save(sd, path)


def load_into(sd):
    with torch.no_grad():
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                m.lora[i][k].A.copy_(sd[f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(sd[f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
    m.load_gates(sd)


# ------------------------------------------------------------------ phase 1: MNTP
if a.mntp > 0 and not os.path.exists(CK + 'mntp.pt'):
    tok = m.p.eng.tok
    MASK = tok.convert_tokens_to_ids('_')
    if MASK is None or MASK == tok.unk_token_id: MASK = tok('_', add_special_tokens=False)['input_ids'][0]
    lm_w = m.embed     # Qwen3.5-2B ties the LM head to the embedding (checked in test_mntp)
    rows = [json.loads(l)['ids'] for l in open(os.path.expanduser('~/work/j2/data/mntp.jsonl'))]
    rng = random.Random(3); rng.shuffle(rows)
    opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_gate, lr=a.mntp_glr)], betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / a.mntp)))))
    ri = 0; t0 = time.time(); acc = collections.defaultdict(float)
    for step in range(a.mntp):
        pack = []; n = 0
        while n < a.mntp_tok:
            r = rows[ri % len(rows)]; ri += 1
            r = r[:a.mntp_tok - n] if n + len(r) > a.mntp_tok else r
            if len(r) < 32: break
            pack.append(r); n += len(r)
        pk = Pack([(r, len(r)) for r in pack], dev, a.mode)
        ids = pk.ids.clone()
        sel = (torch.rand(pk.T, device=dev) < 0.2)
        first = torch.zeros(pk.T, dtype=torch.bool, device=dev); first[pk.cu[:-1]] = True
        sel &= ~first
        ids[sel] = MASK
        h = m.fwd_pk(pk, ckpt=True, ids=ids)
        tgt_pos = sel.nonzero()[:, 0]
        loss = 0.0; nt = tgt_pos.numel()
        for c0 in range(0, nt, 1024):
            tp = tgt_pos[c0:c0 + 1024]
            lg = h[tp - 1].float() @ lm_w.float().t()
            loss = loss + F.cross_entropy(lg, pk.ids[tp], reduction='sum')
        loss = loss / nt
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p_lora + p_gate, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        acc['loss'] += float(loss); acc['n'] += 1
        if (step + 1) % 10 == 0:
            gm = {k: round(float(v.abs().mean()), 4) for k, v in list(m.gam.items())[::4]}
            lm = {k: round(float(v.mean()), 3) for k, v in m.lam.items()}
            rec = dict(phase='mntp', step=step + 1, loss=round(acc['loss'] / acc['n'], 4), t=round(time.time() - t0), gam=gm, lam=lm,
                       mem=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); acc.clear()
    save(CK + 'mntp.pt')
    del opt, sched, h, lg, loss
    import gc; gc.collect(); torch.cuda.empty_cache()
elif a.mntp > 0:
    load_into(torch.load(CK + 'mntp.pt', map_location=dev)); print('loaded mntp.pt', flush=True)

# ------------------------------------------------------------------ phase 2: decider fine-tuning
data = [json.loads(l) for l in open(os.path.expanduser(a.data))]
data = [r for r in data if len(r['s']) + len(r['q']) <= a.maxlen]     # drop (not truncate) longer rows: the teacher saw exactly these ids
need = a.steps * a.rows
data = data[a.skip:a.skip + need]
# length-grouped steps: windows of 16 steps sorted by length, cut into steps of a.rows, step order shuffled within the window
steps = []
W = a.rows * 16
for w0 in range(0, len(data), W):
    win = sorted(data[w0:w0 + W], key=lambda r: len(r['s']) + len(r['q']))
    ch = [win[i:i + a.rows] for i in range(0, len(win), a.rows)]
    random.Random(w0).shuffle(ch); steps += ch
if a.cf:
    cfr = [json.loads(l) for l in open(os.path.expanduser(a.cf))]
    for j, st_ in enumerate(steps):
        k0 = (j * a.cf_per_step) % len(cfr)
        st_[:] = st_[:a.rows - a.cf_per_step] + cfr[k0:k0 + a.cf_per_step]      # pairs stay together (cf.jsonl is pair-adjacent, cf_per_step even)
if a.init and not os.path.exists(CK + 'last.pt'):
    load_into(torch.load(os.path.expanduser(a.init), map_location=dev)); print('init from', a.init, flush=True)
print('ft rows', len(data), 'steps', len(steps), 'tokens', sum(len(r['s']) + len(r['q']) for r in data), flush=True)
opt = torch.optim.AdamW([dict(params=p_lora, lr=a.lr), dict(params=p_head, lr=a.hlr), dict(params=p_gate, lr=a.glr)], betas=(0.9, 0.95), weight_decay=0.0)
NS = len(steps); warm = max(1, int(0.03 * NS))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / NS))))
s0 = 0
if os.path.exists(CK + 'last.pt'):
    st = torch.load(CK + 'last.pt', map_location=dev, weights_only=False)
    load_into(st['sd']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); s0 = st['step']
    print('resumed at step', s0, flush=True)
params = p_lora + p_head + p_gate
acc = collections.defaultdict(float); cnt = collections.Counter(); t0 = time.time(); ntok = 0
for si in range(s0, NS):
    rows = steps[si]
    packs = []; cur = []; n = 0
    for r in rows:
        L = len(r['s']) + len(r['q'])
        if cur and n + L > a.maxpack: packs.append(cur); cur = []; n = 0
        cur.append(r); n += L
    if cur: packs.append(cur)
    for pkr in packs:
        lgs = m.decide(pkr, ckpt=True)
        loss = 0.0
        for r, lg in zip(pkr, lgs):
            lp = F.log_softmax(lg.float(), -1)
            if r['tp'] is not None:
                tp = torch.tensor(r['tp'], device=dev)
                kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
            else:
                tp = None; kl = torch.zeros((), device=dev)
            l = a.wkl * kl
            if r['gold'] >= 0:
                g = r['gold']
                if r['kind'] == 'score' and r['n_slots'] > 2:
                    tgt = torch.zeros_like(lp); nb = [j for j in (g - 1, g + 1) if 0 <= j < r['n_slots']]
                    tgt[g] = 0.9
                    for j in nb: tgt[j] = 0.1 / len(nb)
                    ce = -(tgt * lp).sum()
                else:
                    ce = -lp[g]
                l = l + ce
                acc['ce_' + r['src']] += float(ce); cnt['ce_' + r['src']] += 1
                acc['acc_' + r['src']] += float(int(lp.argmax()) == g); cnt['acc_' + r['src']] += 1
            if tp is not None:
                acc['kl_' + r['src']] += float(kl); cnt['kl_' + r['src']] += 1
                acc['agree_' + r['src']] += float(int(lp.argmax()) == int(tp.argmax())); cnt['agree_' + r['src']] += 1
            loss = loss + l
            ntok += len(r['s']) + len(r['q'])
        (loss / len(rows)).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if (si + 1) % 10 == 0:
        rec = dict(phase='ft', step=si + 1, t=round(time.time() - t0), tok_s=round(ntok / (time.time() - t0)), lr=sched.get_last_lr()[0],
                   mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), **{k: round(acc[k] / max(1, cnt[k]), 4) for k in acc})
        if m.gam: rec['gam'] = round(sum(float(v.abs().mean()) for v in m.gam.values()) / len(m.gam), 4)
        if m.lam: rec['lam'] = round(sum(float(v.mean()) for v in m.lam.values()) / len(m.lam), 4)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); acc.clear(); cnt.clear()
    if (si + 1) % 50 == 0 or si + 1 == NS:
        sd = m.trainable_state(); sd.update(m.gates_state())
        if (si + 1) % a.every == 0 or si + 1 == NS: save(CK + f's{si + 1}.pt')
        torch.save(dict(sd=sd, opt=opt.state_dict(), sched=sched.state_dict(), step=si + 1), CK + 'last.tmp')
        os.replace(CK + 'last.tmp', CK + 'last.pt')
save(CK + 'final.pt')
print('done', time.time() - t0, flush=True)
