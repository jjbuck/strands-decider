"""Resumable LM pretraining at a fixed training-FLOP budget.
usage: python pretrain.py SCALE ARM --pflops BUDGET_PFLOP [--bs 16 --accum 2 --lr 1.2e-3]
Every arm at a scale sees the same stream of micro-batches (same seed); the number of steps is set by BUDGET / train FLOPs per step."""
import os, sys, time, json, math, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('scale'); ap.add_argument('arm')
ap.add_argument('--pflops', type=float, required=True)
ap.add_argument('--bs', type=int, default=16); ap.add_argument('--accum', type=int, default=2)
ap.add_argument('--seq', type=int, default=1024)
ap.add_argument('--lr', type=float, default=None); ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--wd', type=float, default=0.1)
ap.add_argument('--ckpt_min', type=float, default=15.0)
ap.add_argument('--tag', default='')
ap.add_argument('--max_minutes', type=float, default=1e9)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
W = os.path.expanduser('~/work/g4'); D = f'{W}/data'
name = f'{args.scale}_{args.arm}{args.tag}'; RD = f'{W}/runs/{name}'; os.makedirs(RD, exist_ok=True)
cfg = C.Cfg(args.scale, args.arm)
lr = args.lr or {'S60': 1.2e-3, 'S125': 1.0e-3}[args.scale]
mac_tok = C.train_macs_per_token(cfg, args.seq)
flop_tok = 6 * mac_tok
tok_step = args.bs * args.accum * args.seq
steps = int(args.pflops * 1e15 / (flop_tok * tok_step))
log = open(f'{RD}/log.txt', 'a')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


train = np.memmap(f'{D}/train.bin', dtype=np.uint16, mode='r'); val = np.memmap(f'{D}/val.bin', dtype=np.uint16, mode='r')
NTR = len(train) - args.seq - 2


def batch(micro):
    """deterministic random windows: micro-batch index -> (B, seq+1) tokens (same across arms)"""
    g = np.random.default_rng(1234567 + micro)
    off = g.integers(0, NTR, size=args.bs)
    x = np.stack([train[o:o + args.seq + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x).pin_memory().cuda(non_blocking=True)


model = C.build(cfg).cuda()
nparam = sum(p.numel() for p in model.parameters())
groups = C.param_groups(model, args.wd)
opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
step = 0; elapsed = 0.0
ck = f'{RD}/ckpt.pt'
if os.path.exists(ck):
    s = torch.load(ck, map_location='cuda', weights_only=False)
    model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); step = s['step']; elapsed = s['elapsed']
    P(f'resumed at step {step}')
P(f'== {name} cfg {cfg} params {nparam/1e6:.1f}M nonemb {C.nonemb_params(cfg)/1e6:.1f}M train GFLOP/tok {flop_tok/1e9:.3f} '
  f'steps {steps} tokens {steps*tok_step/1e6:.1f}M budget {args.pflops} PFLOP lr {lr}')


def lossf(x):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l = model.lm_loss(x[:, :-1], x[:, 1:])
        if cfg.kind == 'moe':   # switch load-balancing loss, spread over tokens so .mean() adds 0.01 * sum_layers aux
            l = l + 0.01 * sum(b.mlp.aux for b in model.blocks)
        return l


closs = torch.compile(lossf)


def lr_at(s):
    if s < args.warmup: return lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, steps - args.warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def evaluate(nseq=64):
    model.eval(); tot = []; last = []
    g = np.random.default_rng(99)
    offs = g.integers(0, len(val) - args.seq - 2, size=nseq)
    for i in range(0, nseq, args.bs):
        x = torch.from_numpy(np.stack([val[o:o + args.seq + 1] for o in offs[i:i + args.bs]]).astype(np.int64)).cuda()
        l = closs(x).float()
        tot.append(l.mean(1)); last.append(l[:, -128:].mean(1))
    model.train()
    return torch.cat(tot).mean().item(), torch.cat(last).mean().item()


def save(final=False):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, elapsed=elapsed), ck + '.tmp'); os.replace(ck + '.tmp', ck)
    if final:
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}, f'{RD}/final.pt')


if cfg.kind == 'slot': model._pt_setup(args.seq, 'cuda')
model.train()
t_last_ck = time.time(); t0 = time.time(); el0 = elapsed; tl = time.time(); lsum = torch.zeros((), device='cuda'); lcnt = 0; s0 = step
while step < steps:
    for g_ in opt.param_groups: g_['lr'] = lr_at(step) * g_['lr_mult']
    for m in range(args.accum):
        x = batch(step * args.accum + m)
        l = closs(x).mean()
        (l / args.accum).backward()
        lsum += l.detach(); lcnt += 1
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    step += 1
    elapsed = el0 + time.time() - t0
    if step % 50 == 0 or step == s0 + 1:
        torch.cuda.synchronize(); dt = time.time() - tl; tl = time.time()
        tps = (lcnt // args.accum) * tok_step / dt
        P(f'step {step}/{steps} loss {lsum.item()/max(lcnt,1):.4f} gn {gn.item():.2f} lr {lr_at(step):.2e} tok/s {tps:.0f} TFLOPS {tps*flop_tok/1e12:.1f} '
          f'elapsed {elapsed/60:.1f}m mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
        lsum.zero_(); lcnt = 0
    if step % 1000 == 0:
        v, vl = evaluate(64); P(f'VAL step {step} tokens {step*tok_step/1e6:.1f}M loss {v:.4f} lastblock {vl:.4f}')
    if time.time() - t_last_ck > args.ckpt_min * 60:
        save(); t_last_ck = time.time(); P(f'ckpt step {step}')
    if (time.time() - t0) / 60 > args.max_minutes:
        save(); P('max_minutes reached; exiting for resume'); sys.exit(3)
save(final=True)
v, vl = evaluate(256)
res = dict(name=name, scale=args.scale, arm=args.arm, steps=steps, tokens=steps * tok_step, train_pflops=steps * tok_step * flop_tok / 1e15,
           val_loss=v, val_last_block=vl, params=nparam, nonemb=C.nonemb_params(cfg), elapsed_min=elapsed / 60,
           infer_gflop_1000=2 * C.infer_macs(cfg)['total'] / 1e9)
json.dump(res, open(f'{RD}/result.json', 'w'), indent=1)
P('FINAL', json.dumps(res))
