"""Identical decision fine-tune for every arm: probe (CF-probe generator, 2-way) + squad (SQuAD-MC-long, 4-way), alternating
single-task batches, same examples / order / steps / LR / loss for every arm.  Learning curve at 1/3, 2/3, 3/3 of the steps.
usage: python finetune.py SCALE ARM [--steps 2400 --bs 16 --lr 3e-4] [--init RUNNAME] (--init none = no pretraining)"""
import os, sys, time, json, math, argparse, pickle, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('scale'); ap.add_argument('arm')
ap.add_argument('--steps', type=int, default=2400); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--warmup', type=int, default=100)
ap.add_argument('--init', default=None); ap.add_argument('--tag', default='')
ap.add_argument('--curve_n', type=int, default=600)
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
W = os.path.expanduser('~/work/g4'); D = f'{W}/data'
init = args.init or f'{args.scale}_{args.arm}'
name = f'ft_{args.scale}_{args.arm}{args.tag}'; RD = f'{W}/runs/{name}'; os.makedirs(RD, exist_ok=True)
log = open(f'{RD}/log.txt', 'a')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


cfg = C.Cfg(args.scale, args.arm)
model = C.build(cfg)
if init != 'none':
    sd = torch.load(f'{W}/runs/{init}/final.pt', map_location='cpu')
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
model = model.cuda()
heads = nn.ModuleDict(dict(probe=nn.Linear(cfg.d, 2), squad=nn.Linear(cfg.d, 4))).cuda()
for h in heads.values(): nn.init.normal_(h.weight, std=0.02); nn.init.zeros_(h.bias)
data = pickle.load(open(f'{D}/ft.pkl', 'rb'))
PAD = 0
NMAX = 1024


def collate(exs):
    B = len(exs); ids = np.full((B, NMAX), PAD, dtype=np.int64)
    last = np.zeros(B, np.int64); nctx = np.zeros(B, np.int64); nq = np.zeros(B, np.int64)
    for i, e in enumerate(exs):
        q, s = e['q'].astype(np.int64), e['s'].astype(np.int64)
        seq = np.concatenate([q, s, q]); ids[i, :len(seq)] = seq
        last[i] = len(seq) - 1; nctx[i] = len(q) + len(s); nq[i] = len(q)
    t = lambda a: torch.from_numpy(a).cuda(non_blocking=True)
    return t(ids), t(last), t(nctx), t(nq), torch.tensor([e['y'] for e in exs]).cuda()


def fwd(ids, last, nctx, nq):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if cfg.kind == 'slot': return model.dec_hidden(ids, nctx, nq)
        return model.dec_hidden(ids, last)


cfwd = torch.compile(fwd)
params = list(model.parameters()) + list(heads.parameters())
groups = C.param_groups(model, 0.01) + [dict(params=list(heads.parameters()), lr_mult=1.0, weight_decay=0.0)]
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)


def lr_at(s):
    if s < args.warmup: return args.lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(key, n=None):
    model.eval(); exs = data[key][:n] if n else data[key]; task = 'squad' if 'squad' in key else 'probe'
    probs = []
    for i in range(0, len(exs), 32):
        ch = exs[i:i + 32]; m = len(ch); ch = ch + [ch[0]] * (32 - m)
        b = collate(ch)
        h = cfwd(*b[:4]).float()
        probs.append(torch.softmax(heads[task](h), -1)[:m].cpu())
    model.train()
    pr = torch.cat(probs).numpy(); y = np.array([e['y'] for e in exs]); pred = pr.argmax(1)
    res = dict(n=len(exs), acc=float((pred == y).mean()))
    if task == 'probe':
        byp = collections.defaultdict(list)
        for e, ok in zip(exs, pred == y): byp[e['pair']].append((e['kind'], ok))
        pairs = [v for v in byp.values() if len(v) == 2]
        res['flip'] = float(np.mean([v[0][1] and v[1][1] for v in pairs])); res['npairs'] = len(pairs)
        kinds = collections.defaultdict(list)
        for v in pairs: kinds[v[0][0]].append(v[0][1] and v[1][1])
        res['flip_by_kind'] = {k: round(float(np.mean(x)), 3) for k, x in sorted(kinds.items())}
        dist = [v[0][1] and v[1][1] for v in pairs if v[0][0].endswith('_distract')]
        res['flip_distract'] = float(np.mean(dist)) if dist else None
        res['p_true'] = pr[:, 1].round(4).tolist()
    else:
        res['p'] = pr.round(4).tolist()
    return res


P(f'== {name} init {init} cfg {cfg}')
rng = np.random.default_rng(2024)
tr = dict(probe=data['probe_train'], squad=data['squad_train'])
order = {k: rng.permutation(len(v)) for k, v in tr.items()}
ptr = dict(probe=0, squad=0)
curve = []; t0 = time.time(); lacc = collections.defaultdict(list)
marks = {args.steps // 3, 2 * args.steps // 3, args.steps}
model.train()
for step in range(1, args.steps + 1):
    task = 'probe' if step % 2 else 'squad'
    idx = [order[task][(ptr[task] + i) % len(order[task])] for i in range(args.bs)]; ptr[task] += args.bs
    ids, last, nctx, nq, y = collate([tr[task][i] for i in idx])
    for g in opt.param_groups: g['lr'] = lr_at(step) * g['lr_mult']
    h = cfwd(ids, last, nctx, nq).float()
    loss = F.cross_entropy(heads[task](h), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    lacc[task].append(loss.detach())
    if step % 100 == 0:
        P(f'step {step} ' + ' '.join(f'{k} {torch.stack(v).mean().item():.4f}' for k, v in lacc.items()) + f' {time.time()-t0:.0f}s')
        lacc = collections.defaultdict(list)
    if step in marks:
        full = step == args.steps
        r = dict(step=step, examples=step * args.bs, probe=evaluate('probe_eval', None), squad=evaluate('squad_eval', None if full else args.curve_n),
                 cfp_official=evaluate('cfp_official', None))
        curve.append(r)
        P(f'EVAL step {step}: probe acc {r["probe"]["acc"]:.3f} flip {r["probe"]["flip"]:.3f} distract-flip {r["probe"]["flip_distract"]:.3f} '
          f'| squad acc {r["squad"]["acc"]:.3f} (n {r["squad"]["n"]}) | official CF-probe (fits 1k) acc {r["cfp_official"]["acc"]:.3f} flip {r["cfp_official"]["flip"]:.3f} '
          f'n_pairs {r["cfp_official"]["npairs"]} | by kind {r["probe"]["flip_by_kind"]}')
res = dict(name=name, scale=args.scale, arm=args.arm, init=init, steps=args.steps, bs=args.bs, lr=args.lr, minutes=(time.time() - t0) / 60,
           infer_gflop_1000=2 * C.infer_macs(cfg)['total'] / 1e9, curve=curve)
json.dump(res, open(f'{RD}/result.json', 'w'))
P('DONE', name, f'{(time.time()-t0)/60:.1f} min')
