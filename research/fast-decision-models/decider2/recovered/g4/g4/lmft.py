"""Identical LM-loss fine-tune for every arm on the train-split versions of the LM-native probes (recall cloze with/without distractor,
SQuAD-MC-long answer), loss on the gold candidate tokens only; same examples, order, steps, LR.  Eval (lmeval scoring) at 1/3, 2/3, 3/3.
usage: python lmft.py RUNNAME [--steps 1000 --lr 2e-4]"""
import os, sys, json, time, math, pickle, collections, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lmeval as LE
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--bs', type=int, default=16)
args = ap.parse_args()
W = LE.W; D = LE.D
LE.EOS_C = int(json.load(open(f'{D}/meta.json'))['eos_c'])
sc, arm = args.run.split('_')[0], args.run.split('_')[1]
cfg = C.Cfg(sc, arm); m = C.build(cfg)
sd = torch.load(f'{W}/runs/{args.run}/final.pt', map_location='cpu'); m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
m = m.cuda()
if cfg.kind == 'slot': m._pt_setup(LE.N, 'cuda')
f = torch.compile(lambda a, b: m.lm_loss(a, b))
data = pickle.load(open(f'{D}/lmeval.pkl', 'rb'))
opt = torch.optim.AdamW(C.param_groups(m, 0.01), lr=args.lr, betas=(0.9, 0.95), fused=True)
os.makedirs(f'{W}/lmres', exist_ok=True)
log = open(f'{W}/lmres/lmft_{args.run}.log', 'a')


def P(s): print(s, flush=True); log.write(s + '\n'); log.flush()


def lr_at(s):
    if s < 50: return args.lr * (s + 1) / 50
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 50) / max(1, args.steps - 50))))


@torch.no_grad()
def evaluate(nsq=None):
    m.eval(); out = {}
    for task in ('recall', 'squad'):
        its = data[task][:nsq] if (task == 'squad' and nsq) else data[task]
        jobs = [(ii, ci) + LE.layout(e, ci) for ii, e in enumerate(its) for ci in range(len(e['c']))]
        sc_ = collections.defaultdict(dict); B = 16
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for b0 in range(0, len(jobs), B):
                ch = jobs[b0:b0 + B]; ml = len(ch); ch = ch + [ch[0]] * (B - ml)
                x = torch.from_numpy(np.stack([j[2] for j in ch])).cuda()
                l = f(x[:, :-1], x[:, 1:]).float().cpu().numpy()
                for k, (ii, ci, ids, pos) in enumerate(ch[:ml]):
                    lp = -l[k, pos]; sc_[ii][ci] = (float(lp.sum()), float(lp.mean()))
        y = np.array([e['y'] for e in its])
        ps = np.array([max(sc_[i], key=lambda c: sc_[i][c][0]) for i in range(len(its))])
        pm = np.array([max(sc_[i], key=lambda c: sc_[i][c][1]) for i in range(len(its))])
        r = dict(n=len(its), acc_sum=float((ps == y).mean()), acc_mean=float((pm == y).mean()), correct_sum=(ps == y).astype(int).tolist(), correct_mean=(pm == y).astype(int).tolist())
        if task == 'recall':
            byk = collections.defaultdict(list); byp = collections.defaultdict(list)
            for e, ok in zip(its, ps == y): byk[e['kind']].append(ok); byp[e['pair']].append(ok)
            r['by_kind'] = {k: round(float(np.mean(v)), 3) for k, v in sorted(byk.items())}
            r['pair_both'] = float(np.mean([all(v) for v in byp.values() if len(v) == 2]))
            r['acc_distract'] = float(np.mean([ok for e, ok in zip(its, ps == y) if e['kind'].endswith('_distract')]))
        out[task] = r
    m.train(); return out


rng = np.random.default_rng(31)
tr = dict(recall=data['recall_train'], squad=data['squad_train'])
order = {k: rng.permutation(len(v)) for k, v in tr.items()}; ptr = {k: 0 for k in tr}
curve = []; t0 = time.time(); acc = collections.defaultdict(list)
P(f'== lmft {args.run} steps {args.steps} lr {args.lr}')
m.train()
for step in range(1, args.steps + 1):
    task = 'recall' if step % 2 else 'squad'
    exs = [tr[task][order[task][(ptr[task] + i) % len(order[task])]] for i in range(args.bs)]; ptr[task] += args.bs
    lay = [LE.layout(e, e['y']) for e in exs]
    x = torch.from_numpy(np.stack([a for a, _ in lay])).cuda()
    mask = np.zeros((args.bs, LE.N), np.float32)
    for k, (_, pos) in enumerate(lay): mask[k, pos] = 1.0
    mask = torch.from_numpy(mask).cuda()
    for g in opt.param_groups: g['lr'] = lr_at(step) * g['lr_mult']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l = f(x[:, :-1], x[:, 1:]).float()
    loss = (l * mask).sum() / mask.sum()
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    acc[task].append(loss.detach())
    if step % 100 == 0:
        P(f'step {step} ' + ' '.join(f'{k} {torch.stack(v).mean().item():.4f}' for k, v in acc.items()) + f' {time.time()-t0:.0f}s'); acc = collections.defaultdict(list)
    if step in (args.steps // 3, 2 * args.steps // 3, args.steps):
        r = evaluate(None if step == args.steps else 500); r['step'] = step; curve.append(r)
        P(f"EVAL {step}: recall acc {r['recall']['acc_sum']:.3f} distract {r['recall']['acc_distract']:.3f} pair_both {r['recall']['pair_both']:.3f} "
          f"by_kind {r['recall']['by_kind']} | squad acc_sum {r['squad']['acc_sum']:.3f} acc_mean {r['squad']['acc_mean']:.3f} (n {r['squad']['n']})")
json.dump(dict(run=args.run, steps=args.steps, lr=args.lr, minutes=(time.time() - t0) / 60, curve=curve), open(f'{W}/lmres/lmft_{args.run}.json', 'w'))
P(f'DONE {args.run} {(time.time()-t0)/60:.1f} min')
