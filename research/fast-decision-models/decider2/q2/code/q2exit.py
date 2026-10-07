"""Layer-L exit head for a base runtime (J15's recipe, j15learn.cmd_exit): hobson's pointer head + rank-512 residual adapter on the base's
residual after L layers; trained by KL to the base's FINAL distribution on EXIT (train split), early-stopped on DEV (train split, tasks
disjoint from EXIT); then applied to DEV and eval. python q2exit.py BASE L   (reads ~/work/q2/preds/BASE_{exit,dev,eval}.jsonl + taps)"""
import sys, os, time, random, torch
sys.path[:0] = [os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import j15learn as JL
W = os.path.expanduser('~/work/q2/preds')

if __name__ == '__main__':
    random.seed(0); torch.manual_seed(0)
    base_name, L = sys.argv[1], int(sys.argv[2])
    from kitrun import load_P
    P = load_P(); base = P.model.head.float().cpu(); del P; torch.cuda.empty_cache()
    tx = JL.load_preds(f'{W}/{base_name}_exit.jsonl'); td = JL.load_preds(f'{W}/{base_name}_dev.jsonl')
    t0 = time.time()
    fx = JL.load_taps(f'{W}/{base_name}_exit.jsonl', [L]); fd = JL.load_taps(f'{W}/{base_name}_dev.jsonl', [L])
    kx = [k for k in fx if k in tx and L in fx[k]]; kd = [k for k in fd if k in td and L in fd[k]]
    print('EXIT', len(kx), 'DEV', len(kd), flush=True)
    h = JL.EH(base).cuda()
    opt = torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=0.01)
    best = (1e9, None, -1, 0); log = []
    for ep in range(40):
        h.train()
        for kb, d, o, mask, tv, temp in JL.batches(kx, fx, L, tx, 32, True):
            lg = (h(d, o) / temp[:, None]).masked_fill(~mask, -1e9)
            loss = (tv * (torch.log(tv.clamp_min(1e-9)) - torch.log_softmax(lg, -1))).masked_fill(~mask, 0).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        _, (dkl, dag) = JL.run_head(h, kd, fd, L, td)
        log.append((ep, round(dkl, 5), round(dag, 4)))
        if dkl < best[0]: best = (dkl, {k: v.detach().clone() for k, v in h.state_dict().items()}, ep, dag)
        if ep - best[2] >= 6: break
    h.load_state_dict(best[1])
    print(f'exit {base_name} L{L}: best DEV KL {best[0]:.5f} agree {best[3]:.4f} at ep {best[2]} ({time.time()-t0:.0f}s) log {log}', flush=True)
    torch.save(best[1], f'{W}/exithead_{base_name}_L{L}.pt')
    pd, _ = JL.run_head(h, kd, fd, L, td); JL.write_preds(f'{W}/exit_{base_name}_L{L}_dev.jsonl', pd, fd)
    del fx, fd
    fe = JL.load_taps(f'{W}/{base_name}_eval.jsonl', [L]); ke = [k for k in fe if L in fe[k]]
    pe, _ = JL.run_head(h, ke, fe, L); JL.write_preds(f'{W}/exit_{base_name}_L{L}_eval.jsonl', pe, fe)
    print('wrote exit preds', len(pd), len(pe), flush=True)
