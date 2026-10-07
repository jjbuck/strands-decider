"""J15 learned components on frozen deployed-kernel features (box only).

python j15learn.py exit  TEACH LAYERS        depth-exit heads: for each layer L, a pointer head (hobson's head as init + residual adapter) on the
                                               b8 residual after L layers, trained by KL to the b8 FINAL distribution on EXIT (train split, tasks
                                               disjoint from DEV); early-stopped on DEV; then applied to DEV and eval -> preds/exit_L{L}_{dev,eval}.jsonl
python j15learn.py unc                         learned draft uncertainty: MLP on the W4A4 draft's final residual rows + logit features -> P(draft
                                               decision != b8 decision); trained on EXIT, early-stopped on DEV -> preds/unc_{dev,eval}.jsonl
"""
import sys, os, json, glob, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
W = os.path.expanduser('~/work/j15/preds')
KINDS = ['noul', 'choice', 'score']


def load_preds(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out[(r['id'], r['q'])] = r
    return out


def load_taps(prefix, layers):
    feats = {}
    for f in sorted(glob.glob(prefix + '.taps.*.pt')):
        for k, v in torch.load(f).items():
            feats[k] = {L: v[L] for L in layers if L in v}
            feats[k].update(n=v['n'], kind=v['kind'], T=v['T'], temp=v['temp'], labels=v['labels'])
    return feats


def teacher_vec(rec, labels):
    return torch.tensor([rec['probs'][l] for l in labels], dtype=torch.float32)


class EH(nn.Module):
    def __init__(self, base, r=512):
        super().__init__()
        self.pre = nn.LayerNorm(2048, elementwise_affine=False)
        self.a1 = nn.Linear(2048, r); self.a2 = nn.Linear(r, 2048)
        nn.init.zeros_(self.a2.weight); nn.init.zeros_(self.a2.bias)
        self.norm = nn.LayerNorm(2048); self.q = nn.Linear(2048, base.q.out_features); self.k = nn.Linear(2048, base.k.out_features)
        self.norm.load_state_dict(base.norm.state_dict()); self.q.load_state_dict(base.q.state_dict()); self.k.load_state_dict(base.k.state_dict())
        self.scale = base.q.out_features ** -0.5
        self.gain = nn.Parameter(torch.ones(2048) * 8.0)        # LayerNorm output -> roughly the final hidden's scale (refit by training)

    def emb(self, x):
        x = self.pre(x) * self.gain
        return x + self.a2(F.gelu(self.a1(x)))

    def forward(self, d, o):       # d [B, 2048], o [B, K, 2048]
        dq = self.q(self.norm(self.emb(d)))
        ok = self.k(self.norm(self.emb(o)))
        return (ok @ dq.unsqueeze(-1)).squeeze(-1) * self.scale


def batches(keys, feats, L, teach, bs, shuffle, dev='cuda'):
    keys = list(keys)
    if shuffle: random.shuffle(keys)
    for i in range(0, len(keys), bs):
        kb = keys[i:i + bs]
        Kmax = max(feats[k]['n'] for k in kb)
        d = torch.stack([feats[k][L][0].float() for k in kb]).to(dev)
        o = torch.zeros(len(kb), Kmax, 2048, device=dev); mask = torch.zeros(len(kb), Kmax, dtype=torch.bool, device=dev)
        tv = torch.zeros(len(kb), Kmax, device=dev); temp = torch.tensor([feats[k]['temp'] for k in kb], device=dev)
        for j, k in enumerate(kb):
            n = feats[k]['n']; o[j, :n] = feats[k][L][1:1 + n].float().to(dev); mask[j, :n] = True
            if teach is not None: tv[j, :n] = teacher_vec(teach[k], feats[k]['labels']).to(dev)
        yield kb, d, o, mask, tv, temp


def run_head(h, keys, feats, L, teach=None, bs=64):
    out = {}; kl = 0.0; agree = 0; n = 0
    h.eval()
    with torch.no_grad():
        for kb, d, o, mask, tv, temp in batches(keys, feats, L, teach, bs, False):
            lg = (h(d, o) / temp[:, None]).masked_fill(~mask, -1e9)
            p = torch.softmax(lg, -1)
            for j, k in enumerate(kb):
                nn_ = feats[k]['n']; out[k] = p[j, :nn_].cpu()
            if teach is not None:
                kl += (tv * (torch.log(tv.clamp_min(1e-9)) - torch.log_softmax(lg, -1))).masked_fill(~mask, 0).sum().item()
                agree += (p.argmax(-1) == tv.argmax(-1)).sum().item(); n += len(kb)
    return out, (kl / max(n, 1), agree / max(n, 1))


def write_preds(path, probs, feats):
    with open(path, 'w') as f:
        for (iid, q), p in probs.items():
            f.write(json.dumps(dict(id=iid, q=q, T=feats[(iid, q)]['T'], probs={l: float(x) for l, x in zip(feats[(iid, q)]['labels'], p.tolist())})) + '\n')


def cmd_exit(layers):
    from kitrun import load_P
    P = load_P(); base = P.model.head.float().cpu()
    del P; torch.cuda.empty_cache()
    tx = load_preds(f'{W}/b8_exit.jsonl'); td = load_preds(f'{W}/b8_dev.jsonl')
    for L in layers:
        t0 = time.time()
        fx = load_taps(f'{W}/b8_exit.jsonl', [L]); fd = load_taps(f'{W}/b8_dev.jsonl', [L])
        kx = [k for k in fx if k in tx and L in fx[k]]; kd = [k for k in fd if k in td and L in fd[k]]
        h = EH(base).cuda()
        opt = torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=0.01)
        best = (1e9, None); log = []
        for ep in range(40):
            h.train()
            for kb, d, o, mask, tv, temp in batches(kx, fx, L, tx, 32, True):
                lg = (h(d, o) / temp[:, None]).masked_fill(~mask, -1e9)
                loss = (tv * (torch.log(tv.clamp_min(1e-9)) - torch.log_softmax(lg, -1))).masked_fill(~mask, 0).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            _, (dkl, dag) = run_head(h, kd, fd, L, td)
            log.append((ep, round(dkl, 4), round(dag, 4)))
            if dkl < best[0]: best = (dkl, {k: v.detach().clone() for k, v in h.state_dict().items()}, ep, dag)
            if ep - best[2] >= 6: break
        h.load_state_dict(best[1])
        print(f'exit L{L}: best dev KL {best[0]:.4f} agree {best[3]:.4f} at ep {best[2]}  ({time.time()-t0:.0f}s) log {log}', flush=True)
        torch.save(best[1], f'{W}/exithead_L{L}.pt')
        pd, _ = run_head(h, kd, fd, L, td); write_preds(f'{W}/exit_L{L}_dev.jsonl', pd, fd)
        del fx, fd
        fe = load_taps(f'{W}/b8_eval.jsonl', [L]); ke = [k for k in fe if L in fe[k]]
        pe, _ = run_head(h, ke, fe, L); write_preds(f'{W}/exit_L{L}_eval.jsonl', pe, fe)
        del fe; torch.cuda.empty_cache()


class UNC(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.pre = nn.LayerNorm(2048, elementwise_affine=False)
        self.h = nn.Sequential(nn.Linear(2048 + nf, 256), nn.GELU(), nn.Linear(256, 1))
    def forward(self, x, f):
        return self.h(torch.cat([self.pre(x), f], -1)).squeeze(-1)


def unc_feats(feats, preds):
    keys = [k for k in feats if k in preds]
    X = torch.stack([feats[k][24][0].float() for k in keys])
    F_ = []
    for k in keys:
        p = sorted(preds[k]['probs'].values(), reverse=True); p = [x / sum(p) for x in p]
        lg = sorted(preds[k]['logits'], reverse=True)
        ent = -sum(x * math.log(max(x, 1e-9)) for x in p)
        kind = feats[k]['kind']
        F_.append([p[0] - (p[1] if len(p) > 1 else 0), p[0], ent, (lg[0] - lg[1]) if len(lg) > 1 else 10.0, math.log(feats[k]['T']) / 9, len(p) / 23]
                  + [float(kind == kk) for kk in KINDS])
    return keys, X, torch.tensor(F_)


def cmd_unc():
    dx = load_preds(f'{W}/w4a4_exit.jsonl'); vx = load_preds(f'{W}/b8_exit.jsonl')
    dd = load_preds(f'{W}/w4a4_dev.jsonl'); vd = load_preds(f'{W}/b8_dev.jsonl')
    de = load_preds(f'{W}/w4a4_eval.jsonl'); ve = load_preds(f'{W}/b8_eval.jsonl')
    am = lambda r: max(r['probs'], key=r['probs'].get)
    out = {}
    for nm, D, V in (('exit', dx, vx), ('dev', dd, vd), ('eval', de, ve)):
        fe = load_taps(f'{W}/w4a4_{nm}.jsonl', [24])
        keys, X, Fz = unc_feats(fe, D)
        y = torch.tensor([float(V[k] is not None and am(D[k]) != am(V[k])) if k in V else float('nan') for k in keys])
        out[nm] = (keys, X, Fz, y)
        print(nm, len(keys), 'flip rate', float(torch.nanmean(y)), flush=True)
    keys, X, Fz, y = out['exit']; ok = ~torch.isnan(y)
    X, Fz, y = X[ok].cuda(), Fz[ok].cuda(), y[ok].cuda()
    kd, Xd, Fd, yd = out['dev']; okd = ~torch.isnan(yd)
    best = (1e9, None, -1)
    for seed in range(3):
        torch.manual_seed(seed)
        m = UNC(Fz.shape[1]).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        pw = torch.tensor((1 - y.mean()) / y.mean()).cuda()
        for ep in range(60):
            m.train(); perm = torch.randperm(len(y), device='cuda')
            for i in range(0, len(y), 128):
                b = perm[i:i + 128]
                loss = F.binary_cross_entropy_with_logits(m(X[b], Fz[b]), y[b], pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                ld = F.binary_cross_entropy_with_logits(m(Xd[okd].cuda(), Fd[okd].cuda()), yd[okd].cuda(), pos_weight=pw).item()
            if ld < best[0]: best = (ld, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep)
    print('unc best dev loss', best[0], 'ep', best[2], flush=True)
    m.load_state_dict(best[1]); m.eval()
    for nm in ('dev', 'eval', 'exit'):
        ks, Xn, Fn, _ = out[nm]
        with torch.no_grad(): s = torch.sigmoid(m(Xn.cuda(), Fn.cuda())).cpu().tolist()
        with open(f'{W}/unc_{nm}.jsonl', 'w') as f:
            for (iid, q), v in zip(ks, s): f.write(json.dumps(dict(id=iid, q=q, pflip=v)) + '\n')
    torch.save(best[1], f'{W}/unc_head.pt')


def cmd_apply(base_name, layers):
    """apply the b8-trained exit heads to another base's taps: preds/{base}_{dev,eval}.jsonl.taps -> preds/exit_{base}_L{L}_{dev,eval}.jsonl"""
    from kitrun import load_P
    P = load_P(); base = P.model.head.float().cpu(); del P
    for L in layers:
        h = EH(base); h.load_state_dict(torch.load(f'{W}/exithead_L{L}.pt')); h = h.cuda()
        for nm in ('dev', 'eval'):
            f = load_taps(f'{W}/{base_name}_{nm}.jsonl', [L]); ks = [k for k in f if L in f[k]]
            p, _ = run_head(h, ks, f, L); write_preds(f'{W}/exit_{base_name}_L{L}_{nm}.jsonl', p, f)
        print('applied', base_name, L, flush=True)


if __name__ == '__main__':
    random.seed(0); torch.manual_seed(0)
    if sys.argv[1] == 'apply':
        cmd_apply(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
    elif sys.argv[1] == 'exit':
        cmd_exit([int(x) for x in sys.argv[2].split(',')])
    else:
        cmd_unc()
