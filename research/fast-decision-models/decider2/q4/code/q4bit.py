"""Q4: the accuracy cost of the bit Strassen loses, measured at every one of hobson's 96 GEMMs on real activations (train-split requests).
python q4bit.py [n_requests]  -> ~/work/q4/res_bit.json
Hooks H1's bf16 forward (h1lib.H1, rotated H1 FORMAT basis: R1 for Win/Wgu, R2 for Wo, R4 for Wd); for every GEMM call it computes, in fp32
(TF32 off), the reference y = x_r W_r^T and the outputs of these formats, accumulating ||y_fmt - y||^2 and ||y||^2 per GEMM:
  int  : 'a{b}w{b}' per-token activations (absmax, clip 1.0 for >= 6 bits, 0.9 for <= 4 bits) x per-channel weights (absmax for >= 6 bits,
         H1's MSE clip search for <= 4 bits); suffix _p2: Strassen one level, rows t and t+M/2 share one activation scale and channels n and
         n+N/2 share one weight scale; _p4: two levels, groups of four rows / channels share a scale.
         8 = W8A8 (deployed width), 7_p2 = one-level Strassen int8, 6_p4 = two-level Strassen int8; 4 = W4A4, 3_p2 / 1_p4 for int4.
         Also 7 and 3 per token (no sharing) to separate the lost bit from the shared scale, and 7_p2s = channel pairs chosen offline by sorting
         channel absmax (a free permutation for weights; activation rows stay in natural order).
  bf16 : 'bf16' dense (bf16 operands, fp32 accumulation, = the dense kernel); 'bf16_st1' / 'bf16_st2' one / two levels of Strassen with every
         operand sum formed in fp32 and rounded once to bf16, products in fp32, combination in fp32 (the best case for the format)."""
import os, sys, json, math, torch
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import h1lib as HL
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FMTS = ['a8w8', 'a7w7', 'a7w7_p2', 'a7w7_p2s', 'a6w6_p4', 'a4w4', 'a3w3', 'a3w3_p2', 'a1w1_p4', 'bf16', 'bf16_st1', 'bf16_st2']


def pad_rows(X, g):
    M = X.shape[0]; Mp = (M + g - 1) // g * g
    if Mp == M: return X, M
    return torch.cat([X, torch.zeros(Mp - M, X.shape[1], device=X.device, dtype=X.dtype)], 0), M


def q_act(X, qmax, clip, g):
    """per-token absmax; g > 1: rows t, t + M/g, ... share the scale (Strassen blocks of rows)"""
    Xp, M = pad_rows(X, g)
    am = Xp.abs().amax(1).reshape(g, -1).amax(0).clamp_min(1e-8)          # [Mp/g]
    s = (am / torch.full_like(am, qmax)) * clip
    s = s.repeat(g)
    q = torch.round(Xp / s[:, None]).clamp(-qmax, qmax)
    return (q * s[:, None])[:M]


def q_wt(W, qmax, g, search, sort=False):
    N, K = W.shape
    perm = None
    if sort and g > 1:   # pair channels of similar absmax (offline permutation): channel ranks 2j and 2j+1 share a scale
        order = W.abs().amax(1).argsort()
        perm = order.reshape(N // g, g).t().reshape(-1)                    # block j of the permuted matrix = j-th member of every group
        W = W[perm]
    Wg = W.reshape(g, N // g, K).permute(1, 0, 2).reshape(N // g, g * K)   # one row per scale group
    s = HL.rtn_scales(Wg, qmax, search)
    q = torch.round(Wg / s[:, None]).clamp(-qmax, qmax)
    Wq = (q * s[:, None]).reshape(N // g, g, K).permute(1, 0, 2).reshape(N, K)
    if perm is not None:
        inv = torch.empty_like(perm); inv[perm] = torch.arange(N, device=W.device); Wq = Wq[inv]
    return Wq


def blocks(X, g):
    """split [R, C] into g x g blocks (rows padded to a multiple of g; C must divide)"""
    Xp, _ = pad_rows(X, g)
    R, C = Xp.shape
    return [[Xp[i * R // g:(i + 1) * R // g, j * C // g:(j + 1) * C // g] for j in range(g)] for i in range(g)]


# Strassen's original form as coefficient tables over 2x2 blocks: M_k = (sum a_ij A_ij)(sum b_ij B_ij); C_ij = sum c_k M_k
# with C = A B, A = X, B = W^T (so B_ij = W_ji^T)
SA = [{(0, 0): 1, (1, 1): 1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 1): 1}, {(0, 0): 1, (0, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 1): 1, (1, 1): -1}]
SB = [{(0, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (0, 0): -1}, {(1, 1): 1}, {(0, 0): 1, (0, 1): 1}, {(1, 0): 1, (1, 1): 1}]
SC = {(0, 0): {0: 1, 3: 1, 4: -1, 6: 1}, (0, 1): {2: 1, 4: 1}, (1, 0): {1: 1, 3: 1}, (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}}


def lin_ops(levels):
    """coefficient tables for `levels` of Strassen: list of (a: {(bi, bj): c} over 2^L x 2^L blocks, b: same, out: {(ci, cj): c})"""
    ops = [({(0, 0): 1}, {(0, 0): 1}, {(0, 0): 1})]
    for _ in range(levels):
        new = []
        for (a, b, c) in ops:
            for k in range(7):
                na = {(2 * i + di, 2 * j + dj): v * w for (i, j), v in a.items() for (di, dj), w in SA[k].items()}
                nb = {(2 * i + di, 2 * j + dj): v * w for (i, j), v in b.items() for (di, dj), w in SB[k].items()}
                nc = {}
                for (i, j), v in c.items():
                    for (di, dj), cmap in SC.items():
                        if k in cmap: nc[(2 * i + di, 2 * j + dj)] = nc.get((2 * i + di, 2 * j + dj), 0) + v * cmap[k]
                new.append((na, nb, nc))
        ops = new
    return ops


_OPS = {1: lin_ops(1), 2: lin_ops(2)}


def strassen_bf16(X, W, levels):
    """y = X W^T with `levels` of Strassen; every operand sum is formed in fp32 and rounded once to bf16; products and combination fp32"""
    g = 2 ** levels
    M = X.shape[0]
    BT = W.t().contiguous()                                                 # B = W^T [K, N]
    Ab = blocks(X, g); Bb = blocks(BT, g)
    out = None
    Mg, Ng = Ab[0][0].shape[0], Bb[0][0].shape[1]
    C = [[torch.zeros(Mg, Ng, device=X.device) for _ in range(g)] for _ in range(g)]
    for (a, b, c) in _OPS[levels]:
        As = sum(v * Ab[i][j] for (i, j), v in a.items()).to(torch.bfloat16).float()
        Bs = sum(v * Bb[i][j] for (i, j), v in b.items()).to(torch.bfloat16).float()
        P = As @ Bs
        for (i, j), v in c.items(): C[i][j] += v * P
    Y = torch.cat([torch.cat(C[i], 1) for i in range(g)], 0)
    return Y[:M]


def strassen_check():
    """the coefficient tables reproduce A B exactly in fp64"""
    torch.manual_seed(0)
    X = torch.randn(64, 32, dtype=torch.float64); W = torch.randn(48, 32, dtype=torch.float64)
    for L in (1, 2):
        g = 2 ** L
        BT = W.t().contiguous(); Ab = blocks(X, g); Bb = blocks(BT, g)
        C = [[torch.zeros(64 // g, 48 // g, dtype=torch.float64) for _ in range(g)] for _ in range(g)]
        for (a, b, c) in _OPS[L]:
            P = sum(v * Ab[i][j] for (i, j), v in a.items()) @ sum(v * Bb[i][j] for (i, j), v in b.items())
            for (i, j), v in c.items(): C[i][j] += v * P
        Y = torch.cat([torch.cat(C[i], 1) for i in range(g)], 0)
        print('strassen tables, levels', L, 'max err', (Y - X @ W.t()).abs().max().item(), 'products', len(_OPS[L]), flush=True)


class Cap(HL.H1):
    acc = None
    wcache = {}

    def lin(self, i, k, x, xn=None):
        y = super().lin(i, k, x, xn)
        if self.acc is not None: self.measure(i, k, x, xn)
        return y

    def measure(self, i, k, x, xn):
        src = (xn if k in ('Win', 'Wgu') else x).float()
        xr = self.rot_for(i, k)(src)
        if (i, k) not in self.wcache: self.wcache[(i, k)] = self.wfold(i, k).float().contiguous()
        Wr = self.wcache[(i, k)]
        if Wr.shape[0] % 4: Wr = Wr                                         # all hobson N are multiples of 4
        y = xr @ Wr.t()
        ref = y.pow(2).sum().item()
        a = self.acc.setdefault(f'{i}.{k}', {f: 0.0 for f in FMTS + ['ref']})
        a['ref'] += ref
        for f in FMTS:
            if f == 'bf16':
                yf = xr.to(torch.bfloat16).float() @ Wr.to(torch.bfloat16).float().t()
            elif f.startswith('bf16_st'):
                yf = strassen_bf16(xr, Wr, int(f[-1]))
            else:
                b = int(f[1]); qmax = float(2 ** (b - 1) - 1)
                g = 2 if '_p2' in f else (4 if '_p4' in f else 1)
                clip = 1.0 if b >= 6 else 0.9
                xq = q_act(xr, qmax, clip, g)
                wq = q_wt(Wr, qmax, g, b <= 4, sort=f.endswith('s'))
                yf = xq @ wq.t()
            a[f] += (yf - y).pow(2).sum().item()


def main(n):
    strassen_check()
    g = Cap()
    items = HL.cal_items(n, seed=11)
    g.acc = {}
    rows = 0
    with torch.no_grad():
        for st, qd in items:
            pr = g.prep(st, qd); ids = pr['s'] + pr['q']
            if len(ids) > 2500: ids = ids[:500] + ids[-2000:]
            g.fwd(ids, q0=len(pr['s'])); rows += len(ids)
            print('req', len(ids), flush=True)
    res = dict(n_requests=n, rows=rows, per_gemm={}, summary={})
    for key, a in g.acc.items():
        res['per_gemm'][key] = {f: math.sqrt(a[f] / a['ref']) for f in FMTS}
    import statistics as S
    for f in FMTS:
        v = [r[f] for r in res['per_gemm'].values()]
        res['summary'][f] = dict(median=S.median(v), mean=sum(v) / len(v), max=max(v),
                                 energy=math.sqrt(sum(g.acc[k][f] for k in g.acc) / sum(g.acc[k]['ref'] for k in g.acc)))
    for a_, b_ in (('a7w7_p2', 'a8w8'), ('a7w7', 'a8w8'), ('a6w6_p4', 'a8w8'), ('a3w3_p2', 'a4w4'), ('a3w3', 'a4w4'), ('a1w1_p4', 'a4w4'),
                   ('bf16_st1', 'bf16'), ('bf16_st2', 'bf16'), ('a7w7_p2s', 'a7w7_p2')):
        r = [res['per_gemm'][k][a_] / res['per_gemm'][k][b_] for k in res['per_gemm']]
        res['summary'][f'ratio_{a_}_over_{b_}'] = dict(median=S.median(r), min=min(r), max=max(r))
    print(json.dumps(res['summary'], indent=1), flush=True)
    json.dump(res, open(os.path.expanduser('~/work/q4/res_bit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
