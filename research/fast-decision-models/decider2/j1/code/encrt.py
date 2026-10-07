"""J1 fused runtime for the T5Gemma-2B encoder decider (bf16, A10G/GA102), same discipline as d1's lean2 (hobson):
- LoRA merged; pre-norms folded into the next GEMM's weights (row rsqrt applied in the GEMM epilogue from a fp32 sum-of-squares);
- GeGLU in the gate/up GEMM epilogue (rows interleaved g0,u0,g1,u1..);
- post-norm + residual add + next sum-of-squares in ONE row kernel; RoPE + q/k/v split in one kernel;
- flash attention (aten varlen FA2): state rows bidirectional over the state (optionally windowed); question rows over the cached state
  K/V plus their own question (M=1: one call over the contiguous row; M>1: two calls merged by log-sum-exp);
- state computed once per request and shared by all M questions (the masked state cache); everything in one CUDA graph per exact shape.
EncRT(layers_merged, embed, norm1, head).build(Ls, qlens, opts, kinds_T) -> callable graph; see lat_enc.py for timing and checks."""
import math, torch, torch.nn.functional as F
import triton, triton.language as tl

D, NH, NKV, HD, FF = 2304, 8, 4, 256, 9216
EPS = 1e-6


# ---------------------------------------------------------------- GEMM: C[M,N] = A[M,K] @ B[N,K]^T, optional row-scale prologue (folded RMSNorm), GeGLU epilogue
@triton.jit
def _gemm_k(A, B, C, SS, M, N, K, eps, Kd, EPI: tl.constexpr, PRO_RS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + rm[:, None].to(tl.int64) * K + rk[None, :]
    b_ptr = B + rn[None, :].to(tl.int64) * K + rk[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.)
        b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.)
        acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    if PRO_RS:
        ss = tl.load(SS + rm, mask=rm < M, other=1.0)
        acc = acc * tl.rsqrt(ss / Kd + eps)[:, None]
    if EPI == 1:   # GeGLU (tanh approx): gelu(g) * u, interleaved columns
        gg, uu = tl.split(tl.reshape(acc, [BM, BN // 2, 2]))
        gb = gg.to(tl.bfloat16).to(tl.float32)
        ge = gb * tl.sigmoid(1.5957691216057308 * (gb + 0.044715 * gb * gb * gb))
        out = (ge.to(tl.bfloat16).to(tl.float32) * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        cn = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(C + rm[:, None].to(tl.int64) * (N // 2) + cn[None, :], out, mask=rm[:, None] < M)
    else:
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


CFGS = [(128, 64, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (256, 128, 32, 8, 3), (64, 256, 32, 8, 3),
        (32, 64, 128, 4, 4), (16, 64, 128, 4, 4), (32, 128, 64, 4, 4)]


def tgemm(a, b, epi=0, ss=None, cfg=None, out=None):
    M, K = a.shape; N = b.shape[0]
    BM, BN, BK, nw, ns = cfg
    c = out if out is not None else torch.empty(M, N // 2 if epi == 1 else N, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_k[grid](a, b, c, ss if ss is not None else c, M, N, K, EPS, float(K), EPI=epi, PRO_RS=ss is not None,
                  BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=nw, num_stages=ns)
    return c


# ---------------------------------------------------------------- row kernels
@triton.jit
def _addpost_k(X, A, W, SSO, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    # x += rmsnorm(a) * w (w = 1 + weight, fp32);  sso[row] = sum(x_new^2)
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK); m = c < N
    a = tl.load(A + r * N + c, mask=m, other=0.).to(tl.float32)
    rs = tl.rsqrt(tl.sum(a * a, 0) / N + eps)
    w = tl.load(W + c, mask=m, other=0.)
    an = (a * rs * w).to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + r * N + c, mask=m, other=0.).to(tl.float32)
    s = (x + an).to(tl.bfloat16)
    tl.store(X + r * N + c, s, mask=m)
    sf = s.to(tl.float32)
    tl.store(SSO + r, tl.sum(sf * sf, 0))


def addpost(x, a, w1, sso):
    T = x.shape[0]
    _addpost_k[(T,)](x, a, w1, sso, EPS, N=D, BLOCK=4096, num_warps=8)


@triton.jit
def _ss_k(X, SSO, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); c = tl.arange(0, BLOCK); m = c < N
    x = tl.load(X + r * N + c, mask=m, other=0.).to(tl.float32)
    tl.store(SSO + r, tl.sum(x * x, 0))


@triton.jit
def _rope_k(P, COS, SIN, Q, K, V, ps, HDc: tl.constexpr):
    # P [T, 4096] = q(8x256) | k(4x256) | v(4x256); head program h in 0..15
    r = tl.program_id(0).to(tl.int64); h = tl.program_id(1)
    c = tl.arange(0, HDc)
    x = tl.load(P + r * ps + h * HDc + c).to(tl.float32)
    if h < 12:
        pc = tl.where(c < HDc // 2, c + HDc // 2, c - HDc // 2)
        xp = tl.load(P + r * ps + h * HDc + pc).to(tl.float32)
        cs = tl.load(COS + r * HDc + c).to(tl.float32); sn = tl.load(SIN + r * HDc + c).to(tl.float32)
        y = x * cs + tl.where(c < HDc // 2, -xp, xp) * sn
        if h < 8:
            tl.store(Q + r * 8 * HDc + h * HDc + c, y.to(tl.bfloat16))
        else:
            tl.store(K + r * 4 * HDc + (h - 8) * HDc + c, y.to(tl.bfloat16))
    else:
        tl.store(V + r * 4 * HDc + (h - 12) * HDc + c, x.to(tl.bfloat16))


def fa(q, k, v, cu_q, cu_k, mq, mk, window=0, lse=False):
    wl = wr = (window - 1) if window else None
    o, l, *_ = torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, mq, mk, 0.0, False, False, scale=HD ** -0.5,
                                                       window_size_left=wl, window_size_right=wr)
    return (o, l) if lse else o


@triton.jit
def _rowscale_k(X, SS, H, Kd, eps, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); c = tl.arange(0, BLOCK); m = c < N
    x = tl.load(X + r * N + c, mask=m, other=0.).to(tl.float32)
    rs = tl.rsqrt(tl.load(SS + r) / Kd + eps)
    tl.store(H + r * N + c, (x * rs).to(tl.bfloat16), mask=m)


@triton.jit
def _geglu_il_k(GU, M_, I: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); cb = tl.program_id(1)
    c = cb * BLOCK + tl.arange(0, BLOCK); cm = c < I
    g = tl.load(GU + r * 2 * I + 2 * c, mask=cm, other=0.).to(tl.float32)
    u = tl.load(GU + r * 2 * I + 2 * c + 1, mask=cm, other=0.).to(tl.float32)
    ge = g * tl.sigmoid(1.5957691216057308 * (g + 0.044715 * g * g * g))
    tl.store(M_ + r * I + c, (ge.to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16), mask=cm)


def cublas_path(a, b, epi, ss):
    M, K = a.shape
    if ss is not None:
        h = torch.empty_like(a); _rowscale_k[(M,)](a, ss, h, float(K), EPS, N=K, BLOCK=triton.next_power_of_2(K), num_warps=8); a = h
    c = a @ b.t()
    if epi == 1:
        I = b.shape[0] // 2; m = torch.empty(M, I, device=a.device, dtype=torch.bfloat16)
        _geglu_il_k[(M, triton.cdiv(I, 1024))](c, m, I=I, BLOCK=1024, num_warps=4); c = m
    return c


class EncRT:
    def __init__(self, layers, embed, norm1, head, local=(), window=0, dev='cuda'):
        self.dev = dev; self.embed = embed; self.norm1 = norm1.float().contiguous(); self.head = head
        self.local = set(local); self.window = window
        self.L = []
        for d in layers:
            e = {}
            e['Wqkv_f'] = (d['Wqkv'].float() * d['n_pa'][None, :]).to(torch.bfloat16).contiguous()
            Wgu = d['Wgu']; il = torch.stack([Wgu[:FF], Wgu[FF:]], 1).reshape(2 * FF, D)
            e['Wgu_f'] = (il.float() * d['n_pf'][None, :]).to(torch.bfloat16).contiguous()
            e['Wo'] = d['Wo'].contiguous(); e['Wd'] = d['Wd'].contiguous()
            e['w_poa'] = d['n_poa'].float().contiguous(); e['w_pof'] = d['n_pof'].float().contiguous()
            self.L.append(e)
        self.inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
        self.cfg = {}

    # ---------------- GEMM config choice per (M, N, K, epi): fastest of a few Triton tiles (and cuBLAS for plain GEMMs)
    def pick(self, M, N, K, epi, fold):
        key = (M, N, K, epi, fold)
        if key in self.cfg: return self.cfg[key]
        a = torch.randn(M, K, device=self.dev, dtype=torch.bfloat16); b = torch.randn(N, K, device=self.dev, dtype=torch.bfloat16) * 0.02
        ss = torch.full((M,), float(K), device=self.dev)
        best = None
        cands = [('tr', c) for c in CFGS] + [('cublas', None)]
        import os
        if os.environ.get('FORCE_GEMM') == 'cublas': cands = [('cublas', None)]
        if os.environ.get('FORCE_GEMM') == 'tr': cands = [('tr', c) for c in CFGS]
        for kind, c in cands:
            try:
                f = (lambda: cublas_path(a, b, epi, ss if fold else None)) if kind == 'cublas' else (lambda c=c: tgemm(a, b, epi, ss if fold else None, c))
                for _ in range(3): f()
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(10): f()
                e1.record(); torch.cuda.synchronize(); t = e0.elapsed_time(e1) / 10
                if best is None or t < best[0]: best = (t, kind, c)
            except Exception:
                pass
        self.cfg[key] = best
        return best

    def mm(self, a, W, epi=0, ss=None):
        M, K = a.shape; N = W.shape[0]
        t, kind, c = self.pick(M, N, K, epi, ss is not None)
        if kind == 'cublas':
            return cublas_path(a, W, epi, ss)
        return tgemm(a, W, epi, ss, c)

    # ---------------- one request: state [Ls] + M questions (packed after the state), returns final-normed rows at `read` positions
    def body(self, ids, Ls, qlens, read, cos, sin, cu):
        T = ids.shape[0]; M = len(qlens); Nq = T - Ls
        x = F.embedding(ids, self.embed) * torch.tensor(D ** 0.5, dtype=torch.bfloat16)
        ss = torch.empty(T, device=self.dev, dtype=torch.float32)
        _ss_k[(T,)](x, ss, N=D, BLOCK=4096, num_warps=8)
        q = torch.empty(T, NH, HD, device=self.dev, dtype=torch.bfloat16); k = torch.empty(T, NKV, HD, device=self.dev, dtype=torch.bfloat16)
        v = torch.empty(T, NKV, HD, device=self.dev, dtype=torch.bfloat16)
        o = torch.empty(T, NH, HD, device=self.dev, dtype=torch.bfloat16)
        for i, d in enumerate(self.L):
            p = self.mm(x, d['Wqkv_f'], 0, ss)
            _rope_k[(T, 16)](p, cos, sin, q, k, v, p.stride(0), HDc=HD, num_warps=2)
            win = self.window if i in self.local else 0
            o[:Ls] = fa(q[:Ls], k[:Ls], v[:Ls], cu['s'], cu['s'], Ls, Ls, win)
            if M == 1:
                o[Ls:] = fa(q[Ls:], k, v, cu['q1'], cu['all'], Nq, T)
            else:
                oa, la = fa(q[Ls:], k[:Ls], v[:Ls], cu['q1'], cu['s'], Nq, Ls, lse=True)
                ob, lb = fa(q[Ls:], k[Ls:], v[Ls:], cu['qs'], cu['qs'], cu['mq'], cu['mq'], lse=True)
                la = la.reshape(NH, Nq).t()[:, :, None]; lb = lb.reshape(NH, Nq).t()[:, :, None]
                mx = torch.maximum(la, lb); wa = torch.exp(la - mx); wb = torch.exp(lb - mx)
                o[Ls:] = ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(torch.bfloat16)
            a = self.mm(o.reshape(T, NH * HD), d['Wo'])
            addpost(x, a, d['w_poa'], ss)
            m = self.mm(x, d['Wgu_f'], 1, ss)
            dm = self.mm(m, d['Wd'])
            addpost(x, dm, d['w_pof'], ss)
        xr = x[read].float()
        return (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + EPS) * self.norm1).to(torch.bfloat16)

    def build(self, Ls, qlens, opts, temps, nopt_max=None):
        """opts[m] = option row indices relative to question m's first token; answer row = last token of question m.
        -> dict(ids=static input buffer [Ls+sum(qlens)], fn=callable producing probs [M, Kmax])"""
        dev = self.dev; M = len(qlens); T = Ls + sum(qlens)
        starts = [Ls]
        for L in qlens[:-1]: starts.append(starts[-1] + L)
        pos = torch.cat([torch.arange(Ls)] + [Ls + torch.arange(L) for L in qlens]).to(dev).float()
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos = fr.cos().to(torch.bfloat16).contiguous(); sin = fr.sin().to(torch.bfloat16).contiguous()
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
        cqs = [0]
        for L in qlens: cqs.append(cqs[-1] + L)
        cu = dict(s=i32([0, Ls]), all=i32([0, T]), q1=i32([0, T - Ls]), qs=i32(cqs), mq=max(qlens))
        K = nopt_max or max(len(o) for o in opts)
        read = []; oi = []; ai = []
        for m in range(M):
            base = len(read)
            read += [starts[m] + j for j in opts[m]]; oi.append([base + j for j in range(len(opts[m]))] + [base] * (K - len(opts[m])))
            read.append(starts[m] + qlens[m] - 1); ai.append(len(read) - 1)
        read = torch.tensor(read, device=dev); oi = torch.tensor(oi, device=dev); ai = torch.tensor(ai, device=dev)
        valid = torch.tensor([[j < len(opts[m]) for j in range(K)] for m in range(M)], device=dev)
        tdiv = torch.tensor(temps, device=dev, dtype=torch.float32)[:, None]
        ids = torch.zeros(T, dtype=torch.long, device=dev)
        hd = self.head

        def fn():
            hr = self.body(ids, Ls, qlens, read, cos, sin, cu).float()
            lg = hd(hr[ai], hr[oi])                       # [M, K]
            lg = (lg / tdiv).masked_fill(~valid, float('-inf'))
            return torch.softmax(lg, -1)
        return dict(ids=ids, fn=fn, T=T)
