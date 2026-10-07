"""G3 lean MoE runtime for batch-1 prefill (CUDA graph, exact length) + kernels.

Per layer: [fused residual-add + RMSNorm] -> qkv GEMM (cuBLAS) -> fused RoPE -> SDPA (flash, GQA) -> o GEMM -> [fused add + RMSNorm]
-> router GEMM -> top-k gating -> align (sort slots by expert, pad each expert to BM) -> Triton grouped GEMM gate_up with
SwiGLU/ReLU-GLU epilogue (rows gathered by token) -> Triton grouped GEMM down with gate-weight epilogue, scattered to per-slot rows
-> next layer's fused (sum of k slots + residual + RMSNorm).
Granite multipliers are folded into the weights (embedding x12, residual x0.22 into Wo and Wd).

EXP impl: 'triton' (own kernels), 'gmm' (torch._grouped_mm on the sorted slots), 'vllm' (vllm fused_experts if importable).
"""
import os, sys, math, time, json, statistics as st
import torch, torch.nn.functional as F
import triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M


# ------------------------------------------------------------------ elementwise kernels
@triton.jit
def _rms_k(X, W, H, eps, N: tl.constexpr, NP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); offs = tl.arange(0, NP); m = offs < N
    x = tl.load(X + r * N + offs, mask=m, other=0.).to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    w = tl.load(W + offs, mask=m, other=0.).to(tl.float32)
    tl.store(H + r * N + offs, (w * (x * rs).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), mask=m)


@triton.jit
def _add_rms_k(X, D, W, H, eps, N: tl.constexpr, NP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); offs = tl.arange(0, NP); m = offs < N
    s = (tl.load(X + r * N + offs, mask=m, other=0.).to(tl.float32) + tl.load(D + r * N + offs, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16)
    tl.store(X + r * N + offs, s, mask=m)
    sf = s.to(tl.float32)
    rs = tl.rsqrt(tl.sum(sf * sf, 0) / N + eps)
    w = tl.load(W + offs, mask=m, other=0.).to(tl.float32)
    tl.store(H + r * N + offs, (w * (sf * rs).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), mask=m)


@triton.jit
def _red_add_rms_k(X, S, W, H, eps, N: tl.constexpr, NP: tl.constexpr, TOPK: tl.constexpr):
    """x[r] += sum_j S[r*TOPK + j]; h[r] = rms(x[r]) * w"""
    r = tl.program_id(0).to(tl.int64); offs = tl.arange(0, NP); m = offs < N
    acc = tl.zeros([NP], tl.float32)
    for j in tl.static_range(TOPK):
        acc += tl.load(S + (r * TOPK + j) * N + offs, mask=m, other=0.).to(tl.float32)
    s = (tl.load(X + r * N + offs, mask=m, other=0.).to(tl.float32) + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(X + r * N + offs, s, mask=m)
    sf = s.to(tl.float32)
    rs = tl.rsqrt(tl.sum(sf * sf, 0) / N + eps)
    w = tl.load(W + offs, mask=m, other=0.).to(tl.float32)
    tl.store(H + r * N + offs, (w * (sf * rs).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), mask=m)


@triton.jit
def _rope_k(QKV, COS, SIN, Q, K, H: tl.constexpr, KV: tl.constexpr, HD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64); hh = tl.program_id(1)
    c = tl.arange(0, HD); half: tl.constexpr = HD // 2
    pc = tl.where(c < half, c + half, c - half)
    sign = tl.where(c < half, -1.0, 1.0)
    base = QKV + t * (H + 2 * KV) * HD + hh * HD
    x = tl.load(base + c).to(tl.float32); xp = tl.load(base + pc).to(tl.float32)
    cs = tl.load(COS + t * HD + c).to(tl.float32); sn = tl.load(SIN + t * HD + c).to(tl.float32)
    y = x * cs + sign * xp * sn
    if hh < H:
        tl.store(Q + t * H * HD + hh * HD + c, y.to(tl.bfloat16))
    else:
        tl.store(K + t * KV * HD + (hh - H) * HD + c, y.to(tl.bfloat16))


# ------------------------------------------------------------------ grouped GEMM kernels (vLLM-style padded sorted layout)
@triton.jit
def _moe_gu_k(X, SORTED, BEXP, NPAD, W, OUT, NSLOT, es,
              K: tl.constexpr, FFN: tl.constexpr, TOPK: tl.constexpr, ACT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr, NFAST: tl.constexpr):
    pid = tl.program_id(0)
    if NFAST:
        pid_m = pid // NT; pid_n = pid % NT
    else:
        pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(NPAD):
        return
    e = tl.load(BEXP + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm)
    valid = sid < NSLOT
    tok = (sid // TOPK).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = X + tok[:, None] * K + rk[None, :]
    g_ptr = W + e * es + rn[None, :].to(tl.int64) * K + rk[:, None]
    u_ptr = g_ptr + FFN * K
    acc_g = tl.zeros([BM, BN], tl.float32); acc_u = tl.zeros([BM, BN], tl.float32)
    for kk in range(0, K, BK):
        a = tl.load(a_ptr, mask=valid[:, None], other=0.)
        acc_g = tl.dot(a, tl.load(g_ptr), acc_g)
        acc_u = tl.dot(a, tl.load(u_ptr), acc_u)
        a_ptr += BK; g_ptr += BK; u_ptr += BK
    gb = acc_g.to(tl.bfloat16).to(tl.float32)
    if ACT == 0:
        gb = (gb * tl.sigmoid(gb)).to(tl.bfloat16).to(tl.float32)
    else:
        gb = tl.maximum(gb, 0.0)
    out = (gb * acc_u.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rm[:, None].to(tl.int64) * FFN + rn[None, :], out)


@triton.jit
def _moe_dn_k(INTER, SORTED, BEXP, NPAD, W, GATE, OUT, NSLOT, es,
              D: tl.constexpr, FFN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr, NFAST: tl.constexpr):
    pid = tl.program_id(0)
    if NFAST:
        pid_m = pid // NT; pid_n = pid % NT
    else:
        pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(NPAD):
        return
    e = tl.load(BEXP + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm)
    valid = sid < NSLOT
    rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = INTER + rm[:, None].to(tl.int64) * FFN + rk[None, :]
    w_ptr = W + e * es + rn[None, :].to(tl.int64) * FFN + rk[:, None]
    acc = tl.zeros([BM, BN], tl.float32)
    for kk in range(0, FFN, BK):
        acc = tl.dot(tl.load(a_ptr), tl.load(w_ptr), acc)
        a_ptr += BK; w_ptr += BK
    gw = tl.load(GATE + sid, mask=valid, other=0.)
    out = (acc.to(tl.bfloat16).to(tl.float32) * gw[:, None]).to(tl.bfloat16)
    tl.store(OUT + sid[:, None].to(tl.int64) * D + rn[None, :], out, mask=valid[:, None])


@triton.jit
def _stream_k(Xp, n, BLOCK: tl.constexpr, OUT):
    pid = tl.program_id(0); nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(pid * BLOCK, n, nprog * BLOCK):
        o = s + tl.arange(0, BLOCK)
        acc += tl.load(Xp + o, mask=o < n, other=0.).to(tl.float32)
    tl.atomic_add(OUT, tl.sum(acc, 0))


# ------------------------------------------------------------------ runtime
class LeanMoE:
    def __init__(self, g, W, impl="triton", cfg=None, alias=False, synth_route=None):
        self.g, self.impl, self.alias, self.synth_route = g, impl, alias, synth_route
        dev = "cuda"; self.dev = dev
        self.embed = (W["embed"].float() * g.emb_mult).to(torch.bfloat16) if g.emb_mult != 1.0 else W["embed"]
        self.norm = W["norm"]
        self.L = []
        for d in W["layers"]:
            n = dict(ln1=d["ln1"], ln2=d["ln2"], Wr=d["Wr"].contiguous())
            n["Wqkv"] = torch.cat([d["Wq"], d["Wk"], d["Wv"]], 0).contiguous()
            n["Wo"] = (d["Wo"].float() * g.res_mult).to(torch.bfloat16).contiguous() if g.res_mult != 1.0 else d["Wo"].contiguous()
            n["Wgu"] = d["Wgu"].contiguous()
            n["Wd"] = (d["Wd"].float() * g.res_mult).to(torch.bfloat16).contiguous() if g.res_mult != 1.0 else d["Wd"].contiguous()
            self.L.append(n)
        self.cfg = cfg or {}
        self.cos = self.sin = None

    def weight_bytes(self):
        b = 0
        for n in self.L:
            for k in ("ln1", "ln2", "Wr", "Wqkv", "Wo", "Wgu", "Wd"): b += n[k].numel() * 2
        return b + self.norm.numel() * 2

    def pick(self, T):
        S = T * self.g.k
        per = S / self.g.E
        if "BM" in self.cfg: return self.cfg
        BM = 32 if per < 48 else (64 if per < 300 else 128)
        return dict(BM=BM, BN1=64 if self.g.ffn % 128 else 128, BK1=64, BN2=64, BK2=64, w1=4, s1=3, w2=4, s2=3, nf=1)

    def align(self, fe, S, BM):
        g = self.g; E = g.E
        counts = torch.zeros(E, device=self.dev, dtype=torch.int32).scatter_add_(0, fe, torch.ones_like(fe, dtype=torch.int32))
        padded = (counts + BM - 1) // BM * BM
        pend = torch.cumsum(padded, 0); pstart = pend - padded
        cend = torch.cumsum(counts, 0); cstart = cend - counts
        order = torch.argsort(fe, stable=True)
        se = fe[order]
        dest = pstart[se] + (torch.arange(S, device=self.dev) - cstart[se])
        nblk = S // BM + E
        sorted_ids = torch.full((nblk * BM,), S, device=self.dev, dtype=torch.int32)
        sorted_ids.scatter_(0, dest, order.to(torch.int32))
        bexp = torch.searchsorted(pend, torch.arange(nblk, device=self.dev, dtype=pend.dtype) * BM, right=True).clamp_max_(E - 1).to(torch.int32)
        return sorted_ids, bexp, pend[-1:].contiguous(), nblk, counts

    def experts(self, n, h2, gates, fe, T):
        g = self.g; S = T * g.k
        if g.E == 1:   # dense twin: cuBLAS
            hh = h2 @ n["Wgu"][0].t()
            a = (F.silu(hh[:, :g.ffn]) if g.act == "silu" else F.relu(hh[:, :g.ffn])) * hh[:, g.ffn:]
            return a @ n["Wd"][0].t()
        if self.impl == "gmm":
            order = torch.argsort(fe, stable=True); tok = order // g.k
            counts = torch.zeros(g.E, device=self.dev, dtype=torch.int32).scatter_add_(0, fe, torch.ones_like(fe, dtype=torch.int32))
            offs = torch.cumsum(counts, 0).to(torch.int32)
            xs = h2[tok]
            hh = torch._grouped_mm(xs, n["Wgu"].transpose(1, 2), offs=offs)
            a = (F.silu(hh[:, :g.ffn]) if g.act == "silu" else F.relu(hh[:, :g.ffn])) * hh[:, g.ffn:]
            y = torch._grouped_mm(a, n["Wd"].transpose(1, 2), offs=offs) * gates[order].to(torch.bfloat16)[:, None]
            out = torch.empty(S, g.d, device=self.dev, dtype=torch.bfloat16)
            out[order] = y
            return out
        if self.impl == "vllm":
            from vllm.model_executor.layers.fused_moe import fused_experts
            # vllm expects w1 [E, 2N, K] with (gate, up) halves and silu; output already summed over k
            return fused_experts(h2, n["Wgu"], n["Wd"], gates.view(T, g.k), fe.view(T, g.k).to(torch.int32))
        c = self.pick(T); BM = c["BM"]
        sorted_ids, bexp, npad, nblk, _ = self.align(fe, S, BM)
        inter = torch.empty(nblk * BM, g.ffn, device=self.dev, dtype=torch.bfloat16)
        es1 = 0 if self.alias else 2 * g.ffn * g.d
        nf = int(c.get("nf", 0)); nt1 = g.ffn // c["BN1"]; nt2 = g.d // c["BN2"]
        _moe_gu_k[(nblk * nt1, 1) if nf else (nblk, nt1)](h2, sorted_ids, bexp, npad, n["Wgu"], inter, S, es1, K=g.d, FFN=g.ffn, TOPK=g.k,
                                               ACT=0 if g.act == "silu" else 1, BM=BM, BN=c["BN1"], BK=c["BK1"], NT=nt1, NFAST=nf, num_warps=c["w1"], num_stages=c["s1"])
        out = torch.empty(S, g.d, device=self.dev, dtype=torch.bfloat16)
        es2 = 0 if self.alias else g.d * g.ffn
        _moe_dn_k[(nblk * nt2, 1) if nf else (nblk, nt2)](inter, sorted_ids, bexp, npad, n["Wd"], gates, out, S, es2, D=g.d, FFN=g.ffn,
                                             BM=BM, BN=c["BN2"], BK=c["BK2"], NT=nt2, NFAST=nf, num_warps=c["w2"], num_stages=c["s2"])
        return out

    def route(self, n, lg, T):
        g = self.g
        logits = lg.float()
        if g.E == 1:
            return torch.ones(T, device=self.dev, dtype=torch.float32), torch.zeros(T, device=self.dev, dtype=torch.long)
        if self.synth_route is not None:
            fe = self.synth_route[:T * g.k]
            return torch.full((T * g.k,), 1.0 / g.k, device=self.dev, dtype=torch.float32), fe
        v, i = logits.topk(g.k, dim=-1)
        w = torch.softmax(v, -1) if g.gate_fn == "softmax" else (lambda s: s / s.sum(-1, keepdim=True))(torch.sigmoid(v))
        if getattr(self, "record", None) is not None: self.record.append((w.reshape(-1).cpu(), i.reshape(-1).cpu()))
        return w.reshape(-1).contiguous(), i.reshape(-1).contiguous()

    @torch.no_grad()
    def forward(self, ids):
        g = self.g
        T = ids.shape[-1]
        if self.cos is None or self.cos.shape[0] < T:
            self.cos, self.sin = M.rope_cs(max(T, 4096), g.hd, g.theta, self.dev, torch.bfloat16)
        cos, sin = self.cos[:T].contiguous(), self.sin[:T].contiguous()
        x = F.embedding(ids.reshape(-1), self.embed)
        h = torch.empty_like(x)
        _rms_k[(T,)](x, self.L[0]["ln1"], h, g.eps, N=g.d, NP=triton.next_power_of_2(g.d), num_warps=8)
        nq = g.H * g.hd; nk = g.KV * g.hd
        for li, n in enumerate(self.L):
            lpre = (x @ n["Wr"].t()) if g.router_pre else None
            qkv = h @ n["Wqkv"].t()
            q = torch.empty(T, g.H, g.hd, device=self.dev, dtype=torch.bfloat16); k = torch.empty(T, g.KV, g.hd, device=self.dev, dtype=torch.bfloat16)
            _rope_k[(T, g.H + g.KV)](qkv, cos, sin, q, k, H=g.H, KV=g.KV, HD=g.hd, num_warps=1)
            v = qkv[:, nq + nk:].view(1, T, g.KV, g.hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q.view(1, T, g.H, g.hd).transpose(1, 2), k.view(1, T, g.KV, g.hd).transpose(1, 2), v,
                                               is_causal=True, enable_gqa=g.H != g.KV, scale=g.attn_scale)
            o = o.transpose(1, 2).reshape(T, nq)
            ao = o @ n["Wo"].t()
            h2 = torch.empty_like(x)
            _add_rms_k[(T,)](x, ao, n["ln2"], h2, g.eps, N=g.d, NP=triton.next_power_of_2(g.d), num_warps=8)
            gates, fe = self.route(n, lpre if g.router_pre else (h2 @ n["Wr"].t()), T)
            slots = self.experts(n, h2, gates, fe, T)
            nw = self.L[li + 1]["ln1"] if li + 1 < len(self.L) else self.norm
            h = torch.empty_like(x)
            if self.impl == "vllm":
                _add_rms_k[(T,)](x, slots, nw, h, g.eps, N=g.d, NP=triton.next_power_of_2(g.d), num_warps=8)
            else:
                _red_add_rms_k[(T,)](x, slots, nw, h, g.eps, N=g.d, NP=triton.next_power_of_2(g.d), TOPK=g.k, num_warps=8)
        return h

    def stream_graph(self):
        """bare read of every weight byte once (the weight-streaming floor)"""
        out = torch.zeros(1, device=self.dev, dtype=torch.float32)
        ts = [n[k] for n in self.L for k in ("ln1", "ln2", "Wr", "Wqkv", "Wo", "Wgu", "Wd")]
        def fn():
            for t in ts:
                nel = t.numel()
                _stream_k[(min(4 * 80, triton.cdiv(nel, 4096)),)](t, nel, BLOCK=4096, OUT=out, num_warps=8)
            return out
        return fn


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(gr): out = fn()
    return gr, out


def wall(gr, ids_static, pool, reps=20):
    ts = []
    for i in range(reps + 3):
        x = pool[i % len(pool)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids_static.copy_(x, non_blocking=True); gr.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[3:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))


def kernel_split(gr, reps=3):
    from torch.profiler import profile, ProfilerActivity
    tot = {}
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            gr.replay(); torch.cuda.synchronize()
        for e in p.events():
            if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start:
                nm = e.name.lower()
                key = ("moe_gu" if "_moe_gu" in nm else "moe_dn" if "_moe_dn" in nm else "grouped_mm" if "grouped" in nm else
                       "fused_moe" if "fused_moe" in nm else "gemm" if ("gemm" in nm or "cutlass" in nm or "sm80_xmma" in nm or "ampere" in nm) else
                       "attn" if ("flash" in nm or "fmha" in nm or "attention" in nm) else "sort" if ("sort" in nm or "radix" in nm) else "other")
                tot[key] = tot.get(key, 0.0) + (e.time_range.end - e.time_range.start) / 1000 / reps
    return {k: round(v, 3) for k, v in sorted(tot.items())}


# ------------------------------------------------------------------ weight-only quantized expert GEMMs (W8A16 / W4A16, per-output-channel symmetric)
# int8: Wq [E, N, K] int8, scale [E, N] fp32.  int4: Wq [E, N, K//2] uint8; inside every BK-block of K, byte j holds k=j (low nibble) and
# k=j+BK/2 (high nibble), value = nibble - 8.
@triton.jit
def _ldq(ptr, QB: tl.constexpr):
    w = tl.load(ptr)
    return w.to(tl.float32).to(tl.bfloat16)


@triton.jit
def _moe_gu_q_k(X, SORTED, BEXP, NPAD, W, SC, OUT, NSLOT,
                K: tl.constexpr, FFN: tl.constexpr, TOPK: tl.constexpr, ACT: tl.constexpr, QB: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(NPAD):
        return
    e = tl.load(BEXP + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm); valid = sid < NSLOT
    tok = (sid // TOPK).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    acc_g = tl.zeros([BM, BN], tl.float32); acc_u = tl.zeros([BM, BN], tl.float32)
    if QB == 8:
        rk = tl.arange(0, BK)
        a_ptr = X + tok[:, None] * K + rk[None, :]
        g_ptr = W + e * (2 * FFN * K) + rn[None, :].to(tl.int64) * K + rk[:, None]
        u_ptr = g_ptr + FFN * K
        for kk in range(0, K, BK):
            a = tl.load(a_ptr, mask=valid[:, None], other=0.)
            acc_g = tl.dot(a, tl.load(g_ptr).to(tl.bfloat16), acc_g)
            acc_u = tl.dot(a, tl.load(u_ptr).to(tl.bfloat16), acc_u)
            a_ptr += BK; g_ptr += BK; u_ptr += BK
    else:
        HK: tl.constexpr = BK // 2
        rk = tl.arange(0, HK)
        a_ptr = X + tok[:, None] * K + rk[None, :]
        g_ptr = W + e * (2 * FFN * (K // 2)) + rn[None, :].to(tl.int64) * (K // 2) + rk[:, None]
        u_ptr = g_ptr + FFN * (K // 2)
        for kk in range(0, K, BK):
            a0 = tl.load(a_ptr, mask=valid[:, None], other=0.); a1 = tl.load(a_ptr + HK, mask=valid[:, None], other=0.)
            pg = tl.load(g_ptr).to(tl.int32); pu = tl.load(u_ptr).to(tl.int32)
            acc_g = tl.dot(a0, ((pg & 15) - 8).to(tl.bfloat16), acc_g); acc_g = tl.dot(a1, ((pg >> 4) - 8).to(tl.bfloat16), acc_g)
            acc_u = tl.dot(a0, ((pu & 15) - 8).to(tl.bfloat16), acc_u); acc_u = tl.dot(a1, ((pu >> 4) - 8).to(tl.bfloat16), acc_u)
            a_ptr += BK; g_ptr += HK; u_ptr += HK
    sg = tl.load(SC + e * (2 * FFN) + rn); su = tl.load(SC + e * (2 * FFN) + FFN + rn)
    gb = (acc_g * sg[None, :]).to(tl.bfloat16).to(tl.float32)
    if ACT == 0:
        gb = (gb * tl.sigmoid(gb)).to(tl.bfloat16).to(tl.float32)
    else:
        gb = tl.maximum(gb, 0.0)
    out = (gb * (acc_u * su[None, :]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rm[:, None].to(tl.int64) * FFN + rn[None, :], out)


@triton.jit
def _moe_dn_q_k(INTER, SORTED, BEXP, NPAD, W, SC, GATE, OUT, NSLOT,
                D: tl.constexpr, FFN: tl.constexpr, QB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(NPAD):
        return
    e = tl.load(BEXP + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm); valid = sid < NSLOT
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], tl.float32)
    if QB == 8:
        rk = tl.arange(0, BK)
        a_ptr = INTER + rm[:, None].to(tl.int64) * FFN + rk[None, :]
        w_ptr = W + e * (D * FFN) + rn[None, :].to(tl.int64) * FFN + rk[:, None]
        for kk in range(0, FFN, BK):
            acc = tl.dot(tl.load(a_ptr), tl.load(w_ptr).to(tl.bfloat16), acc)
            a_ptr += BK; w_ptr += BK
    else:
        HK: tl.constexpr = BK // 2
        rk = tl.arange(0, HK)
        a_ptr = INTER + rm[:, None].to(tl.int64) * FFN + rk[None, :]
        w_ptr = W + e * (D * (FFN // 2)) + rn[None, :].to(tl.int64) * (FFN // 2) + rk[:, None]
        for kk in range(0, FFN, BK):
            p = tl.load(w_ptr).to(tl.int32)
            acc = tl.dot(tl.load(a_ptr), ((p & 15) - 8).to(tl.bfloat16), acc)
            acc = tl.dot(tl.load(a_ptr + HK), ((p >> 4) - 8).to(tl.bfloat16), acc)
            a_ptr += BK; w_ptr += HK
    sc = tl.load(SC + e * D + rn)
    gw = tl.load(GATE + sid, mask=valid, other=0.)
    out = ((acc * sc[None, :]).to(tl.bfloat16).to(tl.float32) * gw[:, None]).to(tl.bfloat16)
    tl.store(OUT + sid[:, None].to(tl.int64) * D + rn[None, :], out, mask=valid[:, None])


def quantize(W, bits, BK=64):
    """W [E, N, K] bf16 -> (Wq, scale [E, N] fp32); int4 packed per BK-block (low nibble k=j, high k=j+BK/2)."""
    Wf = W.float()
    qmax = 127 if bits == 8 else 7
    sc = (Wf.abs().amax(-1) / qmax).clamp_min(1e-8)
    q = torch.round(Wf / sc[..., None]).clamp(-qmax - (1 if bits == 8 else 1), qmax)
    if bits == 8:
        return q.to(torch.int8).contiguous(), sc.contiguous()
    E, N, K = W.shape
    q = (q + 8).to(torch.int32).view(E, N, K // BK, 2, BK // 2)
    packed = (q[:, :, :, 0, :] | (q[:, :, :, 1, :] << 4)).to(torch.uint8).reshape(E, N, K // 2)
    return packed.contiguous(), sc.contiguous()


def dequantize(Wq, sc, bits, K, BK=64):
    if bits == 8: return (Wq.float() * sc[..., None]).to(torch.bfloat16)
    E, N, _ = Wq.shape
    p = Wq.to(torch.int32).view(E, N, K // BK, BK // 2)
    q = torch.stack([(p & 15) - 8, (p >> 4) - 8], 3).view(E, N, K).float()
    return (q * sc[..., None]).to(torch.bfloat16)


class LeanMoEQ(LeanMoE):
    """expert weights in int8 / int4 (weight-only), everything else as LeanMoE."""

    def __init__(self, g, W, bits, **kw):
        super().__init__(g, W, **kw)
        self.bits = bits
        for n in self.L:
            n["Wgu_q"], n["sgu"] = quantize(n["Wgu"], bits); n["Wd_q"], n["sd"] = quantize(n["Wd"], bits)
            n["Wgu_deq"] = dequantize(n["Wgu_q"], n["sgu"], bits, g.d); n["Wd_deq"] = dequantize(n["Wd_q"], n["sd"], bits, g.ffn)
            del n["Wgu"], n["Wd"]

    def weight_bytes(self):
        b = 0
        for n in self.L:
            for k in ("ln1", "ln2", "Wr", "Wqkv", "Wo"): b += n[k].numel() * 2
            for k in ("Wgu_q", "Wd_q"): b += n[k].numel() * n[k].element_size()
            for k in ("sgu", "sd"): b += n[k].numel() * 4
        return b

    def stream_graph(self):
        out = torch.zeros(1, device=self.dev, dtype=torch.float32)
        ts = [n[k] for n in self.L for k in ("ln1", "ln2", "Wr", "Wqkv", "Wo", "Wgu_q", "Wd_q", "sgu", "sd")]
        def fn():
            for t in ts:
                nel = t.numel()
                _stream_k[(min(4 * 80, triton.cdiv(nel, 4096)),)](t, nel, BLOCK=4096, OUT=out, num_warps=8)
            return out
        return fn

    def experts(self, n, h2, gates, fe, T):
        g = self.g; S = T * g.k
        c = self.pick(T); BM = c["BM"]
        sorted_ids, bexp, npad, nblk, _ = self.align(fe, S, BM)
        inter = torch.empty(nblk * BM, g.ffn, device=self.dev, dtype=torch.bfloat16)
        _moe_gu_q_k[(nblk, g.ffn // c["BN1"])](h2, sorted_ids, bexp, npad, n["Wgu_q"], n["sgu"], inter, S, K=g.d, FFN=g.ffn, TOPK=g.k,
                                                 ACT=0 if g.act == "silu" else 1, QB=self.bits, BM=BM, BN=c["BN1"], BK=64, num_warps=c["w1"], num_stages=c["s1"])
        out = torch.empty(S, g.d, device=self.dev, dtype=torch.bfloat16)
        _moe_dn_q_k[(nblk, g.d // c["BN2"])](inter, sorted_ids, bexp, npad, n["Wd_q"], n["sd"], gates, out, S, D=g.d, FFN=g.ffn, QB=self.bits,
                                               BM=BM, BN=c["BN2"], BK=64, num_warps=c["w2"], num_stages=c["s2"])
        return out
