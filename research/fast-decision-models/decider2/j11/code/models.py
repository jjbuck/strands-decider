"""J11 from-scratch decision architectures (plain PyTorch, SDPA).

Arms (all end in the same pointer readout: an answer vector scored against K option vectors):
  dec    control. Causal decoder over [state ; question+options], hobson's layout and pointer head.
  decv   dec + exact value channels at the input (literal hash / type / key / record / magnitude embeddings). Attribution arm.
  slot   option-centric: a shallow bidirectional token encoder reads the state and (separately) the question; a DEEP stack of
         slot layers (answer slot + K option slots + M scratch slots) attends jointly over [state memory ; question memory ; slots].
         State tokens never see the question (cacheable, question compiled per deployment); depth lives in the K+M+1 slots.
  vslot  slot + exact value channels + pointer/compare operators inside the slot layers (exact payload retrieval, equality of
         canonical hashes, signed differences of retrieved values, sigmoid counting, magnitude-biased pointing).
  belief slot substrate, but the slot stack is ONE weight-tied block applied R rounds; each round emits option beliefs that are
         fed back into the option slots (recurrent inference machine), deep supervision on every round.
Batch layout (dict of cuda tensors): see train.collate().
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

HD = 64  # head dim everywhere


def ffn_hidden(d): return int(round(8 * d / 3 / 64)) * 64


class RMS(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); h = ffn_hidden(d); self.w13 = nn.Linear(d, 2 * h, bias=False); self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x):
        a, b = self.w13(x).chunk(2, -1); return self.w2(F.silu(a) * b)


def rope_cache(T, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, HD, 2, device=device).float() / HD))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return torch.cos(f), torch.sin(f)


def rope(x, cos, sin):  # x [B,H,T,64]
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[: x.shape[2]].to(x.dtype), sin[: x.shape[2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class TokBlock(nn.Module):
    """token layer: causal (dec) or bidirectional with key padding mask (encoders)."""
    def __init__(self, d):
        super().__init__(); self.H = d // HD
        self.n1 = RMS(d); self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False); self.n2 = RMS(d); self.mlp = MLP(d)
        self.qn = RMS(HD); self.kn = RMS(HD)  # qk-norm (as in hobson's attention): bounded logits, stable from-scratch training
    def forward(self, x, cs, mask=None, causal=False, kv_out=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, HD).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), *cs), rope(self.kn(k), *cs)
        if kv_out is not None: kv_out.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class PointerHead(nn.Module):
    def __init__(self, d, dim=256):
        super().__init__(); self.norm = nn.LayerNorm(d); self.q = nn.Linear(d, dim); self.k = nn.Linear(d, dim); self.scale = dim ** -0.5
    def forward(self, dec, opts):  # [B,d], [B,K,d]
        return (self.k(self.norm(opts)) @ self.q(self.norm(dec)).unsqueeze(-1)).squeeze(-1) * self.scale


def hash_bits(h):
    """int64 [..] -> ±1 float [..,64]; 0 hash -> zeros."""
    sh = torch.arange(64, device=h.device)
    b = ((h.unsqueeze(-1) >> sh) & 1).float() * 2 - 1
    return b * (h != 0).unsqueeze(-1).float()


class ValueEmbed(nn.Module):
    """token-level exact channels -> embedding addend: literal identity hash, type, key-name binding, record binding, magnitude."""
    def __init__(self, d):
        super().__init__()
        self.lh = nn.Linear(64, d, bias=False); self.kh = nn.Linear(64, d, bias=False); self.rh = nn.Linear(64, d, bias=False)
        self.typ = nn.Embedding(6, d); self.mag = nn.Linear(9, d, bias=False)
        for m in (self.lh, self.kh, self.rh, self.mag): nn.init.normal_(m.weight, std=0.02 / 8)
        nn.init.normal_(self.typ.weight, std=0.02)
    def forward(self, p):  # p: dict typ [B,T] long, lh/kh/rh int64, val float (nan none)
        v = p["val"]; has = torch.isfinite(v); z = torch.where(has, torch.sign(v) * torch.log1p(v.abs()), torch.zeros_like(v))
        fr = torch.stack([z / 5] + [torch.sin(z * f) for f in (1, 3, 9, 27)] + [torch.cos(z * f) for f in (1, 3, 9, 27)], -1) * has.unsqueeze(-1)
        return (self.lh(hash_bits(p["lh"])) + self.kh(hash_bits(p["kh"])) + self.rh(hash_bits(p["rh"])) + self.typ(p["typ"]) + self.mag(fr))


class Pointer(nn.Module):
    """pointer/compare operator for slots over memory payloads. Uses Hp dedicated heads of the slot layer's attention."""
    SC = (1e-3, 1e-1, 1e1, 1e3)
    def __init__(self, d, Hp=4):
        super().__init__(); self.Hp = Hp
        self.qh = nn.Linear(d, Hp * 64, bias=False); nn.init.normal_(self.qh.weight, std=0.02)
        self.beta = nn.Parameter(torch.ones(Hp)); self.gamma = nn.Parameter(torch.zeros(Hp)); self.tau = nn.Parameter(torch.full((Hp,), 2.0))
        self.b_id = nn.Parameter(torch.tensor([2.0, 0.0, 2.0, 0.0][:Hp])); self.b_rec = nn.Parameter(torch.tensor([0.0, 2.0, 0.0, 2.0][:Hp]))
        self.pairs = [(a, b) for a in range(Hp) for b in range(a + 1, Hp)]
        nf = len(self.pairs) * (1 + len(self.SC)) + Hp * (64 + 9)
        self.out = nn.Linear(nf, d, bias=False); nn.init.zeros_(self.out.weight)
    def logits_bias(self, sn, P):
        """sn [B,S,d] normed slots; P memory payload tensors -> additive logits [B,Hp,S,Tm] (fp32)."""
        B, S, _ = sn.shape
        qh = self.qh(sn).view(B, S, self.Hp, 64).transpose(1, 2).float()  # [B,Hp,S,64]
        hm = P["hb"]  # [B,Tm,64] ±1 / 0
        b = (qh @ hm.transpose(1, 2).unsqueeze(1)) * (self.beta.view(1, -1, 1, 1) / 8)
        b = b + self.gamma.view(1, -1, 1, 1) * P["z"].unsqueeze(1).unsqueeze(1)
        if "match" in P:  # exact binding priors: state tokens whose literal equals one of this slot's own literals, and tokens of that record
            b = b + self.b_id.view(1, -1, 1, 1) * P["match"].unsqueeze(1) + self.b_rec.view(1, -1, 1, 1) * P["rmatch"].unsqueeze(1)
        return b
    def features(self, logits, a, P):
        """logits/a [B,Hp,S,Tm] (a = softmax over valid memory) -> features [B,S,d]. fp32, autocast off: values must stay exact.
        Retrieval of a value / hash is a softmax restricted to tokens that carry one (no division by a vanishing mass)."""
        with torch.autocast("cuda", enabled=False):
            logits = logits.float(); a = a.float()
            mv, hm = P["mv"].float(), P["hb"].float()
            mh = (hm.abs().sum(-1) > 0).float()
            av = torch.softmax(logits.masked_fill(mv[:, None, None, :] <= 0, -1e9), -1)
            vhat = av @ P["v0"].float()[:, None, :, None]  # [B,Hp,S,1]
            ah = torch.softmax(logits.masked_fill(mh[:, None, None, :] <= 0, -1e9), -1)
            hhat = ah @ hm[:, None]  # [B,Hp,S,64]
            massv = a @ mv[:, None, :, None]; massh = a @ mh[:, None, :, None]
            that = a @ P["t1"].float()[:, None]  # [B,Hp,S,6]
            cnt = torch.sigmoid(logits - self.tau.float().view(1, -1, 1, 1)) @ P["ls"].float()[:, None, :, None]
            z = torch.sign(vhat) * torch.log1p(vhat.abs())
            per = torch.cat([hhat, z / 5, massv, torch.log1p(cnt), that], -1)  # 64+1+1+1+6 = 73
            f = [per.transpose(1, 2).flatten(2)]
            for x, y in self.pairs:
                eq = (hhat[:, x] * hhat[:, y]).sum(-1, keepdim=True) / 64
                dv = vhat[:, x] - vhat[:, y]
                f.append(torch.cat([eq] + [torch.tanh(dv * s) for s in self.SC], -1))
            return F.linear(torch.cat(f, -1), self.out.weight.float())


def binding(P, slot_h):
    """slot_h [B,S,8] int64 literal hashes of each slot's own text (0 = none). -> match/rmatch [B,S,Tm] float:
    match = state token's literal hash is one of the slot's literals; rmatch = token lies in the record of the first matched token."""
    lh, rh, st = P["lh"], P["rh"], P["is_state"]  # [B,Tm]
    m = ((lh[:, None, None, :] == slot_h[:, :, :, None]) & (slot_h[:, :, :, None] != 0)).any(2) & st[:, None, :]  # [B,S,Tm]
    has = m & (rh[:, None, :] != 0)
    first = has.float().argmax(-1)  # [B,S]
    rsel = torch.gather(rh, 1, first).masked_fill(~has.any(-1), 0)  # [B,S]
    rm = (rh[:, None, :] == rsel[:, :, None]) & (rsel[:, :, None] != 0)
    return dict(match=m.float(), rmatch=rm.float())


class SlotBlock(nn.Module):
    """slots attend jointly over [memory ; slots] with GQA (kv heads n_kv), then MLP. Optional pointer operator on heads 0..Hp-1."""
    def __init__(self, d, n_kv=2, ptr=False):
        super().__init__(); self.H = d // HD; n_kv = n_kv if self.H % n_kv == 0 else 1; self.nkv = n_kv  # odd head counts (d448): MQA
        self.n1 = RMS(d); self.q = nn.Linear(d, d, bias=False); self.kv_s = nn.Linear(d, 2 * n_kv * HD, bias=False)
        self.kv_m = nn.Linear(d, 2 * n_kv * HD, bias=False); self.o = nn.Linear(d, d, bias=False); self.n2 = RMS(d); self.mlp = MLP(d)
        self.ptr = Pointer(d) if ptr else None
        self.qn = RMS(HD); self.kn = RMS(HD)
    def mem_kv(self, mem):
        B, T, _ = mem.shape
        k, v = self.kv_m(mem).view(B, T, 2, self.nkv, HD).permute(2, 0, 3, 1, 4); return self.kn(k), v
    def forward(self, s, mkv, mask, P=None):
        """s [B,S,d]; mkv (k,v) [B,nkv,Tm,64]; mask bool [B,1,S,Tm+S] (True = attend)."""
        B, S, D = s.shape; sn = self.n1(s)
        q = self.qn(self.q(sn).view(B, S, self.H, HD).transpose(1, 2))
        ks, vs = self.kv_s(sn).view(B, S, 2, self.nkv, HD).permute(2, 0, 3, 1, 4); ks = self.kn(ks)
        k = torch.cat([mkv[0], ks], 2); v = torch.cat([mkv[1], vs], 2)
        rep = self.H // self.nkv
        k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        extra = None
        if self.ptr is None:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            Hp = self.ptr.Hp; Tm = mkv[0].shape[2]
            a_rest = F.scaled_dot_product_attention(q[:, Hp:], k[:, Hp:], v[:, Hp:], attn_mask=mask)
            lg = (q[:, :Hp].float() @ k[:, :Hp, :Tm].float().transpose(-1, -2)) / 8 + self.ptr.logits_bias(sn, P)
            lg = lg.masked_fill(~mask[:, :, :, :Tm], -1e9)
            w = torch.softmax(lg, -1)
            a_p = (w.to(v.dtype) @ v[:, :Hp, :Tm])
            a = torch.cat([a_p, a_rest], 1)
            extra = self.ptr.features(lg, w, P)
        s = s + self.o(a.transpose(1, 2).reshape(B, S, D))
        if extra is not None: s = s + extra.to(s.dtype)
        return s + self.mlp(self.n2(s))


class DecModel(nn.Module):
    def __init__(self, V, d, L, value=False):
        super().__init__(); self.d = d
        self.emb = nn.Embedding(V, d); nn.init.normal_(self.emb.weight, std=0.02)
        self.ve = ValueEmbed(d) if value else None
        self.blocks = nn.ModuleList([TokBlock(d) for _ in range(L)]); self.nf = RMS(d); self.head = PointerHead(d)
        self.cs = None
    def forward(self, b):
        x = self.emb(b["ids"])
        if self.ve is not None: x = x + self.ve(b["pl"])
        T = x.shape[1]
        if self.cs is None or self.cs[0].shape[0] < T: self.cs = rope_cache(max(T, 8192), x.device)
        for blk in self.blocks: x = blk(x, self.cs, causal=True)
        x = self.nf(x); self._h = x
        dec = x[torch.arange(x.shape[0], device=x.device), b["ans"]]
        opts = x.gather(1, b["opt"].clamp_min(0).unsqueeze(-1).expand(-1, -1, self.d))
        return [self.head(dec, opts)]


def aux_dec(m, b, rate=0.1, cap=2048):
    """dense auxiliary for the causal decoder: next-token CE at a random `rate` of positions, tied embedding head (training only)."""
    x = m._h; ids = b["ids"]; B, T, _ = x.shape
    ln = b["ans"] + 1
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    sel = (torch.rand(B, T, device=x.device) < rate) & (pos < (ln - 1).unsqueeze(1))
    bi, ti = sel.nonzero(as_tuple=True)
    if bi.numel() > cap: k = torch.randperm(bi.numel(), device=x.device)[:cap]; bi, ti = bi[k], ti[k]
    lg = F.linear(x[bi, ti], m.emb.weight)
    return F.cross_entropy(lg.float(), ids[bi, ti + 1]), int(bi.numel())


def aux_enc(m, b, mask_id, rate=0.1, cap=2048):
    """dense auxiliary for bidirectional encoders: masked-token CE on a separate pass over the state with `rate` of tokens masked."""
    ids = b["s_ids"]; B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    sel = (torch.rand(B, T, device=ids.device) < rate) & (pos < b["s_len"].unsqueeze(1))
    xin = ids.masked_fill(sel, mask_id)
    pl = b.get("s_pl")
    h, _ = m.encode(xin, b["s_len"], 0, pl)
    bi, ti = sel.nonzero(as_tuple=True)
    if bi.numel() > cap: k = torch.randperm(bi.numel(), device=ids.device)[:cap]; bi, ti = bi[k], ti[k]
    lg = F.linear(h[bi, ti], m.emb.weight)
    return F.cross_entropy(lg.float(), ids[bi, ti]), int(bi.numel())


class SlotModel(nn.Module):
    """slot / vslot / belief."""
    def __init__(self, V, d, Ls, Ld, M=4, value=False, rounds=1, n_kv=2):
        super().__init__(); self.d = d; self.M = M; self.R = rounds
        self.emb = nn.Embedding(V, d); nn.init.normal_(self.emb.weight, std=0.02)
        self.seg = nn.Embedding(3, d); nn.init.normal_(self.seg.weight, std=0.02)
        self.ve = ValueEmbed(d) if value else None
        self.enc = nn.ModuleList([TokBlock(d) for _ in range(Ls)]); self.nf_e = RMS(d)
        self.slots = nn.ModuleList([SlotBlock(d, n_kv, ptr=value and (i % 2 == 1)) for i in range(Ld)])  # pointer ops on alternate layers
        self.styp = nn.Embedding(3, d); nn.init.normal_(self.styp.weight, std=0.02)
        self.scratch = nn.Parameter(torch.randn(M, d)); self.qpool = nn.Linear(d, d, bias=False)  # scratch at unit scale like the other slots (small slots blow up RMSNorm gradients)
        if rounds > 1:
            self.rnd = nn.Embedding(rounds, d); nn.init.normal_(self.rnd.weight, std=0.02)
            self.bel = nn.Linear(3, d, bias=False); nn.init.zeros_(self.bel.weight)
        self.nf = RMS(d); self.head = PointerHead(d); self.cs = None

    def encode(self, ids, mask_len, seg, pl=None):
        x = self.emb(ids) + self.seg.weight[seg]
        if self.ve is not None and pl is not None: x = x + self.ve(pl)
        B, T, _ = x.shape
        if self.cs is None or self.cs[0].shape[0] < T: self.cs = rope_cache(max(T, 8192), x.device)
        valid = torch.arange(T, device=x.device).unsqueeze(0) < mask_len.unsqueeze(1)
        m = valid[:, None, None, :] if not bool(valid.all()) else None
        for blk in self.enc: x = blk(x, self.cs, mask=m)
        return self.nf_e(x), valid

    def forward(self, b):
        ms, vs = self.encode(b["s_ids"], b["s_len"], 0, b.get("s_pl"))
        mq, vq = self.encode(b["q_ids"], b["q_len"], 1, b.get("q_pl"))
        B, K, To = b["o_ids"].shape
        mo, vo = self.encode(b["o_ids"].view(B * K, To), b["o_len"].view(-1).clamp_min(1), 2, b.get("o_pl"))
        return self.decide(ms, vs, mq, vq, mo.view(B, K, To, -1), vo.view(B, K, To) & (b["o_len"] > 0).unsqueeze(-1), b)

    def decide(self, ms, vs, mq, vq, mo, vo, b):
        """position-free: options were encoded separately (no numbering, no order); slots carry no position; memory is a set."""
        B = ms.shape[0]; d = self.d; ar = torch.arange(B, device=ms.device); K, To = mo.shape[1], mo.shape[2]
        ans = mq[ar, b["q_len"] - 1] + self.styp.weight[0]
        last = (b["o_len"] - 1).clamp_min(0)
        opts = mo.gather(2, last.view(B, K, 1, 1).expand(-1, -1, 1, d)).squeeze(2) + self.styp.weight[1]
        qmean = (mq * vq.unsqueeze(-1)).sum(1) / vq.sum(1, keepdim=True)
        scr = self.scratch.unsqueeze(0) + self.qpool(qmean).unsqueeze(1) + self.styp.weight[2]
        s = torch.cat([ans.unsqueeze(1), opts, scr], 1)  # [B, 1+K+M, d]
        oval = b["o_len"] > 0
        sval = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ms.device), oval, torch.ones(B, self.M, dtype=torch.bool, device=ms.device)], 1)
        mem = torch.cat([ms, mq, mo.reshape(B, K * To, d)], 1); mval = torch.cat([vs, vq, vo.reshape(B, K * To)], 1)
        S = s.shape[1]
        mask = torch.cat([mval.unsqueeze(1).expand(-1, S, -1), sval.unsqueeze(1).expand(-1, S, -1)], 2).unsqueeze(1)
        P = b.get("mem_P") if self.ve is not None else None
        if P is not None and "slot_h" in b: P = dict(P, **binding(P, b["slot_h"]))
        mkv = [blk.mem_kv(mem) for blk in self.slots]
        return self.run_slots(s, mkv, mask, P, oval, K)

    def run_slots(self, s, mkv, mask, P, oval, K):
        outs = []; logits = None
        for r in range(self.R):
            if self.R > 1:
                s = s + self.rnd.weight[r]
                if logits is not None:
                    lg = logits.detach().float().masked_fill(~oval, -1e4)
                    p = torch.softmax(lg, -1); f = torch.stack([torch.tanh(lg / 5), p, torch.tanh((lg - lg.max(-1, keepdim=True).values) / 5)], -1)
                    s = torch.cat([s[:, :1], s[:, 1:1 + K] + self.bel(f.to(s.dtype)), s[:, 1 + K:]], 1)
            for blk, kv in zip(self.slots, mkv): s = blk(s, kv, mask, P)
            sn = self.nf(s)
            logits = self.head(sn[:, 0], sn[:, 1:1 + K]); outs.append(logits)
        return outs


def maybe_compile():
    import os
    if os.environ.get("J11_COMPILE", "0") == "1":
        Pointer.features = torch.compile(Pointer.features, dynamic=True)
        Pointer.logits_bias = torch.compile(Pointer.logits_bias, dynamic=True)
    if os.environ.get("J11_COMPILE_BLOCKS", "0") == "1":
        MLP.forward = torch.compile(MLP.forward, dynamic=True)
        RMS.forward = torch.compile(RMS.forward, dynamic=True)


maybe_compile()


def build(arm, V, d, L, **kw):
    """equal-parameter configs relative to a dec of L layers (12 d^2 per token layer; ~10.25 d^2 per slot layer)."""
    if arm == "dec": return DecModel(V, d, L)
    if arm == "decv": return DecModel(V, d, L, value=True)
    Ls = kw.get("Ls", max(2, L // 3))
    per_tok = 12 * d * d; per_slot = 2 * d * d + 2 * 2 * 2 * HD * d + 3 * ffn_hidden(d) * d
    Ld = kw.get("Ld", max(1, round((L - Ls) * per_tok / per_slot)))
    if arm == "slot": return SlotModel(V, d, Ls, Ld)
    if arm == "vslot": return SlotModel(V, d, Ls, Ld, value=True)
    if arm == "belief": return SlotModel(V, d, Ls, Ld, rounds=kw.get("R", 3))
    raise KeyError(arm)


# ------------------------------------------------------------------ analytic forward FLOPs (matmul + attention), per example
def fwd_flops(arm, d, L, Ts, Tq, K, Ls=None, Ld=None, R=3, M=4, compiled_q=False, n_kv=2, To=None, Tqp=None):
    """Tq = hobson-layout question length (dec); for slot arms Tqp = stem length and To = per-option lengths list."""
    h = ffn_hidden(d); tok = 2 * (4 * d * d + 3 * d * h)  # per token per token-layer
    n_kv = n_kv if (d // HD) % n_kv == 0 else 1
    if arm in ("dec", "decv"):
        T = Ts + Tq; return L * (tok * T + 2 * T * T * d) + 2 * (K + 1) * d * 256 * 2 + (2 * 201 * d * T if arm == "decv" else 0)
    Ls = Ls or max(2, L // 3)
    per_slot_par = 2 * d * d + 2 * 2 * n_kv * HD * d + 3 * h * d
    Ld = Ld or max(1, round((L - Ls) * 12 * d * d / per_slot_par))
    rounds = R if arm == "belief" else 1
    S = 1 + K + M; To = To if To is not None else [max(1, Tq // max(K, 1))] * K; Tqp = Tqp if Tqp is not None else Tq
    Tm = Ts + Tqp + K * max(To)
    f = Ls * (tok * Ts + 4 * Ts * Ts * d)
    if not compiled_q: f += Ls * (tok * (Tqp + sum(To)) + 4 * (Tqp * Tqp + sum(t * t for t in To)) * d)
    memkv = Ld * 2 * d * 2 * n_kv * HD * Tm
    per_round = Ld * (2 * S * (2 * d * d + 2 * n_kv * HD * d + 3 * d * h) + 4 * S * (Tm + S) * d)
    if arm == "vslot": per_round += (Ld // 2) * (2 * S * d * 4 * 64 + 4 * 4 * S * Tm * 64 + 2 * S * 322 * d); f += 2 * 201 * d * Tm
    return f + memkv + rounds * per_round + rounds * 2 * (K + 1) * d * 256 * 2
