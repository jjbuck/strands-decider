"""J6: the question in the weights.

hobson-v19 (LoRA merged; H3's differentiable lean forward, fla kernels) split into
  * state_cache(s): the state rows, read by the BASE network (question-blind, exactly hobson's state rows); returns per-layer cache
    (GDN final state S + conv tail; attention K (normed, roped) / V).
  * branch(x, seg, cache, ad): R branch rows that continue the cached state. Rows are grouped in segments; a segment is causal within itself,
    sees all state rows, and its GDN starts from the state's final state (segments never see each other).
      teacher  = hobson in context: one segment per question = the question's full token list (no adapter)
      student  = question in the weights: one segment per question = K option-slot rows + 1 answer row (learned/generated input vectors),
                 with that question's low-rank weight delta applied to those rows (Win, Wo, Wd of all 24 layers).
  * full(x, ad): variant A (question from layer 0): the whole [state][slot set] sequence with the adapter on EVERY row.
LoRA targets Win (fused q/k/v/z/b/a or q+gate/k/v), Wo and Wd: these are the GEMMs whose epilogues the fused runtime can take a delta in
(Wgu's SwiGLU epilogue cannot), so the trained function is exactly what the fused runtime serves.
"""
import os, sys, math
sys.path[:0] = [os.path.expanduser('~/work/j6')]
import torch, torch.nn as nn, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, fla_conv, chunk_gated_delta_rule, StdHead

MODS = ('Win', 'Wo', 'Wd')
KEEP = (5, 11, 17, 23)


def _conv(x, w):
    y = fla_conv(x.contiguous(), w, None, activation='silu')
    return y[0] if isinstance(y, tuple) else y


class Seg:
    """branch layout over one or more cached states. Segment j (rows r0[j]..r0[j]+lens[j]) continues request req[j]'s state (Ls[req[j]] rows):
    causal within itself, sees all of its own request's state rows; its GDN starts from that state's final state, conv history = that
    state's last 3 rows. Rows of all segments are concatenated in order."""

    def __init__(self, Ls, lens, dev, req=None):
        Ls = [Ls] if isinstance(Ls, int) else list(Ls)
        req = req if req is not None else [0] * len(lens)
        self.Ls = Ls; self.lens = list(lens); self.n = len(lens); self.req = list(req); nr = len(Ls)
        r0 = [0]
        for L in lens: r0.append(r0[-1] + L)
        self.r0 = r0; self.R = r0[-1]
        koff = [0]
        for L in Ls: koff.append(koff[-1] + L)
        self.koff = koff; self.KS = koff[-1]
        self.cu = torch.tensor(r0, device=dev, dtype=torch.long)
        self.sidx = torch.tensor(self.req, device=dev, dtype=torch.long)
        g = []
        for j, L in enumerate(lens):
            r = self.req[j]
            for t in range(L):
                g.append([(3 * nr + r0[j] + t - 3 + k) if t - 3 + k >= 0 else (3 * r + 3 + t - 3 + k) for k in range(4)])
        self.gidx = torch.tensor(g, device=dev, dtype=torch.long)
        m = torch.zeros(self.R, self.KS + self.R, dtype=torch.bool, device=dev)
        pos = []
        for j, L in enumerate(lens):
            a = r0[j]; r = self.req[j]
            m[a:a + L, koff[r]:koff[r + 1]] = True
            m[a:a + L, self.KS + a:self.KS + a + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
            pos += list(range(Ls[r], Ls[r] + L))
        self.mask = m
        self.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        Lm = max(lens); self.Lm = Lm
        pad = torch.full((self.n, Lm), self.R, dtype=torch.long)
        unpad = []
        for j, L in enumerate(lens):
            pad[j, :L] = torch.arange(r0[j], r0[j] + L)
            unpad += list(range(j * Lm, j * Lm + L))
        self.pad = pad.to(dev); self.unpad = torch.tensor(unpad, device=dev, dtype=torch.long)


def merge_caches(caches):
    """per-request caches -> one cache whose GDN S / tails are stacked per request and attention K / V concatenated"""
    out = []
    for i in range(len(caches[0])):
        c0 = caches[0][i]
        if 'S' in c0:
            out.append(dict(S=torch.cat([c[i]['S'] for c in caches], 0), tail=torch.cat([c[i]['tail'] for c in caches], 0)))
        else:
            out.append(dict(k=torch.cat([c[i]['k'] for c in caches], 0), v=torch.cat([c[i]['v'] for c in caches], 0)))
    return out


class J6(H.H3):
    def rope_tab(self, pos):
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    @staticmethod
    def _rope(t, cos, sin):
        xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
        c = cos[:, None, :]; s_ = sin[:, None, :]
        return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)

    def _gnorm(self, o, z, d, T):
        of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(T, 2048)

    def _mlp(self, x, i, ad, part):
        d = self.L[i]
        h2 = self.bnorm(x, i, 1)
        gu = h2 @ d['Wgu'].t(); I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        y = m @ d['Wd'].t()
        if ad is not None: y = y + ad(i, 'Wd', m, part)
        return x + y

    # ---------------- state rows (base network, no grad) ----------------
    @torch.no_grad()
    def state_cache(self, s, keep=()):
        """keep: layers whose outputs (all rows) are returned too -> (cache, {layer: [T, 2048]})"""
        dev = self.dev; T = len(s); kept = {}
        x = F.embedding(torch.as_tensor(s, device=dev), self.embed)
        cos, sin = self.rope_tab(torch.arange(T, device=dev, dtype=torch.float32))
        cache = []
        for i in range(24):
            d = self.L[i]
            h = self.bnorm(x, i, 0)
            proj = h @ d['Win'].t()
            if d['type'] == 'linear_attention':
                raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
                beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
                cm = _conv(raw[None], d['conv_w'])[0]
                q, k, v = cm.split(2048, dim=-1)
                o, S = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                              beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
                tail = raw[max(0, T - 3):T]
                if tail.shape[0] < 3: tail = torch.cat([torch.zeros(3 - tail.shape[0], 6144, device=dev, dtype=raw.dtype), tail], 0)
                cache.append(dict(S=S, tail=tail.contiguous()))
                o = self._gnorm(o[0], z, d, T)
            else:
                qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
                kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = self._rope(rms_zc(qh, d['qn'], self.eps), cos, sin); kk = self._rope(rms_zc(kk, d['kn'], self.eps), cos, sin)
                cache.append(dict(k=kk.contiguous(), v=v.contiguous()))
                o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                                   is_causal=True, enable_gqa=True)[0].transpose(0, 1)
                o = (o * torch.sigmoid(gate)).reshape(T, 2048)
            x = x + o @ d['Wo'].t()
            x = self._mlp(x, i, None, None)
            if i in keep: kept[i] = x
        return (cache, kept) if keep else cache

    # ---------------- branch rows (teacher question tokens, or student slot rows with adapters) ----------------
    def branch(self, x, seg, cache, ad=None, keep=(), krows=None):
        R = seg.R; n = seg.n
        cos, sin = self.rope_tab(seg.pos)
        kept = {}
        for i in range(24):
            d = self.L[i]; c = cache[i]
            h = self.bnorm(x, i, 0)
            proj = h @ d['Win'].t()
            if ad is not None: proj = proj + ad(i, 'Win', h, 'b')
            if d['type'] == 'linear_attention':
                raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
                beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
                buf = torch.cat([c['tail'], raw], 0)
                w = d['conv_w'].float()
                acc = sum(buf[seg.gidx[:, j]].float() * w[:, j][None] for j in range(4))
                cs = F.silu(acc).to(raw.dtype)
                q, k, v = cs.split(2048, dim=-1)
                o, _ = chunk_gated_delta_rule(q.reshape(1, R, 16, 128), k.reshape(1, R, 16, 128), v.reshape(1, R, 16, 128), g[None],
                                              beta[None].to(q.dtype), initial_state=c['S'].index_select(0, seg.sidx).contiguous(),
                                              use_qk_l2norm_in_kernel=True, cu_seqlens=seg.cu)
                o = self._gnorm(o[0], z, d, R)
            else:
                qg = proj[:, :4096].reshape(R, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
                kk = proj[:, 4096:4608].reshape(R, 2, 256); v = proj[:, 4608:5120].reshape(R, 2, 256)
                qh = self._rope(rms_zc(qh, d['qn'], self.eps), cos, sin); kk = self._rope(rms_zc(kk, d['kn'], self.eps), cos, sin)
                K = torch.cat([c['k'], kk], 0); V = torch.cat([c['v'], v], 0)
                o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                                   attn_mask=seg.mask[None, None], enable_gqa=True)[0].transpose(0, 1)
                o = (o * torch.sigmoid(gate)).reshape(R, 2048)
            y = o @ d['Wo'].t()
            if ad is not None: y = y + ad(i, 'Wo', o, 'b')
            x = x + y
            x = self._mlp(x, i, ad, 'b')
            if i in keep: kept[i] = x[krows]
        return rms_zc(x, self.norm_w, self.eps), kept

    # ---------------- variant A: one sequence [state][slot set] with the adapter on every row ----------------
    def full(self, s, xslot, ad, ckpt=True):
        """s: state ids (Ls); xslot [L, 2048] slot-set inputs; ad(i, nm, h, 'f') applies to all rows. -> final normed hidden of the slot rows"""
        dev = self.dev; Ls = len(s); L = xslot.shape[0]; T = Ls + L
        x = torch.cat([F.embedding(torch.as_tensor(s, device=dev), self.embed), xslot.to(torch.bfloat16)], 0)
        cos, sin = self.rope_tab(torch.arange(T, device=dev, dtype=torch.float32))
        from torch.utils.checkpoint import checkpoint
        for i in range(24):
            x = checkpoint(self._layer_full, i, x, cos, sin, ad, use_reentrant=False) if ckpt else self._layer_full(i, x, cos, sin, ad)
        return rms_zc(x[Ls:], self.norm_w, self.eps)

    def _layer_full(self, i, x, cos, sin, ad):
        d = self.L[i]; T = x.shape[0]
        h = self.bnorm(x, i, 0)
        proj = h @ d['Win'].t() + ad(i, 'Win', h, 'f')
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            cm = _conv(raw[None], d['conv_w'])[0]
            q, k, v = cm.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                          beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
            o = self._gnorm(o[0], z, d, T)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = self._rope(rms_zc(qh, d['qn'], self.eps), cos, sin); kk = self._rope(rms_zc(kk, d['kn'], self.eps), cos, sin)
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                               is_causal=True, enable_gqa=True)[0].transpose(0, 1)
            o = (o * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + o @ d['Wo'].t() + ad(i, 'Wo', o, 'f')
        return self._mlp(x, i, ad, 'f')

    def head_logits(self, head, h_ans, h_opt, kind):
        return head(h_ans.float()[None], h_opt.float()[None])[0] / self.temp(kind)


# ---------------- adapters ----------------
def lora_pair(out, inp, r, g, dev):
    A = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(dev))
    B = nn.Parameter(torch.zeros(out, r, device=dev))
    return A, B


class QAdapters(nn.Module):
    """(a) a shared LoRA (all questions) + one trained LoRA per deployed question + per-question slot-input vectors.
    delta(i, nm, h, part) applies, per segment, shared + that segment's question LoRA. Set `self.cur` = list of question indices (one per segment)
    and `self.seg` before a branch call. part 'b' = branch rows; 'f' = variant A full sequence (state rows also get the adapter: `state_ad`)."""

    def __init__(self, m, qnames, slot_init, r_s=16, r_q=8, alpha_s=32, alpha_q=16, seed=0):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed); dev = m.dev
        self.qnames = list(qnames); self.qi = {q: j for j, q in enumerate(self.qnames)}
        self.sA = nn.ParameterDict(); self.sB = nn.ParameterDict()
        self.qA = nn.ModuleList(); self.qB = nn.ModuleList()
        for i in range(24):
            for nm in MODS:
                out, inp = m.L[i][nm].shape
                A, B = lora_pair(out, inp, r_s, g, dev); self.sA[f'{i}_{nm}'] = A; self.sB[f'{i}_{nm}'] = B
        for j, q in enumerate(self.qnames):
            pa = nn.ParameterDict(); pb = nn.ParameterDict()
            for i in range(24):
                for nm in MODS:
                    out, inp = m.L[i][nm].shape
                    A, B = lora_pair(out, inp, r_q, g, dev); pa[f'{i}_{nm}'] = A; pb[f'{i}_{nm}'] = B
            self.qA.append(pa); self.qB.append(pb)
        self.slots = nn.ParameterList([nn.Parameter(slot_init[q].float().clone().to(dev)) for q in self.qnames])
        self.ss = alpha_s / r_s; self.sq = alpha_q / r_q
        assert abs(self.ss - self.sq) < 1e-9
        self.Lslot = 0
        self.cur = None; self.seg = None
        self.state_ad = None      # variant A: optional extra adapter for state rows

    def q_params(self, j):
        return list(self.qA[j].parameters()) + list(self.qB[j].parameters()) + [self.slots[j]]

    def shared_params(self):
        return list(self.sA.parameters()) + list(self.sB.parameters())

    def slot_inputs(self, qlist):
        return torch.cat([self.slots[self.qi[q]] for q in qlist], 0)

    def __call__(self, i, nm, h, part):
        """shared + per-question LoRA, batched over segments (S-LoRA / Punica-style: rows padded per segment, one bmm per factor).
        shared and per-question factors are concatenated along the rank (both scales are alpha / r = 2)."""
        k = f'{i}_{nm}'; hd = h.dtype
        if part == 'b':
            seg = self.seg; n = seg.n
            A = torch.cat([self.sA[k][None].expand(n, -1, -1), torch.stack([self.qA[q][k] for q in self.cur])], 1).to(hd)   # [n, r, in]
            B = torch.cat([self.sB[k][None].expand(n, -1, -1), torch.stack([self.qB[q][k] for q in self.cur])], 2).to(hd)   # [n, out, r]
            hp = torch.cat([h, h.new_zeros(1, h.shape[1])], 0)[seg.pad]                                                       # [n, Lm, in]
            y = torch.bmm(torch.bmm(hp, A.transpose(1, 2)), B.transpose(1, 2))                                                 # [n, Lm, out]
            return y.reshape(n * seg.Lm, -1)[seg.unpad] * self.ss
        if part == 'f':
            q = self.cur[0]
            A = torch.cat([self.sA[k], self.qA[q][k]], 0).to(hd); B = torch.cat([self.sB[k], self.qB[q][k]], 1).to(hd)
            y = ((h[self.Lslot:] @ A.t()) @ B.t()) * self.ss
            y = torch.cat([h.new_zeros(self.Lslot, y.shape[1]), y], 0) if self.state_ad is None else torch.cat([self.state_ad(i, nm, h[:self.Lslot]), y], 0)
            return y
        raise ValueError(part)

    def state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}


# ---------------- fused state pass (d1/H4 runtime) ----------------
class FastState:
    """state rows through the fused bf16 runtime (tt_lean.TTL = d1 lean2): same cache layout as J6.state_cache (validated: teacher answers
    within |dp| <= 0.004 of the unfused path). Build BEFORE m.detach_inference() (it needs the HF torso)."""

    def __init__(self, m):
        sys.path.insert(0, os.path.expanduser('~/work/h4'))
        from tt_lean import TTL
        self.rt = TTL(m.p.tm, list(range(10))); self.dev = m.dev

    @torch.no_grad()
    def __call__(self, s):
        self.rt.set_fuse(len(s))
        _, c = self.rt.fwd(torch.tensor([s], device=self.dev), want_cache=True)
        out = []
        for i in range(24):
            if 'S' in c[i]: out.append(dict(S=c[i]['S'], tail=c[i]['tail'][:, :6144].contiguous()))
            else: out.append(dict(k=c[i]['k'], v=c[i]['v']))
        return out


class HyperAdapters(nn.Module):
    """(b) hypernetwork: question spec -> low-rank weight delta for the slot rows + slot-input vectors.
    Encoder: base hobson (frozen) reads the rendered question ALONE (no state) once per question spec; features = its layer-5/11/17/23 outputs at the
    '<answer>' row and their mean over the question rows. z = MLP(features). Per layer l and module m: delta W = U_lm C_lm(z) V_lm (C: r_h x r_h,
    generated, zero-init) on top of a shared LoRA (r16). Slot inputs: option k = mean embedding of its option line + Wo(LN(h_opt_k at layer 23)),
    answer = '<answer>' embedding + Wa(LN(h_ans at layer 23)) (Wo, Wa zero-init). At deployment the generated delta is an ordinary per-question
    LoRA (A = C V, B = U), so serving is identical to (a)."""
    FEAT = (5, 11, 17, 23)

    def __init__(self, m, r_s=16, r_h=24, zdim=1024, seed=0):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed); dev = m.dev
        self.sA = nn.ParameterDict(); self.sB = nn.ParameterDict(); self.U = nn.ParameterDict(); self.V = nn.ParameterDict()
        for i in range(24):
            for nm in MODS:
                out, inp = m.L[i][nm].shape
                A, B = lora_pair(out, inp, r_s, g, dev); self.sA[f'{i}_{nm}'] = A; self.sB[f'{i}_{nm}'] = B
                self.V[f'{i}_{nm}'] = nn.Parameter((torch.randn(r_h, inp, generator=g) / math.sqrt(inp)).to(dev))
                self.U[f'{i}_{nm}'] = nn.Parameter((torch.randn(out, r_h, generator=g) * 0.02).to(dev))
        nf = 2 * len(self.FEAT) * 2048
        self.enc = nn.Sequential(nn.LayerNorm(nf), nn.Linear(nf, zdim), nn.GELU(), nn.Linear(zdim, zdim), nn.GELU()).to(dev)
        self.gen = nn.Linear(zdim, 24 * len(MODS) * r_h * r_h).to(dev)
        nn.init.zeros_(self.gen.weight); nn.init.zeros_(self.gen.bias)
        self.ln_o = nn.LayerNorm(2048).to(dev); self.Wo = nn.Linear(2048, 2048, bias=False).to(dev); nn.init.zeros_(self.Wo.weight)
        self.ln_a = nn.LayerNorm(2048).to(dev); self.Wa = nn.Linear(2048, 2048, bias=False).to(dev); nn.init.zeros_(self.Wa.weight)
        self.r_h = r_h; self.ss = 2.0; self.sh = 1.0
        self.cur = None; self.seg = None

    @torch.no_grad()
    def features(self, m, pq, s0):
        """frozen part (computed once per question spec): base hobson reads [empty state][question] (= hobson's no-state reading)"""
        _, kept = m.state_cache(list(s0) + list(pq['q']), keep=self.FEAT)
        o = len(s0)
        ans = torch.cat([kept[i][-1] for i in self.FEAT]); mean = torch.cat([kept[i][o:].float().mean(0).to(kept[i].dtype) for i in self.FEAT])
        return dict(f=torch.cat([ans, mean]).float(), hopt=kept[23][torch.tensor([o + x for x in pq['opt']], device=m.dev)].float(), hans=kept[23][-1].float())

    def generate(self, feat, sinit):
        """-> code dict: C [24*3, r_h, r_h] and slot inputs [K+1, 2048]"""
        z = self.enc(feat['f'][None])[0]
        C = self.gen(z).reshape(24 * len(MODS), self.r_h, self.r_h)
        slots = torch.cat([sinit[:-1] + self.Wo(self.ln_o(feat['hopt'])), (sinit[-1] + self.Wa(self.ln_a(feat['hans'][None]))[0])[None]], 0)
        return dict(C=C, slots=slots)

    def __call__(self, i, nm, h, part):
        k = f'{i}_{nm}'; hd = h.dtype; seg = self.seg; n = seg.n
        y = ((h @ self.sA[k].t().to(hd)) @ self.sB[k].t().to(hd)) * self.ss
        li = i * len(MODS) + MODS.index(nm)
        C = torch.stack([c['C'][li] for c in self.cur]).to(hd)                       # [n, r_h, r_h]
        hp = torch.cat([h, h.new_zeros(1, h.shape[1])], 0)[seg.pad]                  # [n, Lm, in]
        t = torch.bmm(hp @ self.V[k].t().to(hd), C.transpose(1, 2))                  # [n, Lm, r_h]
        yh = (t @ self.U[k].t().to(hd)).reshape(n * seg.Lm, -1)[seg.unpad]
        return y + yh * self.sh


class QStateAd(nn.Module):
    def __init__(self, m, nq, r=8, seed=1):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.A = nn.ModuleList(); self.B = nn.ModuleList()
        for j in range(nq):
            pa = nn.ParameterDict(); pb = nn.ParameterDict()
            for i in range(24):
                for nm in MODS:
                    out, inp = m.L[i][nm].shape
                    A, B = lora_pair(out, inp, r, g, m.dev); pa[f'{i}_{nm}'] = A; pb[f'{i}_{nm}'] = B
            self.A.append(pa); self.B.append(pb)
        self.sc = 2.0; self.cur = 0

    def __call__(self, i, nm, h):
        k = f'{i}_{nm}'; hd = h.dtype
        return ((h @ self.A[self.cur][k].t().to(hd)) @ self.B[self.cur][k].t().to(hd)) * self.sc
