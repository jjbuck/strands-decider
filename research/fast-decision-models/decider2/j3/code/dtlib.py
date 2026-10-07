"""J3 DT library: the depth-split decision transformer on hobson-v19 weights (H3/H7's differentiable lean forward, fla kernels).

Layout (hobson's own, state-first): rows = [state (Tm)] + one causal branch per question (each question's rows restart at position Tm).
  shallow layers 0..Ls-1 : hobson layers on ALL rows ('seqs' branches: GDN branch from the state's final GDN state, attention = state + own question)
  memory M               : the residual of every state row after layer Ls-1  [Tm, 2048]  (no pooling, no selection)
  deep layers Ls..23     : run on QUESTION rows only. State rows are never updated again. Each deep layer reads M through its OWN projections
                           applied to norm_j(M) (SwiftKV / YOCO pattern) + a trained memory adapter (LoRA on the memory projection rows):
     bridge 'A' : attention layers read memory K/V (state positions, RoPE kept); GDN layers recur over question rows only (zero initial state)
     bridge 'G' : as A, and GDN layers also scan M with their own k, v, beta, g; the question rows continue from that final GDN state (+ conv tail)
  head                   : hobson's pointer head on the final-normed option-end rows and '<answer>' row of each question.
Ls = 24 is exactly hobson (verified by dt_check.py).
"""
import os, sys, math
sys.path[:0] = [os.path.expanduser('~/work/j3')]
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import h3lib as H
from h3lib import rms_zc, DENSE, chunk_gated_delta_rule
from h7lib import H7, Br, _conv

KV = (4096, 5120)          # attention Win rows of k_proj (512) and v_proj (512)
FULL = (0, 8224)           # GDN Win rows (qkv 6144 | z 2048 | b 16 | a 16)


def rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


class DT(H7):
    mem = None
    mem_scale = 1.0

    # ---------------- memory adapters ----------------
    def add_mem(self, Ls_list, bridge='A', r=32, alpha=32, seed=1):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.mem = nn.ModuleDict(); self.mem_scale = alpha / r
        for Ls in Ls_list:
            for i in range(Ls, 24):
                gdn = self.L[i]['type'] == 'linear_attention'
                if gdn and bridge != 'G': continue
                r0, r1 = FULL if gdn else KV
                md = nn.Module()
                md.A = nn.Parameter((torch.randn(r, 2048, generator=g) / math.sqrt(2048)).to(self.dev))
                md.B = nn.Parameter(torch.zeros(r1 - r0, r, device=self.dev))
                self.mem[f'{Ls}_{i}'] = md
        return list(self.mem.parameters())

    def mem_lin(self, hm, i, Ls, rows):
        r0, r1 = rows
        y = hm @ self.L[i]['Win'][r0:r1].t()
        if self.lora is not None:
            lo = self.lora[i]['Win']
            y = y + ((hm @ lo.A.t().to(hm.dtype)) @ lo.B[r0:r1].t().to(hm.dtype)) * self.lora_scale
        key = f'{Ls}_{i}'
        if self.mem is not None and key in self.mem:
            ma = self.mem[key]
            y = y + ((hm @ ma.A.t().to(hm.dtype)) @ ma.B.t().to(hm.dtype)) * self.mem_scale
        return y

    # ---------------- layout ----------------
    def seqs_inputs(self, s, qs):
        Tm = len(s); ids = list(s); pos = list(range(Tm)); seqs = []
        for q in qs:
            seqs.append((len(ids), len(q))); ids += q; pos += list(range(Tm, Tm + len(q)))
        return ids, pos, Br('seqs', Tm, seqs=seqs)

    def deep_meta(self, seqs, Tm):
        """question rows only: conv gather index into [3 tail rows | R question rows], varlen cu_seqlens, attention mask [R, Tm + R]"""
        qseqs = [(s0 - Tm, Lb) for s0, Lb in seqs]
        R = sum(Lb for _, Lb in qseqs)
        gidx = []; cu = [0]
        for r0, Lb in qseqs:
            cu.append(cu[-1] + Lb)
            for t in range(Lb):
                gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (t + j) for j in range(4)])
        mask = torch.zeros(R, Tm + R, dtype=torch.bool, device=self.dev)
        mask[:, :Tm] = True
        for r0, Lb in qseqs:
            mask[r0:r0 + Lb, Tm + r0:Tm + r0 + Lb] = torch.tril(torch.ones(Lb, Lb, dtype=torch.bool, device=self.dev))
        return qseqs, torch.tensor(gidx, device=self.dev, dtype=torch.long), torch.tensor(cu, device=self.dev, dtype=torch.long), mask

    # ---------------- deep layer (question rows only) ----------------
    def layer_deep(self, i, xq, M, Ls, bridge, cos_m, sin_m, cos_q, sin_q, gidx, cu, mask, n):
        d = self.L[i]; R = xq.shape[0]; Tm = M.shape[0]; eps = self.eps
        h = self.bnorm(xq, i, 0)
        proj = self.lin(h, i, 'Win', DENSE)
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            init = None
            tail = torch.zeros(3, 6144, device=raw.device, dtype=raw.dtype)
            if bridge == 'G' and Tm > 0:
                hm = self.bnorm(M, i, 0)
                pm = self.mem_lin(hm, i, Ls, FULL)
                rawm = pm[:, :6144]
                betam = torch.sigmoid(pm[:, 8192:8208].float()); gm = -d['A_log'].float().exp() * F.softplus(pm[:, 8208:8224].float() + d['dt_bias'])
                cm = _conv(rawm[None], d['conv_w'])[0]
                qm, km, vm = cm.split(2048, dim=-1)
                _, S = chunk_gated_delta_rule(qm.reshape(1, Tm, 16, 128), km.reshape(1, Tm, 16, 128), vm.reshape(1, Tm, 16, 128), gm[None],
                                              betam[None].to(qm.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
                init = S.expand(n, -1, -1, -1).contiguous()
                tl_ = rawm[max(0, Tm - 3):Tm]
                tail = torch.cat([torch.zeros(3 - tl_.shape[0], 6144, device=raw.device, dtype=raw.dtype), tl_], 0) if tl_.shape[0] < 3 else tl_
            buf = torch.cat([tail, raw], 0)
            w = d['conv_w'].float()
            acc = sum(buf[gidx[:, j]].float() * w[:, j][None] for j in range(4))
            cs = F.silu(acc).to(raw.dtype)
            q, k, v = cs.split(2048, dim=-1)
            ob, _ = chunk_gated_delta_rule(q.reshape(1, R, 16, 128), k.reshape(1, R, 16, 128), v.reshape(1, R, 16, 128), g[None],
                                           beta[None].to(q.dtype), initial_state=init, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
            o = ob[0]
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(xq.dtype).reshape(R, 2048)
        else:
            qg = proj[:, :4096].reshape(R, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kq = proj[:, 4096:4608].reshape(R, 2, 256); vq = proj[:, 4608:5120].reshape(R, 2, 256)
            qh = rope(rms_zc(qh, d['qn'], eps), cos_q, sin_q); kq = rope(rms_zc(kq, d['kn'], eps), cos_q, sin_q)
            if Tm > 0:
                hm = self.bnorm(M, i, 0)
                pm = self.mem_lin(hm, i, Ls, KV)
                km = rope(rms_zc(pm[:, :512].reshape(Tm, 2, 256), d['kn'], eps), cos_m, sin_m); vm = pm[:, 512:].reshape(Tm, 2, 256)
                K = torch.cat([km, kq], 0); V = torch.cat([vm, vq], 0)
            else:
                K, V = kq, vq
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                               attn_mask=mask[None, None], enable_gqa=True)[0].transpose(0, 1)
            o = (o * torch.sigmoid(gate)).reshape(R, 2048)
        x = xq + self.lin(o, i, 'Wo', DENSE)
        h2 = self.bnorm(x, i, 1)
        gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
        return x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)

    # ---------------- forwards ----------------
    def shallow(self, ids, pos, br, Ls, ckpt=False):
        x = F.embedding(torch.as_tensor(ids, device=self.dev), self.embed)
        cos, sin = self.cos_sin(torch.as_tensor(pos, device=self.dev, dtype=torch.float32))
        for i in range(Ls):
            x = checkpoint(self.layer_br, i, x, cos, sin, br, use_reentrant=False) if (ckpt and torch.is_grad_enabled()) else self.layer_br(i, x, cos, sin, br)
        return x, cos, sin

    def dt_logits(self, s, qs, prs, Ls, bridge='A', ckpt=False, keep=(), head=None, x_shallow=None):
        """-> list of [n_slots] logits (temperature applied). self.kept[i] = question rows [R, 2048] after deep layer i (i in keep).
        x_shallow: optionally the (cos, sin, x) after layer Ls-1 computed elsewhere (shared with the teacher when the shallow stack is frozen)."""
        head = head or self.head or self.head0
        ids, pos, br = self.seqs_inputs(s, qs)
        Tm = br.Tm
        if x_shallow is None:
            x, cos, sin = self.shallow(ids, pos, br, Ls, ckpt)
        else:
            x, cos, sin = x_shallow
        M = x[:Tm]; xq = x[Tm:]
        qseqs, gidx, cu, mask = self.deep_meta(br.seqs, Tm)
        n = len(qs)
        self.kept = {}
        for i in range(Ls, 24):
            args = (i, xq, M, Ls, bridge, cos[:Tm], sin[:Tm], cos[Tm:], sin[Tm:], gidx, cu, mask, n)
            xq = checkpoint(self.layer_deep, *args, use_reentrant=False) if (ckpt and torch.is_grad_enabled() and xq.shape[0] > 256) else self.layer_deep(*args)
            if i in keep: self.kept[i] = xq
        h = rms_zc(xq, self.norm_w, self.eps)
        out = []
        for (r0, Lb), pr in zip(qseqs, prs):
            oi = torch.tensor([r0 + o for o in pr['opt']], device=self.dev)
            lg = head(h[r0 + Lb - 1].float()[None], h[oi].float()[None])[0] / self.temp(pr['rq'].kind)
            out.append(lg[:pr['rq'].n_slots])
        return out

    def teacher_logits(self, s, qs, prs, keep=(), split=None):
        """hobson state-first (call inside a no-LoRA context). self.kept[i] = all question rows after layer i (i in keep).
        split: also return (x, cos, sin) after layer split-1 (to share a frozen shallow stack with the student)."""
        ids, pos, br = self.seqs_inputs(s, qs)
        Tm = br.Tm
        x = F.embedding(torch.as_tensor(ids, device=self.dev), self.embed)
        cos, sin = self.cos_sin(torch.as_tensor(pos, device=self.dev, dtype=torch.float32))
        self.kept = {}; shared = None
        for i in range(24):
            if split is not None and i == split: shared = (x, cos, sin)
            x = self.layer_br(i, x, cos, sin, br)
            if i in keep: self.kept[i] = x[Tm:]
        h = rms_zc(x[Tm:], self.norm_w, self.eps)
        out = []
        for (s0, Lb), pr in zip(br.seqs, prs):
            r0 = s0 - Tm
            oi = torch.tensor([r0 + o for o in pr['opt']], device=self.dev)
            lg = self.head0(h[r0 + Lb - 1].float()[None], h[oi].float()[None])[0] / self.temp(pr['rq'].kind)
            out.append(lg[:pr['rq'].n_slots])
        return (out, shared) if split is not None else out

    # ---------------- io ----------------
    def dt_state(self, meta):
        sd = self.trainable_state()
        if self.mem is not None:
            for k, md in self.mem.items():
                sd[f'mem.{k}.A'] = md.A.detach().cpu(); sd[f'mem.{k}.B'] = md.B.detach().cpu()
        sd['_meta'] = dict(sd['_meta'], mem_scale=self.mem_scale, **meta)
        return sd

    def dt_load(self, path):
        sd = torch.load(os.path.expanduser(path), map_location=self.dev)
        meta = self.load_trainable(path)
        mk = sorted({k.split('.')[1] for k in sd if k.startswith('mem.')})
        if mk:
            self.mem = nn.ModuleDict(); self.mem_scale = sd['_meta'].get('mem_scale', 1.0)
            for k in mk:
                md = nn.Module(); md.A = nn.Parameter(sd[f'mem.{k}.A'].to(self.dev), requires_grad=False); md.B = nn.Parameter(sd[f'mem.{k}.B'].to(self.dev), requires_grad=False)
                self.mem[k] = md
        return sd['_meta']
