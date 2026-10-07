"""H7 library: hobson-v19 (H3's differentiable lean forward: LoRA-merged hobson weights, fla kernels) with BRANCHED layouts.

A forward = one MAIN causal sequence (positions 0..Tm-1) plus branches that continue it from position Tm:
  * kind 'slots' (student, schema-first): main = [Q1' Q2' .. Qn'][state]  (Qk' = question k's tokens minus its final '<answer>' token);
    n one-token branches = the '<answer>' slot of each question. Every slot starts from the main sequence's final GDN state and conv tail
    (so slots never see each other), and at attention layers slot k attends to its OWN question's bundle span + all state rows + itself
    (per-question mask over the bundle). Options are read from the bundle rows (compiled once per deployment).
    With n = 1 this is exactly H6's layout [question minus '<answer>'][state]['<answer>'].
  * kind 'seqs' (teacher, hobson state-first): main = [state]; branch k = question k's full token list, causal within itself and seeing all
    of the state = hobson's own shared-prefix layout. (no-grad use only)
"""
import os, sys, math
sys.path[:0] = [os.path.expanduser('~/work/m2')]
import torch, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, DENSE, fla_conv, chunk_gated_delta_rule


def _conv(x, w):
    y = fla_conv(x.contiguous(), w, None, activation='silu')
    return y[0] if isinstance(y, tuple) else y


class Br:
    def __init__(self, kind, Tm, n=0, allow=None, seqs=None):
        self.kind = kind; self.Tm = Tm; self.n = n; self.allow = allow; self.seqs = seqs or []


class H7(H.H3):
    def cos_sin(self, pos):
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    def layer_br(self, i, x, cos, sin, br):
        d = self.L[i]; T = x.shape[0]; eps = self.eps; Tm = br.Tm
        h = self.bnorm(x, i, 0)
        proj = self.lin(h, i, 'Win', DENSE)
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            cm = _conv(raw[:Tm][None], d['conv_w'])[0]
            q, k, v = cm.split(2048, dim=-1)
            om, S = chunk_gated_delta_rule(q.reshape(1, Tm, 16, 128), k.reshape(1, Tm, 16, 128), v.reshape(1, Tm, 16, 128), g[:Tm][None],
                                           beta[:Tm][None].to(q.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
            outs = [om[0]]
            tail = raw[max(0, Tm - 3):Tm]
            if tail.shape[0] < 3: tail = torch.cat([torch.zeros(3 - tail.shape[0], 6144, device=raw.device, dtype=raw.dtype), tail], 0)
            if br.kind == 'slots' and br.n:
                n = br.n
                ext = torch.cat([tail[None].expand(n, 3, 6144), raw[Tm:Tm + n][:, None]], 1)
                cs = _conv(ext, d['conv_w'])[:, 3:]                                   # [n, 1, 6144]
                q, k, v = cs.split(2048, dim=-1)
                ob, _ = chunk_gated_delta_rule(q.reshape(n, 1, 16, 128), k.reshape(n, 1, 16, 128), v.reshape(n, 1, 16, 128), g[Tm:Tm + n][:, None],
                                               beta[Tm:Tm + n][:, None].to(q.dtype), initial_state=S.expand(n, -1, -1, -1).contiguous(),
                                               use_qk_l2norm_in_kernel=True)
                outs.append(ob[:, 0])
            elif br.kind == 'sets':
                buf = torch.cat([tail, raw[Tm:]], 0)                                   # [3 + R, 6144]
                w = d['conv_w'].float()
                acc = sum(buf[br.gidx[:, j]].float() * w[:, j][None] for j in range(4))
                cs = F.silu(acc).to(raw.dtype)
                R = cs.shape[0]
                q, k, v = cs.split(2048, dim=-1)
                ob, _ = chunk_gated_delta_rule(q.reshape(1, R, 16, 128), k.reshape(1, R, 16, 128), v.reshape(1, R, 16, 128), g[Tm:][None],
                                               beta[Tm:][None].to(q.dtype), initial_state=S.expand(br.n, -1, -1, -1).contiguous(),
                                               use_qk_l2norm_in_kernel=True, cu_seqlens=br.cu)
                outs.append(ob[0])
            elif br.kind == 'seqs':
                for s0, Lb in br.seqs:
                    cs = _conv(torch.cat([tail, raw[s0:s0 + Lb]], 0)[None], d['conv_w'])[0, 3:]
                    q, k, v = cs.split(2048, dim=-1)
                    ob, _ = chunk_gated_delta_rule(q.reshape(1, Lb, 16, 128), k.reshape(1, Lb, 16, 128), v.reshape(1, Lb, 16, 128), g[s0:s0 + Lb][None],
                                                   beta[s0:s0 + Lb][None].to(q.dtype), initial_state=S, use_qk_l2norm_in_kernel=True)
                    outs.append(ob[0])
            o = torch.cat(outs, 0)
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)

            def rope(t):
                xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
                c = cos[:, None, :]; s_ = sin[:, None, :]
                return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            qh, kk = rope(qh), rope(kk)
            om = F.scaled_dot_product_attention(qh[:Tm].transpose(0, 1)[None], kk[:Tm].transpose(0, 1)[None], v[:Tm].transpose(0, 1)[None],
                                                is_causal=True, enable_gqa=True)[0].transpose(0, 1)       # [Tm, 8, 256]
            outs = [om]
            sc = 256 ** -0.5
            if br.kind == 'slots' and br.n:
                n = br.n
                qs = qh[Tm:Tm + n].float().reshape(n, 2, 4, 256)
                lm = torch.einsum('nghd,tgd->nght', qs, kk[:Tm].float()) * sc
                lm = lm.masked_fill(~br.allow[:, None, None, :], float('-inf'))
                ks = kk[Tm:Tm + n].float().reshape(n, 2, 1, 256); vs = v[Tm:Tm + n].float().reshape(n, 2, 1, 256)
                ls = (qs * ks).sum(-1, keepdim=True) * sc
                p = torch.softmax(torch.cat([lm, ls], -1), -1)
                ob = torch.einsum('nght,tgd->nghd', p[..., :Tm], v[:Tm].float()) + p[..., Tm:] * vs
                outs.append(ob.reshape(n, 8, 256).to(om.dtype))
            elif br.kind == 'sets':
                ob = F.scaled_dot_product_attention(qh[Tm:].transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                                    attn_mask=br.mask[None, None], enable_gqa=True)[0].transpose(0, 1)
                outs.append(ob)
            elif br.kind == 'seqs':
                for s0, Lb in br.seqs:
                    K = torch.cat([kk[:Tm], kk[s0:s0 + Lb]], 0); V = torch.cat([v[:Tm], v[s0:s0 + Lb]], 0)
                    ii = torch.arange(Lb, device=x.device)[:, None]; jj = torch.arange(Tm + Lb, device=x.device)[None, :]
                    mask = (jj < Tm) | ((jj - Tm) <= ii)
                    ob = F.scaled_dot_product_attention(qh[s0:s0 + Lb].transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                                        attn_mask=mask[None, None], enable_gqa=True)[0].transpose(0, 1)
                    outs.append(ob)
            o = (torch.cat(outs, 0) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(o, i, 'Wo', DENSE)
        h2 = self.bnorm(x, i, 1)
        gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
        x = x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)
        return x

    def fwd_br(self, ids, pos, br, ckpt=False, keep=(), keep_rows=None):
        ids_t = torch.as_tensor(ids, device=self.dev)
        x = F.embedding(ids_t, self.embed)
        cos, sin = self.cos_sin(torch.as_tensor(pos, device=self.dev, dtype=torch.float32))
        self.kept = {}
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer_br, i, x, cos, sin, br, use_reentrant=False)
            else:
                x = self.layer_br(i, x, cos, sin, br)
            if i in keep: self.kept[i] = x[keep_rows]
        return rms_zc(x, self.norm_w, self.eps)

    # ---------- layouts ----------
    def schema_inputs(self, s, qs):
        """s: state token ids; qs: list of question token lists (each ends with the '<answer>' token).
        -> ids, pos, Br('slots'), offs (bundle offset of each question)"""
        bundle = []; offs = []; spans = []
        for q in qs:
            offs.append(len(bundle)); spans.append((len(bundle), len(bundle) + len(q) - 1)); bundle += q[:-1]
        P = len(bundle); Tm = P + len(s); n = len(qs)
        ids = bundle + list(s) + [q[-1] for q in qs]
        pos = list(range(Tm)) + [Tm] * n
        allow = torch.zeros(n, Tm, dtype=torch.bool, device=self.dev)
        for j, (a0, a1) in enumerate(spans):
            allow[j, a0:a1] = True
        allow[:, P:] = True
        return ids, pos, Br('slots', Tm, n=n, allow=allow), offs

    def schema_logits(self, s, qs, prs, ckpt=False, head=None):
        """-> list of [n_slots] logits (temperature applied), one per question"""
        head = head or self.head or self.head0
        ids, pos, br, offs = self.schema_inputs(s, qs)
        h = self.fwd_br(ids, pos, br, ckpt=ckpt)
        out = []
        for j, pr in enumerate(prs):
            oi = torch.tensor([offs[j] + o for o in pr['opt']], device=self.dev)
            lg = head(h[br.Tm + j].float()[None], h[oi].float()[None])[0] / self.temp(pr['rq'].kind)
            out.append(lg[:pr['rq'].n_slots])
        return out

    def sets_inputs(self, s, qs, prs):
        """slot SET per question = [its option-end tokens .., '<answer>'] re-emitted after the state (state-aware option rows for the pointer head).
        set k rows: positions Tm.., causal within the set, attend to question k's bundle span + all state rows; GDN = a branch from the state's final
        state (conv history = last 3 state rows). -> ids, pos, Br('sets'), list of (row0, L) per set (absolute rows)"""
        bundle = []; spans = []
        for q in qs:
            spans.append((len(bundle), len(bundle) + len(q) - 1)); bundle += q[:-1]
        P = len(bundle); Tm = P + len(s); n = len(qs)
        ids = bundle + list(s); pos = list(range(Tm)); sets = []; gidx = []; cu = [0]
        for q, pr in zip(qs, prs):
            toks = [q[o] for o in pr['opt']] + [q[-1]]; L = len(toks); r0 = len(ids) - Tm
            sets.append((len(ids), L)); ids += toks; pos += list(range(Tm, Tm + L)); cu.append(cu[-1] + L)
            for t in range(L):
                gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (3 + t - 3 + j) for j in range(4)])
        R = len(ids) - Tm
        mask = torch.zeros(R, Tm + R, dtype=torch.bool, device=self.dev)
        for j, ((a0, a1), (r_abs, L)) in enumerate(zip(spans, sets)):
            r = r_abs - Tm
            mask[r:r + L, a0:a1] = True; mask[r:r + L, P:Tm] = True
            mask[r:r + L, Tm + r:Tm + r + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=self.dev))
        br = Br('sets', Tm, n=n)
        br.mask = mask; br.gidx = torch.tensor(gidx, device=self.dev, dtype=torch.long); br.cu = torch.tensor(cu, device=self.dev, dtype=torch.long)
        return ids, pos, br, sets

    def sets_logits(self, s, qs, prs, ckpt=False, head=None, keep=()):
        """keep: layers whose outputs at the slot-set rows are left in self.kept[i] ([R, 2048], set order: option rows then '<answer>')"""
        head = head or self.head or self.head0
        ids, pos, br, sets = self.sets_inputs(s, qs, prs)
        h = self.fwd_br(ids, pos, br, ckpt=ckpt, keep=keep, keep_rows=torch.arange(br.Tm, len(ids), device=self.dev))
        out = []
        for (r0, L), pr in zip(sets, prs):
            lg = head(h[r0 + L - 1].float()[None], h[r0:r0 + L - 1].float()[None])[0] / self.temp(pr['rq'].kind)
            out.append(lg[:pr['rq'].n_slots])
        return out

    def statefirst_logits(self, s, qs, prs, head=None, keep=()):
        """hobson's layout: [state] then each question as its own branch (= one sequence per question).
        keep: layers whose outputs at each question's option rows + its '<answer>' row are left in self.kept[i] (same order as the sets layout rows)"""
        head = head or self.head or self.head0
        Tm = len(s); ids = list(s); pos = list(range(Tm)); seqs = []
        for q in qs:
            seqs.append((len(ids), len(q))); ids += q; pos += list(range(Tm, Tm + len(q)))
        br = Br('seqs', Tm, seqs=seqs)
        rows = [r for (s0, Lb), pr in zip(seqs, prs) for r in [s0 + o for o in pr['opt']] + [s0 + Lb - 1]]
        h = self.fwd_br(ids, pos, br, keep=keep, keep_rows=torch.tensor(rows, device=self.dev))
        out = []
        for (s0, Lb), pr in zip(seqs, prs):
            oi = torch.tensor([s0 + o for o in pr['opt']], device=self.dev)
            lg = head(h[s0 + Lb - 1].float()[None], h[oi].float()[None])[0] / self.temp(pr['rq'].kind)
            out.append(lg[:pr['rq'].n_slots])
        return out
