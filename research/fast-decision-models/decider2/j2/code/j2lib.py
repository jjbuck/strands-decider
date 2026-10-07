"""J2 library: a BIDIRECTIONAL GDN-hybrid encoder made from hobson-v19's own torso (Qwen3.5-2B: 18 GDN + 6 attention layers, d 2048).

Built on H3's differentiable lean hobson forward (LoRA-merged hobson weights, fla kernels). Rows are PACKED into one varlen sequence
(cu_seqlens); each row = [state tokens (q0 of them)] [question tokens] (one question per row), positions restart per row.

Function-preserving init (every new path is gated at 0, so the untrained model IS hobson):
  * GDN layer i in rev_layers:  o = o_fwd + gam_i (.) o_rev            gam_i [2048] (per channel), init 0
      o_rev = the same gated delta rule (shared q, k, v, beta, g and the shared causal-conv output; Caduceus-style shared projections)
      run over a PERMUTED order of the row:
        mode 'qag' (masked state cache, e1b-like): reverse stream = s_T..s_1, q_L..q_1  -> state rows never see the question; each
                   question token's reverse read starts from the state's reverse final state (an early-weighted summary of the state)
        mode 'qa'  (question-aware, e1a-like):     reverse stream = q_L..q_1, s_T..s_1  (plain bidirectional scan over [S;Q])
  * attention layer i (if attn_nc):  o = o_causal + lam_i[h] (o_nc - o_causal)    lam_i [8 heads], init 0
        'qag': o_nc mask = same row & (query is a question row | key is a state row)   (state<->state full; question -> state + whole question)
        'qa' : o_nc mask = same row (fully non-causal over [S;Q])
  * mode 'causal': no new paths (the control).
"""
import os, sys, math
sys.path[:0] = [os.path.expanduser('~/work/j2')]
import torch, torch.nn as nn, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, DENSE, chunk_gated_delta_rule, fla_conv
from fla.modules.fused_norm_gate import rms_norm_gated
FAST = os.environ.get('J2_SLOW') != '1'

GDN_LAYERS = tuple(i for i in range(24) if i not in H.FULL_ATTN)


class Pack:
    """packed rows. rows: list of (ids, q0) with q0 = number of state tokens (q0 == len(ids) for state-only rows)."""

    def __init__(self, rows, dev, mode):
        self.rows = rows
        lens = [len(r[0]) for r in rows]
        self.lens = lens
        cu = [0]
        for L in lens: cu.append(cu[-1] + L)
        self.cu_l = cu
        T = cu[-1]; self.T = T
        self.cu = torch.tensor(cu, device=dev, dtype=torch.long)
        ids = []; pos = []; rid = []; isq = []; perm = []; rstart = []
        for r, (row, q0) in enumerate(rows):
            L = len(row); b = cu[r]
            ids += list(row); pos += list(range(L)); rid += [r] * L; isq += [0] * q0 + [1] * (L - q0); rstart += [b] * L
            S = list(range(b, b + q0)); Q = list(range(b + q0, b + L))
            if mode == 'qa':
                perm += Q[::-1] + S[::-1]
            else:
                perm += S[::-1] + Q[::-1]
        self.ids = torch.tensor(ids, device=dev, dtype=torch.long)
        self.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        self.rid = torch.tensor(rid, device=dev, dtype=torch.long)
        self.isq = torch.tensor(isq, device=dev, dtype=torch.bool)
        self.perm = torch.tensor(perm, device=dev, dtype=torch.long)
        inv = torch.empty_like(self.perm); inv[self.perm] = torch.arange(T, device=dev)
        self.inv = inv
        rs = torch.tensor(rstart, device=dev, dtype=torch.long)
        t = torch.arange(T, device=dev)
        self.cvalid = [((t - j) >= rs) for j in (1, 2, 3)]          # conv tap j steps back stays inside the row
        self.single = len(rows) == 1
        self.mode = mode
        self._mc = None; self._mn = None

    def mask_c(self):
        if self._mc is None:
            same = self.rid[:, None] == self.rid[None, :]
            t = torch.arange(self.T, device=self.rid.device)
            self._mc = same & (t[None, :] <= t[:, None])
        return self._mc

    def mask_n(self):
        if self._mn is None:
            same = self.rid[:, None] == self.rid[None, :]
            if self.mode == 'qa':
                self._mn = same
            else:
                self._mn = same & (self.isq[:, None] | (~self.isq)[None, :])
        return self._mn


def conv_packed(x, w, pk):
    """causal depthwise conv (kernel 4) + SiLU over packed rows. x [T, C] bf16, w [C, 4]. fp32 accumulate."""
    wf = w.float()
    acc = x.float() * wf[:, 3][None]
    for j in (1, 2, 3):
        xs = F.pad(x, (0, 0, j, 0))[:x.shape[0]].float() * pk.cvalid[j - 1][:, None]
        acc = acc + xs * wf[:, 3 - j][None]
    return F.silu(acc).to(x.dtype)


class J2(H.H3):
    def setup_bidir(self, mode='qag', rev_layers=GDN_LAYERS, attn_nc=True):
        """mode in ('causal', 'qag', 'qa'). Creates the gates (zeros) -> function-identical to hobson until trained."""
        self.mode = mode
        self.rev_layers = set(rev_layers) if mode != 'causal' else set()
        self.attn_nc = attn_nc and mode != 'causal'
        self.gam = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(2048, device=self.dev)) for i in sorted(self.rev_layers)})
        self.lam = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(8, device=self.dev)) for i in H.FULL_ATTN} if self.attn_nc else {})
        return list(self.gam.parameters()) + list(self.lam.parameters())

    def gates_state(self):
        sd = {f'gam.{k}': v.detach().cpu() for k, v in self.gam.items()}
        sd.update({f'lam.{k}': v.detach().cpu() for k, v in self.lam.items()})
        return sd

    def load_gates(self, sd):
        with torch.no_grad():
            for k, v in self.gam.items(): v.copy_(sd[f'gam.{k}'])
            for k, v in self.lam.items(): v.copy_(sd[f'lam.{k}'])

    # ------------------------------------------------------------------ packed forward
    def layer_pk(self, i, x, cos, sin, pk):
        d = self.L[i]; T = x.shape[0]; eps = self.eps
        h = self.bnorm(x, i, 0)
        proj = self.lin(h, i, 'Win', DENSE)
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            cu = None if pk.single else pk.cu
            if FAST:
                cm = fla_conv(raw[None].contiguous(), d['conv_w'], None, activation='silu', cu_seqlens=cu)
                cm = (cm[0] if isinstance(cm, tuple) else cm)[0]
            else:
                cm = conv_packed(raw, d['conv_w'], pk)
            q, k, v = cm.split(2048, dim=-1)
            of, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                           beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
            o = of[0].reshape(T, 2048)
            if i in self.rev_layers:
                P_ = pk.perm
                orv, _ = chunk_gated_delta_rule(q[P_].reshape(1, T, 16, 128), k[P_].reshape(1, T, 16, 128), v[P_].reshape(1, T, 16, 128),
                                                g[P_][None], beta[P_][None].to(q.dtype), use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
                orv = orv[0].reshape(T, 2048)[pk.inv]
                o = (o.float() + self.gam[str(i)][None] * orv.float()).to(o.dtype)
            if FAST:
                o = rms_norm_gated(o.reshape(-1, 128), z.reshape(-1, 128), d['gn_w'], None, activation='swish', eps=eps).reshape(T, 2048).to(x.dtype)
            else:
                of_ = o.reshape(-1, 128).float(); of_ = of_ * torch.rsqrt(of_.pow(2).mean(-1, keepdim=True) + eps)
                o = ((d['gn_w'] * of_.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)

            def rope(t):
                xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
                c = cos[:, None, :]; s_ = sin[:, None, :]
                return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            qh, kk = rope(qh), rope(kk)
            Q_ = qh.transpose(0, 1)[None]; K_ = kk.transpose(0, 1)[None]; V_ = v.transpose(0, 1)[None]
            sdpa = F.scaled_dot_product_attention
            outs = []
            for r in range(len(pk.rows)):           # per row: flash kernels, cost ~ sum L_r^2 (not T^2)
                b0, b1 = pk.cu_l[r], pk.cu_l[r + 1]; q0 = pk.rows[r][1]
                Qr, Kr, Vr = Q_[:, :, b0:b1], K_[:, :, b0:b1], V_[:, :, b0:b1]
                oc = sdpa(Qr, Kr, Vr, is_causal=True, enable_gqa=True)
                if self.attn_nc:
                    if self.mode == 'qa' or q0 >= b1 - b0:
                        on = sdpa(Qr, Kr, Vr, enable_gqa=True)
                    else:
                        on = torch.cat([sdpa(Qr[:, :, :q0], Kr[:, :, :q0], Vr[:, :, :q0], enable_gqa=True), sdpa(Qr[:, :, q0:], Kr, Vr, enable_gqa=True)], 2)
                    lam = self.lam[str(i)].to(oc.dtype)[None, :, None, None]
                    oc = oc + lam * (on - oc)
                outs.append(oc)
            oc = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
            o = (oc[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(o, i, 'Wo', DENSE)
        h2 = self.bnorm(x, i, 1)
        gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
        x = x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)
        return x

    def fwd_pk(self, pk, ckpt=False, ids=None, x0=None):
        x = F.embedding(pk.ids if ids is None else ids, self.embed) if x0 is None else x0
        fr = pk.pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer_pk, i, x, cos, sin, pk, use_reentrant=False)
            else:
                x = self.layer_pk(i, x, cos, sin, pk)
        return rms_zc(x, self.norm_w, self.eps)

    def decide(self, items, ckpt=False, head=None):
        """items: list of dicts with s (state ids), q (question ids), opt (option offsets within q), rq (rendered question).
        -> list of temperature-scaled logits [n_slots] (one per item)"""
        head = head or self.head or self.head0
        rows = [(list(it['s']) + list(it['q']), len(it['s'])) for it in items]
        pk = Pack(rows, self.dev, self.mode)
        h = self.fwd_pk(pk, ckpt=ckpt)
        out = []
        for r, it in enumerate(items):
            b = pk.cu_l[r]; q0 = len(it['s']); L = pk.lens[r]
            oi = torch.tensor([b + q0 + o for o in it['opt']], device=self.dev)
            lg = head(h[b + L - 1].float()[None], h[oi].float()[None])[0] / self.temp(it['kind'])
            out.append(lg[:it['n_slots']])
        return out
