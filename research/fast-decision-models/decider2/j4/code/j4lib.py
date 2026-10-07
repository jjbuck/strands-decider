"""J4 library: the Qwen3.5-2B torso (hobson's base, or hobson-v19 merged) as a differentiable lean forward over a FOREST of sequences.

A forward takes roots (independent causal sequences, packed varlen) and children (each continues one root: starts from the root's final
GDN state + conv tail, attends to all of the root's rows and causally to itself, never to siblings).
  * FT rows  : roots = [state + question] rows, no children (one question per sequence, as the evalkit references render).
  * DP rows  : root = state, children = its generated questions (dense decision supervision from one state pass).
  * eval     : root = state, children = the item's questions (= hobson's shared-prefix layout: each question sees state + itself).
LoRA on the four fused weights per layer (Win, Wo, Wgu, Wd), as H3/H7.  Weights: base = PEFT unload() of hobson's torso (= Qwen3.5-2B-Base
exactly), teacher = merge_and_unload().
"""
import os, sys, math, json, glob
sys.path[:0] = [os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

NAMES = ('Win', 'Wo', 'Wgu', 'Wd')


def rms_zc(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


class PointerHead(nn.Module):
    """hobson's PointerHead: LayerNorm -> q / k Linear(2048 -> 256) -> dot / 16 (fresh random init, as head_init: random)"""

    def __init__(self, d=2048, dim=256, dropout=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.q = nn.Linear(d, dim); self.k = nn.Linear(d, dim); self.scale = dim ** -0.5
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, decide, options):  # [B, d], [B, K, d]
        d = self.q(self.drop(self.norm(decide))).unsqueeze(-1)
        o = self.k(self.drop(self.norm(options)))
        return (o @ d).squeeze(-1) * self.scale


def ckpt_path():
    return glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]


class J4:
    def __init__(self, weights='base', dev='cuda', maxlen=16384):
        from strands_decider.infer import load_engine
        import lean as LN
        self.eng = load_engine(ckpt_path(), device=dev)
        self.eng.model.config.max_length = maxlen
        tm = self.eng.model.torso
        if weights == 'base': tm = tm.unload()
        elif weights == 'hobson': tm = tm.merge_and_unload()
        else: raise ValueError(weights)
        tm.eval()
        self.tie = getattr(tm.config, 'tie_word_embeddings', None)
        lean = LN.Lean(tm)
        self.L = lean.layers; self.eps = lean.eps; self.dev = lean.dev
        self.embed = lean.embed.detach().clone(); self.norm_w = lean.norm_w.detach().clone(); self.inv = lean.inv.clone()
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v): d[k] = v.detach().clone()
        self.hob_head = None
        if weights == 'hobson':
            h0 = self.eng.model.head
            self.hob_head = PointerHead(dropout=0.0).to(dev)
            self.hob_head.load_state_dict({k: v.detach().float() for k, v in h0.state_dict().items()}, strict=False)
            for p_ in self.hob_head.parameters(): p_.requires_grad_(False)
        self.eng.model.torso = None
        import gc; gc.collect(); torch.cuda.empty_cache()
        self.tok = self.eng.tok
        self.lora = None; self.lora_scale = 2.0
        self.head = None

    # ---------------- params
    def add_lora(self, r=16, alpha=32, seed=0):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.lora = nn.ModuleList()
        for i in range(len(self.L)):
            md = nn.ModuleDict()
            for k in NAMES:
                out, inp = self.L[i][k].shape
                m = nn.Module()
                m.A = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(self.dev))
                m.B = nn.Parameter(torch.zeros(out, r, device=self.dev))
                md[k] = m
            self.lora.append(md)
        self.lora_scale = alpha / r
        return list(self.lora.parameters())

    def set_head(self, dropout=0.05, seed=0):
        torch.manual_seed(seed)
        self.head = PointerHead(dropout=dropout).to(self.dev)
        return list(self.head.parameters())

    def merge_lora(self):
        """fold LoRA into the weights in fp32 then round to bf16 (used to carry an intermediate phase into the next stage)"""
        with torch.no_grad():
            for i in range(len(self.L)):
                for k in NAMES:
                    lo = self.lora[i][k]
                    self.L[i][k] = (self.L[i][k].float() + self.lora_scale * lo.B.float() @ lo.A.float()).to(torch.bfloat16).contiguous()
        self.lora = None

    def lora_state(self):
        sd = {}
        if self.lora is not None:
            for i in range(len(self.L)):
                for k in NAMES:
                    sd[f'lora.{i}.{k}.A'] = self.lora[i][k].A.detach().cpu(); sd[f'lora.{i}.{k}.B'] = self.lora[i][k].B.detach().cpu()
            sd['_lora_scale'] = self.lora_scale
        if self.head is not None:
            for k, v in self.head.state_dict().items(): sd[f'head.{k}'] = v.detach().cpu()
        return sd

    def load_lora_state(self, sd, trainable=False):
        r = sd['lora.0.Win.A'].shape[0]
        self.add_lora(r=r, alpha=float(sd['_lora_scale']) * r)
        with torch.no_grad():
            for i in range(len(self.L)):
                for k in NAMES:
                    self.lora[i][k].A.copy_(sd[f'lora.{i}.{k}.A']); self.lora[i][k].B.copy_(sd[f'lora.{i}.{k}.B'])
        for p_ in self.lora.parameters(): p_.requires_grad_(trainable)

    def load_head_state(self, sd, trainable=False):
        self.set_head()
        self.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
        for p_ in self.head.parameters(): p_.requires_grad_(trainable)

    def load_stack(self, stack):
        """stack: list of checkpoint paths applied in order. Every one but the last is merged into the weights; the last stays as LoRA.
        The head comes from the last checkpoint that has one."""
        head_sd = None
        for j, pth in enumerate(stack):
            sd = torch.load(os.path.expanduser(pth), map_location='cpu')
            if any(k.startswith('head.') for k in sd): head_sd = sd
            if 'lora.0.Win.A' in sd:
                self.load_lora_state(sd)
                if j < len(stack) - 1: self.merge_lora()
        if head_sd is not None: self.load_head_state(head_sd)

    # ---------------- linear
    def lin(self, x, i, nm):
        W = self.L[i][nm]
        y = x @ W.t()
        if self.lora is not None:
            lo = self.lora[i][nm]
            y = y + ((x @ lo.A.t().to(x.dtype)) @ lo.B.t().to(x.dtype)) * self.lora_scale
        return y

    # ---------------- one layer over the forest
    def layer(self, i, x, cos, sin, F_):
        d = self.L[i]; T = x.shape[0]; eps = self.eps
        TR = F_['TR']
        h = rms_zc(x, d['in_norm'], eps)
        proj = self.lin(h, i, 'Win')
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            # conv over roots (varlen) and children (prefixed by the parent's last 3 raw rows, dropped after the conv)
            cr = fla_conv(raw[:TR][None].contiguous(), d['conv_w'], None, activation='silu', cu_seqlens=F_['cu_r'])
            cr = (cr[0] if isinstance(cr, tuple) else cr).reshape(TR, 6144)
            parts = [cr]
            if F_['nC']:
                ext = raw[F_['cidx']]                                  # [TCx, 6144] (tails of the parent + child rows), -1 -> zero row via mask
                ext = ext * F_['cmask'][:, None].to(ext.dtype)
                cc = fla_conv(ext[None].contiguous(), d['conv_w'], None, activation='silu', cu_seqlens=F_['cu_cx'])
                cc = (cc[0] if isinstance(cc, tuple) else cc).reshape(-1, 6144)[F_['ckeep']]
                parts.append(cc)
            cv = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            q, k, v = cv.split(2048, dim=-1)
            q = q.reshape(1, T, 16, 128); k = k.reshape(1, T, 16, 128); v = v.reshape(1, T, 16, 128)
            orr, S = chunk_gated_delta_rule(q[:, :TR], k[:, :TR], v[:, :TR], g[None, :TR], beta[None, :TR].to(q.dtype),
                                           use_qk_l2norm_in_kernel=True, output_final_state=bool(F_['nC']), cu_seqlens=F_['cu_r'])
            outs = [orr[0]]
            if F_['nC']:
                oc, _ = chunk_gated_delta_rule(q[:, TR:], k[:, TR:], v[:, TR:], g[None, TR:], beta[None, TR:].to(q.dtype),
                                              initial_state=S[F_['cpar']], use_qk_l2norm_in_kernel=True, cu_seqlens=F_['cu_c'])
                outs.append(oc[0])
            o = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
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
            qt = qh.transpose(0, 1); kt = kk.transpose(0, 1); vt = v.transpose(0, 1)        # [H, T, 256]
            outs = [None] * len(F_['roots'])
            couts = []
            for j, (r0, r1) in enumerate(F_['roots']):
                outs[j] = F.scaled_dot_product_attention(qt[None, :, r0:r1], kt[None, :, r0:r1], vt[None, :, r0:r1], is_causal=True,
                                                         enable_gqa=True)[0]
            for (pj, c0, c1, mask) in F_['cgroups']:     # all children of root pj: rows c0:c1, keys = root rows + the children rows
                r0, r1 = F_['roots'][pj]
                K = torch.cat([kt[:, r0:r1], kt[:, c0:c1]], 1); V = torch.cat([vt[:, r0:r1], vt[:, c0:c1]], 1)
                couts.append(F.scaled_dot_product_attention(qt[None, :, c0:c1], K[None], V[None], attn_mask=mask[None, None], enable_gqa=True)[0])
            o = torch.cat(outs + couts, 1).transpose(0, 1)                                   # [T, 8, 256]
            o = (o * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(o, i, 'Wo')
        h2 = rms_zc(x, d['post_norm'], eps)
        gu = self.lin(h2, i, 'Wgu'); I = d['I']
        x = x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd')
        return x

    # ---------------- forest setup
    def forest(self, roots, children):
        """roots: list of token lists; children: list of (root index, token list), children grouped by root in increasing root order.
        -> ids, pos, F_ (metadata), child row offsets"""
        dev = self.dev
        ids = []; pos = []; rspans = []
        for r in roots:
            rspans.append((len(ids), len(ids) + len(r))); ids += r; pos += list(range(len(r)))
        TR = len(ids)
        cspans = []; cpar = []
        cidx = []; cmask = []; ckeep = []; cu_cx = [0]; cu_c = [0]
        for pj, c in children:
            r0, r1 = rspans[pj]
            s0 = len(ids); cspans.append((s0, s0 + len(c))); cpar.append(pj)
            ids += c; pos += list(range(r1 - r0, r1 - r0 + len(c)))
            base = len(cidx)
            for t in range(3):                          # parent's last 3 raw rows (zero if the root is shorter)
                src = r1 - 3 + t
                cidx.append(src if src >= r0 else 0); cmask.append(1.0 if src >= r0 else 0.0)
            for t in range(len(c)):
                cidx.append(s0 + t); cmask.append(1.0); ckeep.append(base + 3 + t)
            cu_cx.append(len(cidx)); cu_c.append(cu_c[-1] + len(c))
        F_ = dict(TR=TR, nC=len(children), roots=rspans,
                  cu_r=torch.tensor([0] + [b for _, b in rspans], device=dev, dtype=torch.int32 if False else torch.long))
        if children:
            # children rows are positions TR.. in ids; cidx refers to absolute rows of `raw`
            F_['cidx'] = torch.tensor(cidx, device=dev); F_['cmask'] = torch.tensor(cmask, device=dev)
            F_['ckeep'] = torch.tensor(ckeep, device=dev); F_['cu_cx'] = torch.tensor(cu_cx, device=dev)
            F_['cu_c'] = torch.tensor(cu_c, device=dev); F_['cpar'] = torch.tensor(cpar, device=dev)
            groups = []
            j = 0
            while j < len(children):
                pj = cpar[j]; k = j
                while k < len(children) and cpar[k] == pj: k += 1
                c0 = cspans[j][0]; c1 = cspans[k - 1][1]
                r0, r1 = rspans[pj]; R = c1 - c0
                mask = torch.zeros(R, (r1 - r0) + R, dtype=torch.bool, device=dev)
                mask[:, :r1 - r0] = True
                for (a0, a1) in cspans[j:k]:
                    L = a1 - a0
                    mask[a0 - c0:a1 - c0, (r1 - r0) + (a0 - c0):(r1 - r0) + (a1 - c0)] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
                groups.append((pj, c0, c1, mask)); j = k
            F_['cgroups'] = groups
        else:
            F_['cgroups'] = []
        return ids, pos, F_, cspans

    def forward(self, roots, children=(), ckpt=False):
        """-> final normed hidden [T, 2048], rspans, cspans"""
        ids, pos, F_, cspans = self.forest(roots, list(children))
        x = F.embedding(torch.tensor(ids, device=self.dev), self.embed)
        p = torch.tensor(pos, device=self.dev, dtype=torch.float32)
        fr = p[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        for i in range(len(self.L)):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer, i, x, cos, sin, F_, use_reentrant=False)
            else:
                x = self.layer(i, x, cos, sin, F_)
        return rms_zc(x, self.norm_w, self.eps), F_['roots'], cspans

    def pointer_logits(self, h, end_row, opt_rows, head=None):
        head = head or self.head
        return head(h[end_row].float()[None], h[opt_rows].float()[None])[0]

    def lm_loss(self, h, ids_next_rows, targets, chunk=1024):
        """next-token CE over rows (tied embedding as the LM head), chunked + checkpointed"""
        from torch.utils.checkpoint import checkpoint
        tot = 0.0; n = 0
        W = self.embed
        def f(hh, tt):
            return F.cross_entropy((hh @ W.t()).float(), tt, reduction='sum')
        for s in range(0, len(ids_next_rows), chunk):
            rr = ids_next_rows[s:s + chunk]; tt = targets[s:s + chunk]
            tot = tot + checkpoint(f, h[rr], tt, use_reentrant=False); n += len(rr)
        return tot / max(1, n)

    # ---------------- prompt prep (exactly as the evalkit references: render_state + render_question, eng._fit, eng._option_idx)
    def prep(self, state, qd, perm=None, maxlen=None):
        from pydantic import TypeAdapter
        import strands_decider.schema as SC
        from strands_decider.prompting import render_question, render_state
        if not hasattr(self, '_ta'): self._ta = TypeAdapter(SC.Question)
        q = self._ta.validate_python(qd)
        rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
        if maxlen: self.eng.model.config.max_length = maxlen
        s, qs = self.eng._fit(render_state(state), [rq.text])
        opt = self.eng._option_idx([rq], 0)[0].tolist()
        return dict(s=s, q=qs[0], opt=opt, rq=rq)

    def temp(self, kind):
        cfg = self.eng.model.config
        return float((getattr(cfg, 'temperature_by_kind', None) or {}).get(kind, cfg.temperature))
