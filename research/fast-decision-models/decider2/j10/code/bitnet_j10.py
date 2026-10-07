"""J10: BitNet b1.58 2B4T torso (own implementation, no remote code) for a natively-ternary decider.

Numerics are exactly the pretrained ones (HF integrations/bitnet.py, online mode):
  weights      per-tensor absmean ternary:  g = mean|W|, Wq = clamp(round(W / g), -1, 1) * g
  activations  per-token absmax int8 at every BitLinear input: s = 127 / max|x|, xq = clamp(round(x s), -128, 127) / s
  layer        x += o_proj(Q(attn_sub_norm(attn(Q(rms(x)))))) ;  x += down(Q(ffn_sub_norm(relu(gate)^2 * up)))  (all Q'd inputs)
  RoPE theta 5e5 (llama default), GQA 20 / 5 heads of 128, causal SDPA, final RMSNorm, LLaMA-3 tokenizer.

Training = LoRA-in-latent QAT ("ternary(W0 + s B A)"):
  W0 = the released bf16 master weights (frozen). The forward uses Wq = ternary(W0 + s*B@A), recomputed after every optimiser step.
  Backward is the straight-through estimator w.r.t. the latent weight, done low-rank (dB = s dy^T (xq A^T), dA = s (dy B)^T xq),
  so no dense weight gradient is ever formed.  The deployed weights are therefore exactly ternary + one fp scale per matrix.
  RMSNorm gains (full precision in BitNet itself) are trainable too.

Activation quantisation is configurable per GEMM class for the 4-bit study:  self.aq = {cls: (bits, block)} with
  cls in qkv / o / gu / down, bits 8 or 4, block 0 = per token, 16/32/... = absmax per K-block (NVFP4-like block scales, int grid).
"""
import os, glob, json, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch._dynamo
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64

REPO = os.environ.get('BITNET_REPO', 'microsoft/bitnet-b1.58-2B-4T-bf16')
HUB = os.path.expanduser('~/.cache/huggingface/hub')
LIN = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
CLS = dict(q='qkv', k='qkv', v='qkv', o='o', gate='gu', up='gu', down='down')


def snap(repo=REPO):
    d = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*")
    if not d:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo)
    return d[0]


def load(dev='cuda'):
    from safetensors import safe_open
    d = snap(); cfg = json.load(open(f'{d}/config.json'))
    W = {}
    for f in sorted(glob.glob(f'{d}/*.safetensors')):
        with safe_open(f, 'pt', device=dev) as fh:
            for k in fh.keys():
                W[k] = fh.get_tensor(k).to(torch.bfloat16)
    return cfg, W


def tern(Wl):
    """latent -> (bf16 ternary*g, int8 codes, g)"""
    g = Wl.float().abs().mean().clamp_min(1e-5)
    q = (Wl.float() / g).round().clamp_(-1, 1)
    return (q * g).to(torch.bfloat16), q.to(torch.int8), g


def _tern_lat(W0, B, A, s: float):
    Wl = W0.float() + s * (B @ A)
    g = Wl.abs().mean().clamp_min(1e-5)
    q = (Wl / g).round().clamp(-1, 1)
    return (q * g).to(torch.bfloat16), q.to(torch.int8), g


tern_lat = torch.compile(_tern_lat, dynamic=False)


def _fq(x, bits: int, block: int):
    qm = 127. if bits == 8 else float(2 ** (bits - 1) - 1)
    xf = x.float()
    sh = xf.shape
    if block:
        xf = xf.reshape(*sh[:-1], sh[-1] // block, block)
    s = qm / xf.abs().amax(-1, keepdim=True).clamp_min(1e-5)
    y = (xf * s).round().clamp_(-qm - 1, qm) / s
    return y.reshape(sh).to(x.dtype)


def _rms(x, w, eps: float):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w.to(x.dtype) * xf.to(x.dtype)


def _relu2mul(g, u):
    return F.relu(g).square() * u


fq = torch.compile(_fq, dynamic=True)
rms = torch.compile(_rms, dynamic=True)
relu2mul = torch.compile(_relu2mul, dynamic=True)


def AQ(x, bits, block):
    """fake-quantised activation with a straight-through gradient"""
    if bits >= 16:
        return x
    return x + (fq(x, bits, block) - x).detach()


class TL(torch.autograd.Function):
    """y = xq @ Wq^T with the low-rank STE backward for W_lat = W0 + s B A."""
    @staticmethod
    def forward(ctx, xq, Wq, A, B, s):
        ctx.save_for_backward(xq, Wq, A, B); ctx.s = s
        return xq @ Wq.t()

    @staticmethod
    def backward(ctx, dy):
        xq, Wq, A, B = ctx.saved_tensors; s = ctx.s
        dx = dy @ Wq
        d2 = dy.reshape(-1, dy.shape[-1]); x2 = xq.reshape(-1, xq.shape[-1])
        xa = x2 @ A.t().to(x2.dtype)                       # [N, r]
        dB = (d2.t() @ xa).float() * s                     # [dout, r]
        dA = ((d2 @ B.to(d2.dtype)).t() @ x2).float() * s  # [r, din]
        return dx, None, dA, dB, None


def rope_cs(T, hd, theta, dev, offset=0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    t = torch.arange(offset, offset + T, device=dev, dtype=torch.float32)
    f = torch.outer(t, inv); e = torch.cat([f, f], -1)
    return e.cos().to(torch.bfloat16), e.sin().to(torch.bfloat16)


def _rot(x, cos, sin):  # x [B, h, T, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


rot = torch.compile(_rot, dynamic=True)


class BitNetTorso(nn.Module):
    def __init__(self, cfg, W, r=16, alpha=32, lora=True, train_norms=True):
        super().__init__()
        self.cfg = cfg
        self.d = cfg['hidden_size']; self.L = cfg['num_hidden_layers']; self.H = cfg['num_attention_heads']; self.KV = cfg['num_key_value_heads']
        self.hd = self.d // self.H; self.eps = cfg['rms_norm_eps']; self.theta = float(cfg['rope_theta']); self.ffn = cfg['intermediate_size']
        self.embed = W['model.embed_tokens.weight']                     # frozen, bf16
        self.W0 = {}                                                      # (i, name) -> bf16 latent master (frozen)
        P = 'model.layers.{}.'
        nm = dict(q='self_attn.q_proj', k='self_attn.k_proj', v='self_attn.v_proj', o='self_attn.o_proj',
                  gate='mlp.gate_proj', up='mlp.up_proj', down='mlp.down_proj')
        for i in range(self.L):
            for n in LIN:
                self.W0[(i, n)] = W[P.format(i) + nm[n] + '.weight']
        normkeys = dict(ln1='input_layernorm', ln2='post_attention_layernorm', sa='self_attn.attn_sub_norm', sf='mlp.ffn_sub_norm')
        self.norms = nn.ParameterDict()
        for i in range(self.L):
            for k, v in normkeys.items():
                self.norms[f'{k}{i}'] = nn.Parameter(W[P.format(i) + v + '.weight'].float(), requires_grad=train_norms)
        self.norms['final'] = nn.Parameter(W['model.norm.weight'].float(), requires_grad=train_norms)
        self.r = r; self.s = alpha / r; self.use_lora = lora
        self.lora = nn.ParameterDict()
        if lora:
            g = torch.Generator().manual_seed(0)
            for (i, n), w in self.W0.items():
                dout, din = w.shape
                A = torch.empty(r, din); nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
                self.lora[f'{i}_{n}_A'] = nn.Parameter(A.to(w.device))
                self.lora[f'{i}_{n}_B'] = nn.Parameter(torch.zeros(dout, r, device=w.device))
        self.aq = dict(qkv=(8, 0), o=(8, 0), gu=(8, 0), down=(8, 0))
        self.ckpt = False
        self.stats = None                                                 # optional callback(layer, cls, x_pre_quant)
        self.Wq = {}; self.gam = {}
        self.track = {}                                                   # sampled code-flip tracking
        self.refresh(first=True)

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def latent(self, i, n):
        w = self.W0[(i, n)]
        if not self.use_lora:
            return w
        A = self.lora[f'{i}_{n}_A']; B = self.lora[f'{i}_{n}_B']
        return (w.float() + self.s * (B @ A)).to(torch.bfloat16)

    @torch.no_grad()
    def refresh(self, first=False):
        flips = 0; tot = 0
        for (i, n) in self.W0:
            if self.use_lora:
                wq, codes, g = tern_lat(self.W0[(i, n)], self.lora[f'{i}_{n}_B'], self.lora[f'{i}_{n}_A'], self.s)
            else:
                wq, codes, g = tern(self.W0[(i, n)])
            self.Wq[(i, n)] = wq; self.gam[(i, n)] = g
            smp = codes.view(-1)[:65536]
            if first: self.track[(i, n)] = smp.clone()
            else:
                flips += int((smp != self.track[(i, n)]).sum()); tot += smp.numel()
        return flips / max(1, tot)

    @torch.no_grad()
    def codes(self):
        """deployable ternary codes (int8) and per-matrix scales"""
        out = {}
        for (i, n) in self.W0:
            if self.use_lora:   # exactly the training-time quantiser (fp32 latent, no bf16 rounding before ternarisation)
                _, c, g = tern_lat(self.W0[(i, n)], self.lora[f'{i}_{n}_B'], self.lora[f'{i}_{n}_A'], self.s)
            else:
                _, c, g = tern(self.W0[(i, n)])
            out[(i, n)] = (c, float(g))
        return out

    def lin(self, i, n, xq):
        Wq = self.Wq[(i, n)]
        if self.use_lora and self.training and torch.is_grad_enabled():
            return TL.apply(xq, Wq, self.lora[f'{i}_{n}_A'], self.lora[f'{i}_{n}_B'], self.s)
        return xq @ Wq.t()

    # ------------------------------------------------------------------ forward
    def Q(self, i, cls, x):
        if self.stats is not None:
            self.stats(i, cls, x)
        return AQ(x, *self.aq[cls])

    def layer(self, i, x, cos, sin):
        B, T, _ = x.shape
        nr = self.norms
        h = self.Q(i, 'qkv', rms(x, nr[f'ln1{i}'], self.eps))
        q = self.lin(i, 'q', h).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.lin(i, 'k', h).view(B, T, self.KV, self.hd).transpose(1, 2)
        v = self.lin(i, 'v', h).view(B, T, self.KV, self.hd).transpose(1, 2)
        q = rot(q, cos, sin); k = rot(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, T, self.d)
        o = self.lin(i, 'o', self.Q(i, 'o', rms(o, nr[f'sa{i}'], self.eps)))
        x = x + o
        h = self.Q(i, 'gu', rms(x, nr[f'ln2{i}'], self.eps))
        m = relu2mul(self.lin(i, 'gate', h), self.lin(i, 'up', h))
        d = self.lin(i, 'down', self.Q(i, 'down', rms(m, nr[f'sf{i}'], self.eps)))
        return x + d

    def forward(self, input_ids, attention_mask=None, hook=None):
        """right-padded batches only (causal: real tokens never see the padding). -> final-normed hidden [B, T, d]"""
        x = F.embedding(input_ids, self.embed)
        T = x.shape[1]
        cos, sin = rope_cs(T, self.hd, self.theta, x.device)
        for i in range(self.L):
            if hook is not None:
                hook(i, x)
            if self.ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(self.layer, i, x, cos, sin, use_reentrant=False)
            else:
                x = self.layer(i, x, cos, sin)
        return rms(x, self.norms['final'], self.eps)

    def trainable_state(self):
        sd = {('lora.' + k): v.detach().cpu() for k, v in self.lora.items()}
        sd.update({('norms.' + k): v.detach().cpu() for k, v in self.norms.items()})
        return sd

    def load_trainable(self, sd):
        with torch.no_grad():
            for k, v in sd.items():
                grp, name = k.split('.', 1)
                getattr(self, grp)[name].copy_(v.to(getattr(self, grp)[name].device))
        self.refresh()
