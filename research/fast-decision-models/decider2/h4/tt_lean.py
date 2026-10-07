"""H4: this-that-model-1.0 in d1's fused lean2 runtime, with an optional schema-first prefix cache.

TTL(torso, label_ids).fwd(ids, pos0, cache, want_cache) -> (final-normed hidden [T, 2048], new cache)
  cache = per layer: GDN {'tail': last 3 rows of in_proj output (conv history), 'S': recurrent state [1,16,128,128] fp32}
                     attention {'k': [P,2,256] (normed + roped), 'v': [P,2,256]}
The cached pass sees exactly what the full pass over prefix+suffix sees: conv history, delta-rule state, and K/V of the prefix,
with RoPE positions offset by P. Head: softmax over the tied-embedding rows of the option-label tokens (A..J) at each answer slot.
"""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/work/systems/g')); sys.path.insert(0, os.path.expanduser('~/work/d1'))
import lean as LM
from lean2 import Lean2, tgemm, conv_l2, gnorm, attn_prep, add_rms, silu_mul
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right
import triton, triton.language as tl


@triton.jit
def _conv_l2_tail_k(P, TL, W, OUT, T, ps, BT: tl.constexpr, HD: tl.constexpr):
    # d1's conv+SiLU+l2norm+split kernel, with the 3 rows of conv history read from TL (the cached prefix tail) instead of zeros
    tb = tl.program_id(0); h = tl.program_id(1)
    t = (tb * BT + tl.arange(0, BT)[:, None]).to(tl.int64); c = tl.arange(0, HD)[None, :]
    ch = h * HD + c
    acc = tl.zeros([BT, HD], tl.float32)
    for j in tl.static_range(4):
        tt = t - 3 + j
        x = tl.load(P + tt * ps + ch, mask=(tt >= 0) & (tt < T), other=0.).to(tl.float32)
        xt = tl.load(TL + (tt + 3) * ps + ch, mask=(tt < 0) & (tt >= -3), other=0.).to(tl.float32)
        w = tl.load(W + ch * 4 + j).to(tl.float32)
        acc += (x + xt) * w
    y = acc * tl.sigmoid(acc)
    y = y.to(tl.bfloat16).to(tl.float32)
    if h < 32:
        y = y / tl.sqrt(tl.sum(y * y, 1)[:, None] + 1e-6)
    grp = h // 16; hh = h % 16
    tl.store(OUT + grp * T * 2048 + t * 2048 + hh * HD + c, y.to(tl.bfloat16), mask=t < T)


def conv_l2_tail(proj, tail, w):
    T = proj.shape[0]
    assert tail.stride(0) == proj.stride(0)
    out = torch.empty(3, T, 16, 128, device=proj.device, dtype=torch.bfloat16)
    _conv_l2_tail_k[(triton.cdiv(T, 32), 48)](proj, tail, w, out, T, proj.stride(0), BT=32, HD=128, num_warps=4)
    return out

FUSE_LONG = 'gnorm,prep,conv,fold'                              # d1's best path at T >= 512
FUSE_SHORT = 'addrms,gnorm,silu,prep,conv,gemm_swiglu'         # d1's best path below 512


def load_tt(name='flock-io/this-that-model-1.0'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    return m, tok


class TTL(Lean2):
    def __init__(self, torso, label_ids, fuse=FUSE_LONG):
        super().__init__(torso, fuse=fuse)
        self.Wlab = self.embed[torch.as_tensor(label_ids[:10], device=self.dev)].contiguous()   # [10, 2048] tied unembedding rows
        self.mask = None

    def set_fuse(self, T):
        self.fuse = set((FUSE_LONG if T >= 512 else FUSE_SHORT).split(','))

    @torch.no_grad()
    def fwd(self, ids, pos0=0, cache=None, want_cache=False):
        f = self.fuse
        B, T = ids.shape
        assert B == 1 and 'conv' in f and 'prep' in f and 'gnorm' in f
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(pos0, pos0 + T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        fold = 'fold' in f
        newc = {} if want_cache else None
        if fold:
            ss = x.float().pow(2).sum(-1)
        else:
            h = LM._rms_zc(x, self.layers[0]['in_norm'], self.eps)
        n = len(self.layers)
        for i, d in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]) if fold else self._mm(h, d['Win'])
            if d['type'] == 'linear_attention':
                if c is not None:
                    qkv3 = conv_l2_tail(proj, c['tail'], d['conv_w'])
                else:
                    qkv3 = conv_l2(proj, d['conv_w'])
                q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                o, S = chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                              A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                              initial_state=None if c is None else c['S'], output_final_state=want_cache)
                if want_cache:
                    newc[i] = {'tail': proj[-3:].contiguous().clone(), 'S': S.clone()}
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                if want_cache:
                    newc[i] = {'k': k.clone(), 'v': v.contiguous().clone()}
                qh = q.reshape(1, T, 8, 256).transpose(1, 2)
                if c is not None:
                    kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                    kh = kk.reshape(1, -1, 2, 256).transpose(1, 2); vh = vv.reshape(1, -1, 2, 256).transpose(1, 2)
                    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=self.mask, enable_gqa=True)
                else:
                    kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
                    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            if fold:
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
                m = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
                continue
            d_out = self._mm(o, d['Wo'])
            x, h2 = add_rms(x, d_out, d['post1'], self.eps)
            m = tgemm(h2, d['Wgu_il'], epi=1, cfg=self.gcfg.get('swiglu'))
            d_mlp = self._mm(m, d['Wd'])
            x, h = add_rms(x, d_mlp, (self.layers[i + 1]['in1'] if i + 1 < n else self.norm1), self.eps)
        if fold:
            h = LM._rms_zc(x, self.norm_w, self.eps)
        return h.reshape(T, -1), newc

    def set_prefix_mask(self, P, T):
        self.mask = causal_lower_right(T, P + T)   # query i sees the whole prefix and new keys <= i (flash path, no dense mask)

    def head(self, h, slots, nopt):
        """slots: LongTensor [N] positions; nopt: LongTensor [N]. -> probs [N, 10] (zero past each question's options)"""
        lg = (h[slots] @ self.Wlab.t()).float()
        lg = lg.masked_fill(torch.arange(10, device=self.dev)[None, :] >= nopt[:, None], float('-inf'))
        return torch.softmax(lg, -1)
