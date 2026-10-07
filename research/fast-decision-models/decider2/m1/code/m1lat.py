"""M1 latency: all-attention hobson in d1/H4's fused bf16 runtime (lean2 Triton GEMMs with folded norms and fused epilogues, TTL kernels),
against hobson in the same harness (J3 dt_lat.py discipline).
  hob1  : hobson, one sequence [state + question] through H4's TTL.fwd with d1's best fuse path for the length (the doc's anchor; Q1 only)
  hobS  : hobson, one sequence, the fold path (same code path as m1S)
  m1S   : M1, one sequence (Q1 only)
  hobB  : hobson, state pass (24 layers, K/V + GDN-state cache) + ONE packed question pass (all questions as branches)
  m1B   : M1, state pass (24 layers of K/V cache) + ONE packed question pass
M1 mixer (18 former GDN layers): folded-norm Win GEMM [T, 8224] -> one Triton prep kernel (causal conv k=4 + SiLU, per-head RMSNorm x gain,
RoPE on 64 of 128 dims, the 6 gate-bias dims of the 136-wide head) -> SDPA (FlashAttention, causal; masked mem-efficient for question branches)
-> gated RMSNorm with SiLU(z) -> Wo GEMM with the residual epilogue.  AUG=0 drops the gate-bias dims (head dim 128, Win still 8224 rows).
Discipline: CUDA graph per exact shape, exclusive GPU, fresh real banking states every rep cut to exactly T tokens, 3 warm-ups discarded,
REPS timed, median + p95; timed span = H2D of the ids + graph replay + D2H of the probabilities.
python m1lat.py time [CKPT]                 -> $LATOUT (default ~/work/m1/lat_m1.json)
python m1lat.py check CKPT PREDS.jsonl      -> fused runtime vs m1lib eager preds (argmax, max|dp|) on 16 items
python m1lat.py prof [CKPT]                 -> kernel-time split (GEMM / attention / GDN / other) at T = 1000, 4000 (torch profiler, eager fused)"""
import os, sys, json, time, statistics as stt
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/m1'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import numpy as np, torch, torch.nn.functional as F
import triton, triton.language as tl
from tt_lean import TTL, tgemm, conv_l2, conv_l2_tail, gnorm, attn_prep, LM, chunk_gated_delta_rule
import evalkit as EK

dev = 'cuda'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'time'
CKPT = sys.argv[2] if len(sys.argv) > 2 else ''
REPS = int(os.environ.get('REPS', 20))
TS = [int(x) for x in os.environ.get('TS', '64,256,1000,4000').split(',')]
AUG = int(os.environ.get('AUG', 2))      # 0: no gate biases; 1: 136-wide heads (6 bias dims appended); 2: bias dims in-head (122 content + 6), head dim 128
D = 136 if AUG == 1 else 128
GDN = tuple(i for i in range(24) if i not in (3, 7, 11, 15, 19, 23))
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4]}


# ---------------------------------------------------------------- M1 kernels
@triton.jit
def _glue_k(P, EA, DTB, G, LB, ps, T, NH: tl.constexpr):
    """per row: g = -exp(A_log) softplus(a + dt_bias) [16], lb = log sigmoid(b) [16] (fp32), stored head-major [16, T]"""
    r = tl.program_id(0).to(tl.int64); hh = tl.arange(0, NH)
    b = tl.load(P + r * ps + 8192 + hh).to(tl.float32); a = tl.load(P + r * ps + 8208 + hh).to(tl.float32)
    x = a + tl.load(DTB + hh)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(G + hh * T + r, -tl.load(EA + hh) * sp)
    lb = tl.where(b > 0, -tl.log(1.0 + tl.exp(-b)), b - tl.log(1.0 + tl.exp(b)))
    tl.store(LB + hh * T + r, lb)


@triton.jit
def _m1prep_k(P, TL, Wc, QG, KG, COS, SIN, GQ, LB, OUT, T, ps, eps, sc,
              HAS_TAIL: tl.constexpr, AUGK: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, HD: tl.constexpr, SLOTS: tl.constexpr):
    """proj [T, 8224] (row stride ps) -> OUT [3, T, 16, D]: q (rms*gain, RoPE, x sc), k (rms*gain, RoPE), v; + gate-bias dims 128..135.
    conv history: zeros, or the 3 cached rows TL (same row stride) if HAS_TAIL. GQ: fp32 [T,16] cumulative log decay G; LB: log sigmoid(b)."""
    tb = tl.program_id(0); h = tl.program_id(1)
    t = (tb * BT + tl.arange(0, BT)[:, None]).to(tl.int64); c = tl.arange(0, HD)[None, :]
    grp = h // 16; hh = h % 16
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    ch = h * HD + c; chp = h * HD + pc
    acc = tl.zeros([BT, HD], tl.float32); accp = tl.zeros([BT, HD], tl.float32)
    for j in tl.static_range(4):
        tt = t - 3 + j
        m_ = (tt >= 0) & (tt < T)
        x = tl.load(P + tt * ps + ch, mask=m_, other=0.).to(tl.float32)
        xp = tl.load(P + tt * ps + chp, mask=m_, other=0.).to(tl.float32)
        if HAS_TAIL:
            mt = (tt < 0) & (tt >= -3)
            x += tl.load(TL + (tt + 3) * ps + ch, mask=mt, other=0.).to(tl.float32)
            xp += tl.load(TL + (tt + 3) * ps + chp, mask=mt, other=0.).to(tl.float32)
        acc += x * tl.load(Wc + ch * 4 + j).to(tl.float32)
        accp += xp * tl.load(Wc + chp * 4 + j).to(tl.float32)
    y = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
    yp = (accp * tl.sigmoid(accp)).to(tl.bfloat16).to(tl.float32)
    om = t < T
    base = OUT + grp * T * 16 * D + t * 16 * D + hh * D
    if grp < 2:
        if SLOTS:
            rs = tl.rsqrt(tl.sum(tl.where(c < 122, y * y, 0.0), 1) / 122 + eps)[:, None]
        else:
            rs = tl.rsqrt(tl.sum(y * y, 1) / HD + eps)[:, None]
        GN = QG if grp == 0 else KG
        w = tl.load(GN + hh * HD + c); wp = tl.load(GN + hh * HD + pc)
        yn = (y * rs * w).to(tl.bfloat16).to(tl.float32)
        ypn = (yp * rs * wp).to(tl.bfloat16).to(tl.float32)
        cm = c < 64
        cs = tl.load(COS + t * 64 + c, mask=cm & om, other=1.0).to(tl.float32)
        sn = tl.load(SIN + t * 64 + c, mask=cm & om, other=0.0).to(tl.float32)
        sign = tl.where(c < 32, -1.0, 1.0)
        y = tl.where(cm, yn * cs + sign * ypn * sn, yn)
        if grp == 0:
            y = y * sc
        if SLOTS:      # dims 122..127: q [G1, G2, G3, 1, 1, 1], k [1, 1, 1, B1, B2, B3] with B = log sigmoid(b) - G
            Gs = tl.load(GQ + hh * T + t, mask=om, other=0.)
            if grp == 0:
                Xv = Gs
            else:
                Xv = tl.load(LB + hh * T + t, mask=om, other=0.) - Gs
            u1 = Xv.to(tl.bfloat16).to(tl.float32); r1 = Xv - u1; u2 = r1.to(tl.bfloat16).to(tl.float32); u3 = r1 - u2
            if grp == 0:
                sv = tl.where(c == 122, u1, tl.where(c == 123, u2, tl.where(c == 124, u3, 1.0)))
            else:
                sv = tl.where(c == 125, u1, tl.where(c == 126, u2, tl.where(c == 127, u3, 1.0)))
            y = tl.where(c < 122, y, sv)
    tl.store(base + c, y.to(tl.bfloat16), mask=om)
    if AUGK:
        e = tl.arange(0, 8)[None, :]
        Gv = tl.load(GQ + hh * T + t, mask=om, other=0.)
        if grp == 0:
            v1 = Gv.to(tl.bfloat16).to(tl.float32); r1 = Gv - v1; v2 = r1.to(tl.bfloat16).to(tl.float32); v3 = r1 - v2
            ext = tl.where(e == 0, v1, tl.where(e == 1, v2, tl.where(e == 2, v3, tl.where(e < 6, 1.0, 0.0))))
        elif grp == 1:
            Bv = tl.load(LB + hh * T + t, mask=om, other=0.) - Gv
            v1 = Bv.to(tl.bfloat16).to(tl.float32); r1 = Bv - v1; v2 = r1.to(tl.bfloat16).to(tl.float32); v3 = r1 - v2
            ext = tl.where(e < 3, 1.0, tl.where(e == 3, v1, tl.where(e == 4, v2, tl.where(e == 5, v3, 0.0))))
        else:
            ext = tl.zeros([BT, 8], tl.float32)
        tl.store(base + 128 + e, ext.to(tl.bfloat16), mask=om)


def m1prep(proj, d, cos, sin, G, LBv, tail=None):
    T = proj.shape[0]
    out = torch.empty(3, T, 16, D, device=proj.device, dtype=torch.bfloat16)
    tl_ = tail if tail is not None else proj
    _m1prep_k[(triton.cdiv(T, 32), 48)](proj, tl_, d['m1_conv'], d['m1_qg'], d['m1_kg'], cos, sin, G, LBv, out, T, proj.stride(0), 1e-6,
                                         (128 ** -0.5) if AUG else 1.0, HAS_TAIL=tail is not None, AUGK=AUG == 1, D=D, BT=32, HD=128, SLOTS=AUG == 2, num_warps=4)
    return out


@triton.jit
def _gnorm2_k(O, Z, W, Y, zs, osr, osh, eps, NH: tl.constexpr, HD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    hh = tl.arange(0, NH)[:, None]; cc = tl.arange(0, HD)[None, :]
    o = tl.load(O + r * osr + hh * osh + cc).to(tl.float32)
    rs = tl.rsqrt(tl.sum(o * o, 1) / HD + eps)
    on = (o * rs[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cc).to(tl.float32)
    y = (w * on).to(tl.bfloat16).to(tl.float32)
    z = tl.load(Z + r * zs + hh * HD + cc).to(tl.float32)
    y = y * z * tl.sigmoid(z)
    tl.store(Y + r * NH * HD + hh * HD + cc, y.to(tl.bfloat16))


def gnorm2(o, z, w, eps):
    """o [T, 16, D] strided view (SDPA output transposed), z [T, 2048] strided view -> [T, 2048]"""
    T = z.shape[0]
    y = torch.empty(T, 2048, device=z.device, dtype=torch.bfloat16)
    _gnorm2_k[(T,)](o, z, w, y, z.stride(0), o.stride(0), o.stride(1), eps, NH=16, HD=128, num_warps=8)
    return y


# ---------------------------------------------------------------- runtime
def rederive(d):
    I = d['I']; W_ = d['Wgu']
    d['Wgu_il'] = torch.stack([W_[:I], W_[I:]], 1).reshape(2 * I, -1).contiguous()
    d['Win_f'] = (d['Win'].float() * d['in1'][None, :]).to(torch.bfloat16).contiguous()
    d['Wgu_f'] = (d['Wgu_il'].float() * d['post1'][None, :]).to(torch.bfloat16).contiguous()


class Rt(TTL):
    def __init__(self, torso, sd=None, tau=1.0):
        super().__init__(torso, list(range(10)))
        self.zt = torch.zeros(3, 8224, device=self.dev, dtype=torch.bfloat16)
        # M1 weights per former GDN layer: from a trained m1lib checkpoint, else the untrained conversion (GDN's own weights)
        if sd is not None and any(k.startswith('lora.') for k in sd):
            sc = sd['_meta']['lora_scale']
            with torch.no_grad():
                for i, d in enumerate(self.layers):
                    for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                        if i in GDN and k in ('Win', 'Wo') and i in sd['_meta']['active']: continue
                        d[k] = (d[k].float() + sc * (sd[f'lora.{i}_{k}.B'].float().to(dev) @ sd[f'lora.{i}_{k}.A'].float().to(dev))).to(torch.bfloat16).contiguous()
                    rederive(d)
        for i in GDN:
            d = self.layers[i]
            g = (lambda n: sd[f'C.{i}.{n}'].to(dev).float()) if sd is not None else None
            Win = g('Win') if g else d['Win'].float(); Wo = g('Wo') if g else d['Wo'].float()
            d['m1_Win_f'] = (Win * d['in1'][None, :]).to(torch.bfloat16).contiguous()
            d['m1_Wo'] = Wo.to(torch.bfloat16).contiguous()
            d['m1_conv'] = (g('conv_w') if g else d['conv_w'].float()).contiguous()
            d['m1_qg'] = (g('qg') if g else torch.full((16, 128), tau, device=dev)).contiguous()
            d['m1_kg'] = (g('kg') if g else torch.ones(16, 128, device=dev)).contiguous()
            d['m1_gn'] = (g('gn_w') if g else d['gn_w'].float()).to(torch.bfloat16).contiguous()
            d['m1_eA'] = (g('A_log') if g else d['A_log'].float()).exp().contiguous()
            d['m1_dtb'] = (g('dt_bias') if g else d['dt_bias'].float()).contiguous()

    def m1_mix(self, d, proj, cos, sin, T, c=None, qinfo=None):
        """M1 mixer on the rows of proj. c: cached state {'tail','k','v','G'} (question pass), qinfo: (mask, sub_idx) for branches."""
        if AUG:
            Gr = torch.empty(16, T, device=self.dev, dtype=torch.float32); LBv = torch.empty_like(Gr)
            _glue_k[(T,)](proj, d['m1_eA'], d['m1_dtb'], Gr, LBv, proj.stride(0), T, NH=16, num_warps=1)
            cs = Gr.cumsum(-1)                     # head-major: innermost-dim scan
            if c is not None:      # branch rows: G = G_state_last + cumsum within the branch
                cs0 = F.pad(cs, (1, 0)); G = (cs - cs0[:, qinfo[1]] + c['G'][:, None]).contiguous()
            else:
                G = cs
        else:
            G = LBv = self.zg if hasattr(self, 'zg') else torch.zeros(1, 16, device=self.dev)
        out = m1prep(proj, d, cos, sin, G, LBv, tail=None if c is None else c['tail'])
        q = out[0].reshape(1, T, 16, D).transpose(1, 2); k = out[1]; v = out[2]
        scale = 1.0 if AUG else 128 ** -0.5
        if c is None:
            o = F.scaled_dot_product_attention(q, k.reshape(1, T, 16, D).transpose(1, 2), v.reshape(1, T, 16, D).transpose(1, 2), is_causal=True, scale=scale)
        else:
            kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
            o = F.scaled_dot_product_attention(q, kk.reshape(1, -1, 16, D).transpose(1, 2), vv.reshape(1, -1, 16, D).transpose(1, 2),
                                               attn_mask=qinfo[0][None, None], scale=scale)
        o = gnorm2(o[0].transpose(0, 1), proj[:, 6144:8192], d['m1_gn'], self.eps)
        return o, (k, v, G)

    @torch.no_grad()
    def seq(self, ids, kind, pos0=0):
        """one causal sequence, all 24 layers, fold path. kind 'hob' or 'm1'. -> final-normed hidden [T, 2048]"""
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(pos0, pos0 + T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            if kind == 'm1' and i in GDN:
                proj = tgemm(x, d['m1_Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                o, _ = self.m1_mix(d, proj, cos, sin, T); Wo = d['m1_Wo']
            elif d['type'] == 'linear_attention':
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                qkv3 = conv_l2(proj, d['conv_w'])
                o, _ = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], proj[:, 8208:8224].reshape(1, T, 16), proj[:, 8192:8208].reshape(1, T, 16),
                                              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps); Wo = d['Wo']
            else:
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), k.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate; Wo = d['Wo']
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(o, Wo, epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps)

    @torch.no_grad()
    def state_pass(self, ids, kind):
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1); cache = {}
        for i, d in enumerate(self.layers):
            if kind == 'm1' and i in GDN:
                proj = tgemm(x, d['m1_Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                o, (k, v, G) = self.m1_mix(d, proj, cos, sin, T)
                cache[i] = {'tail': proj[-3:], 'k': k, 'v': v, 'G': G[:, -1].contiguous() if AUG else G}; Wo = d['m1_Wo']
            elif d['type'] == 'linear_attention':
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                qkv3 = conv_l2(proj, d['conv_w'])
                o, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], proj[:, 8208:8224].reshape(1, T, 16), proj[:, 8192:8208].reshape(1, T, 16),
                                              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'],
                                              use_beta_sigmoid_in_kernel=True, output_final_state=True)
                cache[i] = {'tail': proj[-3:], 'S': S}
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps); Wo = d['Wo']
            else:
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                cache[i] = {'k': k, 'v': v}
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), k.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate; Wo = d['Wo']
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(o, Wo, epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return cache

    @torch.no_grad()
    def q_pass(self, qids, T, cache, n, gidx, cu, mask, sub, cos, sin, kind):
        """all question rows (packed branches) through all 24 layers against the state cache"""
        R = qids.shape[1]
        x = F.embedding(qids, self.embed).reshape(R, -1)
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            c = cache[i]
            if kind == 'm1' and i in GDN:
                proj = tgemm(x, d['m1_Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                # branch conv history: every branch starts from the state's last 3 rows -> gather form (as hobson's q_pass)
                buf = torch.cat([c['tail'][:, :6144], proj[:, :6144]], 0)
                w = d['m1_conv']
                acc = buf[gidx[:, 0]].float() * w[:, 0] + buf[gidx[:, 1]].float() * w[:, 1] + buf[gidx[:, 2]].float() * w[:, 2] + buf[gidx[:, 3]].float() * w[:, 3]
                pc = torch.cat([acc.to(torch.bfloat16), proj[:, 6144:]], 1)          # conv applied; m1prep with identity conv below
                o, _ = self.m1_mix_q(d, pc, cos, sin, R, c, (mask, sub)); Wo = d['m1_Wo']
            elif d['type'] == 'linear_attention':
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                buf = torch.cat([c['tail'][:, :6144], proj[:, :6144]], 0)
                w = d['conv_w'].float()
                acc = buf[gidx[:, 0]].float() * w[:, 0] + buf[gidx[:, 1]].float() * w[:, 1] + buf[gidx[:, 2]].float() * w[:, 2] + buf[gidx[:, 3]].float() * w[:, 3]
                y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).float().reshape(R, 48, 128)
                y = torch.cat([y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6), y[:, 32:]], 1).to(torch.bfloat16).reshape(R, 3, 16, 128)
                init = c['S'].expand(n, -1, -1, -1).contiguous()
                ob, _ = chunk_gated_delta_rule(y[:, 0][None].contiguous(), y[:, 1][None].contiguous(), y[:, 2][None].contiguous(),
                                               proj[:, 8208:8224].reshape(1, R, 16), proj[:, 8192:8208].reshape(1, R, 16), use_qk_l2norm_in_kernel=False,
                                               use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                               initial_state=init, cu_seqlens=cu)
                o = gnorm(ob[0].contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps); Wo = d['Wo']
            else:
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(R, 2, 256)
                kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                o = F.scaled_dot_product_attention(q.reshape(1, R, 8, 256).transpose(1, 2), kk.reshape(1, -1, 2, 256).transpose(1, 2),
                                                   vv.reshape(1, -1, 2, 256).transpose(1, 2), attn_mask=mask[None, None], enable_gqa=True)
                o = o.transpose(1, 2).reshape(R, 2048) * gate; Wo = d['Wo']
            ss = torch.zeros(R, device=self.dev, dtype=torch.float32)
            tgemm(o, Wo, epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(R, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps).reshape(R, -1)

    def m1_mix_q(self, d, pc, cos, sin, R, c, qinfo):
        """question rows: pc = [conv output (pre-SiLU) of q|k|v, z, b, a]; prep with an identity conv (weights e_3) and no tail"""
        if 'm1_id' not in d:
            d['m1_id'] = torch.zeros(6144, 4, device=self.dev); d['m1_id'][:, 3] = 1.0
        d0 = dict(d); d0['m1_conv'] = d['m1_id']
        return self.m1_mix(d0, pc, cos, sin, R, c, qinfo)


def qmeta(prs, T):
    toks = []; gidx = []; cu = [0]; spos = []; rows = []; sub = []
    for p in prs:
        L = len(p['q']); r0 = len(toks); rows.append((r0, L)); toks += p['q']; cu.append(cu[-1] + L); spos += list(range(L)); sub += [r0] * L
        for t in range(L): gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (t + j) for j in range(4)])
    R = len(toks)
    mask = torch.zeros(R, T + R, dtype=torch.bool, device=dev); mask[:, :T] = True
    for r0, L in rows: mask[r0:r0 + L, T + r0:T + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    return (toks, torch.tensor(gidx, device=dev), torch.tensor(cu, device=dev, dtype=torch.long), mask, torch.tensor(spos, device=dev, dtype=torch.float32), rows,
            torch.tensor(sub, device=dev, dtype=torch.long))


def med(ts):
    ts = sorted(ts)
    return dict(median=round(stt.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2), n=len(ts))


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out


def timed(g, buf, out, inputs):
    pins = [torch.from_numpy(np.asarray(x, dtype=np.int64)).pin_memory() for x in inputs]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pins:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        buf[0, :x.numel()].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


def main():
    from kitrun import load_P
    from pydantic import TypeAdapter
    import strands_decider.schema as SC
    from strands_decider.prompting import render_question, render_state
    sys.path.insert(0, os.path.expanduser('~/work/m1'))
    from m1lib import StdHead
    ta = TypeAdapter(SC.Question)
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    head = StdHead(Pm.model.head).to(dev).eval()
    sd = torch.load(os.path.expanduser(CKPT), map_location='cpu') if CKPT else None
    tau = float(os.environ.get('TAU', 1.0))
    rt = Rt(tm, sd, tau); temp = Pm.temp_for
    Pm.model.torso = None; import gc; gc.collect(); torch.cuda.empty_cache()

    def prep(state_text, qd):
        q = ta.validate_python(qd); rq = render_question(q)
        s, qs = eng._fit(state_text, [rq.text])
        return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)

    def make_B(kind, buf, prs, T):
        n = len(prs); toks, gidx, cu, mask, spos, rows, sub = qmeta(prs, T)
        qids = torch.tensor([toks], device=dev)
        fr = (spos + float(T))[:, None] * rt.inv[None, :]; fr = torch.cat([fr, fr], -1)
        qcos, qsin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        tdiv = [temp(p['rq'].kind) for p in prs]; nopt = max(len(p['opt']) for p in prs)
        oidx = [torch.tensor([r0 + o for o in p['opt']], device=dev) for (r0, L), p in zip(rows, prs)]

        def fn():
            cache = rt.state_pass(buf, kind)
            h = rt.q_pass(qids, T, cache, n, gidx, cu, mask, sub, qcos, qsin, kind)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j, (r0, L) in enumerate(rows):
                qv = head.q(head.norm(h[r0 + L - 1].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :ko.shape[0]] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def make_S(kind, buf, p, T):
        L = T + len(p['q']); oi = torch.tensor([T + o for o in p['opt']], device=dev); td = temp(p['rq'].kind)

        def fn():
            if kind == 'hob1':
                hh, _ = rt.fwd(buf)
            else:
                hh = rt.seq(buf, 'hob' if kind == 'hobS' else 'm1')
            lg = (head.k(head.norm(hh[oi].float())) @ head.q(head.norm(hh[L - 1].float()))) * head.scale / td
            return torch.softmax(lg, -1)[None]
        return fn

    if MODE == 'check':
        pd = {}
        for l in open(os.path.expanduser(sys.argv[3])):
            r = json.loads(l); pd.setdefault(r['id'], {})[r['q']] = r['probs']
        its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2 and x['id'] in pd][:8] + [x for x in EK.load_suite('JB-hard') if x['id'] in pd][:4] \
            + [x for x in EK.load_suite('CF-probe') if x['id'] in pd][:4]
        res = []
        for it in its:
            stx = render_state(it['state']); names = [q for q in it['questions'] if q in pd[it['id']]]; prs = [prep(stx, it['questions'][q]) for q in names]
            s = prs[0]['s']; T = len(s)
            with torch.inference_mode():
                pr = make_B('m1', torch.tensor([s], device=dev), prs, T)()
                prS = make_S('m1S', torch.tensor([s + prs[0]['q']], device=dev), prs[0], T)()
            for j, qn in enumerate(names):
                k = len(prs[j]['opt']); a_ = pr[j, :k].tolist(); b_ = [pd[it['id']][qn][lab] for lab in prs[j]['rq'].slot_labels]
                res.append((int(np.argmax(a_)) == int(np.argmax(b_)), max(abs(x - y) for x, y in zip(a_, b_))))
                if j == 0:
                    c_ = prS[0, :k].tolist(); res.append((int(np.argmax(c_)) == int(np.argmax(b_)), max(abs(x - y) for x, y in zip(c_, b_))))
        r = dict(agree=sum(x[0] for x in res), n=len(res), dp_med=float(np.median([x[1] for x in res])), dp_max=max(x[1] for x in res), aug=AUG)
        print('fused runtime vs m1lib preds:', r, flush=True)
        json.dump(r, open(os.path.expanduser('~/work/m1/latcheck.json'), 'w'))
        return

    spec = {}; pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
    toks = [eng.tok(render_state(x), add_special_tokens=False)['input_ids'] for x in pool]
    states = {T: [t[:T] for t in toks if len(t) >= T][:REPS + 3] for T in TS}
    for T in TS: assert len(states[T]) == REPS + 3, (T, len(states[T]))

    if MODE == 'prof':
        from torch.profiler import profile, ProfilerActivity
        prs = [prep('S', spec[k]) for k in QSETS['Q1']]; prs4 = [prep('S', spec[k]) for k in QSETS['Q4']]
        out = {}
        for T in [int(x) for x in os.environ.get('PTS', '1000,4000').split(',')]:
            for kind in os.environ.get('PKINDS', 'hobS,m1S,hobB,m1B').split(','):
                if kind in ('hobS', 'm1S'):
                    buf = torch.tensor([states[T][0] + prs[0]['q']], device=dev); fn = make_S(kind, buf, prs[0], T)
                else:
                    buf = torch.tensor([states[T][0]], device=dev); fn = make_B('hob' if kind == 'hobB' else 'm1', buf, prs4, T)
                with torch.inference_mode():
                    for _ in range(3): fn()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as pf:
                        for _ in range(5): fn()
                        torch.cuda.synchronize()
                cat = {}
                for ev in pf.key_averages():
                    nm = ev.key.lower(); t = ev.device_time_total / 5 / 1000.0 if hasattr(ev, 'device_time_total') else ev.cuda_time_total / 5 / 1000.0
                    if t <= 0: continue
                    if 'flash' in nm or 'fmha' in nm or 'attention' in nm or 'efficient' in nm: k_ = 'attention'
                    elif '_gemm_k' in nm or 'gemm' in nm or 'cutlass' in nm or 'xmma' in nm: k_ = 'gemm'
                    elif 'chunk' in nm or 'delta' in nm or 'fwd_kernel' in nm or 'recompute' in nm or 'solve' in nm or 'kkt' in nm or 'wy' in nm: k_ = 'gdn'
                    elif 'conv' in nm or 'm1prep' in nm: k_ = 'conv_prep'
                    else: k_ = 'other'
                    cat[k_] = cat.get(k_, 0.0) + t
                    cat.setdefault('_kernels', {})
                    cat['_kernels'][ev.key[:60]] = round(cat['_kernels'].get(ev.key[:60], 0) + t, 3)
                kk = cat.pop('_kernels')
                out[f'{kind}_T{T}'] = dict({k: round(v, 3) for k, v in cat.items()}, total=round(sum(cat.values()), 3),
                                           top=dict(sorted(kk.items(), key=lambda x: -x[1])[:25]))
                print(kind, T, {k: v for k, v in out[f'{kind}_T{T}'].items() if k != 'top'}, flush=True)
        json.dump(out, open(os.path.expanduser(os.environ.get('PROFOUT', '~/work/m1/prof_m1.json')), 'w'), indent=1)
        return

    OUT = os.path.expanduser(os.environ.get('LATOUT', '~/work/m1/lat_m1.json'))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['meta'] = dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT, aug=AUG, head_dim=D,
                       note='bf16 lean2/TTL fused kernels, CUDA graph, exact T, fresh banking states per rep')
    KINDS = os.environ.get('KINDS', 'hob1,hobS,m1S,hobB,m1B').split(',')
    for qn, names in QSETS.items():
        prs = [prep('S', spec[k]) for k in names]
        for T in TS:
            cfgs = ([c for c in ('hob1', 'hobS', 'm1S') if c in KINDS] if qn == 'Q1' else []) + [c for c in ('hobB', 'm1B') if c in KINDS]
            for c in cfgs:
                key = f'{qn}_T{T}_{c}'
                if key in res: continue
                if c in ('hob1', 'hobS', 'm1S'):
                    if c == 'hob1': rt.set_fuse(T + len(prs[0]['q'])); rt.mask = None
                    buf = torch.tensor([states[T][0] + prs[0]['q']], device=dev); fn = make_S(c, buf, prs[0], T)
                else:
                    buf = torch.tensor([states[T][0]], device=dev); fn = make_B('hob' if c == 'hobB' else 'm1', buf, prs, T)
                g, out = capture(fn)
                res[key] = dict(timed(g, buf, out, states[T]), q_tokens=sum(len(p['q']) for p in prs))
                del g, out; torch.cuda.empty_cache()
                print(key, res[key], flush=True)
                json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
