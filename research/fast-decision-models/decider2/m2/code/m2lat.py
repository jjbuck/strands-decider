"""M2 latency: the decision-native layout in d1/H4's fused bf16 runtime (tt_lean.TTL = lean2 kernels), against hobson in the same harness.

Layout 'm2' (= m2lib Cfg 'gran=..;iso=all;pos=local;freeze=k;comp=docfirst'):
  * U = the 4 tokens of '<state>\\n': compiled once: per shallow layer its K/V, S_U and conv tail; per deep layer the same from its frozen residual.
  * compiled segments (KB documents + hook notes): compiled once each, standalone ([U + segment], local positions): per layer K (roped at
    local positions; re-rotated by a scalar offset at request time) and V; per GDN layer the transfer A_d and end state E_d from S_U.
  * live state rows, layers 0..k-1: attention over U + own segment (local positions; dense bool mask, or one batched lower-right causal
    flash call when every segment is a B-token block); GDN conv per segment (U tail history) and one varlen scan, every segment from S_U.
    Question GDN state: fold of the compiled segments from S_U (S <- A_d (S - S_U) + E_d), then one native-order scan of the live rows.
  * deep layers k..23: J3's bridge G over the frozen live rows (one GEMM for all deep attention K/V; per deep GDN layer GEMM + conv + scan).
  * question pass: J3's RTD.q_pass (every question a varlen branch through all 24 layers reading the caches).
Anchors in the same harness: hob1 (hobson, one sequence; TTL.fwd), hobB (hobson state pass + packed question pass), dtG{k} (J3 depth split, G).
Discipline (d1/J3/J9): CUDA graph per exact shape, exclusive GPU, fresh inputs every rep, 3 warm-ups, REPS timed, median + p95;
timed span = H2D of the ids + graph replay + D2H of the probabilities.
python m2lat.py grid [CKPT]          -> res/lat_grid.json  (T = 64, 256, 1000, 4000; 1 and 4 questions; B-token blocks; compiled share c)
python m2lat.py real [CKPT]          -> res/lat_real.json  (J9's 84 eval-split requests, natural segments, compiled docs + notes)
python m2lat.py check PREDS [CKPT]   -> runtime (all-live and compiled) vs m2lib preds"""
import os, sys, json, time, statistics as stt, random
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import numpy as np, torch, torch.nn.functional as F
from tt_lean import TTL, tgemm, conv_l2, gnorm, attn_prep, LM, chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right
import evalkit as EK
import m2seg as SG

dev = 'cuda'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'grid'
ARG = sys.argv[2] if len(sys.argv) > 2 else ''
CKPT = os.environ.get('CKPT', '')
REPS = int(os.environ.get('REPS', 15))
KS = [int(x) for x in os.environ.get('KS', '8,12').split(',')]
QFUSED = os.environ.get('QFUSED', '0') == '1'     # one question: run it as TTL's fused cached pass (as hob1 / J9) instead of J3's q_pass glue
GRAN = os.environ.get('GRAN', 'nat')
RO = os.environ.get('RO', '0') == '1'           # R-order runtime (gran const;ro=1): live rows = one reader stream after the documents
Q4 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses']


def cgdr(q, k, v, a, b, d, init=None, cu=None, final=False):
    return chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'],
                                  use_beta_sigmoid_in_kernel=True, initial_state=init, cu_seqlens=cu, output_final_state=final)


def rope_rows(k, cos, sin):
    xr, xp = k[..., :64], k[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


class RTM(TTL):
    def __init__(self, torso, ks, mem=None):
        super().__init__(torso, list(range(10)))
        mem = mem or {}
        self.zt = torch.zeros(3, 8224, device=self.dev, dtype=torch.bfloat16)
        self.memw = {}; self.memg = {}
        for Ls in ks:
            blocks = []
            for i in range(Ls, 24):
                d = self.layers[i]
                dW = mem.get(f'{Ls}_{i}')
                if d['type'] == 'linear_attention':
                    W = torch.cat([d['Win'][:6144], d['Win'][8192:8224]], 0).float()
                    if dW is not None: W = W + torch.cat([dW[:6144], dW[8192:8224]], 0)
                    self.memg[(Ls, i)] = (W * d['in1'][None, :]).to(torch.bfloat16).contiguous()
                    continue
                W = d['Win'][4096:5120].float()
                if dW is not None: W = W + dW[4096:5120]
                blocks.append((W * d['in1'][None, :]).to(torch.bfloat16))
            self.memw[Ls] = torch.cat(blocks, 0).contiguous()
        self.GDN = [i for i, d in enumerate(self.layers) if d['type'] == 'linear_attention']
        self.ATT = [i for i, d in enumerate(self.layers) if d['type'] != 'linear_attention']

    def cs(self, pos):
        fr = pos.float()[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()

    def _mlp(self, x, o, d, T):
        ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
        tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
        mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
        ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
        tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return ss

    # ------------------------------------------------------------------ the isolated state pass (live rows)
    @torch.no_grad()
    def spass(self, ids, k, M, C, want_A=False):
        """ids [1, T] state rows (U excluded: compiled in C['U']). M: meta. C: {'U': per-layer U cache, 'S0': per-layer initial state of
        the question composition scan, 'docs_k'/'docs_v': per attention layer keys/values of the compiled segments (global positions)}.
        -> cache per layer for the question pass (+ per GDN layer 'E' end state from S_U and, if want_A, 'A' transfer: compile use)"""
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        ss = x.float().pow(2).sum(-1); cache = {}
        dk = C.get('docs_k'); dv = C.get('docs_v')
        for i in range(24):
            d = self.layers[i]; Uc = C['U'][i]
            deep = i >= k
            if deep:
                if i == k: y = tgemm(x, self.memw[k], epi=0, ss=ss, Kd=x.shape[1]); jdeep = 0
                if d['type'] != 'linear_attention':
                    kr = y[:, jdeep * 1024:jdeep * 1024 + 512].reshape(T, 2, 256); v = y[:, jdeep * 1024 + 512:(jdeep + 1) * 1024].reshape(T, 2, 256).contiguous(); jdeep += 1
                    kn = LM._rms_zc(kr, d['kn'], self.eps)
                    kg = rope_rows(kn, M['cos_g'], M['sin_g'])
                    ent = {'k': torch.cat([Uc['k']] + ([dk[i]] if dk else []) + [kg], 0), 'v': torch.cat([Uc['v']] + ([dv[i]] if dv else []) + [v], 0)}
                    if want_A: ent['k_loc'] = rope_rows(kn, M['cos_l'], M['sin_l']); ent['v_loc'] = v
                    cache[i] = ent
                    continue
                proj = tgemm(x, self.memg[(k, i)], epi=0, ss=ss, Kd=x.shape[1])          # [T, 6176]: qkv | b | a
                a = proj[:, 6160:6176]; b = proj[:, 6144:6160]
            else:
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                if d['type'] == 'linear_attention':
                    a = proj[:, 8208:8224]; b = proj[:, 8192:8208]
            if d['type'] == 'linear_attention' and M.get('stream'):      # R order: the live rows are one reader stream after U + documents
                W_ = proj.shape[1]
                buf = torch.cat([C['hist'][i], self.zt[:1, :W_], proj], 0)[M['cin']]
                qkv3 = conv_l2(buf.contiguous(), d['conv_w'])[:, M['cout']]
                ob, S = cgdr(qkv3[0][None], qkv3[1][None], qkv3[2][None], a[None], b[None], d, init=C['S0'][i], final=True)
                cache[i] = {'tail': proj[-3:], 'S': S}
                if deep: continue
                o = gnorm(ob[0].contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps)
            elif d['type'] == 'linear_attention':
                W_ = proj.shape[1]
                buf = torch.cat([Uc['tail'], self.zt[:1, :W_], proj], 0)[M['cin']]
                qkv3 = conv_l2(buf.contiguous(), d['conv_w'])[:, M['cout']]
                ent = {}
                if not deep or want_A:
                    so = M['so']
                    ob, E = cgdr(qkv3[0][so][None], qkv3[1][so][None], qkv3[2][so][None], a[so][None], b[so][None], d,
                                 init=Uc['S'].expand(M['nseg'], -1, -1, -1).contiguous(), cu=M['cu'], final=True)
                    ent['E'] = E
                if want_A:
                    I0 = torch.eye(128, device=self.dev, dtype=torch.float32)[None, None].expand(1, 16, 128, 128).contiguous()
                    _, A = cgdr(qkv3[0][None], qkv3[1][None], torch.zeros_like(qkv3[2][None]), a[None], b[None], d, init=I0, final=True)
                    ent['A'] = A
                _, S = cgdr(qkv3[0][None], qkv3[1][None], qkv3[2][None], a[None], b[None], d, init=C['S0'][i], final=True)
                ent['tail'] = proj[-3:]; ent['S'] = S
                cache[i] = ent
                if deep: continue
                o = torch.empty(T, 16, 128, device=self.dev, dtype=torch.bfloat16); o[so] = ob[0]
                o = gnorm(o.contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps)
            elif M.get('stream'):
                q, kg, gate = attn_prep(proj, d['qn'], d['kn'], M['cos_g'], M['sin_g'], self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256).contiguous()
                K = torch.cat([Uc['k']] + ([dk[i]] if dk else []) + [kg], 0); V = torch.cat([Uc['v']] + ([dv[i]] if dv else []) + [v], 0)
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), K.reshape(1, -1, 2, 256).transpose(1, 2),
                                                   V.reshape(1, -1, 2, 256).transpose(1, 2), attn_mask=causal_lower_right(T, K.shape[0]), enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
                cache[i] = {'k': K, 'v': V}
            else:
                q, kl, gate = attn_prep(proj, d['qn'], d['kn'], M['cos_l'], M['sin_l'], self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256).contiguous()
                Ukl, Uv = Uc['k'], Uc['v']                   # U at positions 0..u-1 (local == global)
                q3 = q.reshape(T, 8, 256); uu = Ukl.shape[0]
                o = torch.empty(T, 8, 256, device=self.dev, dtype=torch.bfloat16)
                if M.get('B'):          # fixed B-token blocks: one batched lower-right causal flash call (+ one for a partial last block)
                    B = M['B']; nb = T // B; nf = nb * B
                    if nb:
                        qb = q3[:nf].reshape(nb, B, 8, 256).transpose(1, 2)
                        kb = torch.cat([Ukl[None].expand(nb, -1, -1, -1), kl[:nf].reshape(nb, B, 2, 256)], 1).transpose(1, 2)
                        vb = torch.cat([Uv[None].expand(nb, -1, -1, -1), v[:nf].reshape(nb, B, 2, 256)], 1).transpose(1, 2)
                        o[:nf] = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=causal_lower_right(B, B + uu), enable_gqa=True).transpose(1, 2).reshape(nf, 8, 256)
                    segl = [slice(nf, T)] if nf < T else []
                else:
                    segl = M['segl']
                for sl in segl:         # natural segments: one lower-right causal flash call each, keys [U | segment]
                    L = (sl.stop - sl.start) if isinstance(sl, slice) else sl.numel()
                    ks = torch.cat([Ukl, kl[sl]], 0); vs = torch.cat([Uv, v[sl]], 0)
                    o[sl] = F.scaled_dot_product_attention(q3[sl][None].transpose(1, 2), ks[None].transpose(1, 2), vs[None].transpose(1, 2),
                                                           attn_mask=causal_lower_right(L, L + uu), enable_gqa=True)[0].transpose(0, 1)
                o = o.reshape(T, 2048) * gate
                kg = rope_rows(LM._rms_zc(proj[:, 4096:4608].reshape(T, 2, 256), d['kn'], self.eps), M['cos_g'], M['sin_g'])
                ent = {'k': torch.cat([Ukl] + ([dk[i]] if dk else []) + [kg], 0), 'v': torch.cat([Uv] + ([dv[i]] if dv else []) + [v], 0)}
                if want_A: ent['k_loc'] = kl; ent['v_loc'] = v
                cache[i] = ent
            ss = self._mlp(x, o, d, T)
        return cache

    @torch.no_grad()
    def u_pass(self, ids, k):
        """U alone: layers < k native (U sees only itself), then frozen; per layer K/V (positions 0..u-1), S_U (from zero), conv tail"""
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        cos, sin = self.cs(torch.arange(T, device=self.dev))
        ss = x.float().pow(2).sum(-1); C = {}
        for i in range(24):
            d = self.layers[i]; deep = i >= k
            if deep and i == k: y = tgemm(x, self.memw[k], epi=0, ss=ss, Kd=x.shape[1]); jdeep = 0
            if d['type'] == 'linear_attention':
                if deep:
                    proj = tgemm(x, self.memg[(k, i)], epi=0, ss=ss, Kd=x.shape[1]); a = proj[:, 6160:6176]; b = proj[:, 6144:6160]
                else:
                    proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]); a = proj[:, 8208:8224]; b = proj[:, 8192:8208]
                qkv3 = conv_l2(proj, d['conv_w'])
                o, S = cgdr(qkv3[0][None], qkv3[1][None], qkv3[2][None], a[None], b[None], d, final=True)
                tl = proj[-3:]
                if tl.shape[0] < 3: tl = torch.cat([torch.zeros(3 - tl.shape[0], proj.shape[1], device=self.dev, dtype=proj.dtype), tl], 0)
                C[i] = {'S': S, 'tail': tl.contiguous()}
                if deep: continue
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                if deep:
                    kr = y[:, jdeep * 1024:jdeep * 1024 + 512].reshape(T, 2, 256); v = y[:, jdeep * 1024 + 512:(jdeep + 1) * 1024].reshape(T, 2, 256).contiguous(); jdeep += 1
                    C[i] = {'k': rope_rows(LM._rms_zc(kr, d['kn'], self.eps), cos, sin), 'v': v}
                    continue
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                q, kk, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256).contiguous()
                C[i] = {'k': kk, 'v': v}
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), kk.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            ss = self._mlp(x, o, d, T)
        return C

    # ------------------------------------------------------------------ native anchors (J3 RTD)
    @torch.no_grad()
    def native_state(self, ids, k):
        """hobson state pass (native causal) through layers < k, then bridge G for the deep layers (k = 24: hobson)"""
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        cos, sin = self.cs(torch.arange(T, device=self.dev))
        ss = x.float().pow(2).sum(-1); cache = {}
        for i in range(24):
            d = self.layers[i]; deep = i >= k
            if deep and i == k: y = tgemm(x, self.memw[k], epi=0, ss=ss, Kd=x.shape[1]); jdeep = 0
            if d['type'] == 'linear_attention':
                if deep:
                    proj = tgemm(x, self.memg[(k, i)], epi=0, ss=ss, Kd=x.shape[1]); a = proj[:, 6160:6176]; b = proj[:, 6144:6160]
                else:
                    proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]); a = proj[:, 8208:8224]; b = proj[:, 8192:8208]
                qkv3 = conv_l2(proj, d['conv_w'])
                o, S = cgdr(qkv3[0][None], qkv3[1][None], qkv3[2][None], a[None], b[None], d, final=True)
                cache[i] = {'tail': proj[-3:], 'S': S}
                if deep: continue
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                if deep:
                    kr = y[:, jdeep * 1024:jdeep * 1024 + 512].reshape(T, 2, 256); v = y[:, jdeep * 1024 + 512:(jdeep + 1) * 1024].reshape(T, 2, 256).contiguous(); jdeep += 1
                    cache[i] = {'k': rope_rows(LM._rms_zc(kr, d['kn'], self.eps), cos, sin), 'v': v}
                    continue
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
                q, kk, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256).contiguous()
                cache[i] = {'k': kk, 'v': v}
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), kk.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            ss = self._mlp(x, o, d, T)
        return cache

    @torch.no_grad()
    def q_fused(self, qids, cache, Ttot):
        """one question as TTL's fused cached pass (conv with cached tail, lower-right causal flash over the state keys)"""
        R = qids.shape[1]
        c2 = {}
        for i, c in cache.items():
            if 'S' in c:
                tl_ = c['tail']
                if tl_.shape[1] != 8224: tl_ = torch.cat([tl_[:, :6144], torch.zeros(3, 8224 - 6144, device=self.dev, dtype=tl_.dtype)], 1)
                c2[i] = {'tail': tl_.contiguous(), 'S': c['S']}
            else:
                c2[i] = {'k': c['k'], 'v': c['v']}
        self.mask = causal_lower_right(R, Ttot + R); self.set_fuse(R)
        h, _ = self.fwd(qids, pos0=Ttot, cache=c2)
        self.mask = None
        return h

    @torch.no_grad()
    def q_pass(self, qids, cache, n, gidx, cu, mask, cos, sin):
        """J3 RTD.q_pass: all question rows (packed varlen branches) through all 24 layers reading the caches"""
        R = qids.shape[1]
        x = F.embedding(qids, self.embed).reshape(R, -1)
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            c = cache[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if d['type'] == 'linear_attention':
                buf = torch.cat([c['tail'][:, :6144], proj[:, :6144]], 0)
                w = d['conv_w'].float()
                acc = buf[gidx[:, 0]].float() * w[:, 0] + buf[gidx[:, 1]].float() * w[:, 1] + buf[gidx[:, 2]].float() * w[:, 2] + buf[gidx[:, 3]].float() * w[:, 3]
                y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).float().reshape(R, 48, 128)
                y = torch.cat([y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6), y[:, 32:]], 1).to(torch.bfloat16).reshape(R, 3, 16, 128)
                init = c['S'].expand(n, -1, -1, -1).contiguous()
                ob, _ = cgdr(y[:, 0][None].contiguous(), y[:, 1][None].contiguous(), y[:, 2][None].contiguous(), proj[:, 8208:8224].reshape(1, R, 16),
                             proj[:, 8192:8208].reshape(1, R, 16), d, init=init, cu=cu)
                o = gnorm(ob[0].contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(R, 2, 256)
                kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                o = F.scaled_dot_product_attention(q.reshape(1, R, 8, 256).transpose(1, 2), kk.reshape(1, -1, 2, 256).transpose(1, 2),
                                                   vv.reshape(1, -1, 2, 256).transpose(1, 2), attn_mask=mask[None, None], enable_gqa=True)
                o = o.transpose(1, 2).reshape(R, 2048) * gate
            ss = self._mlp(x, o, d, R)
        return LM._rms_zc(x, self.norm_w, self.eps).reshape(R, -1)


# ====================================================================== request meta
def seg_meta(rt, seg, gpos, u, B=None):
    """seg: per live row its segment id (>= 0, any order of first appearance); gpos: per live row its global position.
    -> M for RTM.spass (conv gather with U tail history, segment order + cu_seqlens, local / global cos-sin, mask)"""
    T = len(seg)
    remap = {}; sg = []
    for x in seg:
        if x not in remap: remap[x] = len(remap)
        sg.append(remap[x])
    nseg = len(remap); cnt = [0] * nseg; lpos = []
    for x in sg: lpos.append(u + cnt[x]); cnt[x] += 1
    order = sorted(range(T), key=lambda t: (sg[t], t))
    cu = [0]
    for j in range(nseg): cu.append(cu[-1] + cnt[j])
    cin = []; cout = [0] * T
    for j in range(nseg):
        cin += [0, 1, 2]
        for t in order[cu[j]:cu[j + 1]]: cout[t] = len(cin); cin.append(4 + t)
    M = dict(T=T, nseg=nseg, so=torch.tensor(order, device=dev), cu=torch.tensor(cu, device=dev, dtype=torch.long),
             cin=torch.tensor(cin, device=dev), cout=torch.tensor(cout, device=dev))
    M['cos_l'], M['sin_l'] = rt.cs(torch.tensor(lpos, device=dev))
    M['cos_g'], M['sin_g'] = rt.cs(torch.tensor(gpos, device=dev))
    if B: M['B'] = B
    segl = []
    for j in range(nseg):
        rows = order[cu[j]:cu[j + 1]]
        segl.append(slice(rows[0], rows[-1] + 1) if rows[-1] - rows[0] + 1 == len(rows) else torch.tensor(rows, device=dev))
    M['segl'] = segl
    return M


def stream_meta(rt, gpos):
    T = len(gpos)
    M = dict(T=T, stream=True, cin=torch.tensor([0, 1, 2] + [4 + t for t in range(T)], device=dev), cout=torch.arange(3, 3 + T, device=dev))
    M['cos_g'], M['sin_g'] = rt.cs(torch.tensor(gpos, device=dev))
    return M


def qmeta(prs, Ttot, end_ids):
    toks = []; gidx = []; cu = [0]; spos = []; rows = []
    for p in prs:
        q = list(end_ids) + list(p['q']); L = len(q); r0 = len(toks); rows.append((r0, L)); toks += q; cu.append(cu[-1] + L); spos += list(range(L))
        for t in range(L): gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (t + j) for j in range(4)])
    R = len(toks)
    mask = torch.zeros(R, Ttot + R, dtype=torch.bool, device=dev); mask[:, :Ttot] = True
    for r0, L in rows: mask[r0:r0 + L, Ttot + r0:Ttot + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    return toks, torch.tensor(gidx, device=dev), torch.tensor(cu, device=dev, dtype=torch.long), mask, torch.tensor(spos, device=dev, dtype=torch.float32) + Ttot, rows


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


def merge_ckpt(tm, path):
    """merge an m2train LoRA (all layers) into the HF torso; return head state"""
    sd = torch.load(os.path.expanduser(path), map_location=dev)
    sc = sd['_meta']['lora_scale']
    with torch.no_grad():
        for i, L in enumerate(tm.layers):
            dl = {k: sc * (sd[f'lora.{i}.{k}.B'].float() @ sd[f'lora.{i}.{k}.A'].float()) for k in ('Win', 'Wo', 'Wgu', 'Wd')}
            if tm.config.layer_types[i] == 'linear_attention':
                a = L.linear_attn; mods = [a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a]; wo = a.out_proj
            else:
                a = L.self_attn; mods = [a.q_proj, a.k_proj, a.v_proj]; wo = a.o_proj
            r = 0
            for mm in mods:
                n = mm.weight.shape[0]; mm.weight.copy_((mm.weight.float() + dl['Win'][r:r + n]).to(mm.weight.dtype)); r += n
            wo.weight.copy_((wo.weight.float() + dl['Wo']).to(wo.weight.dtype))
            m = L.mlp; I = m.gate_proj.weight.shape[0]
            m.gate_proj.weight.copy_((m.gate_proj.weight.float() + dl['Wgu'][:I]).to(m.gate_proj.weight.dtype))
            m.up_proj.weight.copy_((m.up_proj.weight.float() + dl['Wgu'][I:]).to(m.up_proj.weight.dtype))
            m.down_proj.weight.copy_((m.down_proj.weight.float() + dl['Wd']).to(m.down_proj.weight.dtype))
    ms = sd['_meta'].get('mem_scale', 1.0)
    mem = {k.split('.')[1]: None for k in sd if k.startswith('mem.')}
    for k in mem: mem[k] = ms * (sd[f'mem.{k}.B'].float() @ sd[f'mem.{k}.A'].float())
    return {k[5:]: v for k, v in sd.items() if k.startswith('head.')}, mem


def main():
    from kitrun import load_P
    from pydantic import TypeAdapter
    import strands_decider.schema as SC
    from strands_decider.prompting import render_question, render_state
    from h3lib import StdHead
    ta = TypeAdapter(SC.Question)
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng; tok = eng.tok
    head = StdHead(Pm.model.head).to(dev).eval(); mem = None
    if CKPT:
        hs, mem = merge_ckpt(tm, CKPT); head.load_state_dict(hs)
    rt = RTM(tm, KS, mem); temp = Pm.temp_for
    U = tok('<state>\n', add_special_tokens=False)['input_ids']; u = len(U)
    END = tok('</state>\n', add_special_tokens=False)['input_ids']
    UC = {k: rt.u_pass(torch.tensor([U], device=dev), k) for k in KS + [24]}

    def prep(state_text, qd):
        q = ta.validate_python(qd); rq = render_question(q)
        s, qs = eng._fit(state_text, [rq.text])
        return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)

    LIB = {}

    def compile_seg(ids, k):
        """a compiled segment: standalone [U + ids] through the layout (local positions u..u+L-1)"""
        key = (k, tuple(ids))
        if key in LIB: return LIB[key]
        L = len(ids)
        M = seg_meta(rt, [0] * L, list(range(u, u + L)), u)
        C = dict(U=UC[k], S0={i: UC[k][i]['S'] for i in rt.GDN})
        with torch.inference_mode():
            c = rt.spass(torch.tensor([ids], device=dev), k, M, C, want_A=True)
        ent = dict(L=L, A=torch.cat([c[i]['A'] for i in rt.GDN], 0), E=torch.cat([c[i]['S'] for i in rt.GDN], 0),
                   tail={i: c[i]['tail'] for i in rt.GDN}, k={i: c[i]['k_loc'] for i in rt.ATT}, v={i: c[i]['v_loc'] for i in rt.ATT})
        LIB[key] = ent
        return ent

    def rot(kc, dpos):
        fr = float(dpos) * rt.inv; c = torch.cat([fr, fr]).cos().to(kc.dtype); s_ = torch.cat([fr, fr]).sin().to(kc.dtype)
        xr, xp = kc[..., :64], kc[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
        return torch.cat([torch.cat([x1 * c[:32] - x2 * s_[:32], x2 * c[32:] + x1 * s_[32:]], -1), xp], -1)

    def make_m2(k, live_seg, live_gpos, comp_list, prs, Ttot, B=None, stream=False):
        """comp_list: list of (ids, global start) compiled segments in native order. Returns fn(buf) and the live buffer length."""
        n = len(prs); toks, gidx, cu, mask, spos, rows = qmeta(prs, Ttot, END)
        qids = torch.tensor([toks], device=dev); qcos, qsin = rt.cs(spos)
        tdiv = [temp(p['rq'].kind) for p in prs]; nopt = max(len(p['opt']) for p in prs)
        oidx = [torch.tensor([r0 + len(END) + o for o in p['opt']], device=dev) for (r0, L), p in zip(rows, prs)]
        M = seg_meta(rt, live_seg, live_gpos, u, B) if not stream else stream_meta(rt, live_gpos)
        ents = []
        for ids, gp in comp_list:      # per-row position offsets (a document segment has holes where its rank / score tokens were)
            dlt = torch.tensor([g - (u + j) for j, g in enumerate(gp)], device=dev)
            ents.append((compile_seg(ids, k), rt.cs(dlt)))
        SU = torch.cat([UC[k][i]['S'] for i in rt.GDN], 0)
        Uk = UC[k]
        last_is_comp = bool(comp_list) and (comp_list[-1][1][-1] == Ttot - 1)

        def fn(buf):
            S = SU
            for e, _ in ents: S = torch.matmul(e['A'], S - SU) + e['E']
            C = dict(U=Uk, S0={i: S[j:j + 1] for j, i in enumerate(rt.GDN)})
            if stream:      # conv history of the reader stream = the last 3 rows before it in R order (last document, else U)
                C['hist'] = {i: (ents[-1][0]['tail'][i] if ents else Uk[i]['tail']) for i in rt.GDN}
            if ents:
                C['docs_k'] = {i: torch.cat([rope_rows(e['k'][i], cd, sd) for e, (cd, sd) in ents], 0) for i in rt.ATT}
                C['docs_v'] = {i: torch.cat([e['v'][i] for e, _ in ents], 0) for i in rt.ATT}
            cache = rt.spass(buf, k, M, C)
            if last_is_comp:
                for i in rt.GDN: cache[i]['tail'] = ents[-1][0]['tail'][i]
            h = rt.q_fused(qids, cache, Ttot) if (QFUSED and n == 1) else rt.q_pass(qids, cache, n, gidx, cu, mask, qcos, qsin)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j, (r0, L) in enumerate(rows):
                qv = head.q(head.norm(h[r0 + L - 1].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :ko.shape[0]] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def make_native(k, prs, Ttot):
        n = len(prs); toks, gidx, cu, mask, spos, rows = qmeta(prs, Ttot, END)
        qids = torch.tensor([toks], device=dev); qcos, qsin = rt.cs(spos)
        tdiv = [temp(p['rq'].kind) for p in prs]; nopt = max(len(p['opt']) for p in prs)
        oidx = [torch.tensor([r0 + len(END) + o for o in p['opt']], device=dev) for (r0, L), p in zip(rows, prs)]

        def fn(buf):
            cache = rt.native_state(buf, k)
            h = rt.q_fused(qids, cache, Ttot) if (QFUSED and n == 1) else rt.q_pass(qids, cache, n, gidx, cu, mask, qcos, qsin)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j, (r0, L) in enumerate(rows):
                qv = head.q(head.norm(h[r0 + L - 1].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :ko.shape[0]] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def make_hob1(p, T):
        L = T + len(p['q']); oi = torch.tensor([T + o for o in p['opt']], device=dev); td = temp(p['rq'].kind)

        def fn(buf):
            rt.set_fuse(L); rt.mask = None
            hh, _ = rt.fwd(buf)
            lg = (head.k(head.norm(hh[oi].float())) @ head.q(head.norm(hh[L - 1].float()))) * head.scale / td
            return torch.softmax(lg, -1)[None]
        return fn

    def req_struct(st, s, gran):
        """natural segmentation of a real request -> (live ids, live seg, live gpos, compiled list [(ids, g0)], core length)"""
        enc = tok(st, add_special_tokens=True, return_offsets_mapping=True)
        assert list(enc['input_ids']) == list(s)
        ts = SG.token_segments(enc['offset_mapping'], st, gran)
        core = s[:len(s) - ts['ne']]; seg = ts['seg']
        live_ids, live_seg, live_g = [], [], []; comp = collections_od()
        isc = [x >= 0 and ts['kinds'][x] in ('doc', 'note') for x in seg]
        if RO:      # R order: U, then every compiled row (native order), then the live rows: positions follow that order
            pc = sum(isc); rk = [0, 0]; pos_of = []
            for t in range(len(seg)):
                if isc[t]: pos_of.append(u + rk[0]); rk[0] += 1
                else: pos_of.append(u + pc + rk[1]); rk[1] += 1
        for t, x in enumerate(seg):
            gpos = pos_of[t] if RO else u + t; tid = core[u + t]
            if isc[t]:
                e_ = comp.setdefault(x, [[], []]); e_[0].append(tid); e_[1].append(gpos)
            else:
                live_ids.append(tid); live_seg.append(x); live_g.append(gpos)
        # compiled segments must be contiguous runs to be re-rotated by one offset: split non-contiguous ones (never in 'nat')
        comp_list = [(v[0], v[1]) for v in comp.values()]
        return live_ids, live_seg, live_g, comp_list, len(core)

    def collections_od():
        import collections
        return collections.OrderedDict()

    res_path = os.path.expanduser(f'~/work/m2/res/lat_{MODE}{os.environ.get("TAG", "")}.json')
    res = json.load(open(res_path)) if os.path.exists(res_path) and MODE != 'check' else {}
    res['meta'] = dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT, ks=KS, gran=GRAN,
                       note='bf16 TTL fused kernels, CUDA graph per exact shape, fresh inputs')

    if MODE == 'check':
        # (a) all-live m2 runtime vs m2lib preds; (b) compiled docs vs all-live (precompilation exactness in the runtime)
        pd = json.load(open(os.path.expanduser(ARG))); k = KS[0]
        its = [x for x in EK.load_suite('REAL-agree') if x['id'] in pd and x['domain'] == 'banking_knowledge'][:12] + \
              [x for x in EK.load_suite('CF-probe') if x['id'] in pd][:4]
        rows = []
        for it in its:
            qn = list(it['questions'])[0]; st = render_state(it['state']); p = prep(st, it['questions'][qn]); s = p['s']
            li, ls_, lg_, cl, Tc = req_struct(st, s, GRAN)
            # all live: compiled segments become live rows (same segment ids, same global positions)
            buf2 = torch.tensor([li], device=dev)
            with torch.inference_mode():
                pc = make_m2(k, ls_, lg_, cl, [p], Tc, stream=RO)(buf2)[0, :len(p['opt'])].tolist()
            ref = [pd[it['id']][qn][lab] for lab in p['rq'].slot_labels]
            rows.append(dict(id=it['id'], agree=int(np.argmax(pc)) == int(np.argmax(ref)), dp=max(abs(a_ - b_) for a_, b_ in zip(pc, ref)), live=len(li), comp=sum(len(x[0]) for x in cl)))
            print(rows[-1], flush=True)
        out = dict(agree=sum(r['agree'] for r in rows), n=len(rows), dp_med=float(np.median([r['dp'] for r in rows])), dp_max=max(r['dp'] for r in rows), rows=rows)
        print('runtime (compiled docs) vs m2lib preds:', {k_: v for k_, v in out.items() if k_ != 'rows'}, flush=True)
        json.dump(out, open(os.path.expanduser(f'~/work/m2/res/latcheck_{k}.json'), 'w'), indent=1)
        return

    spec = {}; pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            pool.append((s_, it))

    if MODE == 'grid':
        banking = [it for s_, it in pool if it['domain'] == 'banking_knowledge']
        toks_ = [tok(render_state(it['state']), add_special_tokens=False)['input_ids'] for it in banking[:150] + banking[-60:]]
        toks_ += [a_ + b_ for a_, b_ in zip(toks_[::2], toks_[1::2])]
        B = int(os.environ.get('BLK', 256))
        for qset in ((Q4[:1],) if os.environ.get('Q1ONLY') else (Q4[:1], Q4)):
            prs = [prep('S', spec[q]) for q in qset]
            for T in [int(x) for x in os.environ.get('TS', '64,256,1000,4000').split(',')]:
                srcs = [t[u:] for t in toks_ if len(t) >= T + u][:REPS + 3]
                Ts = T - u          # T state tokens = U + (T - u) content rows (END rides with the questions)
                inputs = [x[:Ts] for x in srcs]
                Ttot = T
                def run(key, fn, inp):
                    if key in res: return
                    buf = torch.tensor([inp[0]], device=dev)
                    g, out = capture(lambda: fn(buf))
                    res[key] = dict(timed(g, buf, out, inp), q_tokens=sum(len(p['q']) + len(END) for p in prs)); del g, out; torch.cuda.empty_cache()
                    print(key, res[key], flush=True); json.dump(res, open(res_path, 'w'), indent=1)
                nq = len(qset)
                if nq == 1:
                    p0 = prs[0]
                    run(f'Q{nq}_T{T}_hob1', make_hob1(p0, T), [list(U) + x + list(END) + p0['q'] for x in inputs])
                run(f'Q{nq}_T{T}_hobB', make_native(24, prs, Ttot), [list(U) + x for x in inputs])
                for k in KS:
                    run(f'Q{nq}_T{T}_dtG{k}', make_native(k, prs, Ttot), [list(U) + x for x in inputs])
                for k in KS + [24]:
                    for c in (0.0, 0.55):
                        nc = int(round(c * Ts)) // B * B if T >= 2 * B else 0
                        if c > 0 and nc == 0: continue
                        nl = Ts - nc
                        Bk = B
                        live_seg = [j // B for j in range(nl)]; live_g = [u + nc + j for j in range(nl)]
                        comp = [(srcs[0][j0:j0 + B], list(range(u + j0, u + j0 + B))) for j0 in range(0, nc, B)]
                        fn = make_m2(k, live_seg, live_g, comp, prs, Ttot, B=Bk, stream=RO)
                        run(f'Q{nq}_T{T}_m2k{k}_c{c}', fn, [x[nc:] for x in inputs])

    if MODE == 'real':
        ids84 = [r['id'] for r in json.load(open(os.path.expanduser('~/work/m2/j9_lat_real_ids.json')))]
        byid = {it['id']: (s_, it) for s_, it in pool}
        rows = res.get('rows', [])
        doneids = {r['id'] for r in rows}
        for iid in ids84:
            if iid in doneids: continue
            s_, it = byid[iid]; qn = list(it['questions'])[0]
            st = render_state(it['state']); p = prep(st, it['questions'][qn]); s = p['s']
            li, ls_, lg_, cl, Tc = req_struct(st, s, GRAN)
            full = s + p['q']
            inp_n = [full] + [[full[(j * 7919 + r) % len(full)] for j in range(len(full))] for r in range(REPS + 2)]
            buf = torch.tensor([full], device=dev)
            fh = make_hob1(p, len(s))
            g, out = capture(lambda: fh(buf)); rn = timed(g, buf, out, inp_n); del g, out
            row = dict(suite=s_, id=iid, domain=it['domain'], T=len(full), state=len(s), live=len(li), compiled=sum(len(x[0]) for x in cl), nseg=len(set(ls_)), ncomp=len(cl),
                       hob1=rn['median'], hob1_p95=rn['p95'])
            for k in KS:
                inp_c = [li] + [[li[(j * 7919 + r) % len(li)] for j in range(len(li))] for r in range(REPS + 2)]
                for tag, clist, lids, lseg, lgp in (('m2c', cl, li, ls_, lg_),):
                    buf2 = torch.tensor([lids], device=dev)
                    fn = make_m2(k, lseg, lgp, clist, [p], Tc, stream=RO)
                    g, out = capture(lambda: fn(buf2)); r2 = timed(g, buf2, out, inp_c); del g, out
                    row[f'{tag}k{k}'] = r2['median']; row[f'{tag}k{k}_p95'] = r2['p95']
            LIB.clear(); import gc; gc.collect(); torch.cuda.empty_cache()
            rows.append(row); print(row, flush=True)
            res['rows'] = rows; json.dump(res, open(res_path, 'w'), indent=1)
        summ = {}
        for key in ['hob1'] + [f'm2ck{k}' for k in KS]:
            v = [r[key] for r in rows]; summ[key] = dict(mean=float(np.mean(v)), median=float(np.median(v)))
        for k in KS:
            summ[f'speedup_of_means_k{k}'] = summ['hob1']['mean'] / summ[f'm2ck{k}']['mean']
            for dom in ('banking_knowledge', 'retail'):
                rr = [r for r in rows if r['domain'] == dom]
                if rr: summ[f'{dom}_k{k}'] = dict(n=len(rr), hob1=float(np.mean([r['hob1'] for r in rr])), m2=float(np.mean([r[f'm2ck{k}'] for r in rr])))
        summ['tok_state'] = float(np.mean([r['state'] for r in rows])); summ['tok_live'] = float(np.mean([r['live'] for r in rows]))
        res['summary'] = summ; print(summ, flush=True)
        json.dump(res, open(res_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
