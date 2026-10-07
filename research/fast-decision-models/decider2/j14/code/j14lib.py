"""J14 library: STATE-FIRST compiled question bundle ("in-place partial compile").

Request layout (hobson's own order and positions):
  [state rows: hobson, exact]  [question rows at their natural positions T .. T+Lq-1]
Question rows are of two kinds:
  compiled : computed ONCE per deployment in a universal context U = render_state('') (hobson's own no-state prompt), by the
             compile path (hobson weights + optional compile adapter / full-FT compile weights). The cache holds, per layer,
             GDN: pre-conv qkv rows + beta + g  (replayed from the state's final GDN state at runtime = exact affine composition),
             attention: post-norm PRE-RoPE K and V (rotated to the natural position at runtime).
  live     : option-end rows, the final '<answer>' row and optional extra rows; hobson weights (optional live-row LoRA),
             attend to the state + the question span (compiled + live) causally; GDN state = state's final state then the span.
The compiled rows' hidden states are context-free; everything else is hobson's computation.

Training: the state prefix is frozen hobson -> computed once without grad (cache: K/V, GDN final state, conv tail); the question
suffix (compile pass over U+Q and live pass) is differentiable.
"""
import os, sys, math
sys.path[:0] = [os.path.expanduser('~/work/j14')]
import torch, torch.nn as nn, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, fla_conv, chunk_gated_delta_rule

NAMES = ('Win', 'Wo', 'Wgu', 'Wd')


def _rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def conv_span(tail, raw, w):
    """depthwise causal conv (k=4) + SiLU over [tail(3); raw(L)] -> L rows. Differentiable (manual)."""
    buf = torch.cat([tail, raw], 0)
    L = raw.shape[0]; wf = w.float()
    acc = buf[0:L].float() * wf[:, 0] + buf[1:L + 1].float() * wf[:, 1] + buf[2:L + 2].float() * wf[:, 2] + buf[3:L + 3].float() * wf[:, 3]
    return F.silu(acc).to(raw.dtype)


class LoRA(nn.Module):
    def __init__(self, L, r, alpha, dev, seed=0, layers=range(24)):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.A = nn.ParameterDict(); self.B = nn.ParameterDict(); self.layers = list(layers)
        for i in self.layers:
            for k in NAMES:
                out, inp = L[i][k].shape
                self.A[f'{i}_{k}'] = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(dev))
                self.B[f'{i}_{k}'] = nn.Parameter(torch.zeros(out, r, device=dev))
        self.scale = alpha / r; self.r = r

    def delta(self, x, i, k):
        key = f'{i}_{k}'
        if key not in self.A: return None
        return ((x @ self.A[key].t().to(x.dtype)) @ self.B[key].t().to(x.dtype)) * self.scale


class J14(H.H3):
    def setup(self):
        self.tok = self.p.tok
        from strands_decider.prompting import render_state
        self.U = self.tok(render_state(''), add_special_tokens=False)['input_ids']
        self.Lc = None        # full-FT compile weights: list of {nm: tensor}
        self.clora = None     # compile-path LoRA
        self.llora = None     # live-row LoRA (question live rows only; never the state)
        self.chead = None     # optional trainable head
        self.live_ft = False      # True: ONE full-FT weight set (self.Lc) for state, live and compile rows
        self.conv_mode = 'natural'   # 'natural': conv over the natural span (compiled pre-conv rows); 'frozen': compiled post-conv rows constant

    # ------------------------------------------------------------------ weights
    def Ws(self, i, nm):
        """weights of the live path (state rows + live question rows): hobson, or the full-FT weights when live_ft"""
        return self.Lc[i][nm] if (self.live_ft and self.Lc is not None) else self.L[i][nm]

    def lin_h(self, x, i, nm, live=False):
        y = x @ self.Ws(i, nm).t()
        if live and self.llora is not None:
            dl = self.llora.delta(x, i, nm)
            if dl is not None: y = y + dl
        return y

    def lin_c(self, x, i, nm):
        W = self.Lc[i][nm] if self.Lc is not None else self.L[i][nm]
        y = x @ W.t()
        if self.clora is not None:
            dl = self.clora.delta(x, i, nm)
            if dl is not None: y = y + dl
        return y

    def make_fullft(self):
        """trainable bf16 copies of all 96 GEMM weights for the compile path (anchor = self.L)"""
        self.Lc = []
        ps = []
        for i in range(24):
            d = {}
            for nm in NAMES:
                p_ = nn.Parameter(self.L[i][nm].detach().clone()); d[nm] = p_; ps.append(p_)
            self.Lc.append(d)
        return ps

    def cos_sin(self, pos):
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    # ------------------------------------------------------------------ state prefix (hobson, cached)
    def prefix(self, s):
        grad = self.live_ft and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            dev = self.dev; T = len(s)
            x = F.embedding(torch.as_tensor(s, device=dev), self.embed)
            cos, sin = self.cos_sin(torch.arange(T, device=dev, dtype=torch.float32))
            cache = []
            for i in range(24):
                if grad:
                    from torch.utils.checkpoint import checkpoint
                    x, c = checkpoint(self._pre_layer, i, x, cos, sin, use_reentrant=False)
                else:
                    x, c = self._pre_layer(i, x, cos, sin)
                cache.append(c)
        return cache, T

    def _pre_layer(self, i, x, cos, sin):
        dev = self.dev; T = x.shape[0]; eps = self.eps
        if True:
            d = self.L[i]
            h = rms_zc(x, d['in_norm'], eps)
            proj = h @ self.Ws(i, 'Win').t()
            if d['type'] == 'linear_attention':
                raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
                beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
                cv = fla_conv(raw[None].contiguous(), d['conv_w'], None, activation='silu')
                cv = cv[0] if isinstance(cv, tuple) else cv
                cv = cv.reshape(T, 6144)
                q, k, v = cv.split(2048, dim=-1)
                o, S = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                              beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
                tail = raw[max(0, T - 3):]
                if tail.shape[0] < 3: tail = torch.cat([torch.zeros(3 - tail.shape[0], 6144, device=dev, dtype=raw.dtype), tail], 0)
                c = dict(S=S, tail=tail.contiguous())
                of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
                o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
            else:
                qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
                kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
                qh, kk = _rope(qh, cos, sin), _rope(kk, cos, sin)
                o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
                o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
                c = dict(K=kk.contiguous(), V=v.contiguous())
            x = x + o @ self.Ws(i, 'Wo').t()
            h2 = rms_zc(x, d['post_norm'], eps)
            gu = h2 @ self.Ws(i, 'Wgu').t(); I = d['I']
            x = x + (F.silu(gu[:, :I]) * gu[:, I:]) @ self.Ws(i, 'Wd').t()
        return x, c

    # ------------------------------------------------------------------ question suffix
    def suffix(self, cache, T, q, lp, gdn_mode='replay', keep=(), ckpt=False, comp=None, conv_mode=None):
        """q: question ids (Lq); lp: sorted live positions (must contain Lq-1 and the option rows).
        Returns final normed hidden of the live rows [nl, 2048]; self.kept[i] = live rows after layer i (for i in keep).
        comp: optional precomputed compile cache (list per layer) -> skips the compile pass (inference)."""
        dev = self.dev; Lq = len(q)
        lp_t = torch.as_tensor(lp, device=dev, dtype=torch.long)
        nl = len(lp); all_live = nl == Lq
        xl = F.embedding(torch.as_tensor([q[j] for j in lp], device=dev), self.embed)
        need_c = (not all_live) and comp is None
        u = len(self.U)
        if need_c:
            xc = F.embedding(torch.as_tensor(list(self.U) + list(q), device=dev), self.embed)
            csc = self.cos_sin(torch.arange(u + Lq, device=dev, dtype=torch.float32))
        else:
            xc = None; csc = None
        csn = self.cos_sin(torch.arange(T, T + Lq, device=dev, dtype=torch.float32))      # natural positions of the span
        cmask = torch.ones(Lq, dtype=torch.bool, device=dev); cmask[lp_t] = False         # compiled positions
        # attention mask for live queries over [state T | span Lq]
        jj = torch.arange(T + Lq, device=dev)[None, :]
        amask = (jj < T) | ((jj - T) <= lp_t[:, None])
        ctx = dict(lp=lp_t, Lq=Lq, u=u, T=T, nl=nl, all_live=all_live, csc=csc, csn=csn, cmask=cmask, amask=amask, gdn_mode=gdn_mode, comp=comp,
                   conv_mode=conv_mode or self.conv_mode)
        self.kept = {}; self.comp_out = [] if (need_c and not torch.is_grad_enabled()) else None
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                xl, xc = checkpoint(self._sfx_layer, i, xl, xc, cache[i], ctx, use_reentrant=False)
            else:
                xl, xc = self._sfx_layer(i, xl, xc, cache[i], ctx)
            if i in keep: self.kept[i] = xl
        return rms_zc(xl, self.norm_w, self.eps)

    def _sfx_layer(self, i, xl, xc, ci, ctx):
        d = self.L[i]; eps = self.eps
        lp = ctx['lp']; Lq = ctx['Lq']; u = ctx['u']; T = ctx['T']; all_live = ctx['all_live']; comp = ctx['comp']
        hl = rms_zc(xl, d['in_norm'], eps)
        projl = self.lin_h(hl, i, 'Win', live=True)
        if xc is not None:
            hc = rms_zc(xc, d['in_norm'], eps)
            projc = self.lin_c(hc, i, 'Win')
        gdn = d['type'] == 'linear_attention'
        cpre = None
        if gdn:
            rawl = projl[:, :6144]; zl = projl[:, 6144:8192]
            bl = projl[:, 8192:8208]; al = projl[:, 8208:8224]
            betal = torch.sigmoid(bl.float()); gl = -d['A_log'].float().exp() * F.softplus(al.float() + d['dt_bias'])
            if xc is not None:
                rawc = projc[:, :6144]; zc = projc[:, 6144:8192]; bc = projc[:, 8192:8208]; ac = projc[:, 8208:8224]
                betac = torch.sigmoid(bc.float()); gc = -d['A_log'].float().exp() * F.softplus(ac.float() + d['dt_bias'])
                # compile pass (standard causal over U+Q from zero state)
                Tc = rawc.shape[0]
                cvc = conv_span(torch.zeros(3, 6144, device=rawc.device, dtype=rawc.dtype), rawc, d['conv_w'])
                qc_, kc_, vc_ = cvc.split(2048, dim=-1)
                oc, _ = chunk_gated_delta_rule(qc_.reshape(1, Tc, 16, 128), kc_.reshape(1, Tc, 16, 128), vc_.reshape(1, Tc, 16, 128), gc[None],
                                               betac[None].to(qc_.dtype), use_qk_l2norm_in_kernel=True)
                ofc = oc.reshape(-1, 128).float(); ofc = ofc * torch.rsqrt(ofc.pow(2).mean(-1, keepdim=True) + eps)
                oc = ((d['gn_w'] * ofc.to(oc.dtype)).float() * F.silu(zc.reshape(-1, 128).float())).to(xc.dtype).reshape(Tc, 2048)
                raw_s, beta_s, g_s = rawc[u:], betac[u:], gc[u:]
                if self.comp_out is not None: self.comp_out.append(dict(raw=raw_s.detach(), beta=beta_s.detach(), g=g_s.detach(), cv=cvc[u:].detach(), ab=projc[u:, 8192:8224].detach()))
            elif comp is not None:
                raw_s, beta_s, g_s = comp[i]['raw'], comp[i]['beta'], comp[i]['g']
            if all_live:
                raw_n, beta_n, g_n = rawl, betal, gl
            else:
                raw_n = raw_s.index_copy(0, lp, rawl.to(raw_s.dtype))
                beta_n = beta_s.index_copy(0, lp, betal)
                g_n = g_s.index_copy(0, lp, gl)
                if ctx['gdn_mode'] == 'skip':
                    keepm = (~ctx['cmask']).float()
                    beta_n = beta_n * keepm[:, None]; g_n = g_n * keepm[:, None]
            cv = conv_span(ci['tail'], raw_n, d['conv_w'])
            if ctx['conv_mode'] == 'frozen' and not all_live:
                # compiled rows keep their compile-time post-conv q/k/v (fully constant cache); only live rows see natural neighbours
                cvs = cvc[u:] if xc is not None else comp[i]['cv']
                cv = cvs.index_copy(0, lp, cv.index_select(0, lp).to(cvs.dtype))
            q_, k_, v_ = cv.split(2048, dim=-1)
            on, _ = chunk_gated_delta_rule(q_.reshape(1, Lq, 16, 128), k_.reshape(1, Lq, 16, 128), v_.reshape(1, Lq, 16, 128), g_n[None],
                                           beta_n[None].to(q_.dtype), initial_state=ci['S'], use_qk_l2norm_in_kernel=True)
            ol = on[0] if all_live else on[0][lp]
            of = ol.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            ol = ((d['gn_w'] * of.to(ol.dtype)).float() * F.silu(zl.reshape(-1, 128).float())).to(xl.dtype).reshape(-1, 2048)
        else:
            nl = xl.shape[0]
            qg = projl[:, :4096].reshape(nl, 8, 512); ql, gatel = qg[..., :256], qg[..., 256:]
            kl = projl[:, 4096:4608].reshape(nl, 2, 256); vl = projl[:, 4608:5120].reshape(nl, 2, 256)
            ql = rms_zc(ql, d['qn'], eps); kl = rms_zc(kl, d['kn'], eps)
            if xc is not None:
                Tc = xc.shape[0]
                qgc = projc[:, :4096].reshape(Tc, 8, 512); qcc, gatec = qgc[..., :256], qgc[..., 256:]
                kcc = projc[:, 4096:4608].reshape(Tc, 2, 256); vcc = projc[:, 4608:5120].reshape(Tc, 2, 256)
                qcc = rms_zc(qcc, d['qn'], eps); kcc = rms_zc(kcc, d['kn'], eps)
                k_s, v_s = kcc[u:], vcc[u:]                     # pre-RoPE K, V of the question rows
                if self.comp_out is not None: self.comp_out.append(dict(K=k_s.detach(), V=v_s.detach()))
                cc, sc = ctx['csc']
                oc = F.scaled_dot_product_attention(_rope(qcc, cc, sc).transpose(0, 1)[None], _rope(kcc, cc, sc).transpose(0, 1)[None],
                                                    vcc.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
                oc = (oc[0].transpose(0, 1) * torch.sigmoid(gatec)).reshape(Tc, 2048)
            elif comp is not None:
                k_s, v_s = comp[i]['K'], comp[i]['V']
            if all_live:
                k_n, v_n = kl, vl
            else:
                k_n = k_s.index_copy(0, lp, kl.to(k_s.dtype)); v_n = v_s.index_copy(0, lp, vl.to(v_s.dtype))
            cn, sn = ctx['csn']
            k_n = _rope(k_n, cn, sn)
            ql = _rope(ql, cn[lp], sn[lp])
            K = torch.cat([ci['K'], k_n], 0); V = torch.cat([ci['V'], v_n], 0)
            ol = F.scaled_dot_product_attention(ql.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                                attn_mask=ctx['amask'][None, None], enable_gqa=True)
            ol = (ol[0].transpose(0, 1) * torch.sigmoid(gatel)).reshape(nl, 2048)
        xl = xl + self.lin_h(ol, i, 'Wo', live=True)
        h2 = rms_zc(xl, d['post_norm'], eps)
        gu = self.lin_h(h2, i, 'Wgu', live=True); I = d['I']
        xl = xl + self.lin_h(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', live=True)
        if xc is not None:
            if i == 23:
                xc = None      # last layer: the compile rows' outputs are not needed
            else:
                xc = xc + self.lin_c(oc, i, 'Wo')
                h2c = rms_zc(xc, d['post_norm'], eps)
                guc = self.lin_c(h2c, i, 'Wgu')
                xc = xc + self.lin_c(F.silu(guc[:, :I]) * guc[:, I:], i, 'Wd')
        return xl, xc

    # ------------------------------------------------------------------ readout
    def readout(self, hl, lp, opt, Lq, kind, n_slots, head=None):
        head = head or self.chead or self.head0
        pos = {p: j for j, p in enumerate(lp)}
        oi = torch.tensor([pos[o] for o in opt], device=self.dev)
        lg = head(hl[pos[Lq - 1]].float()[None], hl[oi].float()[None])[0] / self.temp(kind)
        return lg[:n_slots]


# ---------------------------------------------------------------------- live sets
def live_positions(tok, pr, mode):
    """pr: plib prep dict (q ids, opt offsets, rq). mode tokens joined by '+':
      'oa'  : option-end rows + final row (minimum for the pointer head)
      'sfx' : + every row after the last option end ('\\n</options>\\n</question>\\n<answer>')
      'okN' : + the last N rows of every option line
      'ob'  : + the whole options block
      'qN'  : + the last N rows before the options block (end of the instructions + '<options>' line)
      'all' : every row (= hobson)"""
    q = pr['q']; Lq = len(q); opt = list(pr['opt'])
    parts = mode.split('+')
    if 'all' in parts: return list(range(Lq))
    S = set(opt); S.add(Lq - 1)
    # first token of the options block
    rq = pr['rq']
    pre = rq.text[:rq.option_spans[0][0]]
    npre = len(tok(pre, add_special_tokens=False)['input_ids'])
    # front truncation never happens with max_length 16384; sanity: prefix ids must match
    ob0 = npre if q[:npre] == tok(pre, add_special_tokens=False)['input_ids'] else None
    if ob0 is None:   # fall back: one row before the first option end
        ob0 = max(0, opt[0] - 1)
    for p in parts:
        if p == 'oa': pass
        elif p == 'sfx': S.update(range(max(opt) + 1, Lq))
        elif p == 'ob' or p.startswith('ob@'):
            kmax = int(p[3:]) if '@' in p else 10 ** 9      # 'ob@K': whole options block only for questions with <= K options
            if len(opt) <= kmax: S.update(range(ob0, Lq))
        elif p.startswith('ok'):
            spec = p[2:]; kmin = 0
            if '@' in spec: spec, kmin = spec.split('@'); kmin = int(kmin)
            n = int(spec); prev = ob0
            if len(opt) <= kmin: continue        # 'okN@K': option tails only for questions with > K options
            for o in opt:
                S.update(range(max(prev, o - n + 1), o + 1)); prev = o + 1
        elif p.startswith('q') and p[1:].isdigit():
            n = int(p[1:]); S.update(range(max(0, ob0 - n), ob0))
        else:
            raise ValueError(p)
    return sorted(S)
