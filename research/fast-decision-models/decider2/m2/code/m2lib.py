"""M2 library: segment-isolated state encoding, frozen-depth state rows and global question readers on hobson-v19
(H3/H7's differentiable lean forward: LoRA-merged hobson weights, fla kernels).

Rows: [state rows: U | content rows (segments)] then one branch per question = [END rows ('</state>\n')] + question tokens.
With every flag off this is exactly hobson's shared-prefix layout (H7 'seqs'); moving END into the branches is exact (state rows never
see END; END sees the whole state, as before).

Per layer i (Cfg):
  iso_attn[i] : attention, state rows: an ISOLATED row (segment >= 0) attends to U + earlier rows of its own segment; a READER state row
                (U, or seg -2 in gran 'const') attends causally to everything. pos 'local': isolated rows use segment-local RoPE
                positions (U at 0..u-1, segment rows at u, u+1, ..) -> a segment's computation is independent of where it sits (exactly
                precomputable); 'global': native positions.
  iso_gdn[i]  : GDN, state rows: conv history and delta-rule recurrence per segment, every segment starting from U's state S_U (U's conv
                tail). Reader rows use the native-order scan over all rows (= J9's exact affine composition of the segments).
  frozen[i]   : state rows are not updated in layer i (depth split: state rows go through layers < freeze only). Later layers still
                read them through their own projections (J3's bridge G for GDN, K/V for attention), from the frozen residual.
Question branches read ALL state rows in every layer: attention over all state keys (global RoPE positions), GDN from a composed state:
  comp 'exact' : the native-order scan over all state rows' (isolated) activations   = sum_j-free exact composition S <- A_j(S - S_U) + E_j
  comp 'sum'   : S = S_U + sum_j (E_j - S_U)    (order-free; no transfer matrices)
  comp 'last'  : S = E_last;  comp 'zero': no GDN read of the state (J3 bridge A)
  comp 'docfirst': exact composition of the reordered state [U][documents + hook notes][other segments], each part in native order
                 (compiled documents compose first, as J9's layout R; then one scan over the live rows)
  topk         : question rows keep only the top-k state keys per head at attention layers (M5, exact for what it keeps)
"""
import os, sys, math, json
sys.path[:0] = [os.path.expanduser('~/work/m2')]
import torch, torch.nn as nn, torch.nn.functional as F
import h3lib as H
from h3lib import rms_zc, DENSE, chunk_gated_delta_rule
from h7lib import H7, _conv
import m2seg as SG

ATT = H.FULL_ATTN


def _rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


class Cfg:
    """name grammar (all optional, ';'-separated):  gran=nat|sec|const|blkN  iso=all|attn|gdn|none  lay=a-b (iso layers, inclusive)
    pos=local|global  comp=exact|sum|last|zero  freeze=k (state rows through layers < k)  dcomp=exact|sum|zero (GDN read in frozen layers)
    topk=K"""

    def __init__(self, spec=''):
        kv = dict(x.split('=') for x in spec.split(';') if x)
        self.spec = spec
        self.gran = kv.get('gran', 'nat'); self.iso = kv.get('iso', 'none'); self.pos = kv.get('pos', 'local')
        self.comp = kv.get('comp', 'exact'); self.freeze = int(kv.get('freeze', 24)); self.dcomp = kv.get('dcomp', self.comp)
        self.topk = int(kv.get('topk', 0))
        self.ro = kv.get('ro', '0') == '1'          # R order (J9 layout R): reader rows come after U + all documents (positions too)
        if self.ro: self.comp = 'docfirst'; self.dcomp = 'docfirst' if self.dcomp in ('exact', 'docfirst') else self.dcomp
        a, b = (int(x) for x in kv.get('lay', '0-23').split('-'))
        rng = set(range(a, b + 1))
        self.iso_attn = [(i in rng) and self.iso in ('all', 'attn') and i in ATT for i in range(24)]
        self.iso_gdn = [(i in rng) and self.iso in ('all', 'gdn') and i not in ATT for i in range(24)]
        self.frozen = [i >= self.freeze for i in range(24)]

    def __repr__(self): return f'Cfg({self.spec})'


class Item:
    """one request: state rows + question branches, with all index tensors. Built by M2.prep_item."""
    pass


class M2(H7):
    mem = None
    mem_scale = 1.0

    def add_mem(self, ks, r=32, alpha=32, seed=1):
        """J3-style memory adapters: LoRA on the Win projection of the frozen state rows, per (freeze depth, deep layer)"""
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.mem = nn.ModuleDict(); self.mem_scale = alpha / r
        for k in ks:
            for i in range(k, 24):
                out = self.L[i]['Win'].shape[0]
                md = nn.Module()
                md.A = nn.Parameter((torch.randn(r, 2048, generator=g) / math.sqrt(2048)).to(self.dev))
                md.B = nn.Parameter(torch.zeros(out, r, device=self.dev))
                self.mem[f'{k}_{i}'] = md
        return list(self.mem.parameters())

    # ------------------------------------------------------------------ inputs
    def setup(self):
        self.eng = self.p.eng; self.tok = self.eng.tok
        from pydantic import TypeAdapter
        import strands_decider.schema as SC
        from strands_decider.prompting import render_question, render_state
        self._ta = TypeAdapter(SC.Question); self._rq = render_question; self._rs = render_state

    def prep_q(self, st_text, qd, perm=None):
        q = self._ta.validate_python(qd)
        rq = self._rq(q, option_order=perm) if perm is not None else self._rq(q)
        s, qs = self.eng._fit(st_text, [rq.text])
        return dict(s=s, q=qs[0], opt=self.eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)

    def segs_for(self, st_text, s, gran):
        enc = self.tok(st_text, add_special_tokens=True, return_offsets_mapping=True)
        if list(enc['input_ids']) != list(s):
            return None
        return SG.token_segments(enc['offset_mapping'], st_text, gran)

    def build(self, s, ts, prs, cfg, cut=None):
        """s: state ids (with U ... END); ts: token_segments dict (or None -> one segment = native); prs: list of prep_q dicts.
        cut: optional (k_head, k_tail) to keep only s[:k_head] + s[-k_tail:] (training truncation; segments cut identically)."""
        dev = self.dev
        u, ne = (ts['u'], ts['ne']) if ts is not None else (0, 0)
        if ts is None:      # fall back: one segment covering the content, END stays in the state (exact hobson)
            seg = [0] * len(s); u = 0; ne = 0
        else:
            seg = [-1] * u + list(ts['seg'])
        core = list(s[:len(s) - ne]); end = list(s[len(s) - ne:])
        if cut is not None:
            kh, kt = cut
            if len(core) > kh + kt:
                core = core[:kh] + core[len(core) - kt:]; seg = seg[:kh] + seg[len(seg) - kt:]
        Tm = len(core)
        assert len(seg) == Tm
        it = Item(); it.Tm = Tm; it.u = u; it.ids_state = core; it.seg = seg
        # renumber segments in order of first appearance; -1 = U, -2 = reader
        remap = {}; sg = []
        for x in seg:
            if x >= 0:
                if x not in remap: remap[x] = len(remap)
                sg.append(remap[x])
            else:
                sg.append(x)
        nseg = len(remap); it.nseg = nseg
        kinds0 = ts['kinds'] if ts is not None else ['all']
        it.kinds = [None] * nseg
        for x0, x1 in remap.items(): it.kinds[x1] = kinds0[x0] if x0 < len(kinds0) else 'all'
        segt = torch.tensor(sg, device=dev, dtype=torch.long) if Tm else torch.zeros(0, dtype=torch.long, device=dev)
        it.segt = segt
        # local positions: U rows 0..u-1, isolated rows u + rank within segment, readers: global index
        lpos = []; cnt = [0] * nseg
        for t, x in enumerate(sg):
            if x >= 0: lpos.append(u + cnt[x]); cnt[x] += 1
            else: lpos.append(t)
        it.lpos = torch.tensor(lpos, device=dev, dtype=torch.float32)
        it.gpos = torch.arange(Tm, device=dev, dtype=torch.float32)
        iso_rows = [t for t, x in enumerate(sg) if x >= 0]
        rd_rows = [t for t, x in enumerate(sg) if x < 0]
        it.iso_idx = torch.tensor(iso_rows, device=dev, dtype=torch.long); it.rd_idx = torch.tensor(rd_rows, device=dev, dtype=torch.long)
        # isolated rows in segment order (for the per-segment scan), cu_seqlens
        order = sorted(iso_rows, key=lambda t: (sg[t], t))
        it.seg_order = torch.tensor(order, device=dev, dtype=torch.long)
        isdoc = [x >= 0 and it.kinds[x] in ('doc', 'note') for x in sg]
        perm = [t for t in range(Tm) if sg[t] == -1] + [t for t in range(Tm) if isdoc[t]] + [t for t in range(Tm) if sg[t] != -1 and not isdoc[t]]
        it.perm_doc = torch.tensor(perm, device=dev, dtype=torch.long)
        ppos = [0] * Tm
        for r_, t in enumerate(perm): ppos[t] = r_
        it.ppos = torch.tensor(ppos if cfg.ro else list(range(Tm)), device=dev, dtype=torch.long)
        cu = [0]
        for j in range(nseg): cu.append(cu[-1] + cnt[j])
        it.cu = torch.tensor(cu, device=dev, dtype=torch.long)
        # conv runs (J9's gather trick): each run = 3 history rows (-1 = zero) + its rows; cout[t] = position of row t's output
        runs = []
        uh = list(range(u))[-3:]; uh = [-1] * (3 - len(uh)) + uh
        if u: runs.append(([-1, -1, -1], list(range(u))))
        for j in range(nseg):
            runs.append((uh, order[cu[j]:cu[j + 1]]))
        # reader rows (after U): maximal native-contiguous blocks, history = previous 3 native rows
        rr = [t for t in rd_rows if t >= u]
        blk = []
        for t in rr:
            if blk and blk[-1][-1] == t - 1: blk[-1].append(t)
            else: blk.append([t])
        if cfg.ro and rr:       # one reader stream after U + documents (perm order): history = the last 3 rows before it in that order
            before = [t for t in perm if sg[t] == -1 or isdoc[t]][-3:]
            runs.append(([-1] * (3 - len(before)) + before, rr))
        else:
            for b in blk:
                h = list(range(max(0, b[0] - 3), b[0])); h = [-1] * (3 - len(h)) + h
                runs.append((h, b))
        cin = []; cout = [0] * Tm
        for h, rows in runs:
            cin += h
            for t in rows: cout[t] = len(cin); cin.append(t)
        it.cin = torch.tensor(cin, device=dev, dtype=torch.long); it.cout = torch.tensor(cout, device=dev, dtype=torch.long)
        # attention mask for isolated query rows over state keys: own segment causal, or key in U
        if Tm:
            ar = torch.arange(Tm, device=dev)
            causal = ar[None, :] <= ar[:, None]
            same = (segt[:, None] == segt[None, :]) & (segt[:, None] >= 0)
            keyU = (segt[None, :] == -1)
            it.mask_iso = (same | keyU) & causal          # used for isolated query rows
            it.mask_iso = it.mask_iso[it.iso_idx] if len(iso_rows) else None
        # branches
        it.br = []; ids = list(core); pos = list(ppos) if cfg.ro else list(range(Tm))
        for p in prs:
            q = end + list(p['q'])
            it.br.append(dict(r0=len(ids), L=len(q), opt=[len(end) + o for o in p['opt']], rq=p['rq']))
            ids += q; pos += list(range(Tm, Tm + len(q)))
        it.ids = ids; it.pos = pos; it.ne = len(end)
        it.cfg = cfg
        return it

    # ------------------------------------------------------------------ forward
    def layer_m2(self, i, x, it):
        cfg = it.cfg; d = self.L[i]; T = x.shape[0]; Tm = it.Tm; eps = self.eps; u = it.u
        frz = cfg.frozen[i]
        if frz and not it.br: return x
        h = self.bnorm(x, i, 0)
        proj = self.lin(h, i, 'Win', DENSE)
        mem = getattr(self, 'mem', None)
        if frz and mem is not None and f'{cfg.freeze}_{i}' in mem and Tm:     # memory adapter: deep layers reading the frozen state rows (J3)
            md = mem[f'{cfg.freeze}_{i}']
            dm = ((h[:Tm] @ md.A.t().to(h.dtype)) @ md.B.t().to(h.dtype)) * self.mem_scale
            proj = torch.cat([proj[:Tm] + dm, proj[Tm:]], 0)
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            iso = cfg.iso_gdn[i] and it.nseg > 0
            comp = cfg.dcomp if frz else cfg.comp
            outs = []
            S = None
            if Tm:
                if iso:
                    rs = raw[:Tm]
                    rp = torch.cat([rs, torch.zeros(1, rs.shape[1], device=rs.device, dtype=rs.dtype)], 0)
                    xin = rp[torch.where(it.cin < 0, torch.full_like(it.cin, Tm), it.cin)]
                    cv = _conv(xin[None], d['conv_w'])[0][it.cout]
                else:
                    cv = _conv(raw[:Tm][None], d['conv_w'])[0]
                q, k, v = cv.split(2048, dim=-1)
                q = q.reshape(Tm, 16, 128); k = k.reshape(Tm, 16, 128); v = v.reshape(Tm, 16, 128)
                gm = g[:Tm]; bm = beta[:Tm].to(q.dtype)
                need_full = (not frz) or comp in ('exact', 'docfirst') or not iso
                if need_full and cfg.ro and iso:     # R order: one scan over [U][documents][readers]; reader outputs + the question state
                    pd = it.perm_doc
                    oP, S = chunk_gated_delta_rule(q[pd][None], k[pd][None], v[pd][None], gm[pd][None], bm[pd][None], use_qk_l2norm_in_kernel=True,
                                                   output_final_state=True)
                    o_st = torch.empty_like(oP[0]).index_copy(0, pd, oP[0])
                elif need_full:
                    oA, S = chunk_gated_delta_rule(q[None], k[None], v[None], gm[None], bm[None], use_qk_l2norm_in_kernel=True, output_final_state=True)
                    o_st = oA[0]
                if iso and ((not frz) or comp in ('sum', 'last')):
                    if u:
                        _, SU = chunk_gated_delta_rule(q[None, :u], k[None, :u], v[None, :u], gm[None, :u], bm[None, :u], use_qk_l2norm_in_kernel=True,
                                                       output_final_state=True)
                    else:
                        SU = torch.zeros(1, 16, 128, 128, device=x.device, dtype=torch.float32)
                    so = it.seg_order
                    oB, E = chunk_gated_delta_rule(q[so][None], k[so][None], v[so][None], gm[so][None], bm[so][None],
                                                   initial_state=SU.expand(it.nseg, -1, -1, -1).contiguous(), use_qk_l2norm_in_kernel=True,
                                                   output_final_state=True, cu_seqlens=it.cu)
                    if not frz:
                        o_st = o_st.index_copy(0, so, oB[0])
                    if comp == 'sum': S = SU + (E - SU).sum(0, keepdim=True)
                    elif comp == 'last': S = E[-1:]
                if comp == 'docfirst' and iso and not cfg.ro:      # exact composition of the reordered state [U][documents + notes][everything else]
                    pd = it.perm_doc
                    _, S = chunk_gated_delta_rule(q[pd][None], k[pd][None], v[pd][None], gm[pd][None], bm[pd][None], use_qk_l2norm_in_kernel=True,
                                                  output_final_state=True)
                if comp == 'zero': S = None
                if not frz: outs.append(o_st)
                tail = raw[it.perm_doc[-3:]] if cfg.ro else raw[max(0, Tm - 3):Tm]
            else:
                tail = raw[:0]
            if tail.shape[0] < 3: tail = torch.cat([torch.zeros(3 - tail.shape[0], 6144, device=raw.device, dtype=raw.dtype), tail], 0)
            for br in it.br:
                s0, Lb = br['r0'], br['L']
                cs = _conv(torch.cat([tail, raw[s0:s0 + Lb]], 0)[None], d['conv_w'])[0, 3:]
                qb, kb, vb = cs.split(2048, dim=-1)
                ob, _ = chunk_gated_delta_rule(qb.reshape(1, Lb, 16, 128), kb.reshape(1, Lb, 16, 128), vb.reshape(1, Lb, 16, 128), g[s0:s0 + Lb][None],
                                               beta[s0:s0 + Lb][None].to(qb.dtype), initial_state=S, use_qk_l2norm_in_kernel=True)
                outs.append(ob[0])
            o = torch.cat(outs, 0)
            zz = z if not frz else z[Tm:]
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(zz.reshape(-1, 128).float())).to(x.dtype).reshape(-1, 2048)
        else:
            qg_ = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg_[..., :256], qg_[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
            cos, sin = it.cs_g
            qr = _rope(qh, cos, sin); kr = _rope(kk, cos, sin)
            outs = []
            if Tm and not frz:
                if cfg.iso_attn[i] and it.nseg > 0:
                    o_st = torch.empty(Tm, 8, 256, device=x.device, dtype=x.dtype)
                    ii = it.iso_idx
                    if cfg.pos == 'local':
                        cl, sl = it.cs_l
                        ql = _rope(qh[:Tm], cl, sl); kl = _rope(kk[:Tm], cl, sl)
                    else:
                        ql, kl = qr[:Tm], kr[:Tm]
                    Kx = kl.repeat_interleave(4, dim=1); Vx = v[:Tm].repeat_interleave(4, dim=1)
                    oi = F.scaled_dot_product_attention(ql[ii].transpose(0, 1)[None], Kx.transpose(0, 1)[None], Vx.transpose(0, 1)[None],
                                                        attn_mask=it.mask_iso[None, None])[0].transpose(0, 1)
                    o_st[ii] = oi
                    if len(it.rd_idx):
                        rd = it.rd_idx
                        pp = it.ppos
                        mr = pp[None, :] <= pp[rd][:, None]
                        Kg = kr[:Tm].repeat_interleave(4, dim=1); Vg = v[:Tm].repeat_interleave(4, dim=1)
                        o_st[rd] = F.scaled_dot_product_attention(qr[rd].transpose(0, 1)[None], Kg.transpose(0, 1)[None], Vg.transpose(0, 1)[None],
                                                                  attn_mask=mr[None, None])[0].transpose(0, 1)
                else:
                    o_st = F.scaled_dot_product_attention(qr[:Tm].transpose(0, 1)[None], kr[:Tm].transpose(0, 1)[None], v[:Tm].transpose(0, 1)[None],
                                                          is_causal=True, enable_gqa=True)[0].transpose(0, 1)
                outs.append(o_st)
            for br in it.br:
                s0, Lb = br['r0'], br['L']
                if cfg.topk and Tm > cfg.topk:
                    outs.append(self._topk_branch(qr[s0:s0 + Lb], kr[:Tm], v[:Tm], kr[s0:s0 + Lb], v[s0:s0 + Lb], cfg.topk))
                    continue
                K = torch.cat([kr[:Tm], kr[s0:s0 + Lb]], 0); V = torch.cat([v[:Tm], v[s0:s0 + Lb]], 0)
                ii_ = torch.arange(Lb, device=x.device)[:, None]; jj = torch.arange(Tm + Lb, device=x.device)[None, :]
                mask = (jj < Tm) | ((jj - Tm) <= ii_)
                ob = F.scaled_dot_product_attention(qr[s0:s0 + Lb].transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                                    attn_mask=mask[None, None], enable_gqa=True)[0].transpose(0, 1)
                outs.append(ob)
            gg = gate if not frz else gate[Tm:]
            o = (torch.cat(outs, 0) * torch.sigmoid(gg)).reshape(-1, 2048)
        if frz:
            xq = x[Tm:]
            xq = xq + self.lin(o, i, 'Wo', DENSE)
            h2 = self.bnorm(xq, i, 1)
            gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
            xq = xq + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)
            return torch.cat([x[:Tm], xq], 0)
        x = x + self.lin(o, i, 'Wo', DENSE)
        h2 = self.bnorm(x, i, 1)
        gu = self.lin(h2, i, 'Wgu', DENSE); I = d['I']
        return x + self.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)

    def _topk_branch(self, q, Ks, Vs, Kb, Vb, kk):
        """q [Lb, 8, 256]; state keys Ks [Tm, 2, 256]; branch keys Kb [Lb, 2, 256]. Keep the top-kk state keys per (row, head)."""
        Lb = q.shape[0]; sc = 256 ** -0.5
        qf = q.float().reshape(Lb, 2, 4, 256)
        ls = torch.einsum('lghd,tgd->lght', qf, Ks.float()) * sc                      # [Lb, 2, 4, Tm]
        thr = ls.topk(kk, dim=-1).values[..., -1:]
        ls = ls.masked_fill(ls < thr, float('-inf'))
        lb = torch.einsum('lghd,tgd->lght', qf, Kb.float()) * sc                      # [Lb, 2, 4, Lb]
        cm = torch.ones(Lb, Lb, dtype=torch.bool, device=q.device).tril()
        lb = lb.masked_fill(~cm[:, None, None, :], float('-inf'))
        p = torch.softmax(torch.cat([ls, lb], -1), -1)
        Tm = Ks.shape[0]
        o = torch.einsum('lght,tgd->lghd', p[..., :Tm], Vs.float()) + torch.einsum('lght,tgd->lghd', p[..., Tm:], Vb.float())
        return o.reshape(Lb, 8, 256).to(q.dtype)

    def fwd_m2(self, it, ckpt=False, keep=(), keep_rows=None):
        x = F.embedding(torch.as_tensor(it.ids, device=self.dev), self.embed)
        it.cs_g = self.cos_sin(torch.as_tensor(it.pos, device=self.dev, dtype=torch.float32))
        it.cs_l = self.cos_sin(it.lpos) if it.Tm else None
        self.kept = {}
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer_m2, i, x, it, use_reentrant=False)
            else:
                x = self.layer_m2(i, x, it)
            ov = getattr(it, 'override', None)
            if ov is not None and i in ov:          # splice precomputed rows (compiled segments) into the request
                rows, vals = ov[i]; x = x.index_copy(0, rows, vals.to(x.dtype))
            if i in keep: self.kept[i] = x[keep_rows] if keep_rows is not None else x[it.Tm:]
        return rms_zc(x, self.norm_w, self.eps)

    def logits_m2(self, it, h, head=None):
        head = head or self.head or self.head0
        out = []
        for br in it.br:
            r0, L = br['r0'], br['L']
            oi = torch.tensor([r0 + o for o in br['opt']], device=self.dev)
            lg = head(h[r0 + L - 1].float()[None], h[oi].float()[None])[0] / self.temp(br['rq'].kind)
            out.append(lg[:br['rq'].n_slots])
        return out

    def qrows(self, it):
        """row indices of every branch's option rows + answer row (dense-loss rows), in branch order"""
        rows = []
        for br in it.br:
            rows += [br['r0'] + o for o in br['opt']] + [br['r0'] + br['L'] - 1]
        return torch.tensor(rows, device=self.dev, dtype=torch.long)
