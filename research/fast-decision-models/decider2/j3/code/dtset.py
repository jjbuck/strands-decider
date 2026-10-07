"""J3 DT-set: option-order invariance BY CONSTRUCTION, on the depth-split decision transformer (dtlib.DT).

Each question becomes a small tree over the shared state:
  stem   = '<question type=..>\n{header}\n{instructions}\n<options>\n'          (one branch off the state)
  option = '{name} -- {desc}'  for each option, NO number, every option branch starts at the SAME position (one branch off the stem each)
  tail   = '\n</options>\n</question>\n<answer>'                              (one branch off the stem, positioned after the longest option)
  GDN    : stem continues the state's final GDN state; options and tail continue the stem's final state + conv tail (so no option sees another)
  attention: stem -> state + own causal; option -> state + stem + own causal; tail -> state + stem + ALL its options (a set: equal positions) + own causal
  head   : hobson's pointer head, decision row = last tail row, option k's row = its last token.
Permuting the options permutes identical, independent branches, and the tail reads them as a set => the distribution over labels is invariant
(up to floating-point summation order). Shallow layers 0..Ls-1 run the state rows; deep layers read the layer-Ls memory exactly as dtlib (bridge A/G).
"""
import os, sys
sys.path[:0] = [os.path.expanduser('~/work/j3')]
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from h3lib import rms_zc, DENSE, chunk_gated_delta_rule
from h7lib import _conv
from dtlib import rope, KV, FULL

HEAD = {'noul': 'Decide whether the statement is true of the state.', 'choice': 'Select exactly one option.',
        'score': 'Rate the state against the ordered levels below (lowest first).'}


def set_tokens(tok, pr):
    """pr = a prepped hobson question (dict with rq, qd). -> stem ids, [option ids], tail ids"""
    rq = pr['rq']; qd = pr['qd']
    ins = qd['instructions'] if isinstance(qd['instructions'], str) else __import__('json').dumps(qd['instructions'], indent=2, ensure_ascii=False)
    stem = f'<question type="{rq.kind}">\n{HEAD[rq.kind]}\n{ins.strip()}\n<options>\n'
    enc = lambda t: tok(t, add_special_tokens=False)['input_ids']
    opts = []
    for name, desc in zip(rq.slot_labels, rq.slot_descriptions):
        d = ' '.join((desc or '').split())
        opts.append(enc(f'{name} — {d}' if d else f'{name}'))
    return enc(stem), opts, enc('\n</options>\n</question>\n<answer>')


def build(m, s, prs):
    """-> ids, pos (absolute), meta for the question rows (R rows after the Tm state rows)"""
    tok = m.p.eng.tok; Tm = len(s); dev = m.dev
    ids = list(s); pos = list(range(Tm))
    segs = []      # (kind, q, start_row_in_R, length, parent_seg_index or -1)
    R = 0; readout = []
    for qi, pr in enumerate(prs):
        st, opts, tl = set_tokens(tok, pr)
        Ls_ = len(st); mx = max(len(o) for o in opts)
        si = len(segs); segs.append(('stem', qi, R, Ls_, -1)); ids += st; pos += list(range(Tm, Tm + Ls_)); R += Ls_
        orows = []
        for o in opts:
            segs.append(('opt', qi, R, len(o), si)); ids += o; pos += list(range(Tm + Ls_, Tm + Ls_ + len(o))); orows.append(R + len(o) - 1); R += len(o)
        segs.append(('tail', qi, R, len(tl), si)); ids += tl; pos += list(range(Tm + Ls_ + mx, Tm + Ls_ + mx + len(tl))); R += len(tl)
        readout.append((orows, R - 1))
    # conv gather into buf = [3 state-tail rows | R question rows]
    gidx = [None] * R
    for (kd, qi, r0, L, par) in segs:
        for t in range(L):
            row = []
            for j in range(4):
                u = t - 3 + j
                if u >= 0: row.append(3 + r0 + u)
                elif par < 0: row.append(u + 3)
                else:
                    _, _, p0, pl, _ = segs[par]; row.append(3 + p0 + pl + u)
            gidx[r0 + t] = row
    # attention mask [R, Tm + R]
    mask = torch.zeros(R, Tm + R, dtype=torch.bool, device=dev); mask[:, :Tm] = True
    for (kd, qi, r0, L, par) in segs:
        mask[r0:r0 + L, Tm + r0:Tm + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        if par >= 0:
            _, _, p0, pl, _ = segs[par]; mask[r0:r0 + L, Tm + p0:Tm + p0 + pl] = True
        if kd == 'tail':
            for (k2, q2, a0, L2, p2) in segs:
                if k2 == 'opt' and q2 == qi: mask[r0:r0 + L, Tm + a0:Tm + a0 + L2] = True
    lv1 = [i for i, sg in enumerate(segs) if sg[4] < 0]; lv2 = [i for i, sg in enumerate(segs) if sg[4] >= 0]
    rows1 = torch.tensor([r for i in lv1 for r in range(segs[i][2], segs[i][2] + segs[i][3])], device=dev)
    rows2 = torch.tensor([r for i in lv2 for r in range(segs[i][2], segs[i][2] + segs[i][3])], device=dev)
    cu1 = torch.tensor([0] + list(torch.cumsum(torch.tensor([segs[i][3] for i in lv1]), 0).tolist()), device=dev)
    cu2 = torch.tensor([0] + list(torch.cumsum(torch.tensor([segs[i][3] for i in lv2]), 0).tolist()), device=dev)
    par2 = torch.tensor([lv1.index(segs[i][4]) for i in lv2], device=dev)          # level-2 segment -> its stem's index among level-1
    meta = dict(Tm=Tm, R=R, gidx=torch.tensor(gidx, device=dev), mask=mask, rows1=rows1, rows2=rows2, cu1=cu1, cu2=cu2, par2=par2, n1=len(lv1),
                readout=readout)
    return ids, pos, meta


def gdn_tree(d, raw, g, beta, tail, S0, mt):
    """question rows of one GDN layer: conv (history from parent), level 1 (stems) from S0, level 2 (options, tails) from each stem's final state"""
    buf = torch.cat([tail, raw], 0); w = d['conv_w'].float()
    acc = sum(buf[mt['gidx'][:, j]].float() * w[:, j][None] for j in range(4))
    cs = F.silu(acc).to(raw.dtype)
    q, k, v = cs.split(2048, dim=-1)
    out = torch.empty(raw.shape[0], 16, 128, device=raw.device, dtype=raw.dtype)
    r1, r2 = mt['rows1'], mt['rows2']; n1 = mt['n1']

    def run(rows, cu, init):
        n = rows.numel()
        return chunk_gated_delta_rule(q[rows].reshape(1, n, 16, 128), k[rows].reshape(1, n, 16, 128), v[rows].reshape(1, n, 16, 128), g[rows][None],
                                      beta[rows][None].to(q.dtype), initial_state=init, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, output_final_state=True)
    o1, S1 = run(r1, mt['cu1'], S0.expand(n1, -1, -1, -1).contiguous() if S0 is not None else None)
    o2, _ = run(r2, mt['cu2'], S1[mt['par2']].contiguous())
    out = out.index_put((r1,), o1[0]).index_put((r2,), o2[0])
    return out


def layer_set(m, i, x, Tm, cos, sin, mt, deep, M, Ls, bridge):
    """shallow (deep=False): x = [state rows | question rows], state rows updated. deep: x = question rows only, state = memory M."""
    d = m.L[i]; eps = m.eps; T = x.shape[0]
    h = m.bnorm(x, i, 0); proj = m.lin(h, i, 'Win', DENSE)
    Ts = 0 if deep else Tm; R = T - Ts
    if d['type'] == 'linear_attention':
        raw = proj[:, :6144]; z = proj[:, 6144:8192]
        beta = torch.sigmoid(proj[:, 8192:8208].float()); g = -d['A_log'].float().exp() * F.softplus(proj[:, 8208:8224].float() + d['dt_bias'])
        outs = []; S0 = None; tail = torch.zeros(3, 6144, device=x.device, dtype=x.dtype)
        if not deep:
            cm = _conv(raw[:Tm][None], d['conv_w'])[0]; qm, km, vm = cm.split(2048, dim=-1)
            om, S0 = chunk_gated_delta_rule(qm.reshape(1, Tm, 16, 128), km.reshape(1, Tm, 16, 128), vm.reshape(1, Tm, 16, 128), g[:Tm][None],
                                            beta[:Tm][None].to(qm.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
            outs.append(om[0]); tl_ = raw[max(0, Tm - 3):Tm]
        elif bridge == 'G':
            hm = m.bnorm(M, i, 0); pm = m.mem_lin(hm, i, Ls, FULL); rawm = pm[:, :6144]
            bm = torch.sigmoid(pm[:, 8192:8208].float()); gm = -d['A_log'].float().exp() * F.softplus(pm[:, 8208:8224].float() + d['dt_bias'])
            cm = _conv(rawm[None], d['conv_w'])[0]; qm, km, vm = cm.split(2048, dim=-1)
            _, S0 = chunk_gated_delta_rule(qm.reshape(1, Tm, 16, 128), km.reshape(1, Tm, 16, 128), vm.reshape(1, Tm, 16, 128), gm[None],
                                           bm[None].to(qm.dtype), use_qk_l2norm_in_kernel=True, output_final_state=True)
            tl_ = rawm[max(0, Tm - 3):Tm]
        else:
            tl_ = tail
        if tl_.shape[0] < 3: tl_ = torch.cat([torch.zeros(3 - tl_.shape[0], 6144, device=x.device, dtype=x.dtype), tl_], 0)
        oq = gdn_tree(d, raw[Ts:], g[Ts:], beta[Ts:], tl_, S0, mt)
        o = torch.cat(outs + [oq], 0) if outs else oq
        of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
        o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
    else:
        qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
        kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = rope(rms_zc(qh, d['qn'], eps), cos, sin); kk = rope(rms_zc(kk, d['kn'], eps), cos, sin)
        outs = []
        if not deep:
            om = F.scaled_dot_product_attention(qh[:Tm].transpose(0, 1)[None], kk[:Tm].transpose(0, 1)[None], v[:Tm].transpose(0, 1)[None],
                                                is_causal=True, enable_gqa=True)[0].transpose(0, 1)
            outs.append(om); K, V = kk, v
        else:
            cmm, smm = mt['cos_m'], mt['sin_m']
            hm = m.bnorm(M, i, 0); pm = m.mem_lin(hm, i, Ls, KV)
            km = rope(rms_zc(pm[:, :512].reshape(Tm, 2, 256), d['kn'], eps), cmm, smm); vm = pm[:, 512:].reshape(Tm, 2, 256)
            K = torch.cat([km, kk], 0); V = torch.cat([vm, v], 0)
        ob = F.scaled_dot_product_attention(qh[Ts:].transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None],
                                            attn_mask=mt['mask'][None, None], enable_gqa=True)[0].transpose(0, 1)
        o = (torch.cat(outs + [ob], 0) * torch.sigmoid(gate)).reshape(T, 2048)
    x = x + m.lin(o, i, 'Wo', DENSE)
    h2 = m.bnorm(x, i, 1); gu = m.lin(h2, i, 'Wgu', DENSE); I = d['I']
    return x + m.lin(F.silu(gu[:, :I]) * gu[:, I:], i, 'Wd', DENSE)


def set_logits(m, s, prs, Ls, bridge='A', ckpt=False, keep=(), head=None):
    """-> list of [n_slots] logits; m.kept[i] = readout rows (per question: option rows in label order, then the answer row) after layer i"""
    head = head or m.head or m.head0
    ids, pos, mt = build(m, s, prs); Tm = mt['Tm']
    x = F.embedding(torch.as_tensor(ids, device=m.dev), m.embed)
    cos, sin = m.cos_sin(torch.as_tensor(pos, device=m.dev, dtype=torch.float32))
    mt['cos_m'], mt['sin_m'] = cos[:Tm], sin[:Tm]
    rd = torch.tensor([r for orows, a in mt['readout'] for r in orows + [a]], device=m.dev)
    m.kept = {}
    for i in range(Ls):
        args = (m, i, x, Tm, cos, sin, mt, False, None, Ls, bridge)
        x = checkpoint(layer_set, *args, use_reentrant=False) if (ckpt and torch.is_grad_enabled()) else layer_set(*args)
        if i in keep: m.kept[i] = x[Tm:][rd]
    M = x[:Tm]; xq = x[Tm:]; cq, sq = cos[Tm:], sin[Tm:]
    for i in range(Ls, 24):
        args = (m, i, xq, Tm, cq, sq, mt, True, M, Ls, bridge)
        xq = checkpoint(layer_set, *args, use_reentrant=False) if (ckpt and torch.is_grad_enabled() and xq.shape[0] > 256) else layer_set(*args)
        if i in keep: m.kept[i] = xq[rd]
    h = rms_zc(xq, m.norm_w, m.eps)
    out = []
    for (orows, a), pr in zip(mt['readout'], prs):
        lg = head(h[a].float()[None], h[torch.tensor(orows, device=m.dev)].float()[None])[0] / m.temp(pr['rq'].kind)
        out.append(lg[:pr['rq'].n_slots])
    return out
