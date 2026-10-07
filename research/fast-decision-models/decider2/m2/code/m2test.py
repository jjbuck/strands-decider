"""M2 checks (box):
  1. native config (no isolation) == hobson's H7 shared-prefix teacher path (argmax, max|dp|) and vs evalkit hobson refs
  2. segmentation: whole-state tokenization == engine state ids (offset mapping usable) on the sweep items; segment statistics
  3. exact composition: native-order scan over segments == fold S <- A_j (S - S_U) + E_j (fla kernels, random activations)
  4. isolation sanity: with iso=all, a segment's state-row outputs do not change when another segment's tokens change
python m2test.py"""
import os, sys, json, collections, hashlib
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from m2lib import M2, Cfg
from h3lib import chunk_gated_delta_rule
from strands_decider.prompting import render_state

m = M2(); m.free_hf(); m.setup(); m.head = m.head0
out = {}
refs = {}
for s in ('REAL-agree', 'JB-all', 'CF-probe'):
    for iid, r in EK.load_refs(s).items(): refs[iid] = r.get('hobson', {})
its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2][:8] + EK.load_suite('JB-hard')[:4] + EK.load_suite('CF-probe')[:4]
# ---- 1
agree = n = 0; maxd = 0.0; aref = 0; cfg0 = Cfg('')
with torch.inference_mode():
    for it in its:
        st = render_state(it['state']); names = list(it['questions']); prs = [m.prep_q(st, it['questions'][q]) for q in names]; s = prs[0]['s']
        lt = m.statefirst_logits(s, [p['q'] for p in prs], prs)
        ts = m.segs_for(st, s, 'nat')
        I = m.build(s, ts, prs, cfg0); hh = m.fwd_m2(I); lm = m.logits_m2(I, hh)
        for qn, p, x, y in zip(names, prs, lt, lm):
            px, py = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
            maxd = max(maxd, float((px - py).abs().max())); n += 1; agree += int(px.argmax() == py.argmax())
            rd = refs.get(it['id'], {}).get(qn)
            if rd: aref += int(p['rq'].slot_labels[int(py.argmax())] == max(rd, key=rd.get))
out['native_vs_teacher'] = dict(agree=agree, n=n, max_dp=maxd, agree_evalkit_refs=aref)
print('1. native cfg vs H7 teacher:', out['native_vs_teacher'], flush=True)
# ---- 2
mism = 0; tot = 0; segstat = collections.defaultdict(list); kinds = collections.Counter(); u_ne = collections.Counter()
for suite in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-all'):
    for it in EK.load_suite(suite)[:150]:
        st = render_state(it['state']); qn = next(iter(it['questions']))
        p = m.prep_q(st, it['questions'][qn]); s = p['s']; tot += 1
        for gran in ('nat', 'sec', 'const'):
            ts = m.segs_for(st, s, gran)
            if ts is None:
                if gran == 'nat': mism += 1
                continue
            segstat[(suite, gran)].append((len(s), ts['nseg'], sum(1 for x in ts['seg'] if x >= 0)))
            if gran == 'nat':
                u_ne[(ts['u'], ts['ne'])] += 1
                for x in ts['seg']:
                    if x >= 0: kinds[ts['kinds'][x]] += 1
out['seg'] = dict(mismatch=mism, n=tot, u_ne={str(k): v for k, v in u_ne.items()}, kinds_tokens=dict(kinds.most_common()),
                  per={f'{a}|{b}': dict(n=len(v), mean_tok=sum(x[0] for x in v) / len(v), mean_seg=sum(x[1] for x in v) / len(v),
                                         iso_share=sum(x[2] for x in v) / max(1, sum(x[0] for x in v)),
                                         mean_seg_len=sum(x[2] for x in v) / max(1, sum(x[1] for x in v))) for (a, b), v in segstat.items()})
print('2. segmentation:', json.dumps(out['seg'], indent=1)[:3000], flush=True)
# ---- 3
torch.manual_seed(0); dev = m.dev
lens = [3, 57, 200, 129, 64, 301]
T = sum(lens); Hh = 16
q = torch.nn.functional.normalize(torch.randn(T, Hh, 128, device=dev), dim=-1).to(torch.bfloat16)
k = torch.nn.functional.normalize(torch.randn(T, Hh, 128, device=dev), dim=-1).to(torch.bfloat16)
v = torch.randn(T, Hh, 128, device=dev).to(torch.bfloat16)
g = -torch.rand(T, Hh, device=dev) * 0.3; beta = torch.rand(T, Hh, device=dev).to(torch.bfloat16)
with torch.inference_mode():
    _, S_full = chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None], use_qk_l2norm_in_kernel=True, output_final_state=True)
    u = lens[0]
    _, SU = chunk_gated_delta_rule(q[None, :u], k[None, :u], v[None, :u], g[None, :u], beta[None, :u], use_qk_l2norm_in_kernel=True, output_final_state=True)
    cu = [u]
    for L in lens[1:]: cu.append(cu[-1] + L)
    cut = torch.tensor([c - u for c in cu], device=dev)
    sl = slice(u, T); nb = len(lens) - 1
    _, E = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl], initial_state=SU.expand(nb, -1, -1, -1).contiguous(),
                                  use_qk_l2norm_in_kernel=True, output_final_state=True, cu_seqlens=cut)
    I0 = torch.eye(128, device=dev)[None, None].expand(nb, Hh, 128, 128).contiguous()
    _, A = chunk_gated_delta_rule(q[None, sl], k[None, sl], torch.zeros_like(v[None, sl]), g[None, sl], beta[None, sl], initial_state=I0,
                                  use_qk_l2norm_in_kernel=True, output_final_state=True, cu_seqlens=cut)
    S = SU.clone()
    for j in range(nb):
        S = torch.matmul(A[j:j + 1], S - SU) + E[j:j + 1]
    rel = float((S - S_full).norm() / S_full.norm())
    Ssum = SU + (E - SU).sum(0, keepdim=True)
    rels = float((Ssum - S_full).norm() / S_full.norm())
out['composition'] = dict(rel_err_affine_vs_scan=rel, rel_err_sum_vs_scan=rels)
print('3. composition:', out['composition'], flush=True)
# ---- 4
it = [x for x in EK.load_suite('REAL-agree') if x.get('domain') == 'banking_knowledge'][0]
st = render_state(it['state']); qn = next(iter(it['questions'])); p = m.prep_q(st, it['questions'][qn]); s = p['s']
ts = m.segs_for(st, s, 'nat'); cfg = Cfg('gran=nat;iso=all')
with torch.inference_mode():
    I1 = m.build(s, ts, [p], cfg); m.fwd_m2(I1, keep=(23,), keep_rows=torch.arange(I1.Tm, device=dev)); h1 = m.kept[23]
    seg = I1.segt; j_last = int(seg[-1]); j_other = int(seg[I1.u + 5])
    # perturb a token of a different segment (token in segment j_other) -> rows of segment j_last must be unchanged
    s2 = list(s); pos = I1.u + 5; s2[pos] = (s2[pos] + 17) % 150000
    I2 = m.build(s2, ts, [p], cfg); m.fwd_m2(I2, keep=(23,), keep_rows=torch.arange(I2.Tm, device=dev)); h2 = m.kept[23]
    rows_last = (seg == j_last).nonzero()[:, 0]; rows_other = (seg == j_other).nonzero()[:, 0]
    d_last = float((h1[rows_last].float() - h2[rows_last].float()).abs().max()); d_other = float((h1[rows_other].float() - h2[rows_other].float()).abs().max())
out['isolation'] = dict(j_last=j_last, j_other=j_other, max_change_other_segment_rows=d_last, max_change_perturbed_segment_rows=d_other)
print('4. isolation:', out['isolation'], flush=True)
json.dump(out, open(os.path.expanduser('~/work/m2/res/test.json'), 'w'), indent=1)
