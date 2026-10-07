"""H7 latency: hobson-v19 (+ H7 LoRA merged) in H4's fused runtime (tt_lean.TTL = d1 lean2 + cached-history conv + lower-right causal flash),
with the compiled question schema and H7's slot layout:
  compile once per deployment: bundle [Q1' .. Qn'] -> GDN states, conv tails, attention K/V, and the pointer head's option keys (state-independent)
  per request: state rows (continuing the cache) + n '<answer>' slot rows. Slot k: GDN = one step from the state's final GDN state (conv history = last
  3 state rows); attention = own question span + state + itself (dense mask, n rows); head = q(slot) . K_opt(k).
Baseline in the same harness ('plain' = hobson's own layout): state pass once, then each question as its own branch continuing the state cache.
Discipline (d1/F7/H4): CUDA graph per exact shape, exclusive GPU, fresh real states every rep, 3 warm-ups discarded, n=20 timed, median + p95;
timed span = H2D of the state ids + graph replay + D2H of the probabilities.
python h7lat.py check CKPT   (runtime vs h7lib reference dump)  |  python h7lat.py time CKPT"""
import os, sys, json, time, statistics as st, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
import numpy as np, torch, torch.nn.functional as F
from tt_lean import TTL, tgemm, conv_l2, conv_l2_tail, gnorm, attn_prep, add_rms, LM, chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right
import evalkit as EK

dev = 'cuda'
MODE = sys.argv[1]; CKPT = sys.argv[2] if len(sys.argv) > 2 else ''
REPS = 20
TS = [1000, 4000]
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4], 'Q15': Q15}
SMOKE = os.environ.get('H7LAT_SMOKE') == '1'
if SMOKE: REPS = 3; TS = [1000]; QSETS = {'Q15': Q15}


def merge_lora(tm, path):
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
            assert r == dl['Win'].shape[0]
            wo.weight.copy_((wo.weight.float() + dl['Wo']).to(wo.weight.dtype))
            m = L.mlp; I = m.gate_proj.weight.shape[0]
            m.gate_proj.weight.copy_((m.gate_proj.weight.float() + dl['Wgu'][:I]).to(m.gate_proj.weight.dtype))
            m.up_proj.weight.copy_((m.up_proj.weight.float() + dl['Wgu'][I:]).to(m.up_proj.weight.dtype))
            m.down_proj.weight.copy_((m.down_proj.weight.float() + dl['Wd']).to(m.down_proj.weight.dtype))
    return {k[5:]: v for k, v in sd.items() if k.startswith('head.')}


class RT(TTL):
    """TTL + H7 slot rows.  fwd_req(ids [Ls+n], P, cache, n, allow [n, P+Ls+n]) -> final normed hidden [Ls+n, 2048]"""

    @torch.no_grad()
    def fwd_req(self, ids, P, cache, n, allow):
        f = self.fuse; T = ids.shape[1]; Ls = T - n
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.cat([torch.arange(P, P + Ls, device=self.dev, dtype=torch.float32), torch.full((n,), float(P + Ls), device=self.dev)])
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        fold = 'fold' in f
        if fold: ss = x.float().pow(2).sum(-1)
        else: h = LM._rms_zc(x, self.layers[0]['in_norm'], self.eps)
        nl = len(self.layers)
        for i, d in enumerate(self.layers):
            c = cache[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]) if fold else self._mm(h, d['Win'])
            if d['type'] == 'linear_attention':
                pm = proj[:Ls]
                qkv3 = conv_l2_tail(pm, c['tail'], d['conv_w'])
                a = pm[:, 8208:8224].reshape(1, Ls, 16); b = pm[:, 8192:8208].reshape(1, Ls, 16)
                om, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                               A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True, initial_state=c['S'], output_final_state=True)
                # slots: causal conv (kernel 4) over [last 3 state rows, own row], SiLU, bf16 round, l2norm on q/k heads
                w = d['conv_w'].float()                                                      # [6144, 4]
                hist = (proj[Ls - 3:Ls, :6144].float() * w[:, :3].t()).sum(0)
                y = hist[None] + proj[Ls:, :6144].float() * w[:, 3][None]
                y = (y * torch.sigmoid(y)).to(torch.bfloat16).float().reshape(n, 48, 128)
                y[:, :32] = y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6)
                y = y.to(torch.bfloat16).reshape(n, 3, 16, 128)
                ps = proj[Ls:]
                ob, _ = chunk_gated_delta_rule(y[:, 0][:, None].contiguous(), y[:, 1][:, None].contiguous(), y[:, 2][:, None].contiguous(),
                                               ps[:, 8208:8224].reshape(n, 1, 16), ps[:, 8192:8208].reshape(n, 1, 16), use_qk_l2norm_in_kernel=False,
                                               use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                               initial_state=S.expand(n, -1, -1, -1).contiguous())
                o = torch.cat([om[0], ob[:, 0]], 0).contiguous()
                o = gnorm(o, proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)              # [P+Ls+n, 2, 256]
                kh = kk.reshape(1, -1, 2, 256).transpose(1, 2); vh = vv.reshape(1, -1, 2, 256).transpose(1, 2)
                qm = q[:Ls].reshape(1, Ls, 8, 256).transpose(1, 2)
                om = F.scaled_dot_product_attention(qm, kh[:, :, :P + Ls], vh[:, :, :P + Ls], attn_mask=self.mask, enable_gqa=True)
                qs = q[Ls:].reshape(1, n, 8, 256).transpose(1, 2)
                osl = F.scaled_dot_product_attention(qs, kh, vh, attn_mask=allow[None, None], enable_gqa=True)
                o = torch.cat([om, osl], 2).transpose(1, 2).reshape(T, 2048) * gate
            if fold:
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
                mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
                continue
            d_out = self._mm(o, d['Wo'])
            x, h2 = add_rms(x, d_out, d['post1'], self.eps)
            from tt_lean import silu_mul
            mm = tgemm(h2, d['Wgu_il'], epi=1, cfg=self.gcfg.get('swiglu'))
            x, h = add_rms(x, self._mm(mm, d['Wd']), (self.layers[i + 1]['in1'] if i + 1 < nl else self.norm1), self.eps)
        if fold: h = LM._rms_zc(x, self.norm_w, self.eps)
        return h.reshape(T, -1)


class RTS(RT):
    """slot SETS: fwd_sets(ids [Ls+R], P, cache, n, gidx [R,4], cu [n+1], mask [R, P+Ls+R]) -> final normed hidden [Ls+R, 2048]"""

    @torch.no_grad()
    def fwd_sets(self, ids, P, cache, n, gidx, cu, mask, spos):
        f = self.fuse; T = ids.shape[1]; R = gidx.shape[0]; Ls = T - R
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.cat([torch.arange(P, P + Ls, device=self.dev, dtype=torch.float32), spos + float(P + Ls)])
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1)          # always the folded-norm path (d1's long path); also used below 512 rows here
        for i, d in enumerate(self.layers):
            c = cache[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if d['type'] == 'linear_attention':
                pm = proj[:Ls]
                qkv3 = conv_l2_tail(pm, c['tail'], d['conv_w'])
                a = pm[:, 8208:8224].reshape(1, Ls, 16); b = pm[:, 8192:8208].reshape(1, Ls, 16)
                om, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                               A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True, initial_state=c['S'], output_final_state=True)
                w = d['conv_w'].float()
                buf = proj[Ls - 3:, :6144]                                                   # [3 + R, 6144]: state tail, then slot rows
                acc = buf[gidx[:, 0]].float() * w[:, 0] + buf[gidx[:, 1]].float() * w[:, 1] + buf[gidx[:, 2]].float() * w[:, 2] + buf[gidx[:, 3]].float() * w[:, 3]
                y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).float().reshape(R, 48, 128)
                y = torch.cat([y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6), y[:, 32:]], 1).to(torch.bfloat16).reshape(R, 3, 16, 128)
                ps = proj[Ls:]
                ob, _ = chunk_gated_delta_rule(y[:, 0][None].contiguous(), y[:, 1][None].contiguous(), y[:, 2][None].contiguous(),
                                               ps[:, 8208:8224].reshape(1, R, 16), ps[:, 8192:8208].reshape(1, R, 16), use_qk_l2norm_in_kernel=False,
                                               use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                               initial_state=S.expand(n, -1, -1, -1).contiguous(), cu_seqlens=cu)
                o = torch.cat([om[0], ob[0]], 0).contiguous()
                o = gnorm(o, proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                kh = kk.reshape(1, -1, 2, 256).transpose(1, 2); vh = vv.reshape(1, -1, 2, 256).transpose(1, 2)
                qm = q[:Ls].reshape(1, Ls, 8, 256).transpose(1, 2)
                om = F.scaled_dot_product_attention(qm, kh[:, :, :P + Ls], vh[:, :, :P + Ls], attn_mask=self.mask, enable_gqa=True)
                qs = q[Ls:].reshape(1, R, 8, 256).transpose(1, 2)
                osl = F.scaled_dot_product_attention(qs, kh, vh, attn_mask=mask[None, None], enable_gqa=True)
                o = torch.cat([om, osl], 2).transpose(1, 2).reshape(T, 2048) * gate
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps).reshape(T, -1)


def sets_meta(prs, spans, Pn, Ls):
    toks = []; gidx = []; cu = [0]; spos = []; rows = []
    for p in prs:
        t_ = [p['q'][o] for o in p['opt']] + [p['q'][-1]]; L = len(t_); r0 = len(toks)
        rows.append((r0, L)); toks += t_; cu.append(cu[-1] + L); spos += list(range(L))
        for t in range(L): gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (3 + t - 3 + j) for j in range(4)])
    R = len(toks)
    mask = torch.zeros(R, Pn + Ls + R, dtype=torch.bool, device=dev)
    for (a0, a1), (r0, L) in zip(spans, rows):
        mask[r0:r0 + L, a0:a1] = True; mask[r0:r0 + L, Pn:Pn + Ls] = True
        mask[r0:r0 + L, Pn + Ls + r0:Pn + Ls + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    return toks, torch.tensor(gidx, device=dev), torch.tensor(cu, device=dev, dtype=torch.long), mask, torch.tensor(spos, device=dev, dtype=torch.float32), rows


def med(ts):
    ts = sorted(ts)
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2), n=len(ts))


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
    ta = TypeAdapter(SC.Question)
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    from h3lib import StdHead
    head = StdHead(Pm.model.head).to(dev).eval()
    if CKPT:
        head.load_state_dict(merge_lora(tm, CKPT))
    rt = RTS(tm, list(range(10)))
    temp = Pm.temp_for

    def prep(state_text, qd):
        q = ta.validate_python(qd); rq = render_question(q)
        s, qs = eng._fit(state_text, [rq.text])
        return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)

    def compile_bundle(prs, Lmax):
        bundle = []; spans = []; offs = []
        for p in prs:
            offs.append(len(bundle)); spans.append((len(bundle), len(bundle) + len(p['q']) - 1)); bundle += p['q'][:-1]
        Pn = len(bundle)
        with torch.inference_mode():
            rt.set_fuse(Pn); hP, cache = rt.fwd(torch.tensor([bundle], device=dev), want_cache=True)
            Kopt = [head.k(head.norm(hP[torch.tensor([offs[j] + o for o in p['opt']], device=dev)].float())) for j, p in enumerate(prs)]
        return Pn, cache, spans, Kopt

    def req_fn(buf, Pn, cache, spans, Kopt, prs, Ls):
        n = len(prs)
        allow = torch.zeros(n, Pn + Ls + n, dtype=torch.bool, device=dev)
        for j, (a0, a1) in enumerate(spans): allow[j, a0:a1] = True
        allow[:, Pn:Pn + Ls] = True; allow[torch.arange(n), Pn + Ls + torch.arange(n)] = True
        tdiv = torch.tensor([temp(p['rq'].kind) for p in prs], device=dev)
        nopt = max(len(p['opt']) for p in prs)

        def fn():
            h = rt.fwd_req(buf, Pn, cache, n, allow)
            qv = head.q(head.norm(h[Ls:].float()))                                  # [n, 256]
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j in range(n):
                out[j, :Kopt[j].shape[0]] = (Kopt[j] @ qv[j]) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def req_fn_sets(buf, Pn, cache, spans, prs, Ls):
        n = len(prs)
        toks, gidx, cu, mask, spos, rows = sets_meta(prs, spans, Pn, Ls)
        tdiv = [temp(p['rq'].kind) for p in prs]
        nopt = max(len(p['opt']) for p in prs)
        oidx = [torch.arange(Ls + r0, Ls + r0 + L - 1, device=dev) for r0, L in rows]

        def fn():
            h = rt.fwd_sets(buf, Pn, cache, n, gidx, cu, mask, spos)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j, (r0, L) in enumerate(rows):
                qv = head.q(head.norm(h[Ls + r0 + L - 1].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :L - 1] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn, toks

    if MODE == 'checksets':
        pd = json.load(open(os.path.expanduser(sys.argv[3])))
        ref = [dict(state=it['state'], qs=it['questions'], p=pd[it['id']]) for it in [x for x in EK.load_suite('REAL-agree') if len(x['questions']) == 4][:8] + [x for x in EK.load_suite('LONG') if len(x['questions']) == 3][:4] + EK.load_suite('JB-all')[:6]]
        res = []
        for r in ref:
            st_ = render_state(r['state']); prs = [prep(st_, r['qs'][qn]) for qn in r['qs']]
            s = prs[0]['s']; Ls = len(s)
            Pn, cache, spans, _ = compile_bundle(prs, Ls + 64)
            toks = sets_meta(prs, spans, Pn, Ls)[0]
            rt.set_fuse(Ls + len(toks)); rt.mask = causal_lower_right(Ls, Pn + Ls)
            buf = torch.tensor([s + toks], device=dev)
            with torch.inference_mode(): pr = req_fn_sets(buf, Pn, cache, spans, prs, Ls)[0]()
            for j, qn in enumerate(r['qs']):
                k = len(prs[j]['opt']); a = pr[j, :k].tolist(); b = [r['p'][qn][lab] for lab in prs[j]['rq'].slot_labels]
                res.append((int(np.argmax(a)) == int(np.argmax(b)), max(abs(x - y) for x, y in zip(a, b))))
        print('sets runtime vs h7lib: argmax agree %d/%d, max|dp| median %.4f max %.4f' % (sum(x[0] for x in res), len(res), float(np.median([x[1] for x in res])), max(x[1] for x in res)), flush=True)
        json.dump(dict(agree=sum(x[0] for x in res), n=len(res), dp_med=float(np.median([x[1] for x in res])), dp_max=max(x[1] for x in res)),
                  open(os.path.expanduser('~/work/h7/latcheck_sets.json'), 'w'))
        return

    if MODE == 'check':
        pd = json.load(open(os.path.expanduser(sys.argv[3])))
        ref = [dict(state=it['state'], qs=it['questions'], p=pd[it['id']]) for it in [x for x in EK.load_suite('REAL-agree') if len(x['questions']) == 4][:8] + [x for x in EK.load_suite('LONG') if len(x['questions']) == 3][:4]]
        res = []
        for r in ref:
            st_ = render_state(r['state']); prs = [prep(st_, r['qs'][qn]) for qn in r['qs']]
            s = prs[0]['s']; Ls = len(s); n = len(prs)
            Pn, cache, spans, Kopt = compile_bundle(prs, Ls + n)
            rt.set_fuse(Ls + n); rt.mask = causal_lower_right(Ls, Pn + Ls)
            buf = torch.tensor([s + [p['q'][-1] for p in prs]], device=dev)
            with torch.inference_mode(): pr = req_fn(buf, Pn, cache, spans, Kopt, prs, Ls)()
            for j, qn in enumerate(r['qs']):
                k = len(prs[j]['opt']); a = pr[j, :k].tolist(); b = [r['p'][qn][lab] for lab in prs[j]['rq'].slot_labels]
                res.append((int(np.argmax(a)) == int(np.argmax(b)), max(abs(x - y) for x, y in zip(a, b))))
        print('runtime vs h7lib: argmax agree %d/%d, max|dp| median %.4f max %.4f' % (sum(x[0] for x in res), len(res), float(np.median([x[1] for x in res])), max(x[1] for x in res)), flush=True)
        json.dump(dict(agree=sum(x[0] for x in res), n=len(res), dp_med=float(np.median([x[1] for x in res])), dp_max=max(x[1] for x in res)),
                  open(os.path.expanduser('~/work/h7/latcheck.json'), 'w'))
        return

    # ---- timing
    spec = {}; pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
    toks = [eng.tok(render_state(x), add_special_tokens=False)['input_ids'] for x in pool]
    states = {T: [t[:T] for t in toks if len(t) >= T][:REPS + 3] for T in TS}
    for T in TS: assert len(states[T]) == REPS + 3, (T, len(states[T]))
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT, note='hobson-v19 + H7 LoRA merged, TTL fused runtime')}
    OUT = os.path.expanduser('~/work/h7/lat_h7%s.json' % ('_smoke' if SMOKE else ''))
    for qn, names in QSETS.items():
        prs = [prep('S', spec[k]) for k in names]; n = len(prs)
        for T in TS:
            Pn, cache, spans, Kopt = compile_bundle(prs, T + n)
            torch.cuda.synchronize(); t0 = time.perf_counter(); compile_bundle(prs, T + n); torch.cuda.synchronize(); tc = (time.perf_counter() - t0) * 1000
            rt.set_fuse(T + n); rt.mask = causal_lower_right(T, Pn + T)
            buf = torch.tensor([states[T][0] + [p['q'][-1] for p in prs]], device=dev)
            g, out = capture(req_fn(buf, Pn, cache, spans, Kopt, prs, T))
            res[f'{qn}_T{T}_schema'] = dict(timed(g, buf, out, states[T]), rows=T + n, bundle=Pn, compile_once_ms=round(tc, 1))
            del g, out; torch.cuda.empty_cache()
            # sets layout (the trained H7 layout): state + per-question slot sets
            toks = sets_meta(prs, spans, Pn, T)[0]
            rt.set_fuse(T + len(toks)); rt.mask = causal_lower_right(T, Pn + T)
            bufs = torch.tensor([states[T][0] + toks], device=dev)
            fn, _ = req_fn_sets(bufs, Pn, cache, spans, prs, T)
            g, out = capture(fn)
            res[f'{qn}_T{T}_sets'] = dict(timed(g, bufs, out, states[T]), rows=T + len(toks), bundle=Pn, compile_once_ms=round(tc, 1))
            del g, out; torch.cuda.empty_cache()
            # plain (hobson state-first): state pass with cache, then each question as a branch continuing the state cache
            rt.set_fuse(T); bufp = torch.tensor([states[T][0]], device=dev)
            qids = [torch.tensor([p['q']], device=dev) for p in prs]
            masks = [causal_lower_right(len(p['q']), T + len(p['q'])) for p in prs]
            oidx = [torch.tensor(p['opt'], device=dev) for p in prs]

            def plain():
                hs, cs = rt.fwd(bufp, want_cache=True)
                outs = []
                for j, p in enumerate(prs):
                    rt.mask = masks[j]; rt.set_fuse(len(p['q']))
                    hq, _ = rt.fwd(qids[j], pos0=T, cache=cs)
                    lg = (head.k(head.norm(hq[oidx[j]].float())) @ head.q(head.norm(hq[-1].float()))) * head.scale
                    outs.append(torch.softmax(lg, -1))
                rt.set_fuse(T)
                return torch.cat(outs)
            g, out = capture(plain)
            res[f'{qn}_T{T}_plain'] = dict(timed(g, bufp, out, states[T]), q_tokens=sum(len(p['q']) for p in prs))
            del g, out; torch.cuda.empty_cache()
            print(qn, T, 'schema1', res[f'{qn}_T{T}_schema'], 'sets', res[f'{qn}_T{T}_sets'], 'plain', res[f'{qn}_T{T}_plain'], flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
