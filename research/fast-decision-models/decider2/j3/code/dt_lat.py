"""J3 latency: the depth-split decision transformer (DT) in d1/H4's fused bf16 runtime (tt_lean.TTL = lean2 kernels + cached-history conv),
against hobson in the same harness.
  hob1     : hobson, one sequence [state + question] (d1's anchor layout; 1 question only), TTL best fuse path for the length
  hobB     : hobson, state pass (24 layers, cache) + ONE packed question pass (all questions as varlen branches, 24 layers)
  dtA{Ls}  : DT bridge A: state pass (Ls layers, cache) + memory K/V for the deep attention layers (ONE GEMM over the state rows, norms folded)
             + the same packed question pass (shallow layers continue the state cache; deep layers read the memory; deep GDN start from zero)
Discipline (d1/F7/H4/H7): CUDA graph per exact shape, exclusive GPU, fresh real states every rep (banking states cut to exactly T tokens),
3 warm-ups discarded, REPS timed, median + p95; timed span = H2D of the state ids + graph replay + D2H of the probabilities.
python dt_lat.py time [CKPT]          -> ~/work/j3/lat_dt.json
python dt_lat.py check CKPT PREDS LS  -> runtime vs dtlib preds (argmax, max|dp|) on 16 items"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import numpy as np, torch, torch.nn.functional as F
from tt_lean import TTL, tgemm, conv_l2_tail, gnorm, attn_prep, LM, chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right
import evalkit as EK

dev = 'cuda'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'time'
CKPT = sys.argv[2] if len(sys.argv) > 2 else ''
REPS = int(os.environ.get('REPS', 20))
TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,4000').split(',')]
LSS = [int(x) for x in os.environ.get('LSS', '4,8,12').split(',')]
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4]}


def merge_ckpt(tm, path):
    """merge J3 LoRA (all layers) into the HF torso; return (head state, mem adapters {f'{Ls}_{i}': dB@A*scale [rows, 2048] fp32})"""
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


class RTD(TTL):
    def __init__(self, torso, mem=None):
        super().__init__(torso, list(range(10)))
        self.memw = {}
        self.zt = torch.zeros(3, 8224, device=self.dev, dtype=torch.bfloat16)
        for Ls in LSS:      # folded memory K/V weights of every deep attention layer, concatenated: ONE GEMM over the state rows
            blocks = []
            for i in range(Ls, 24):
                d = self.layers[i]
                if d['type'] == 'linear_attention': continue
                W = d['Win'][4096:5120].float()
                if mem and f'{Ls}_{i}' in mem: W = W + mem[f'{Ls}_{i}']
                blocks.append((W * d['in1'][None, :]).to(torch.bfloat16))
            self.memw[Ls] = torch.cat(blocks, 0).contiguous() if blocks else None
        self.memg = {}
        if os.environ.get('BRIDGE_G'):
            for Ls in LSS:      # bridge G: per deep GDN layer, folded [qkv | b | a] rows (z skipped) for the state rows
                for i in range(Ls, 24):
                    d = self.layers[i]
                    if d['type'] != 'linear_attention': continue
                    W = torch.cat([d['Win'][:6144], d['Win'][8192:8224]], 0).float()
                    if mem and f'{Ls}_{i}' in mem:
                        dW = mem[f'{Ls}_{i}']; W = W + torch.cat([dW[:6144], dW[8192:8224]], 0)
                    self.memg[(Ls, i)] = (W * d['in1'][None, :]).to(torch.bfloat16).contiguous()

    @torch.no_grad()
    def state_pass(self, ids, nl):
        """state rows through layers 0..nl-1 (folded-norm path). -> x residual [T, 2048], ss (row sum of squares of x), cache, cos, sin"""
        T = ids.shape[1]
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1); cache = {}
        for i in range(nl):
            d = self.layers[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if d['type'] == 'linear_attention':
                qkv3 = LM_conv(proj, d['conv_w'])
                a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                o, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                              A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True, output_final_state=True)
                cache[i] = {'tail': proj[-3:], 'S': S}
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                cache[i] = {'k': k, 'v': v}
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), k.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return x, ss, cache, cos, sin

    @torch.no_grad()
    def mem_pass(self, x, ss, Ls, cos, sin, cache, bridge='A'):
        """bridge A: K/V of every deep attention layer from the layer-Ls residual of the state rows (one GEMM, folded norms).
        bridge G: also each deep GDN layer scans the state rows with its own (folded) q,k,v,b,a projections -> final state + conv tail"""
        T = x.shape[0]
        y = tgemm(x, self.memw[Ls], epi=0, ss=ss, Kd=x.shape[1])
        j = 0
        for i in range(Ls, 24):
            d = self.layers[i]
            if d['type'] == 'linear_attention':
                if bridge == 'G':
                    buf = tgemm(x, self.memg[(Ls, i)], epi=0, ss=ss, Kd=x.shape[1])
                    qkv3 = LM_conv(buf, d['conv_w'])
                    _, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], buf[:, 6160:6176].reshape(1, T, 16), buf[:, 6144:6160].reshape(1, T, 16),
                                                  use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'],
                                                  use_beta_sigmoid_in_kernel=True, output_final_state=True)
                    cache[i] = {'tail': buf[-3:], 'S': S}
                else:
                    cache[i] = None
                continue
            kr = y[:, j * 1024:j * 1024 + 512].reshape(T, 2, 256); v = y[:, j * 1024 + 512:(j + 1) * 1024].reshape(T, 2, 256); j += 1
            k = LM._rms_zc(kr, d['kn'], self.eps)
            xr, xp = k[..., :64], k[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
            c = cos[:, None, :]; s_ = sin[:, None, :]
            k = torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            cache[i] = {'k': k.contiguous(), 'v': v.contiguous()}
        return cache

    @torch.no_grad()
    def q_pass(self, qids, T, cache, n, gidx, cu, mask, cos, sin, tree=None):
        """all question rows (packed varlen branches) through all 24 layers; cache[i]: GDN {'tail','S'} or None (zero state), attention {'k','v'}"""
        R = qids.shape[1]
        x = F.embedding(qids, self.embed).reshape(R, -1)
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            c = cache[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if d['type'] == 'linear_attention':
                tail = c['tail'] if c is not None else self.zt
                buf = torch.cat([tail[:, :6144], proj[:, :6144]], 0)
                w = d['conv_w'].float()
                acc = buf[gidx[:, 0]].float() * w[:, 0] + buf[gidx[:, 1]].float() * w[:, 1] + buf[gidx[:, 2]].float() * w[:, 2] + buf[gidx[:, 3]].float() * w[:, 3]
                y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).float().reshape(R, 48, 128)
                y = torch.cat([y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6), y[:, 32:]], 1).to(torch.bfloat16).reshape(R, 3, 16, 128)
                if tree is None:
                    init = c['S'].expand(n, -1, -1, -1).contiguous() if c is not None else None
                    ob, _ = chunk_gated_delta_rule(y[:, 0][None].contiguous(), y[:, 1][None].contiguous(), y[:, 2][None].contiguous(),
                                                   proj[:, 8208:8224].reshape(1, R, 16), proj[:, 8192:8208].reshape(1, R, 16), use_qk_l2norm_in_kernel=False,
                                                   use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                                   initial_state=init, cu_seqlens=cu)
                    oq = ob[0]
                else:                                   # DT-set: stems from the state's GDN state, then options + tails from each stem's final state
                    oq = torch.empty(R, 16, 128, device=self.dev, dtype=torch.bfloat16); a_ = proj[:, 8208:8224]; b_ = proj[:, 8192:8208]
                    def run(rows, cu_, init):
                        k_ = rows.numel()
                        return chunk_gated_delta_rule(y[rows, 0][None].contiguous(), y[rows, 1][None].contiguous(), y[rows, 2][None].contiguous(),
                                                      a_[rows].reshape(1, k_, 16), b_[rows].reshape(1, k_, 16), use_qk_l2norm_in_kernel=False,
                                                      use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                                      initial_state=init, cu_seqlens=cu_, output_final_state=True)
                    o1, S1 = run(tree['rows1'], tree['cu1'], c['S'].expand(tree['n1'], -1, -1, -1).contiguous() if c is not None else None)
                    o2, _ = run(tree['rows2'], tree['cu2'], S1[tree['par2']].contiguous())
                    oq[tree['rows1']] = o1[0]; oq[tree['rows2']] = o2[0]
                o = gnorm(oq.contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(R, 2, 256)
                kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                o = F.scaled_dot_product_attention(q.reshape(1, R, 8, 256).transpose(1, 2), kk.reshape(1, -1, 2, 256).transpose(1, 2),
                                                   vv.reshape(1, -1, 2, 256).transpose(1, 2), attn_mask=mask[None, None], enable_gqa=True)
                o = o.transpose(1, 2).reshape(R, 2048) * gate
            ss = torch.zeros(R, device=self.dev, dtype=torch.float32)
            tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(R, device=self.dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps).reshape(R, -1)


def LM_conv(proj, w):
    from tt_lean import conv_l2
    return conv_l2(proj, w)


def qmeta(prs, T):
    toks = []; gidx = []; cu = [0]; spos = []; rows = []
    for p in prs:
        L = len(p['q']); r0 = len(toks); rows.append((r0, L)); toks += p['q']; cu.append(cu[-1] + L); spos += list(range(L))
        for t in range(L): gidx.append([(3 + r0 + t - 3 + j) if t - 3 + j >= 0 else (t + j) for j in range(4)])
    R = len(toks)
    mask = torch.zeros(R, T + R, dtype=torch.bool, device=dev); mask[:, :T] = True
    for r0, L in rows: mask[r0:r0 + L, T + r0:T + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
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
    from h3lib import StdHead
    ta = TypeAdapter(SC.Question)
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    head = StdHead(Pm.model.head).to(dev).eval(); mem = None
    if CKPT:
        hs, mem = merge_ckpt(tm, CKPT); head.load_state_dict(hs)
    rt = RTD(tm, mem); temp = Pm.temp_for

    def prep(state_text, qd):
        q = ta.validate_python(qd); rq = render_question(q)
        s, qs = eng._fit(state_text, [rq.text])
        return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)

    class _Shim: pass
    shim = _Shim(); shim.p = Pm; shim.dev = dev

    def make_set(buf, prs, T, Ls):
        import dtset
        ids, pos, mt = dtset.build(shim, [0] * T, prs); n = len(prs)
        toks = ids[T:]; qids = torch.tensor([toks], device=dev)
        fr = torch.tensor(pos[T:], device=dev, dtype=torch.float32)[:, None] * rt.inv[None, :]; fr = torch.cat([fr, fr], -1)
        qcos, qsin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        tdiv = [temp(p['rq'].kind) for p in prs]; nopt = max(len(p['opt']) for p in prs)
        tree = {k: mt[k] for k in ('rows1', 'rows2', 'cu1', 'cu2', 'par2', 'n1')}
        oidx = [torch.tensor(orows, device=dev) for orows, a in mt['readout']]; arow = [a for orows, a in mt['readout']]

        def fn():
            x, ss, cache, cos, sin = rt.state_pass(buf, Ls)
            if Ls < 24: rt.mem_pass(x, ss, Ls, cos, sin, cache)
            h = rt.q_pass(qids, T, cache, n, mt['gidx'], None, mt['mask'], qcos, qsin, tree=tree)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j in range(n):
                qv = head.q(head.norm(h[arow[j]].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :ko.shape[0]] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def make(kind, buf, prs, T):
        """kind 'hobB' or ('dt', Ls). returns fn() -> probs [n, nopt]"""
        n = len(prs); toks, gidx, cu, mask, spos, rows = qmeta(prs, T)
        qids = torch.tensor([toks], device=dev)
        fr = (spos + float(T))[:, None] * rt.inv[None, :]; fr = torch.cat([fr, fr], -1)
        qcos, qsin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        tdiv = [temp(p['rq'].kind) for p in prs]; nopt = max(len(p['opt']) for p in prs)
        oidx = [torch.tensor([r0 + o for o in p['opt']], device=dev) for (r0, L), p in zip(rows, prs)]
        Ls = 24 if kind == 'hobB' else kind[1]; br = 'G' if kind != 'hobB' and kind[0] == 'dtG' else 'A'

        def fn():
            x, ss, cache, cos, sin = rt.state_pass(buf, Ls)
            if Ls < 24: rt.mem_pass(x, ss, Ls, cos, sin, cache, br)
            h = rt.q_pass(qids, T, cache, n, gidx, cu, mask, qcos, qsin)
            out = torch.full((n, nopt), float('-inf'), device=dev)
            for j, (r0, L) in enumerate(rows):
                qv = head.q(head.norm(h[r0 + L - 1].float())); ko = head.k(head.norm(h[oidx[j]].float()))
                out[j, :ko.shape[0]] = (ko @ qv) * head.scale / tdiv[j]
            return torch.softmax(out, -1)
        return fn

    def make_hob1(buf, p, T):
        L = T + len(p['q']); oi = torch.tensor([T + o for o in p['opt']], device=dev); td = temp(p['rq'].kind)

        def fn():
            hh, _ = rt.fwd(buf)
            lg = (head.k(head.norm(hh[oi].float())) @ head.q(head.norm(hh[L - 1].float()))) * head.scale / td
            return torch.softmax(lg, -1)[None]
        return fn

    if MODE == 'check':
        pd = json.load(open(os.path.expanduser(sys.argv[3]))); Lc = int(sys.argv[4])
        its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2 and x['id'] in pd][:8] + [x for x in EK.load_suite('JB-hard') if x['id'] in pd][:4] \
            + [x for x in EK.load_suite('CF-probe') if x['id'] in pd][:4]
        res = []
        for it in its:
            stt = render_state(it['state']); names = list(it['questions']); prs = [prep(stt, it['questions'][q]) for q in names]
            s = prs[0]['s']; T = len(s); buf = torch.tensor([s], device=dev)
            with torch.inference_mode():
                pr = (make_set(buf, prs, T, Lc) if os.environ.get('LAYOUT') == 'set' else make((('dtG' if os.environ.get('BRIDGE_G') else 'dt'), Lc) if Lc < 24 else 'hobB', buf, prs, T))()
            for j, qn in enumerate(names):
                k = len(prs[j]['opt']); a_ = pr[j, :k].tolist(); b_ = [pd[it['id']][qn][lab] for lab in prs[j]['rq'].slot_labels]
                res.append((int(np.argmax(a_)) == int(np.argmax(b_)), max(abs(x - y) for x, y in zip(a_, b_))))
        r = dict(agree=sum(x[0] for x in res), n=len(res), dp_med=float(np.median([x[1] for x in res])), dp_max=max(x[1] for x in res))
        print('runtime vs dtlib preds:', r, flush=True)
        json.dump(r, open(os.path.expanduser(f'~/work/j3/latcheck_{os.environ.get("LAYOUT", "seqs")}_{Lc}.json'), 'w'))
        return

    spec = {}; pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
    toks = [eng.tok(render_state(x), add_special_tokens=False)['input_ids'] for x in pool]
    states = {T: [t[:T] for t in toks if len(t) >= T][:REPS + 3] for T in TS}
    for T in TS: assert len(states[T]) == REPS + 3, (T, len(states[T]))
    OUT = os.path.expanduser(os.environ.get('LATOUT', '~/work/j3/lat_dt.json'))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['meta'] = dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT, note='bf16 TTL fused kernels, CUDA graph, exact T')
    for qn, names in QSETS.items():
        prs = [prep('S', spec[k]) for k in names]
        for T in TS:
            KINDS = os.environ.get('KINDS', 'hob1,hobB,dt').split(',')
            cfgs = (['hob1'] if qn == 'Q1' and 'hob1' in KINDS else []) + (['hobB'] if 'hobB' in KINDS else []) + \
                [(k, L) for k in ('dt', 'dtG', 'set') if k in KINDS for L in LSS]
            for c in cfgs:
                key = f"{qn}_T{T}_{c if isinstance(c, str) else {'dt': 'dtA%d', 'dtG': 'dtG%d', 'set': 'setA%d'}[c[0]] % c[1]}"
                if key in res: continue
                if c == 'hob1':
                    rt.set_fuse(T + len(prs[0]['q'])); rt.mask = None
                    buf = torch.tensor([states[T][0] + prs[0]['q']], device=dev); fn = make_hob1(buf, prs[0], T)
                elif c[0] == 'set':
                    buf = torch.tensor([states[T][0]], device=dev); fn = make_set(buf, prs, T, c[1])
                else:
                    buf = torch.tensor([states[T][0]], device=dev); fn = make(c, buf, prs, T)
                g, out = capture(fn)
                res[key] = dict(timed(g, buf, out, states[T]), q_tokens=sum(len(p['q']) for p in prs))
                del g, out; torch.cuda.empty_cache()
                print(key, res[key], flush=True)
                json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
