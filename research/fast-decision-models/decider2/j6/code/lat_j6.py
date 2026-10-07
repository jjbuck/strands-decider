"""J6 latency: question in the weights vs question in the context, A10G, fused bf16 runtime (H4's TTL = d1 lean2 + cached-history conv).
Arms (all read out by the same batched pointer head):
  ctx_packed : hobson, question in context. One pass: state rows + every question's full token list as its own segment (causal within itself,
               sees the state; GDN branches from the state's final state)  = packed multi-question, rows T + sum|q|.
  ctx_plain  : hobson, 1 question, one ordinary sequence [state][question] with d1's best per-length fusion (the doc's anchor).
  w_late     : question in the weights, state read question-blind (trained (a) layout): state rows + per question K option-slot rows + 1 answer row;
               per-question LoRA (shared r16 + question r8, Win/Wo/Wd, 24 layers) applied to slot rows only, batched over questions
               (padded bmm, S-LoRA/Punica style). rows T + sum(K+1).
  w_early    : question in the weights from layer 0 (variant A): every question's network reads its own copy of the state.
               1 question: LoRA merged into the weights, one sequence of T + K + 1 rows (d1's best fusion).
               n questions: one batched pass over n sequences of T + K + 1 rows, per-sequence LoRA via bmm on every row.
Discipline: CUDA graph per exact shape, exclusive GPU, fresh real states every rep (banking states cut to exactly T tokens), 3 warm-ups discarded,
n=REPS timed, median + p95; timed span = H2D of state ids + graph replay + D2H of probabilities.
python lat_j6.py time [CKPT]   |   python lat_j6.py check CKPT PREDS.json   (runtime vs j6lib on real items)"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
import numpy as np, torch, torch.nn.functional as F
from tt_lean import TTL, tgemm, conv_l2_tail, gnorm, attn_prep, LM, chunk_gated_delta_rule, conv_l2, add_rms
from torch.nn.attention.bias import causal_lower_right
from fla.modules.convolution import causal_conv1d as fla_conv
import evalkit as EK, qtab

dev = 'cuda'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'time'
CKPT = sys.argv[2] if len(sys.argv) > 2 else ''
REPS = int(os.environ.get('REPS', '20'))
TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,2000,4000').split(',')]
QSETS = {'Q1s': ['states_amount'], 'Q1p': ['procedure'],
         'Q4': ['details_match', 'failure_cause', 'needed_procedure', 'rule_bound_values'],
         'Q15': ['code_abusive_customer_behavior', 'code_accessibility_or_special_needs', 'code_account_closure_request', 'code_account_ownership_dispute',
                 'code_complex_billing_dispute', 'code_customer_demands_after_unavailable_offer_refusal', 'code_deceased_account_holder',
                 'code_fraud_or_security_concern', 'code_kb_search_unsuccessful_customer_requests_transfer', 'code_legal_or_regulatory_matter',
                 'code_specialized_department_required', 'code_technical_system_error', 'code_third_party_inquiry',
                 'code_unconfirmed_external_communication', 'identity_verified']}
if os.environ.get('QS'): QSETS = {k: v for k, v in QSETS.items() if k in os.environ['QS'].split(',')}
ARMS = os.environ.get('ARMS', 'ctx_packed,ctx_plain,w_late,w_early').split(',')
MODS = ('Win', 'Wo', 'Wd')


class SegMeta:
    """set rows after a state of Ls rows: segment j has lens[j] rows, causal within itself, sees the whole state (P = 0)."""

    def __init__(self, Ls, lens):
        r0 = [0]
        for L in lens: r0.append(r0[-1] + L)
        R = r0[-1]; self.R = R; self.n = len(lens); self.r0 = r0; self.lens = lens
        g = []
        for j, L in enumerate(lens):
            for t in range(L): g.append([(3 + r0[j] + t - 3 + k) if t - 3 + k >= 0 else (3 + t - 3 + k) for k in range(4)])
        self.gidx = torch.tensor(g, device=dev)
        self.cu = torch.tensor(r0, device=dev, dtype=torch.long)
        m = torch.zeros(R, Ls + R, dtype=torch.bool, device=dev); m[:, :Ls] = True
        for j, L in enumerate(lens): m[r0[j]:r0[j] + L, Ls + r0[j]:Ls + r0[j] + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        self.mask = m
        self.spos = torch.tensor([t for L in lens for t in range(L)], device=dev, dtype=torch.float32)
        Lm = max(lens); self.Lm = Lm
        pad = torch.zeros(self.n, Lm, dtype=torch.long)
        unpad = []
        for j, L in enumerate(lens):
            pad[j] = torch.arange(r0[j], r0[j] + Lm).clamp(max=r0[j] + L - 1); unpad += list(range(j * Lm, j * Lm + L))
        self.pad = pad.to(dev); self.unpad = torch.tensor(unpad, device=dev)
        self.big = R > 64


class StackLoRA:
    """per-deployment compiled adapters for a question set: A [n, in, r], B [n, r, out] (scale folded), bf16. rows: 'seg' (slot rows, padded
    per segment) or 'seq' (n equal-length sequences, every row)."""

    def __init__(self, AB, rows, meta=None):
        self.AB = AB; self.rows = rows; self.meta = meta

    def __call__(self, i, nm, h):
        A, B = self.AB[i][nm]
        n = A.shape[0]
        if self.rows == 'seg':
            mt = self.meta
            hp = h[mt.pad] if n > 1 else h[None]
            y = torch.bmm(torch.bmm(hp, A), B)
            return y.reshape(-1, B.shape[2])[mt.unpad] if n > 1 else y[0]
        y = torch.bmm(torch.bmm(h.reshape(n, -1, h.shape[1]), A), B)
        return y.reshape(-1, B.shape[2])


class RTJ(TTL):
    def empty_cache(self):
        c = []
        for d in self.layers:
            if d['type'] == 'linear_attention':
                c.append(dict(tail=torch.zeros(3, 8224, device=dev, dtype=torch.bfloat16), S=torch.zeros(1, 16, 128, 128, device=dev, dtype=torch.float32)))
            else:
                c.append(dict(k=torch.zeros(0, 2, 256, device=dev, dtype=torch.bfloat16), v=torch.zeros(0, 2, 256, device=dev, dtype=torch.bfloat16)))
        return c

    @torch.no_grad()
    def fwd_sets(self, ids, xs, cache, mt, lora=None):
        """ids [1, Ls] state ids; xs [R, 2048] set-row inputs; -> final normed hidden [Ls+R, 2048]"""
        Ls = ids.shape[1]; R = xs.shape[0]; T = Ls + R; n = mt.n
        x = torch.cat([F.embedding(ids[0], self.embed), xs], 0)
        pos = torch.cat([torch.arange(Ls, device=dev, dtype=torch.float32), mt.spos + float(Ls)])
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            c = cache[i]
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if lora is not None: proj[Ls:] += lora(i, 'Win', LM._rms_zc(x[Ls:], d['in_norm'], self.eps))
            if d['type'] == 'linear_attention':
                pm = proj[:Ls]
                qkv3 = conv_l2_tail(pm, c['tail'], d['conv_w'])
                a = pm[:, 8208:8224].reshape(1, Ls, 16); b = pm[:, 8192:8208].reshape(1, Ls, 16)
                om, S = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                               A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True, initial_state=c['S'], output_final_state=True)
                ps = proj[Ls:]
                if mt.big:   # long segments (in-context questions): d1's fused conv+SiLU+l2norm kernel per segment, history = last 3 state rows
                    y3 = torch.cat([conv_l2_tail(proj[Ls + a:Ls + a + L], proj[Ls - 3:Ls], d['conv_w']) for a, L in zip(mt.r0[:-1], mt.lens)], 1)
                    yq, yk, yv = y3[0][None], y3[1][None], y3[2][None]
                else:        # a few slot rows: gather form
                    w = d['conv_w'].float()
                    buf = proj[Ls - 3:, :6144]
                    gi = mt.gidx
                    acc = buf[gi[:, 0]].float() * w[:, 0] + buf[gi[:, 1]].float() * w[:, 1] + buf[gi[:, 2]].float() * w[:, 2] + buf[gi[:, 3]].float() * w[:, 3]
                    y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).float().reshape(R, 48, 128)
                    y = torch.cat([y[:, :32] / torch.sqrt(y[:, :32].pow(2).sum(-1, keepdim=True) + 1e-6), y[:, 32:]], 1).to(torch.bfloat16).reshape(R, 3, 16, 128)
                    yq, yk, yv = y[:, 0][None].contiguous(), y[:, 1][None].contiguous(), y[:, 2][None].contiguous()
                ob, _ = chunk_gated_delta_rule(yq, yk, yv,
                                               ps[:, 8208:8224].reshape(1, R, 16), ps[:, 8192:8208].reshape(1, R, 16), use_qk_l2norm_in_kernel=False,
                                               use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True,
                                               initial_state=S.expand(n, -1, -1, -1).contiguous(), cu_seqlens=mt.cu)
                o = torch.cat([om[0], ob[0]], 0).contiguous()
                o = gnorm(o, proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
                qm = q[:Ls].reshape(1, Ls, 8, 256).transpose(1, 2)
                om = F.scaled_dot_product_attention(qm, kh[:, :, :Ls], vh[:, :, :Ls], is_causal=True, enable_gqa=True)
                qs = q[Ls:].reshape(1, R, 8, 256).transpose(1, 2)
                osl = F.scaled_dot_product_attention(qs, kh, vh, attn_mask=mt.mask[None, None], enable_gqa=True)
                o = torch.cat([om, osl], 2).transpose(1, 2).reshape(T, 2048) * gate
            if lora is not None: x[Ls:] += lora(i, 'Wo', o[Ls:])
            ss = torch.zeros(T, device=dev, dtype=torch.float32)
            tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            if lora is not None: x[Ls:] += lora(i, 'Wd', mm[Ls:])
            ss = torch.zeros(T, device=dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps)

    @torch.no_grad()
    def fwd_plain(self, ids, xs, lora):
        """1 question, late-bound: ONE ordinary causal sequence [state ids][K+1 slot vectors] through d1's best per-length fusion path,
        with the question's unmerged delta added on the last R rows only (state rows = base hobson)."""
        f = self.fuse; Ls = ids.shape[1]; R = xs.shape[0]; T = Ls + R
        x = torch.cat([F.embedding(ids[0], self.embed), xs], 0)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        fold = 'fold' in f
        if fold: ss = x.float().pow(2).sum(-1)
        else: h = LM._rms_zc(x, self.layers[0]['in_norm'], self.eps)
        n = len(self.layers)
        for i, d in enumerate(self.layers):
            if fold:
                proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]); proj[Ls:] += lora(i, 'Win', LM._rms_zc(x[Ls:], d['in_norm'], self.eps))
            else:
                proj = self._mm(h, d['Win']); proj[Ls:] += lora(i, 'Win', h[Ls:])
            if d['type'] == 'linear_attention':
                qkv3 = conv_l2(proj, d['conv_w'])
                a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                o, _ = chunk_gated_delta_rule(qkv3[0][None], qkv3[1][None], qkv3[2][None], a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                              A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                o = F.scaled_dot_product_attention(q.reshape(1, T, 8, 256).transpose(1, 2), k.reshape(1, T, 2, 256).transpose(1, 2),
                                                   v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            if fold:
                x[Ls:] += lora(i, 'Wo', o[Ls:])
                ss = torch.zeros(T, device=dev, dtype=torch.float32)
                tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
                m = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
                x[Ls:] += lora(i, 'Wd', m[Ls:])
                ss = torch.zeros(T, device=dev, dtype=torch.float32)
                tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
                continue
            d_out = self._mm(o, d['Wo']); d_out[Ls:] += lora(i, 'Wo', o[Ls:])
            x, h2 = add_rms(x, d_out, d['post1'], self.eps)
            m = tgemm(h2, d['Wgu_il'], epi=1, cfg=self.gcfg.get('swiglu'))
            d_mlp = self._mm(m, d['Wd']); d_mlp[Ls:] += lora(i, 'Wd', m[Ls:])
            x, h = add_rms(x, d_mlp, (self.layers[i + 1]['in1'] if i + 1 < n else self.norm1), self.eps)
        if fold: h = LM._rms_zc(x, self.norm_w, self.eps)
        return h

    @torch.no_grad()
    def fwd_batch(self, x, B, lora):
        """variant A, n > 1 questions: x [B*T, 2048] inputs of B equal-length sequences; lora on every row (per sequence). -> normed hidden"""
        N = x.shape[0]; T = N // B
        pos = torch.arange(T, device=dev, dtype=torch.float32).repeat(B)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        x = x.clone()
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=2048)
            proj += lora(i, 'Win', LM._rms_zc(x, d['in_norm'], self.eps))
            if d['type'] == 'linear_attention':
                cm = fla_conv(proj[:, :6144].reshape(B, T, 6144).contiguous(), d['conv_w'], None, activation='silu')
                cm = cm[0] if isinstance(cm, tuple) else cm
                q, k, v = cm.split(2048, dim=-1)
                o, _ = chunk_gated_delta_rule(q.reshape(B, T, 16, 128), k.reshape(B, T, 16, 128), v.reshape(B, T, 16, 128),
                                              proj[:, 8208:8224].reshape(B, T, 16), proj[:, 8192:8208].reshape(B, T, 16), use_qk_l2norm_in_kernel=True,
                                              use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
                o = gnorm(o.reshape(N, 16, 128).contiguous(), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120]
                o = F.scaled_dot_product_attention(q.reshape(B, T, 8, 256).transpose(1, 2), k.reshape(B, T, 2, 256).transpose(1, 2),
                                                   v.reshape(B, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(N, 2048) * gate
            x += lora(i, 'Wo', o)
            ss = torch.zeros(N, device=dev, dtype=torch.float32)
            tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=2048)
            x += lora(i, 'Wd', mm)
            ss = torch.zeros(N, device=dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps)


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
        buf.view(-1)[:x.numel()].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


class Readout:
    """batched pointer head over n question segments: answer row = segment's last row, option rows = given absolute rows"""

    def __init__(self, head, ans, opts, Ks, tdiv):
        self.head = head; n = len(Ks); Km = max(Ks)
        self.ans = torch.tensor(ans, device=dev)
        oi = torch.zeros(n, Km, dtype=torch.long); msk = torch.zeros(n, Km, dtype=torch.bool)
        for j, o in enumerate(opts): oi[j, :len(o)] = torch.tensor(o); msk[j, len(o):] = True
        self.oi = oi.to(dev); self.msk = msk.to(dev); self.tdiv = torch.tensor(tdiv, device=dev)[:, None]

    def __call__(self, h):
        hd = self.head
        qv = hd.q(hd.norm(h[self.ans].float()))                                   # [n, 256]
        ko = hd.k(hd.norm(h[self.oi].float()))                                    # [n, Km, 256]
        lg = (ko @ qv[:, :, None])[..., 0] * hd.scale / self.tdiv
        return torch.softmax(lg.masked_fill(self.msk, float('-inf')), -1)


def main():
    from kitrun import load_P
    from h3lib import StdHead
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    head0 = StdHead(Pm.model.head).to(dev).eval()
    rt = RTJ(tm, list(range(10)))
    pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
    PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
    ck = None
    if CKPT:
        ck = torch.load(os.path.expanduser(CKPT), map_location=dev, weights_only=False)
        headj = StdHead(Pm.model.head).to(dev).eval(); headj.load_state_dict(ck['head'])
    else:
        headj = head0

    def adapters(names, rows, meta=None):
        """stacked bf16 adapters for a question set; random if no checkpoint (latency does not depend on the values)"""
        AB = []
        g = torch.Generator(device='cpu'); g.manual_seed(0)
        for i, d in enumerate(rt.layers):
            e = {}
            for nm in MODS:
                W = d[nm]; out, inp = W.shape
                As, Bs = [], []
                for q in names:
                    if ck is not None:
                        sd = ck['ad']; j = ck['qnames'].index(q)
                        A = torch.cat([sd[f'sA.{i}_{nm}'], sd[f'qA.{j}.{i}_{nm}']], 0); B = torch.cat([sd[f'sB.{i}_{nm}'], sd[f'qB.{j}.{i}_{nm}']], 1) * 2.0
                    else:
                        A = torch.randn(24, inp, generator=g) / inp ** 0.5; B = torch.randn(out, 24, generator=g) * 1e-3
                    As.append(A.t()); Bs.append(B.t())
                e[nm] = (torch.stack(As).to(dev, torch.bfloat16).contiguous(), torch.stack(Bs).to(dev, torch.bfloat16).contiguous())
            AB.append(e)
        return StackLoRA(AB, rows, meta)

    def slot_x(names):
        if ck is not None:
            sd = ck['ad']; return torch.cat([sd[f'slots.{ck["qnames"].index(q)}'] for q in names], 0).to(dev, torch.bfloat16)
        return torch.cat([F.embedding(torch.tensor([PQ[q]['q'][-1]] * (PQ[q]['K'] + 1), device=dev), rt.embed) for q in names], 0)

    temp = Pm.temp_for

    if MODE == 'check':
        # runtime vs j6lib reference predictions (eval_a output) on real items: ctx_packed vs teacher, w_late vs student
        ref = json.load(open(os.path.expanduser(sys.argv[3])))
        its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2][:12] + EK.load_suite('LONG')[:4]
        res = {'ctx': [], 'w': [], 'w1': []}
        for it in its:
            names = [q for q in it['questions'] if q in PQ]
            s = qtab.state_ids(eng, it['state']); Ls = len(s); ids = torch.tensor([s], device=dev)
            cache = rt.empty_cache(); rt.mask = None
            for arm in ('ctx', 'w'):
                if arm == 'ctx':
                    lens = [len(PQ[q]['q']) for q in names]; mt = SegMeta(Ls, lens)
                    xs = F.embedding(torch.tensor([t for q in names for t in PQ[q]['q']], device=dev), rt.embed)
                    ro = Readout(head0, [Ls + mt.r0[j] + lens[j] - 1 for j in range(len(names))], [[Ls + mt.r0[j] + o for o in PQ[q]['opt']] for j, q in enumerate(names)],
                                 [PQ[q]['K'] for q in names], [temp(PQ[q]['kind']) for q in names])
                    lo = None; refp = ref['teacher']
                else:
                    lens = [PQ[q]['K'] + 1 for q in names]; mt = SegMeta(Ls, lens)
                    xs = slot_x(names)
                    ro = Readout(headj, [Ls + mt.r0[j] + lens[j] - 1 for j in range(len(names))], [[Ls + mt.r0[j] + o for o in range(PQ[q]['K'])] for j, q in enumerate(names)],
                                 [PQ[q]['K'] for q in names], [temp(PQ[q]['kind']) for q in names])
                    lo = adapters(names, 'seg', mt); refp = ref['student']
                with torch.inference_mode(): pr = ro(rt.fwd_sets(ids, xs, cache, mt, lo))
                for j, q in enumerate(names):
                    K = PQ[q]['K']; a_ = pr[j, :K].tolist(); b_ = [refp[it['id']][q][lab] for lab in PQ[q]['rq'].slot_labels]
                    res[arm].append((int(np.argmax(a_)) == int(np.argmax(b_)), max(abs(x - y) for x, y in zip(a_, b_))))
            for q in names:   # 1-question plain-path late-bound runtime
                K = PQ[q]['K']; mt1 = SegMeta(Ls, [K + 1]); rt.set_fuse(Ls + K + 1)
                ro = Readout(headj, [Ls + K], [[Ls + o for o in range(K)]], [K], [temp(PQ[q]['kind'])])
                with torch.inference_mode(): pr = ro(rt.fwd_plain(ids, slot_x([q]), adapters([q], 'seg', mt1)))
                a_ = pr[0, :K].tolist(); b_ = [ref['student'][it['id']][q][lab] for lab in PQ[q]['rq'].slot_labels]
                res['w1'].append((int(np.argmax(a_)) == int(np.argmax(b_)), max(abs(x - y) for x, y in zip(a_, b_))))
        out = {k: dict(agree=sum(x[0] for x in v), n=len(v), dp_med=float(np.median([x[1] for x in v])), dp_max=float(max(x[1] for x in v))) for k, v in res.items()}
        print('runtime vs j6lib:', out, flush=True)
        json.dump(out, open(os.path.expanduser('~/work/j6/latcheck.json'), 'w'))
        return

    # ---- timing
    pool_states = [it['state'] for it in EK.load_suite('LONG')] + [it['state'] for it in EK.load_suite('REAL-agree') if it.get('domain') == 'banking_knowledge']
    toks = [qtab.state_ids(eng, x) for x in pool_states]
    TMAX = max(TS)
    long_ = [t for t in toks if len(t) >= TMAX]
    assert len(long_) >= REPS + 3, len(long_)
    states = {T: [t[:T] for t in long_[:REPS + 3]] for T in TS}
    OUT = os.path.expanduser(os.environ.get('LATOUT', '~/work/j6/lat_j6.json'))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['meta'] = dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT)
    for qn, names in QSETS.items():
        n = len(names); Ks = [PQ[q]['K'] for q in names]; tdiv = [temp(PQ[q]['kind']) for q in names]
        for T in TS:
            key = f'{qn}_T{T}'
            r = res.get(key, {})
            ids = torch.tensor([states[T][0]], device=dev)
            cache = rt.empty_cache()
            if 'ctx_packed' in ARMS and 'ctx_packed' not in r:
                lens = [len(PQ[q]['q']) for q in names]; mt = SegMeta(T, lens)
                xs = F.embedding(torch.tensor([t for q in names for t in PQ[q]['q']], device=dev), rt.embed)
                ro = Readout(head0, [T + mt.r0[j] + lens[j] - 1 for j in range(n)], [[T + mt.r0[j] + o for o in PQ[q]['opt']] for j, q in enumerate(names)], Ks, tdiv)
                g, out = capture(lambda: ro(rt.fwd_sets(ids, xs, cache, mt)))
                r['ctx_packed'] = dict(timed(g, ids, out, states[T]), rows=T + mt.R); del g, out
            if 'ctx_plain' in ARMS and n == 1 and 'ctx_plain' not in r:
                q = names[0]; L = T + len(PQ[q]['q'])
                buf = torch.tensor([states[T][0] + PQ[q]['q']], device=dev); rt.set_fuse(L)
                ro = Readout(head0, [L - 1], [[T + o for o in PQ[q]['opt']]], Ks, tdiv)
                g, out = capture(lambda: ro(rt.fwd(buf)[0]))
                r['ctx_plain'] = dict(timed(g, buf, out, states[T]), rows=L); del g, out
            if 'ctx_seq' in ARMS and 'ctx_seq' not in r:
                rt.set_fuse(T); bufp = torch.tensor([states[T][0]], device=dev)
                qids = [torch.tensor([PQ[q]['q']], device=dev) for q in names]
                masks = [causal_lower_right(len(PQ[q]['q']), T + len(PQ[q]['q'])) for q in names]
                oidx = [torch.tensor(PQ[q]['opt'], device=dev) for q in names]

                def seq():
                    rt.set_fuse(T); hs_, cs = rt.fwd(bufp, want_cache=True)
                    outs = []
                    for j, q in enumerate(names):
                        rt.mask = masks[j]; rt.set_fuse(len(PQ[q]['q']))
                        hq, _ = rt.fwd(qids[j], pos0=T, cache=cs)
                        lg = (head0.k(head0.norm(hq[oidx[j]].float())) @ head0.q(head0.norm(hq[-1].float()))) * head0.scale / tdiv[j]
                        outs.append(torch.softmax(lg, -1))
                    rt.set_fuse(T)
                    return torch.cat(outs)
                g, out = capture(seq)
                r['ctx_seq'] = dict(timed(g, bufp, out, states[T]), rows=T + sum(len(PQ[q]['q']) for q in names)); del g, out
            if 'w_late' in ARMS and 'w_late' not in r:
                lens = [k + 1 for k in Ks]; mt = SegMeta(T, lens)
                xs = slot_x(names); lo = adapters(names, 'seg', mt)
                ro = Readout(headj, [T + mt.r0[j] + lens[j] - 1 for j in range(n)], [[T + mt.r0[j] + o for o in range(Ks[j])] for j in range(n)], Ks, tdiv)
                g, out = capture(lambda: ro(rt.fwd_sets(ids, xs, cache, mt, lo)))
                r['w_late'] = dict(timed(g, ids, out, states[T]), rows=T + mt.R); del g, out, lo
            if 'w_late1' in ARMS and n == 1 and 'w_late1' not in r:
                Lq = Ks[0] + 1; L = T + Lq; rt.set_fuse(L)
                mt1 = SegMeta(T, [Lq]); xs = slot_x(names); lo = adapters(names, 'seg', mt1)
                ro = Readout(headj, [L - 1], [[T + o for o in range(Ks[0])]], Ks, tdiv)
                g, out = capture(lambda: ro(rt.fwd_plain(ids, xs, lo)))
                r['w_late1'] = dict(timed(g, ids, out, states[T]), rows=L); del g, out, lo
            if 'w_early' in ARMS and 'w_early' not in r:
                Lq = max(Ks) + 1; L = T + Lq
                if n == 1:
                    buf = torch.tensor([states[T][0] + [0] * Lq], device=dev); rt.set_fuse(L)     # merged weights: cost of one sequence of T + K + 1 rows
                    ro = Readout(headj, [L - 1], [[T + o for o in range(Ks[0])]], Ks, tdiv)
                    g, out = capture(lambda: ro(rt.fwd(buf)[0]))
                    r['w_early'] = dict(timed(g, buf, out, states[T]), rows=L); del g, out
                else:
                    buf = torch.tensor([states[T][0]], device=dev); xsl = slot_x(names)
                    sl = []; r0 = 0
                    for k in Ks: sl.append(torch.arange(r0, r0 + Lq).clamp(max=r0 + k)); r0 += k + 1
                    sl = torch.stack(sl).to(dev)                                                     # [n, Lq] slot rows per sequence (padded)
                    lo = adapters(names, 'seq')
                    ro = Readout(headj, [b * L + T + Ks[b] for b in range(n)], [[b * L + T + o for o in range(Ks[b])] for b in range(n)], Ks, tdiv)

                    def fa():
                        e = F.embedding(buf[0], rt.embed)
                        x = torch.cat([e[None].expand(n, -1, -1), xsl[sl]], 1).reshape(n * L, 2048)
                        return ro(rt.fwd_batch(x, n, lo))
                    g, out = capture(fa)
                    r['w_early'] = dict(timed(g, buf, out, states[T]), rows=n * L); del g, out, lo
            torch.cuda.empty_cache()
            res[key] = r
            print(key, {k: (v['median'], v['p95'], v['rows']) for k, v in r.items()}, flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
