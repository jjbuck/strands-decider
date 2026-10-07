"""J12 latency: d1 lean2 fused bf16 runtime (all fusions incl. norm folding), full 24-layer hobson, CUDA graph per input, exact lengths,
with and without the XR modules (layers 11, 17; question rows only; XR projections in bf16, readout in fp32).
For each T in (64,128,256,400,1000,4000) and 1 / 4 questions: NREP distinct real inputs (REAL-agree / LONG states cut to exactly T state
tokens + real questions of 60-200 tokens), each captured into its own graph, 3 warm replays, then median of 5 timed replays.
Also: host-side XR prep (literal extraction + slot index arrays + H2D) per input.  python lat_xr.py [--ck CKPT] [--nrep 14]"""
import os, sys, json, time, math, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
import lean2 as L2M
from lean2 import Lean2, tgemm, conv_l2, gnorm, attn_prep, chunk_gated_delta_rule
import lean as LM
from plib import P
import xlit
from xrlib import XR, make_ix, NT, NF
import evalkit as EK
from strands_decider.prompting import render_question, render_state
from pydantic import TypeAdapter
import strands_decider.schema as SC
ta = TypeAdapter(SC.Question)

ap = argparse.ArgumentParser(); ap.add_argument('--ck', default=''); ap.add_argument('--nrep', type=int, default=14)
ap.add_argument('--Ts', default='64,128,256,400,1000,4000'); ap.add_argument('--out', default=os.path.expanduser('~/work/j12/lat_xr.json'))
a = ap.parse_args()
p = P(); p.model.config.max_length = 16384
FUSE = 'addrms,gnorm,silu,prep,conv,gemm_swiglu,fold'
R = Lean2(p.tm, fuse=FUSE); dev = R.dev
head = p.model.head
XL = (11, 17)
xrs = [XR(seed=17 + j).to(dev) for j in range(len(XL))]
if a.ck:
    sd = torch.load(os.path.expanduser(a.ck), map_location=dev)
    for j, x in enumerate(xrs):
        x.load_state_dict({k[len(f'xr.{j}.'):]: v for k, v in sd.items() if k.startswith(f'xr.{j}.')})
else:
    for x in xrs: torch.nn.init.normal_(x.wo.weight, 0, 1e-3)
XB = []
for x in xrs:
    XB.append(dict(qa=x.qa.weight.to(torch.bfloat16), qb=x.qb.weight.to(torch.bfloat16), ka=x.ka.weight.to(torch.bfloat16), kb=x.kb.weight.to(torch.bfloat16),
                   ca=x.cat_a.weight.float(), cb=x.cat_b.weight.float(), na=x.null_a.float(), nb=x.null_b.float(), nbias=x.null_bias.float(),
                   nw=x.nw.to(torch.bfloat16), wo=x.wo.weight.to(torch.bfloat16), H=x.H, dk=x.dk))


def xr_infer(W, x, q0, ix):
    T = x.shape[0]; Q = T - q0; H, dk = W['H'], W['dk']
    xf = x.float(); hn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * W['nw']
    rows = hn[q0:]
    Qa = (rows @ W['qa'].t()).float().view(Q, H, dk); Qb = (rows @ W['qb'].t()).float().view(Q, H, dk)
    sc = 1.0 / math.sqrt(dk)
    na = torch.einsum('qhd,hd->hq', Qa, W['na']) * sc + W['nbias'][0][:, None]
    nb = torch.einsum('qhd,hd->hq', Qb, W['nb']) * sc + W['nbias'][1][:, None]
    N = ix['N']
    S = hn[ix['pos']]
    Ka = (S @ W['ka'].t()).float().view(N, H, dk) + W['ca'][ix['cat']].view(N, H, dk)
    Kb = (S @ W['kb'].t()).float().view(N, H, dk) + W['cb'][ix['cat']].view(N, H, dk)
    la = torch.einsum('qhd,nhd->hqn', Qa, Ka) * sc; lb = torch.einsum('qhd,nhd->hqn', Qb, Kb) * sc
    bad = ix['bad']
    la = la.masked_fill(bad, float('-inf')); lb = lb.masked_fill(bad, float('-inf'))
    pa = torch.softmax(torch.cat([la, na[..., None]], -1), -1); pb = torch.softmax(torch.cat([lb, nb[..., None]], -1), -1)
    f = XR.readout(pa, pb, ix)
    return (f.permute(1, 0, 2).reshape(Q, H * NF).to(torch.bfloat16) @ W['wo'].t())


def forward(ids, q0, opt_abs, ix, use_xr):
    T = ids.shape[1]
    x = F.embedding(ids, R.embed).reshape(T, -1)
    pos = torch.arange(T, device=dev, dtype=torch.float32)
    fr = pos[:, None] * R.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
    ss = x.float().pow(2).sum(-1)
    for i, d in enumerate(R.layers):
        proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
        if d['type'] == 'linear_attention':
            qkv3 = conv_l2(proj, d['conv_w'])
            q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
            aa = proj[:, 8208:8224].reshape(1, T, 16); bb = proj[:, 8192:8208].reshape(1, T, 16)
            o, _ = chunk_gated_delta_rule(q, k, v, aa, bb, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                          A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
            o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], R.eps)
        else:
            q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, R.eps)
            v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
            o = o.transpose(1, 2).reshape(T, 2048) * gate
        ss = torch.zeros(T, device=dev, dtype=torch.float32)
        tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
        m = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
        ss = torch.zeros(T, device=dev, dtype=torch.float32)
        tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
        if use_xr and i in XL:
            y = xr_infer(XB[XL.index(i)], x, q0, ix)
            x[q0:] += y
            ss[q0:] = x[q0:].float().pow(2).sum(-1)
    h = LM._rms_zc(x, R.norm_w, R.eps)
    return head(h[T - 1:T].float(), h[opt_abs].float()[None])


def build_inputs(T, nq, n):
    """n inputs: exactly T state tokens (cut from real states) + nq real questions (60-200 tokens each)"""
    srcs = EK.load_suite('LONG') if T > 2000 else EK.load_suite('REAL-agree') + EK.load_suite('LONG')
    tok = p.tok; out = []
    for it in srcs:
        if len(out) >= n: break
        st = render_state(it['state'])
        enc = tok(st, add_special_tokens=True, return_offsets_mapping=True)
        if len(enc['input_ids']) < T: continue
        s = enc['input_ids'][:T]; soff = enc['offset_mapping'][:T]
        qs = []
        for qn, spec in list(it['questions'].items()):
            rq = render_question(ta.validate_python(spec)); e = tok(rq.text, add_special_tokens=False, return_offsets_mapping=True)
            if 60 <= len(e['input_ids']) <= 200: qs.append((rq, e))
        pool_q = qs
        if len(pool_q) < nq:
            continue
        qs = pool_q[:nq]
        out.append((st, s, soff, qs))
    return out


def prep_ix(st, s, soff, qs):
    t0 = time.perf_counter()
    slits = xlit.extract(st[:soff[-1][1]])
    q0 = len(s); ids = list(s); qlits = []; qoff = []; qtext = ''
    base = 0
    for rq, e in qs:      # concatenate questions (4q: one packed-equivalent pass)
        ql = xlit.extract(rq.text) ; ql = ql + xlit.qwords(rq.text, ql, slits)
        for l in ql: l.start += base; l.end += base
        qlits += ql; qoff += [(o[0] + base, o[1] + base) for o in e['offset_mapping']]
        qtext += rq.text; base += len(rq.text); ids += e['input_ids']
    ix = make_ix(slits, soff, qlits, qoff, q0, qtext, dev)
    T = len(ids)
    ix['bad'] = (ix['pos'][None, :] > torch.arange(q0, T, device=dev)[:, None])[None]
    torch.cuda.synchronize()
    return ids, q0, ix, (time.perf_counter() - t0) * 1000


def time_graph(fn, nwarm=3, ntime=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    for _ in range(nwarm): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ntime):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    del g
    return statistics.median(ts), out


def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


res = {}
with torch.inference_mode():
    for nq in (1, 4):
        for T in [int(x) for x in a.Ts.split(',')]:
            ins = build_inputs(T, nq, a.nrep)
            tb, tx, tp, Ns, dev_logit = [], [], [], [], []
            for st, s, soff, qs in ins:
                ids, q0, ix, ms = prep_ix(st, s, soff, qs)
                tp.append(ms); Ns.append(ix['N'])
                idt = torch.tensor([ids], device=dev)
                opt_abs = torch.tensor([len(ids) - 2], device=dev)
                t1, o1 = time_graph(lambda: forward(idt, q0, opt_abs, ix, False))
                t2, o2 = time_graph(lambda: forward(idt, q0, opt_abs, ix, True))
                tb.append(t1); tx.append(t2)
            if not ins: continue
            r = dict(n=len(ins), state_T=T, rows=len(ids), N_slots_med=statistics.median(Ns), base_med=statistics.median(tb), base_p95=pct(tb, .95),
                     xr_med=statistics.median(tx), xr_p95=pct(tx, .95), delta_med=statistics.median([b - c for b, c in zip(tx, tb)]),
                     host_prep_med=statistics.median(tp), host_prep_p95=pct(tp, .95))
            res[f'{T}x{nq}'] = r
            print(json.dumps(r), flush=True)
            json.dump(res, open(a.out, 'w'), indent=1)
print('done')
