"""hobson-v19 (LoRA merged) latency in the fused runtime (d1 lean2 via H4's TTL: all fusions, folded norms, CUDA graph), same harness as
lat_enc.py: exact T, banking states (fresh each rep), h7lat's Q1/Q4/Q15 banking questions, 3 warm-ups, n=REPS, H2D + replay + D2H.
  single : one causal pass over state T + all question tokens (for Q1 this IS hobson's deployed 1-question pass; for Q4/Q15 it is a LOWER
           BOUND on a packed multi-question pass: same rows through the same weights, no branch overhead)
  plain  : state pass with cache, then each question as its own branch continuing the cache (h7lat 'plain'; the existing fused path)"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
import numpy as np, torch
from tt_lean import TTL
from torch.nn.attention.bias import causal_lower_right
import evalkit as EK
from kitrun import load_P
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state

dev = 'cuda'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'lat_hob.json'
TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,2000,4000').split(',')]
REPS = int(os.environ.get('REPS', 20))
MODES = os.environ.get('MODES', 'single,plain').split(',')
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4], 'Q15': Q15}
ta = TypeAdapter(SC.Question)
Pm = load_P(); tm = Pm.tm; eng = Pm.eng; head = Pm.model.head.eval()
if os.environ.get('AUTOTUNE') == '1':   # fairness: give hobson's Triton GEMMs the same per-shape tile autotuning the encoder runtime gets
    import lean2, triton
    CF = [(128, 64, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (256, 128, 32, 8, 3), (64, 256, 32, 8, 3),
          (32, 64, 128, 4, 4), (16, 64, 128, 4, 4), (32, 128, 64, 4, 4)]
    _orig = lean2.pick_cfg; _cache = {}

    def pick_auto(M, N, K):
        if (M, N, K) in _cache: return _cache[(M, N, K)]
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        best = None
        for c in [_orig(M, N, K)] + CF:
            try:
                for _ in range(2): lean2.tgemm(a, b, epi=0, cfg=c)
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
                for _ in range(8): lean2.tgemm(a, b, epi=0, cfg=c)
                e1.record(); torch.cuda.synchronize(); t = e0.elapsed_time(e1)
                if best is None or t < best[0]: best = (t, c)
            except Exception: pass
        _cache[(M, N, K)] = best[1]; return best[1]
    lean2.pick_cfg = pick_auto
rt = TTL(tm, list(range(10)))
temp = Pm.temp_for


def prep(qd):
    q = ta.validate_python(qd); rq = render_question(q)
    s, qs = eng._fit('S', [rq.text])
    return dict(q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)


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


def timed(g, buf, out, inputs, tail):
    pins = [torch.from_numpy(np.asarray(x + tail, dtype=np.int64)).pin_memory() for x in inputs]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pins:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        buf[0, :x.numel()].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


spec = {}; pool = []
for s_ in ('REAL-agree', 'LONG'):
    for it in EK.load_suite(s_):
        for q, sp in it['questions'].items(): spec.setdefault(q, sp)
        if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
toks = [eng.tok(render_state(x), add_special_tokens=False)['input_ids'] for x in pool]
res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, note='hobson-v19 merged, d1 lean2 / H4 TTL fused runtime')}
if os.path.exists(OUT): res.update(json.load(open(OUT)))
for qn, names in QSETS.items():
    prs = [prep(spec[k]) for k in names]
    tail = [t for p in prs for t in p['q']]
    for T in TS:
        states = [t[:T] for t in toks if len(t) >= T][:REPS + 3]
        assert len(states) == REPS + 3
        if 'single' in MODES and f'{qn}_T{T}_single' not in res:
            L = T + len(tail); rt.set_fuse(L)
            buf = torch.tensor([states[0] + tail], device=dev)
            p = prs[-1]; oi = torch.tensor([L - len(p['q']) + o for o in p['opt']], device=dev); tdv = temp(p['rq'].kind)

            def single():
                h, _ = rt.fwd(buf)
                lg = head(h[-1:].float(), h[oi][None].float()) / tdv
                return torch.softmax(lg, -1)
            g, out = capture(single)
            res[f'{qn}_T{T}_single'] = dict(timed(g, buf, out, states, tail), q_tokens=len(tail), rows=L)
            del g, out; torch.cuda.empty_cache()
            print(qn, T, 'single', res[f'{qn}_T{T}_single'], flush=True)
        if 'plain' in MODES and f'{qn}_T{T}_plain' not in res:
            rt.set_fuse(T); bufp = torch.tensor([states[0]], device=dev)
            qids = [torch.tensor([p['q']], device=dev) for p in prs]
            masks = [causal_lower_right(len(p['q']), T + len(p['q'])) for p in prs]
            oidx = [torch.tensor(p['opt'], device=dev) for p in prs]
            tds = [temp(p['rq'].kind) for p in prs]

            def plain():
                hs, cs = rt.fwd(bufp, want_cache=True)
                outs = []
                for j, p in enumerate(prs):
                    rt.mask = masks[j]; rt.set_fuse(len(p['q']))
                    hq, _ = rt.fwd(qids[j], pos0=T, cache=cs)
                    lg = head(hq[-1:].float(), hq[oidx[j]][None].float()) / tds[j]
                    outs.append(torch.softmax(lg, -1)[0, :2])
                rt.set_fuse(T)
                return torch.cat(outs)
            g, out = capture(plain)
            res[f'{qn}_T{T}_plain'] = dict(timed(g, bufp, out, states, []), q_tokens=len(tail))
            del g, out; torch.cuda.empty_cache()
            print(qn, T, 'plain', res[f'{qn}_T{T}_plain'], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
json.dump(res, open(OUT, 'w'), indent=1)
