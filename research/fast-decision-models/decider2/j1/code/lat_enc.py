"""J1 latency + runtime check for the encoder decider in the fused runtime (encrt.EncRT).
python lat_enc.py check CKPT          runtime probs vs the training-path reference (EncTorso.forward_packed, LoRA unmerged) on real items, 1 and M questions
python lat_enc.py time [CKPT] [OUT]   exact T in TS x question sets Q1/Q4/Q15 (h7lat's banking questions); CUDA graph per exact shape, fresh real
                                      states each rep, 3 warm-ups discarded, n=REPS timed; timed span = H2D of the ids + replay + D2H of the probs.
env: TS=64,128,.. REPS=20 LOCAL=layers WINDOW=W (state->state window on those layers)"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/sd/src')]
import numpy as np, torch
import encj1 as E, encrt as R
import evalkit as EK
from transformers import AutoTokenizer
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderConfig, build_head

dev = 'cuda'
MODE = sys.argv[1]; CK = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else ''
OUT = sys.argv[3] if len(sys.argv) > 3 else 'lat_enc.json'
TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,2000,4000').split(',')]
REPS = int(os.environ.get('REPS', 20))
LOCAL = [int(x) for x in os.environ.get('LOCAL', '').split(',') if x != '']; WINDOW = int(os.environ.get('WINDOW', 0))
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4], 'Q15': Q15}
ta = TypeAdapter(SC.Question)
tok = AutoTokenizer.from_pretrained(E.find_ckpt())

NL = int(os.environ.get('NL', 26))
W, _ = E.load_weights(n_layers=NL)
torso = E.EncTorso(W, local_layers=LOCAL, window=WINDOW, n_layers=NL).to(dev); del W
torso.mode = 'masked'
hc = StrandsDeciderConfig(base_model='t5gemma', head_type='pointer', pointer_dim=256, head_dropout=0.0)
head = build_head(hc, E.D).to(dev).eval()
temps = dict(noul=1.0, choice=1.0, score=1.0)
if CK.endswith('.pt'):   # a trainer resume file (LoRA + head of a run in progress)
    ck = torch.load(CK, weights_only=False)
    torso.lora.load_state_dict({k: v.to(dev) for k, v in ck['lora'].items() if int(k.split('.')[0]) < NL}); head.load_state_dict(ck['head'])
elif CK:
    torso.lora.load_state_dict({k: v.to(dev) for k, v in torch.load(os.path.join(CK, 'lora.pt')).items() if int(k.split('.')[0]) < NL})
    head.load_state_dict(torch.load(os.path.join(CK, 'slot_head.pt')))
    mp = os.path.join(CK, 'temps.json')
    if os.path.exists(mp): temps = json.load(open(mp))
else:
    with torch.no_grad():
        for p in torso.lora.parameters(): p.normal_(0, 0.01)
torso.eval()
rt = R.EncRT(torso.merged(), torso.embed, torso.norm1, head, local=LOCAL, window=WINDOW)
if MODE == 'time':
    torso.L = None; torch.cuda.empty_cache()


def qprep(spec):
    rq = render_question(ta.validate_python(spec))
    e = tok([rq.text], add_special_tokens=False, return_offsets_mapping=True)
    return dict(q=e['input_ids'][0], opt=_option_token_index(e['offset_mapping'][0], rq.option_spans, 0), rq=rq, kind=rq.kind)


def sprep(state):
    return [tok.bos_token_id] + tok(render_state(state), add_special_tokens=False)['input_ids']


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


def timed(g, buf, out, inputs, qtail):
    pins = [torch.from_numpy(np.asarray(x + qtail, dtype=np.int64)).pin_memory() for x in inputs]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pins:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        buf.copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


@torch.no_grad()
def ref_probs(s, prs):
    """training-path reference: one packed row per question (state never sees the question), LoRA unmerged"""
    rows = [dict(ids=torch.tensor(s + p['q']), nstate=len(s)) for p in prs]
    lens = [len(r['ids']) for r in rows]; meta = E.pack_meta(lens, [len(s)] * len(rows), dev)
    hid = torso.forward_packed(torch.cat([r['ids'] for r in rows]).to(dev), meta)
    st_ = meta['starts'].tolist(); out = []
    for j, p in enumerate(prs):
        oi = torch.tensor([st_[j] + len(s) + o for o in p['opt']], device=dev)
        lg = head(hid[meta['last'][j]].float()[None], hid[oi].float()[None])[0] / temps.get(p['kind'], 1.0)
        out.append(torch.softmax(lg, -1).cpu())
    return out


if MODE == 'check':
    its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) == 4][:8] + [x for x in EK.load_suite('LONG') if len(x['questions']) >= 3][:3] + EK.load_suite('JB-all')[:6]
    res = []
    if NL < 26:   # debug while training holds the GPU: short items only
        its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2 and len(sprep(x['state'])) < 700][:5] + EK.load_suite('JB-all')[:4]
    for it in its:
        s = sprep(it['state']); prs = [qprep(sp) for sp in it['questions'].values()]
        ref = ref_probs(s, prs)
        b = rt.build(len(s), [len(p['q']) for p in prs], [p['opt'] for p in prs], [temps.get(p['kind'], 1.0) for p in prs])
        b['ids'].copy_(torch.tensor(s + [t for p in prs for t in p['q']], device=dev))
        with torch.inference_mode(): pr = b['fn']().cpu()
        # also M=1 path per question
        for j, p in enumerate(prs):
            b1 = rt.build(len(s), [len(p['q'])], [p['opt']], [temps.get(p['kind'], 1.0)])
            b1['ids'].copy_(torch.tensor(s + p['q'], device=dev))
            with torch.inference_mode(): p1 = b1['fn']().cpu()[0]
            k = len(p['opt']); a_ = pr[j, :k]; r_ = ref[j]
            res.append(dict(M=len(prs), T=len(s), agreeM=int(a_.argmax() == r_.argmax()), dpM=float((a_ - r_).abs().max()),
                            agree1=int(p1[:k].argmax() == r_.argmax()), dp1=float((p1[:k] - r_).abs().max())))
    s_ = dict(n=len(res), agree_multi=sum(r['agreeM'] for r in res), agree_single=sum(r['agree1'] for r in res),
              dp_multi_med=float(np.median([r['dpM'] for r in res])), dp_multi_max=max(r['dpM'] for r in res),
              dp_single_med=float(np.median([r['dp1'] for r in res])), dp_single_max=max(r['dp1'] for r in res))
    print('CHECK fused runtime vs training-path reference:', s_, flush=True)
    json.dump(dict(summary=s_, rows=res), open('latcheck_enc.json', 'w'), indent=1)
    sys.exit(0)

# ---------------- timing
spec = {}; pool = []
for s_ in ('REAL-agree', 'LONG'):
    for it in EK.load_suite(s_):
        for q, sp in it['questions'].items(): spec.setdefault(q, sp)
        if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
toks = [sprep(x) for x in pool]
res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ck=CK, local=LOCAL, window=WINDOW,
                    note='T5Gemma-2B encoder decider, fused EncRT, masked state cache, exact T')}
if os.path.exists(OUT): res.update(json.load(open(OUT)))
for qn, names in QSETS.items():
    prs = [qprep(spec[k]) for k in names]
    qtail = [t for p in prs for t in p['q']]
    for T in TS:
        key = f'{qn}_T{T}'
        if key in res: continue
        states = [t[:T] for t in toks if len(t) >= T][:REPS + 3]
        assert len(states) == REPS + 3, (T, len(states))
        b = rt.build(T, [len(p['q']) for p in prs], [p['opt'] for p in prs], [temps.get(p['kind'], 1.0) for p in prs])
        b['ids'].copy_(torch.tensor(states[0] + qtail, device=dev))
        g, out = capture(b['fn'])
        res[key] = dict(timed(g, b['ids'], out, states, qtail), q_tokens=len(qtail), rows=b['T'])
        del g, out, b; torch.cuda.empty_cache()
        print(key, res[key], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
json.dump(res, open(OUT, 'w'), indent=1)
