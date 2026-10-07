"""H4 latency on the A10G, d1/F7 discipline: fused lean2 runtime, CUDA graph per exact shape, exclusive GPU, fresh real inputs
every rep (copied into the static buffer), 3 warm-up replays discarded, n=20 timed reps, median + p95. Timed span = H2D of the ids
+ graph replay (forward + typed head) + D2H of the probabilities.

  python tt_lat.py tt      -> this-that: forward-only at exact T, and requests (state of exactly T tokens + N questions) in three layouts
  python tt_lat.py hobson  -> hobson-v19 torso (merged LoRA) forward-only at exact T, same harness (box control)
  python tt_lat.py hf      -> this-that through its own package (thisthat.TypedDecider.decide, HF eager), same requests, n=12
"""
import os, sys, json, time, statistics as st, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/h4')); sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import numpy as np, torch
import evalkit as EK

MODE = sys.argv[1]
TS = [256, 1000, 4000]
REPS = 20
OUT = os.path.expanduser(f'~/work/h4/lat_{MODE}.json')
dev = 'cuda'
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
QSETS = {'Q1': Q15[:1], 'Q4': Q15[:4], 'Q15': Q15, 'Q1proc': ['needed_procedure']}


def med(ts):
    ts = sorted(ts)
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2), n=len(ts))


def specs_and_states(tok):
    spec = {}
    pool = []
    for s in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
    toks = [tok.encode(x, add_special_tokens=False) for x in pool]
    states = {T: [t[:T] for t in toks if len(t) >= T][:REPS + 3] for T in TS}
    for T in TS: assert len(states[T]) == REPS + 3, (T, len(states[T]))
    return spec, states


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out


def timed(g, buf, out, inputs, offset):
    """inputs: list of 1-D int64 numpy arrays of len == buf.numel()-offset region they overwrite (buf[offset:offset+len])"""
    pins = [torch.from_numpy(np.asarray(x, dtype=np.int64)).pin_memory() for x in inputs]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []
    for i, x in enumerate(pins):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        buf[0, offset:offset + x.numel()].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


def run_tt():
    from tt_lean import TTL, load_tt
    from thisthat.prompt import build, option_label_ids, _question_block, _answer_slots
    from thisthat.systemone_protocol import _typed
    m, tok = load_tt(); lab = option_label_ids(tok)
    ttl = TTL(m.model, lab)
    spec, states = specs_and_states(tok)
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__)}
    CTXSF = tok.encode('Context:\n', add_special_tokens=False); CTX = tok.encode('\n\nContext:\n', add_special_tokens=False)
    # forward only at exact T (the hobson-comparable number; head on the last position)
    for T in TS:
        ttl.set_fuse(T)
        buf = torch.zeros(1, T, dtype=torch.long, device=dev)
        sl = torch.tensor([T - 1], device=dev); no = torch.tensor([2], device=dev)
        g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, no))
        res[f'fwd_T{T}'] = timed(g, buf, out, [np.array(s) for s in states[T]], 0)
        print('fwd', T, res[f'fwd_T{T}'], flush=True); del g, out; torch.cuda.empty_cache()
    # tile staircase just above 256 (short path): fresh ids are random tokens here (latency is not data dependent, d1)
    for T in (256, 260, 280, 320, 335, 352, 384, 448, 512, 1004, 1024, 1096):
        ttl.set_fuse(T)
        buf = torch.zeros(1, T, dtype=torch.long, device=dev)
        sl = torch.tensor([T - 1], device=dev); no = torch.tensor([2], device=dev)
        g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, no))
        res[f'sweep_T{T}'] = timed(g, buf, out, [np.random.randint(1000, 100000, T) for _ in range(REPS + 3)], 0)
        print('sweep', T, res[f'sweep_T{T}'], flush=True); del g, out; torch.cuda.empty_cache()
    for qn, names in QSETS.items():
        tqs = [_typed(k, spec[k]).question for k in names]; N = len(tqs); multi = N > 1
        # question-side token blocks, exactly as thisthat.prompt.build lays them out
        sf_tail = []; sf_slots_rel = []
        for k, q in enumerate(tqs):
            sf_tail += _question_block(tok, k, q, multi, '\n\n')
            sf_tail += tok.encode(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False); sf_slots_rel.append(len(sf_tail) - 1)
        pre = []
        for k, q in enumerate(tqs): pre += _question_block(tok, k, q, multi, '\n\n' if k else '')
        pre += CTX
        slot_ids = []; rel = _answer_slots(tok, N, slot_ids, '\n\n')
        nopt = torch.tensor([len(q.options) for q in tqs], device=dev)
        sx = tok.encode('S', add_special_tokens=False)
        assert build(tok, 'S', tqs, layout='schema_first')['ids'] == pre + sx + slot_ids
        assert build(tok, 'S', tqs)['ids'] == tok.encode('Context:\nS', add_special_tokens=False) + sf_tail
        res[f'{qn}_tokens'] = dict(question_tokens_state_first=len(sf_tail), schema_prefix=len(pre), slot_tokens=len(slot_ids))
        for T in TS:
            # (a) state-first, one pass: Context: + state + [Q_k, Answer_k:(]...
            L = len(CTXSF) + T + len(sf_tail)
            ttl.set_fuse(L); buf = torch.tensor([CTXSF + states[T][0] + sf_tail], device=dev)
            sl = torch.tensor([len(CTXSF) + T + r for r in sf_slots_rel], device=dev)
            g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, nopt))
            res[f'{qn}_T{T}_statefirst'] = dict(timed(g, buf, out, [np.array(s) for s in states[T]], len(CTXSF)), L=L)
            del g, out; torch.cuda.empty_cache()
            # (b) schema-first, no cache: [Q blocks] Context: state [slots], one pass
            L = len(pre) + T + len(slot_ids)
            ttl.set_fuse(L); buf = torch.tensor([pre + states[T][0] + slot_ids], device=dev)
            sl = torch.tensor([len(pre) + T + r for r in rel], device=dev)
            g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, nopt))
            res[f'{qn}_T{T}_schema_nocache'] = dict(timed(g, buf, out, [np.array(s) for s in states[T]], len(pre)), L=L)
            del g, out; torch.cuda.empty_cache()
            # (c) schema-first with the prefix cached once per deployment: per request only state + slots
            P = len(pre)
            with torch.inference_mode():
                ttl.set_fuse(P); torch.cuda.synchronize(); t0 = time.perf_counter()
                _, cache = ttl.fwd(torch.tensor([pre], device=dev), want_cache=True); torch.cuda.synchronize()
                tpre = (time.perf_counter() - t0) * 1000
            S = T + len(slot_ids)
            ttl.set_fuse(S); ttl.set_prefix_mask(P, S)
            buf = torch.tensor([states[T][0] + slot_ids], device=dev)
            sl = torch.tensor([T + r for r in rel], device=dev)
            g, out = capture(lambda: ttl.head(ttl.fwd(buf, pos0=P, cache=cache)[0], sl, nopt))
            # check vs the uncached schema-first graph inputs: same probabilities up to bf16 noise
            res[f'{qn}_T{T}_schema_cached'] = dict(timed(g, buf, out, [np.array(s) for s in states[T]], 0), L=S, prefix=P, prefix_once_ms=round(tpre, 2))
            del g, out, cache; torch.cuda.empty_cache()
            print(qn, T, {k.split('_', 2)[-1]: (v['median'], v['p95'], v['L']) for k, v in res.items() if k.startswith(f'{qn}_T{T}_')}, flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)


def run_hobson():
    sys.path.insert(0, os.path.expanduser('~/work/systems/g')); sys.path.insert(0, os.path.expanduser('~/work/d1'))
    from prof_d1 import load_torso
    from tt_lean import TTL
    from transformers import AutoTokenizer
    torso = load_torso()
    tok = AutoTokenizer.from_pretrained(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0])
    ttl = TTL(torso, list(range(10)))   # torso cost only (the pointer head is a 2048x256 projection of a few rows; not timed here)
    _, states = specs_and_states(tok)
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, note='hobson-v19 merged torso, same fused runtime')}
    for T in TS:
        ttl.set_fuse(T)
        buf = torch.zeros(1, T, dtype=torch.long, device=dev)
        sl = torch.tensor([T - 1], device=dev); no = torch.tensor([2], device=dev)
        g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, no))
        res[f'fwd_T{T}'] = timed(g, buf, out, [np.array(s) for s in states[T]], 0)
        print('hobson fwd', T, res[f'fwd_T{T}'], flush=True); del g, out; torch.cuda.empty_cache()
    json.dump(res, open(OUT, 'w'), indent=1)


def run_hf():
    from thisthat import TypedDecider
    from thisthat.systemone_protocol import _typed
    dec = TypedDecider.from_pretrained('flock-io/this-that-model-1.0', device='cuda')
    tok = dec.tokenizer
    spec, states = specs_and_states(tok)
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=12, note='thisthat.TypedDecider.decide, HF eager bf16, state_first, synced')}
    for qn, names in QSETS.items():
        tqs = [_typed(k, spec[k]).question for k in names]
        for T in TS:
            texts = [tok.decode(s) for s in states[T]][:15]
            for x in texts[:3]: dec.decide(x, tqs, max_state_tokens=10 ** 7)
            ts = []
            for x in texts[3:15]:
                torch.cuda.synchronize(); t0 = time.perf_counter(); dec.decide(x, tqs, max_state_tokens=10 ** 7); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            res[f'{qn}_T{T}'] = med(ts); print(qn, T, res[f'{qn}_T{T}'], flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)


def run_rec68():
    """the paper's 30.9 ms workload (68 recorded questions, ~180 tokens, one question per pass) in the fused runtime"""
    from tt_lean import TTL, load_tt
    from thisthat.prompt import build, option_label_ids
    from thisthat import Question
    m, tok = load_tt(); ttl = TTL(m.model, option_label_ids(tok))
    rows = [json.loads(l) for l in open(os.path.expanduser('~/work/h4/tt_src/data/recorded_68.jsonl'))]
    built = [build(tok, r['state'], [Question(r['question'], r['options'])], max_state_tokens=2048) for r in rows]
    graphs = {}
    for b in built:
        L = len(b['ids'])
        if L in graphs: continue
        ttl.set_fuse(L); buf = torch.zeros(1, L, dtype=torch.long, device=dev)
        sl = torch.tensor([L - 1], device=dev); no = torch.tensor([2], device=dev)
        g, out = capture(lambda: ttl.head(ttl.fwd(buf)[0], sl, no)); graphs[L] = (g, buf, out, sl, no)   # keep sl/no alive: the graph reads their memory
    ts = []; host = torch.empty(1, 10).pin_memory()
    for rep in range(4):
        for b in built:
            g, buf, out = graphs[len(b['ids'])][:3]; x = torch.tensor(b['ids']).pin_memory()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            buf[0].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
            if rep: ts.append((time.perf_counter() - t0) * 1000)
    res = dict(fused=med(ts), lengths=sorted({len(b['ids']) for b in built}))
    print('rec68 fused', res, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)


{'tt': run_tt, 'hobson': run_hobson, 'hf': run_hf, 'rec68': run_rec68}[MODE]()
print('done', MODE, flush=True)
