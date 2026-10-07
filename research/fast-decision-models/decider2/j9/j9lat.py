"""J9 latency: the compiled-deployment layout R in the fused runtime (d1 lean2 + H4 TTL cached-history conv + lower-right causal flash).
Compile once per deployment: the hook frame (exact prefix) and every constant block [U + block] -> per GDN layer E (final state from U), A (transfer,
v = 0 run from I), conv tail; per attention layer K (normed + roped at compile positions) and V.
Per request (all inside one CUDA graph): compose S <- A_j (S - S_U) + E_j over the request's blocks (one batched matmul over 18 layers per block),
re-rotate each block's K by its runtime offset, concatenate [frame | blocks] K/V, then the fused forward over the LIVE tokens only (dynamic
state + question) continuing that cache, and the pointer head.
Baseline in the same harness: hobson's own layout, one fused pass over [state + question] (1 q) / state pass + one branch per question (4 q).
Discipline: CUDA graph per exact shape, exclusive GPU, fresh inputs every rep (other real states of the same shape), 3 warm-ups, n >= 12 timed,
median + p95. Timed span = H2D of ids + graph replay + D2H of probabilities.
python j9lat.py check PREDS.json | grid | real   [CKPT]"""
import os, sys, json, time, statistics as stt, glob, random
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
import numpy as np, torch, torch.nn.functional as F
import tt_lean
from tt_lean import TTL, causal_lower_right
import evalkit as EK
import j9lib as J

dev = 'cuda'
MODE = sys.argv[1]; ARG = sys.argv[2] if len(sys.argv) > 2 else ''; CKPT = sys.argv[3] if len(sys.argv) > 3 else ''
REPS = int(os.environ.get('REPS', 15))

# ---- capture A during block compile: wrap the delta-rule call inside tt_lean
_orig = tt_lean.chunk_gated_delta_rule
COMPILE = {'on': False, 'A': []}


def _wrapped(q, k, v, a, b, **kw):
    out = _orig(q, k, v, a, b, **kw)
    if COMPILE['on']:
        I0 = torch.eye(128, device=q.device, dtype=torch.float32)[None, None].expand(1, 16, 128, 128).contiguous()
        kw2 = dict(kw); kw2['initial_state'] = I0; kw2['output_final_state'] = True
        _, A = _orig(q, k, torch.zeros_like(v), a, b, **kw2)
        COMPILE['A'].append(A)
    return out


tt_lean.chunk_gated_delta_rule = _wrapped


def med(ts):
    ts = sorted(ts)
    return dict(median=round(stt.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2), n=len(ts))


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
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng; tok = eng.tok
    from h3lib import StdHead
    head = StdHead(Pm.model.head).to(dev).eval()
    if CKPT:
        sys.argv = [sys.argv[0], 'x']
        from h7lat import merge_lora
        head.load_state_dict(merge_lora(tm, CKPT))
    rt = TTL(tm, list(range(10)))
    GDN = [i for i, d in enumerate(rt.layers) if d['type'] == 'linear_attention']; ATT = [i for i, d in enumerate(rt.layers) if d['type'] != 'linear_attention']
    U = tok('<state>\n', add_special_tokens=False)['input_ids']; u = len(U)
    inv = rt.inv
    LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
    temp = Pm.temp_for

    def prep(state_text, qd):
        q = ta.validate_python(qd); rq = render_question(q)
        s, qs = eng._fit(state_text, [rq.text])
        return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)

    LIB = {}

    @torch.inference_mode()
    def compile_block(key, ids):
        if key in LIB: return LIB[key]
        COMPILE['on'] = True; COMPILE['A'] = []
        rt.set_fuse(u + len(ids)); _, c = rt.fwd(torch.tensor([U + ids], device=dev), want_cache=True)
        COMPILE['on'] = False
        A = torch.cat(COMPILE['A'], 0)                                                   # [18, 16, 128, 128]
        E = torch.cat([c[i]['S'] for i in GDN], 0)
        ent = dict(A=A, E=E, tail=[c[i]['tail'] for i in GDN], k=[c[i]['k'][u:] for i in ATT], v=[c[i]['v'][u:] for i in ATT], L=len(ids), p0=u)
        LIB[key] = ent
        return ent

    @torch.inference_mode()
    def compile_frame(ids):
        rt.set_fuse(len(ids)); _, c = rt.fwd(torch.tensor([ids], device=dev), want_cache=True)
        return dict(S=torch.cat([c[i]['S'] for i in GDN], 0), tail=[c[i]['tail'] for i in GDN], k=[c[i]['k'] for i in ATT], v=[c[i]['v'] for i in ATT], L=len(ids))

    _SU = {}

    def S_U():
        if 'S' not in _SU:
            with torch.inference_mode():
                rt.set_fuse(u); _, c = rt.fwd(torch.tensor([U], device=dev), want_cache=True)
            _SU['S'] = torch.cat([c[i]['S'] for i in GDN], 0)
        return _SU['S']

    def rot(kc, d):
        """rotate roped K by a position offset d (rotate-half on the first 64 dims)"""
        fr = d * inv; c = torch.cat([fr, fr]).cos().to(kc.dtype); s_ = torch.cat([fr, fr]).sin().to(kc.dtype)
        xr, xp = kc[..., :64], kc[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
        return torch.cat([torch.cat([x1 * c[:32] - x2 * s_[:32], x2 * c[32:] + x1 * s_[32:]], -1), xp], -1)

    def req_parts(item_state, qd):
        st = render_state(item_state)
        req = J.tokenize_pieces(tok, J.pieces(st, LL))
        p = prep(st, qd)
        assert [t for _, ids, _ in req for t in ids] == p['s']
        fr = [t for k, ids, _ in req if k == 'frame' for t in ids]
        if not fr:  # no frame: the first dyn piece is the prefix segment (as j9lib layout R)
            j0 = next(i for i, (k, _, _) in enumerate(req) if k == 'dyn'); fr = req[j0][1]; req = req[:j0] + req[j0 + 1:]
        blks = []; seen = set()
        for k, ids, key in req:
            if k == 'blk' and key not in seen: seen.add(key); blks.append((key, ids))
        dyn = [t for k, ids, _ in req if k == 'dyn' for t in ids]
        return p, fr, blks, dyn

    def make_req_fn(fr, blks, live_ids_buf, prs, nlive_state):
        """R layout request: compose cache from compiled frame + blocks, then live pass (+ question branches if several questions)"""
        F_ = compile_frame(fr); ents = [compile_block(k, ids) for k, ids in blks]
        SU = S_U()
        offs = []; pos = F_['L']
        for e in ents: offs.append(pos - e['p0']); pos += e['L']
        Ptot = pos
        dts = [torch.tensor(float(o), device=dev) for o in offs]
        n = len(prs); nst = nlive_state
        tdiv = [temp(p['rq'].kind) for p in prs]
        oidx = [torch.tensor(p['opt'], device=dev) for p in prs]
        last_tail = ents[-1]['tail'] if ents else F_['tail']

        comp = os.environ.get('COMP', 'affine')

        def fn():
            S = F_['S']
            if comp == 'affine':
                for e in ents: S = torch.matmul(e['A'], S - SU) + e['E']
            elif comp == 'last' and ents:
                S = ents[-1]['E']
            cache = {}
            for gi, i in enumerate(GDN): cache[i] = {'tail': last_tail[gi], 'S': S[gi:gi + 1]}
            for ai, i in enumerate(ATT):
                cache[i] = {'k': torch.cat([F_['k'][ai]] + [rot(e['k'][ai], dts[j]) for j, e in enumerate(ents)], 0),
                            'v': torch.cat([F_['v'][ai]] + [e['v'][ai] for e in ents], 0)}
            if n == 1:
                T = live_ids_buf.shape[1]
                rt.mask = causal_lower_right(T, Ptot + T); rt.set_fuse(T)
                h, _ = rt.fwd(live_ids_buf, pos0=Ptot, cache=cache)
                q0 = T - len(prs[0]['q'])
                lg = (head.k(head.norm(h[q0 + oidx[0]].float())) @ head.q(head.norm(h[-1].float()))) * head.scale / tdiv[0]
                return torch.softmax(lg, -1)
            T = nst
            rt.mask = causal_lower_right(T, Ptot + T); rt.set_fuse(T)
            hs, cs = rt.fwd(live_ids_buf[:, :T], pos0=Ptot, cache=cache, want_cache=True)
            for i in ATT:
                cs[i] = {'k': torch.cat([cache[i]['k'], cs[i]['k']], 0), 'v': torch.cat([cache[i]['v'], cs[i]['v']], 0)}
            outs = []; q0 = T
            for j, p in enumerate(prs):
                L = len(p['q'])
                rt.mask = causal_lower_right(L, Ptot + T + L); rt.set_fuse(L)
                hq, _ = rt.fwd(live_ids_buf[:, q0:q0 + L], pos0=Ptot + T, cache=cs); q0 += L
                lg = (head.k(head.norm(hq[oidx[j]].float())) @ head.q(head.norm(hq[-1].float()))) * head.scale / tdiv[j]
                outs.append(torch.softmax(lg, -1))
            return torch.cat(outs)
        return fn, Ptot

    def make_native_fn(buf, prs, Ls):
        n = len(prs); tdiv = [temp(p['rq'].kind) for p in prs]
        oidx = [torch.tensor(p['opt'], device=dev) for p in prs]

        def fn():
            if n == 1:
                T = buf.shape[1]; rt.set_fuse(T); rt.mask = None
                h, _ = rt.fwd(buf)
                q0 = T - len(prs[0]['q'])
                lg = (head.k(head.norm(h[q0 + oidx[0]].float())) @ head.q(head.norm(h[-1].float()))) * head.scale / tdiv[0]
                return torch.softmax(lg, -1)
            rt.set_fuse(Ls); rt.mask = None
            hs, cs = rt.fwd(buf[:, :Ls], want_cache=True)
            outs = []; q0 = Ls
            for j, p in enumerate(prs):
                L = len(p['q']); rt.mask = causal_lower_right(L, Ls + L); rt.set_fuse(L)
                hq, _ = rt.fwd(buf[:, q0:q0 + L], pos0=Ls, cache=cs); q0 += L
                lg = (head.k(head.norm(hq[oidx[j]].float())) @ head.q(head.norm(hq[-1].float()))) * head.scale / tdiv[j]
                outs.append(torch.softmax(lg, -1))
            return torch.cat(outs)
        return fn

    if MODE == 'prof':
        from torch.profiler import profile, ProfilerActivity
        banking = [it for it in EK.load_suite('LONG')]
        it = banking[0]; qn = list(it['questions'])[0]
        p, fr, blks, dyn = req_parts(it['state'], it['questions'][qn])
        dyn = dyn[:450]
        buf = torch.tensor([dyn + p['q']], device=dev)
        fn, Ptot = make_req_fn(fr, blks[:11], buf, [p], len(dyn))
        full = (fr + [t for _, ids in blks[:11] for t in ids] + dyn)[:1000 - len(p['q'])] + p['q']
        bufn = torch.tensor([full], device=dev); fnn = make_native_fn(bufn, [p], len(full) - len(p['q']))
        for name, f_ in (('compiled', fn), ('native', fnn)):
            with torch.inference_mode():
                for _ in range(3): f_()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    f_(); torch.cuda.synchronize()
            print(name, 'live', buf.shape[1] if name == 'compiled' else bufn.shape[1], 'prefix', Ptot)
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=14))
        return

    if MODE == 'check':
        pd = json.load(open(os.path.expanduser(ARG)))
        its = [x for x in EK.load_suite('REAL-agree') if x['id'] in pd and x['domain'] == 'banking_knowledge'][:10]
        res = []
        for it in its:
            qn = list(it['questions'])[0]
            p, fr, blks, dyn = req_parts(it['state'], it['questions'][qn])
            buf = torch.tensor([dyn + p['q']], device=dev)
            fn, Ptot = make_req_fn(fr, blks, buf, [p], len(dyn))
            with torch.inference_mode(): pr = fn()
            a = pr.tolist(); b = [pd[it['id']][qn][lab] for lab in p['rq'].slot_labels]
            res.append((int(np.argmax(a)) == int(np.argmax(b)), max(abs(x - y) for x, y in zip(a, b)), len(blks), Ptot, len(dyn)))
            print(res[-1], flush=True)
        print('runtime R vs j9lib R: argmax %d/%d, max|dp| median %.4f max %.4f' % (sum(r[0] for r in res), len(res), float(np.median([r[1] for r in res])), max(r[1] for r in res)))
        json.dump(dict(agree=sum(r[0] for r in res), n=len(res), dp_med=float(np.median([r[1] for r in res])), dp_max=max(r[1] for r in res)),
                  open(os.path.expanduser('~/work/j9/latcheck.json'), 'w'))
        return

    # pool of real requests (eval split; banking + retail) with their pieces
    spec = {}
    pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            pool.append((s_, it))
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ckpt=CKPT, note='R layout, TTL fused runtime, bf16')}
    OUT = os.path.expanduser(f"~/work/j9/lat_{MODE}_{os.environ.get('COMP', 'affine')}.json")
    if MODE == 'grid':
        # exact T with constant share c: state = frame(45) + blocks (c*T, split into nb blocks) + live; question(s) from the banking spec
        Q4 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses']
        rng = random.Random(0)
        banking = [it for s_, it in pool if it['domain'] == 'banking_knowledge']
        toks = [tok(render_state(it['state']), add_special_tokens=False)['input_ids'] for it in banking[:150] + banking[-60:]]
        toks += [a_ + b_ for a_, b_ in zip(toks[::2], toks[1::2])]     # concatenations for the longest grid points
        for qset in (Q4[:1], Q4):
            prs = [prep('S', spec[k]) for k in qset]; qtok = [t for p in prs for t in p['q']]
            for T in (64, 128, 256, 400, 1000, 4000):
                srcs = [t for t in toks if len(t) >= T][:REPS + 3]
                if len(srcs) < REPS + 3: srcs = (srcs * (REPS + 3))[:REPS + 3]
                inputs = [t[:T] + qtok for t in srcs]
                buf = torch.tensor([inputs[0]], device=dev)
                g, out = capture(make_native_fn(buf, prs, T)); r0 = timed(g, buf, out, inputs); del g, out
                res[f'native_q{len(qset)}_T{T}'] = r0
                for c, nb in ((0.37, 4), (0.55, 11)):
                    P_ = int(round(c * T)); Fl = min(45, max(4, T // 8)); bl = P_ - Fl
                    if bl < nb * 8: nb = max(1, bl // 16)
                    if bl < 8: continue
                    sizes = [bl // nb] * nb; sizes[-1] += bl - sum(sizes)
                    src = toks[0]
                    fr = src[:Fl]; blks = []; o = Fl
                    for j, sz in enumerate(sizes): blks.append((f'g{T}_{c}_{j}', src[o:o + sz])); o += sz
                    Lst = T - P_
                    inputs2 = [t[P_:P_ + Lst] + qtok for t in srcs]
                    buf2 = torch.tensor([inputs2[0]], device=dev)
                    fn, Ptot = make_req_fn(fr, blks, buf2, prs, Lst)
                    g, out = capture(fn); r1 = timed(g, buf2, out, inputs2); del g, out
                    res[f'R_c{c}_q{len(qset)}_T{T}'] = dict(r1, live=Lst, prefix=Ptot, nblk=len(blks))
                    print(T, len(qset), 'native', r0, 'R c', c, r1, flush=True)
                torch.cuda.empty_cache()
                json.dump(res, open(OUT, 'w'), indent=1)
    if MODE == 'real':
        # the real length distribution: a stratified sample of eval-split requests (first question), native vs compiled, each at its exact shape
        rng = random.Random(1)
        sample = rng.sample([x for x in pool if x[0] == 'REAL-agree'], int(os.environ.get('NREAL', 60))) + rng.sample([x for x in pool if x[0] == 'LONG'], int(os.environ.get('NLONG', 24)))
        rows = []
        for s_, it in sample:
            qn = list(it['questions'])[0]
            p, fr, blks, dyn = req_parts(it['state'], it['questions'][qn])
            full = p['s'] + p['q']; live = dyn + p['q']
            # fresh inputs: the same shape with other real token ids (random tokens from the state) every rep
            inp_n = [full] + [[full[(j * 7919 + r) % len(full)] for j in range(len(full))] for r in range(REPS + 2)]
            inp_c = [live] + [[live[(j * 7919 + r) % len(live)] for j in range(len(live))] for r in range(REPS + 2)]
            buf = torch.tensor([full], device=dev)
            g, out = capture(make_native_fn(buf, [p], len(p['s']))); rn = timed(g, buf, out, inp_n); del g, out
            buf2 = torch.tensor([live], device=dev)
            fn, Ptot = make_req_fn(fr, blks, buf2, [p], len(dyn))
            g, out = capture(fn); rc = timed(g, buf2, out, inp_c); del g, out
            torch.cuda.empty_cache()
            rows.append(dict(suite=s_, id=it['id'], domain=it['domain'], hook=it['hook'], T=len(full), live=len(live), prefix=Ptot, nblk=len(blks),
                             native=rn['median'], native_p95=rn['p95'], compiled=rc['median'], compiled_p95=rc['p95']))
            print(rows[-1], flush=True)
            res['rows'] = rows
            json.dump(res, open(OUT, 'w'), indent=1)
        nat = [r['native'] for r in rows]; com = [r['compiled'] for r in rows]
        res['summary'] = dict(native_median=stt.median(nat), compiled_median=stt.median(com), native_mean=float(np.mean(nat)), compiled_mean=float(np.mean(com)),
                              speedup_of_means=float(np.mean(nat) / np.mean(com)), median_speedup=float(np.median([a / b for a, b in zip(nat, com)])),
                              tok_total=float(np.mean([r['T'] for r in rows])), tok_live=float(np.mean([r['live'] for r in rows])))
        print(res['summary'])
        json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
