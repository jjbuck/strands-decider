"""J10 runtime checks + latency (A10G, exclusive GPU).
  python bench_j10.py check CKPT|base PRECS            -> hidden cosine + decision agreement of RT(prec) vs the emulated torso on real questions
  python bench_j10.py lat   CKPT|base PRECS TS NQS [tag] -> res_lat{tag}.jsonl: wall median/p95 (fresh ids H2D + graph replay + D2H probs),
                                                          kernel split (gemm / attn / glue) from one profiled replay
  python bench_j10.py gemm  PRECS MS                    -> per-GEMM-class microbenchmarks at exact M
PRECS: comma list of  bf16 | i8 | w2 | s4 | mix (qkv,gu s4; o,down i8) | mixw2 (qkv,gu s4; o,down w2) | auto (w2 below M 256 else i8)
"""
import os, sys, json, time, statistics as st, collections, random
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/sd/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
import bitnet_j10 as BJ
import rt_j10 as R
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
from strands_decider.prompting import render_question, render_state
from pydantic import TypeAdapter
import strands_decider.schema as SC
from transformers import AutoTokenizer
ta = TypeAdapter(SC.Question)
PRECS = dict(bf16='bf16', i8='i8', w2='w2', s4='s4', mix=dict(qkv='s4', gu='s4', o='i8', down='i8'), mixw2=dict(qkv='s4', gu='s4', o='w2', down='w2'),
             w2s4=dict(qkv='w2', gu='w2', o='w2', down='s4'), c8='c8', mixc=dict(qkv='s4', gu='s4', o='c8', down='c8'))


def load(ck):
    if os.environ.get('NLAYERS'):   # small-memory kernel check: first N layers of the base model only
        cfg, W = BJ.load(dev='cpu'); n = int(os.environ['NLAYERS'])
        cfg = dict(cfg, num_hidden_layers=n)
        W = {k: v.cuda() for k, v in W.items() if not k.startswith('model.layers.') or int(k.split('.')[2]) < n}
        t = BJ.BitNetTorso(cfg, W, lora=False).cuda(); del W
        h = build_head(StrandsDeciderConfig(head_type='pointer', pointer_dim=256), t.d)
        return t.eval(), h.cuda().eval()
    cfg, W = BJ.load()
    t = BJ.BitNetTorso(cfg, W, r=16, alpha=32, lora=(ck != 'base')).cuda(); del W
    if ck != 'base':
        t.load_trainable(torch.load(os.path.join(ck, 'torso_trainable.pt')))
        c = StrandsDeciderConfig.from_json(os.path.join(ck, 'strands_decider_config.json'))
        h = build_head(c, t.d); h.load_state_dict(torch.load(os.path.join(ck, 'slot_head.pt')))
    else:
        h = build_head(StrandsDeciderConfig(head_type='pointer', pointer_dim=256), t.d)
    t.eval(); h = h.cuda().eval()
    return t, h


def real_questions(tok, n_req=40, seed=0, maxT=3000):
    """real REAL-agree requests -> list of dict(s=state ids, qs=[dict(ids, opt)]) with the BitNet tokenizer"""
    from prep_g3 import fit
    its = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl'))]
    random.Random(seed).shuffle(its)
    out = []
    for it in its:
        st_ = render_state(it['state']); qs = []
        for qn, qd in it['questions'].items():
            rq = render_question(ta.validate_python(qd))
            s, q, opt = fit(tok, st_, rq.text, rq, max_len=16384)
            qs.append(dict(ids=q, opt=[o - len(s) for o in opt], rq=rq, qn=qn))
        if len(s) > maxT: continue
        out.append(dict(id=it['id'], s=s, qs=qs))
        if len(out) >= n_req: break
    return out


class Req:
    """state of T ids + the given questions in one packed pass; returns softmax probs per question [nq, Kmax]"""
    def __init__(self, rt, s_ids, qs):
        self.rt = rt; T = len(s_ids); self.T = T
        self.lay = R.Packed(T, [len(q['ids']) for q in qs])
        ids = list(s_ids)
        ans, opt = [], []; o = T; K = max(len(q['opt']) for q in qs)
        for q in qs:
            ids += q['ids']; ans.append(o + len(q['ids']) - 1)
            opt.append([o + j for j in q['opt']] + [o + q['opt'][0]] * (K - len(q['opt']))); o += len(q['ids'])
        self.ids = torch.tensor(ids, device='cuda'); self.ans = torch.tensor(ans, device='cuda'); self.opt = torch.tensor(opt, device='cuda')
        self.ns = torch.tensor([len(q['opt']) for q in qs], device='cuda')
        self.B = rt.buffers(self.lay.M)

    def run(self):
        h = self.rt.forward(self.ids, self.lay.pos, self.B, self.lay.attn)
        rows = torch.cat([self.ans, self.opt.reshape(-1)])
        hn = self.rt.final(h[rows]).float()
        nq = self.ans.shape[0]
        dec = hn[:nq]; opts = hn[nq:].reshape(nq, -1, hn.shape[-1])
        return masked_log_softmax(self.rt.head(dec, opts), self.ns).exp()


@torch.inference_mode()
def emulated(t, h, s_ids, q):
    ids = torch.tensor([list(s_ids) + q['ids']], device='cuda')
    hid = t(ids)
    opt = torch.tensor([[len(s_ids) + j for j in q['opt']]], device='cuda')
    return masked_log_softmax(h(hid[:, -1].float(), gather_options(hid, opt).float()), torch.tensor([len(q['opt'])], device='cuda')).exp()[0], hid[0]


def check(ck, precs):
    t, h = load(ck); tok = AutoTokenizer.from_pretrained(BJ.snap())
    reqs = real_questions(tok, int(os.environ.get('NCHECK', 30)))
    ref = []
    for r in reqs:
        q = r['qs'][0]; p, hid = emulated(t, h, r['s'], q); ref.append((p, hid[-1]))
    res = {}
    for pn in precs:
        rt = R.RT(t, h, PRECS[pn]); agree = 0; tv = []; cos = []
        with torch.inference_mode():
            for r, (p0, h0) in zip(reqs, ref):
                rq = Req(rt, r['s'], r['qs'][:1]); p = rq.run()[0, :len(p0)]
                hl = rt.final(rq.rt.forward(rq.ids, rq.lay.pos, rq.B, rq.lay.attn)[-1:])[0]
                agree += int(p.argmax() == p0.argmax()); tv.append(0.5 * float((p - p0).abs().sum()))
                cos.append(float(F.cosine_similarity(hl.float(), h0.float(), dim=0)))
            # packed 4-question path vs single-question passes (same runtime)
            pk = []
            for r in reqs[:12]:
                if len(r['qs']) < 2: continue
                qs = r['qs'][:4]; pp = Req(rt, r['s'], qs).run()
                for j, q in enumerate(qs):
                    p1 = Req(rt, r['s'], [q]).run()[0]
                    pk.append(0.5 * float((pp[j, :len(q['opt'])] - p1[:len(q['opt'])]).abs().sum()))
        res[pn] = dict(n=len(reqs), agree=agree / len(reqs), tv_mean=sum(tv) / len(tv), tv_max=max(tv), cos_min=min(cos), cos_mean=sum(cos) / len(cos),
                       packed_vs_single_tv_mean=(sum(pk) / len(pk)) if pk else None, packed_n=len(pk))
        print(pn, json.dumps(res[pn]), flush=True)
        del rt; torch.cuda.empty_cache()
    json.dump(res, open(f'check_{os.path.basename(ck.rstrip("/"))}.json', 'w'), indent=1)


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        out = fn()
    return g, out


def wall(req, g, out, reps=30, warm=5):
    T = req.T
    pool = [torch.randint(1000, 120000, (T,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids[:T].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warm:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))


def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl or 'softmax' in nl: return 'attn'
    if '_i8mm' in nl or '_w2mm' in nl or 'gemm' in nl or 'cutlass' in nl or 'ampere_' in nl or 'sm80' in nl: return 'gemm'
    if any(k in nl for k in ('_addnormq', '_relu2normq', '_attnnormq', '_rope')): return 'glue'
    return 'other'


def kprof(g):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); n = 0
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        agg[kcat(e.name)] += dt; n += 1
    r = {k: round(v, 3) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 3)
    return r


def lat(ck, precs, Ts, nqs, tag=''):
    t, h = load(ck); tok = AutoTokenizer.from_pretrained(BJ.snap())
    # a real 4-question banking request supplies the question token sequences (BitNet tokenizer)
    reqs = [r for r in real_questions(tok, 400) if len(r['qs']) >= 4]
    qs4 = reqs[0]['qs'][:4]
    print('question tokens', [len(q['ids']) for q in qs4], flush=True)
    out_path = f'res_lat{tag}.jsonl'
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path): done.add(json.loads(l)['key'])
    for pn in precs:
        rt = R.RT(t, h, PRECS[pn]) if pn != 'auto' else None
        for T in Ts:
            for nq in nqs:
                key = f'{pn}|{T}|{nq}'
                if key in done: continue
                if pn == 'auto':
                    rt = R.RT(t, h, 'w2' if T <= 256 else 'i8')
                req = Req(rt, [5] * T, qs4[:nq])
                g, out = capture(req.run)
                w = wall(req, g, out); kp = kprof(g)
                rec = dict(key=key, prec=pn, T=T, nq=nq, qtok=sum(len(q['ids']) for q in qs4[:nq]), rows=req.lay.M, wall=w, kern=kp,
                           mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                print(json.dumps(rec), flush=True)
                with open(out_path, 'a') as f: f.write(json.dumps(rec) + '\n')
                del g, out, req; torch.cuda.empty_cache()
                if pn == 'auto': del rt; rt = None; torch.cuda.empty_cache()
        del rt; torch.cuda.empty_cache()


def gemm(precs, Ms):
    shapes = dict(qkv=(3840, 2560), o=(2560, 2560), gu=(13824, 2560), down=(2560, 6912))
    res = []
    for M in Ms:
        for cls, (N, K) in shapes.items():
            codes = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8)
            sw = torch.full((N,), 0.02, device='cuda'); r = dict(M=M, cls=cls, N=N, K=K)
            a8 = torch.randint(-127, 128, (M, K), device='cuda', dtype=torch.int8); sa = torch.rand(M, device='cuda')
            for pn in precs:
                if pn == 'bf16':
                    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16); Wt = (codes.to(torch.bfloat16) * 0.02).t().contiguous()
                    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16); fn = lambda: torch.matmul(A, Wt, out=out)
                elif pn == 'i8':
                    Wt = codes.t().contiguous(); out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16); fn = lambda: R.i8mm(a8, Wt, sa, sw, out)
                elif pn == 'w2':
                    Wp = R.pack2(codes); out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16); fn = lambda: R.w2mm(a8, Wp, sa, sw, out)
                elif pn == 'c8':
                    out = torch.empty(M, N, device='cuda', dtype=torch.float16); W8 = codes.contiguous(); fn = lambda: R.c8mm(a8, W8, 1 / 16, out)
                elif pn == 's4':
                    a4 = R.pack4(torch.randint(-8, 8, (M, K), device='cuda', dtype=torch.int8)); W4 = R.pack4(codes)
                    out = torch.empty(M, N, device='cuda', dtype=torch.float16); fn = lambda: R.s4mm(a4, W4, 1 / 16, out, R.S4CFG)
                for _ in range(5): fn()
                torch.cuda.synchronize(); ts = []
                for _ in range(30):
                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
                r[pn] = round(st.median(ts), 1)
            print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open('res_gemm_j10.json', 'w'), indent=1)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check': check(sys.argv[2], sys.argv[3].split(','))
    elif mode == 'lat': lat(sys.argv[2], sys.argv[3].split(','), [int(x) for x in sys.argv[4].split(',')], [int(x) for x in sys.argv[5].split(',')], sys.argv[6] if len(sys.argv) > 6 else '')
    elif mode == 'gemm': gemm(sys.argv[2].split(','), [int(x) for x in sys.argv[3].split(',')])
