"""Latency of the depth cascade k64rr -> layer-16 exit (A10G, exclusive GPU, CUDA graphs, fresh state ids, 3 warm + 20 timed, median/p95 ms).
python q2xbench.py grid NAMES TS BUNDLES      NAMES: b8 (H2 deployed, J15's QRTJ) | k64 (QRT2C k64rr map). modes timed per (T, bundle):
     full = whole forward + hobson head; pre16 = layers 0-15 + exit head (the exit path); post16 = layers 16-23 + head from a residual.
python q2xbench.py real TAU                   J15's 120 real requests (j15/res/res_dcasc_16.jsonl ids) at exact lengths: segment graphs
     [0,16)+exit head -> host margin check -> [16,24)+head (k64rr), full k64rr graph, full b8 graph. -> res_xreal.jsonl"""
import os, sys, json, time, statistics as st, torch, copy
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch.nn.functional as F
import qrt as Q, q2rt as R
from q2bench import capture, kprof
from j15learn import EH
M = os.path.expanduser('~/work/q2')
CODES = os.environ.get('CODES', 'w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')
L = 16


def load_codes(spec):
    return {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in spec.split(',')}


def build(name, ln, head, wc):
    if name == 'b8':
        from j15run import QRTJ
        m = QRTJ(ln, head=head, prec='map:~/work/h1/precmap_w8a8_b8.json:w8a8', wcodes=wc)
    elif name == 'k64':
        m = R.QRT2C(ln, head=head, fmt=f'map:{M}/q2map_k64rr.json', wcodes=wc)
    else: raise ValueError(name)
    m.tune = True
    return m


def exit_head(P, path):
    h = EH(copy.deepcopy(P.model.head).float().cpu())
    if path and os.path.exists(path): h.load_state_dict(torch.load(path))
    return h.cuda().eval()


class XReq:
    """one request: ids (state + question bundle), readout rows per question; fn(mode) runs full / pre16 / post16"""
    def __init__(self, m, Ts, qs, eh, ids_state=None, dev='cuda'):
        self.m = m; self.Ts = Ts; Mq = len(qs); self.eh = eh
        Kmax = max(len(q['opt']) for q in qs)
        self.temps = torch.tensor([q['temp'] for q in qs], device=dev)
        self.nsl = torch.tensor([q['n_slots'] for q in qs], device=dev)
        self.lay = Q.Lay('single', Ts + len(qs[0]['ids'])) if Mq == 1 else Q.Lay('packed', Ts, qlens=[len(q['ids']) for q in qs])
        tail = [t for q in qs for t in q['ids']]
        pr = []; orow = []; s = Ts
        for q in qs:
            pr.append(s + len(q['ids']) - 1)
            orow.append([s + j for j in q['opt']] + [s + q['opt'][0]] * (Kmax - len(q['opt']))); s += len(q['ids'])
        self.pool_rows = torch.tensor(pr, device=dev); self.opt_rows = torch.tensor(orow, device=dev)
        st_ids = torch.randint(1000, 100000, (Ts,), device=dev) if ids_state is None else torch.tensor(ids_state, device=dev)
        self.ids = torch.cat([st_ids, torch.tensor(tail, device=dev)])
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]
        self.xbuf = torch.empty(self.lay.T, 2048, device=dev, dtype=torch.bfloat16)

    def _readout(self, h_dec, h_opt, head):
        lg = head(h_dec, h_opt) / self.temps[:, None]
        return torch.softmax(lg.masked_fill(~self.kmask, float('-inf')), -1)

    @torch.no_grad()
    def run(self, mode):
        m = self.m
        if hasattr(m, 'q0'): m.q0 = self.Ts
        if mode == 'full':
            self.xbuf.copy_(F.embedding(self.ids, m.embed))
            hn = m.forward(self.ids, self.lay, None, x0=self.xbuf)
            dec = m.unrot(hn[self.pool_rows]).float()
            opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
            return self._readout(dec, opts, m.head)
        if mode == 'pre16':
            self.xbuf.copy_(F.embedding(self.ids, m.embed))
            m.forward(self.ids, self.lay, None, x0=self.xbuf, i0=0, i1=L)
            dec = m.unrot_raw(self.xbuf[self.pool_rows])
            opts = m.unrot_raw(self.xbuf[self.opt_rows.reshape(-1)]).reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
            return self._readout(dec, opts, self.eh)
        if mode == 'post16':
            hn = m.forward(self.ids, self.lay, None, x0=self.xbuf, i0=L, i1=24)
            dec = m.unrot(hn[self.pool_rows]).float()
            opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
            return self._readout(dec, opts, m.head)


def wall(req, g, out, reps=20, warm=3, prefill=None):
    Ts = req.Ts
    pool = [torch.randint(1000, 100000, (Ts,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pool:
        if prefill is not None: prefill()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids[:Ts].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warm:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))


def grid(names, Tss, bnames):
    out = os.path.expanduser('~/work/q2/res_xgrid.jsonl')
    done = set(json.loads(l)['key'] for l in open(out)) if os.path.exists(out) else set()
    bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
    from kitrun import load_P
    from lean2 import Lean2
    P = load_P(); ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval(); wc = load_codes(CODES)
    for name in names:
        m = build(name, ln, head, wc)
        eh = exit_head(P, f'{M}/preds/exithead_{name}_L16.pt')
        for T in Tss:
            for bn in bnames:
                for mode in ('full', 'pre16', 'post16'):
                    key = f'{name}|{T}|{bn}|{mode}'
                    if key in done: continue
                    req = XReq(m, T, bundles[bn], eh)
                    if mode == 'post16':
                        with torch.no_grad(): req.run('pre16')        # a valid residual in xbuf
                    g, o = capture(lambda: req.run(mode))
                    w = wall(req, g, o)
                    rec = dict(key=key, name=name, T=T, bundle=bn, mode=mode, rows=req.lay.T, wall=w, kern=kprof(g))
                    print(json.dumps(rec), flush=True)
                    with open(out, 'a') as f: f.write(json.dumps(rec) + '\n')
                    del g, o, req; torch.cuda.empty_cache()
        del m; torch.cuda.empty_cache()


def real(tau):
    import evalkit as EK
    from kitrun import load_P, prep_question
    from lean2 import Lean2
    items = [json.loads(l) for l in open(os.path.expanduser('~/work/q2/res_dcasc_16_j15.jsonl'))]
    byid = {}
    for s in ('REAL-agree', 'LONG', 'JB-all'):
        for it in EK.load_suite(s): byid[(s, it['id'])] = it
    P = load_P(); ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval(); wc = load_codes(CODES)
    mk = build('k64', ln, head, wc); mb = build('b8', ln, head, wc)
    eh = exit_head(P, f'{M}/preds/exithead_k64_L16.pt')
    outp = os.path.expanduser('~/work/q2/res_xreal.jsonl')
    done = set((json.loads(l)['id'], json.loads(l)['q']) for l in open(outp)) if os.path.exists(outp) else set()
    for rj in items:
        if (rj['id'], rj['q']) in done: continue
        it = byid[(rj['suite'], rj['id'])]; pr = prep_question(P, it, rj['q'])
        qs = [dict(ids=pr['q'], opt=pr['opt'], temp=P.temp_for(pr['rq'].kind), n_slots=pr['rq'].n_slots)]
        Ts = len(pr['s'])
        rk = XReq(mk, Ts, qs, eh, ids_state=pr['s']); rb = XReq(mb, Ts, qs, eh, ids_state=pr['s'])
        with torch.inference_mode():
            ga, oa = capture(lambda: rk.run('pre16')); gb, ob = capture(lambda: rk.run('post16')); gf, of = capture(lambda: rk.run('full'))
            g8, o8 = capture(lambda: rb.run('full'))
            g8a, o8a = capture(lambda: rb.run('pre16')); g8b, o8b = capture(lambda: rb.run('post16'))
            ha = torch.empty(oa.shape, dtype=oa.dtype).pin_memory(); hb_ = torch.empty(ob.shape, dtype=ob.dtype).pin_memory()
            n = pr['rq'].n_slots; tc = []; tf = []; t8 = []; ex = None
            for r in range(15):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ga.replay(); ha.copy_(oa, non_blocking=True); torch.cuda.synchronize()
                p = sorted(ha[0, :n].tolist(), reverse=True); mg = (p[0] - (p[1] if len(p) > 1 else 0)) / max(sum(p), 1e-9)
                if mg >= tau: ex = 16
                else:
                    gb.replay(); hb_.copy_(ob, non_blocking=True); torch.cuda.synchronize(); ex = 24
                tc.append((time.perf_counter() - t0) * 1000)
            t8x = []; h8a = torch.empty(o8a.shape, dtype=o8a.dtype).pin_memory(); h8b = torch.empty(o8b.shape, dtype=o8b.dtype).pin_memory()
            for r in range(10):
                torch.cuda.synchronize(); t0 = time.perf_counter(); gf.replay(); torch.cuda.synchronize(); tf.append((time.perf_counter() - t0) * 1000)
                torch.cuda.synchronize(); t0 = time.perf_counter(); g8.replay(); torch.cuda.synchronize(); t8.append((time.perf_counter() - t0) * 1000)
                # J15's b8 + exit-16 on this box: the same two-graph path, with J15's recorded exit decision for this request
                torch.cuda.synchronize(); t0 = time.perf_counter()
                g8a.replay(); h8a.copy_(o8a, non_blocking=True); torch.cuda.synchronize()
                _ = sorted(h8a[0, :n].tolist(), reverse=True)
                if rj['exit'] != 16:
                    g8b.replay(); h8b.copy_(o8b, non_blocking=True); torch.cuda.synchronize()
                t8x.append((time.perf_counter() - t0) * 1000)
        rec = dict(suite=rj['suite'], id=rj['id'], q=rj['q'], T=Ts + len(pr['q']), exit=ex, margin=round(mg, 4), casc=round(st.median(tc[3:]), 3),
                   k64=round(st.median(tf[2:]), 3), b8=round(st.median(t8[2:]), 3), b8x=round(st.median(t8x[2:]), 3),
                   j15_b8=rj['b8'], j15_casc=rj['casc'], j15_exit=rj['exit'])
        print(json.dumps(rec), flush=True)
        with open(outp, 'a') as f: f.write(json.dumps(rec) + '\n')
        del ga, gb, gf, g8, g8a, g8b, oa, ob, of, o8, o8a, o8b, rk, rb; torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == 'grid':
        grid(sys.argv[2].split(','), [int(x) for x in sys.argv[3].split(',')], sys.argv[4].split(','))
    else:
        real(float(sys.argv[2]))
