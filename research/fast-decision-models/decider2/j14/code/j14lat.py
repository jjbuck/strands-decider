"""J14 latency / runtime fidelity on the A10G.
python j14lat.py fid CACHE.pt [prec]                    runtime (CUDA-graph-free) vs j14lib probabilities on real REAL-agree states
python j14lat.py time CACHE.pt PRECS TS SETS MODES TAG  latency matrix; PRECS bf16,w8a8b8 ; MODES plain,replay,affine
Discipline: exact shapes, CUDA graph per shape, fresh random state ids each rep (pinned H2D), 3 warm + REPS timed, median / p95."""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2/code'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g')]
import torch
import j14rt as RT
import qrt as Q
from lean2 import Lean2

REPS = int(os.environ.get('REPS', '20'))
SPEC = {'bf16': 'bf16', 'w8a8b8': 'w8a8:~/work/h1/precmap_w8a8_b8.json:w8a8'}


def load_model():
    from strands_decider.modeling import StrandsDeciderModel
    from common import CKPT
    m = StrandsDeciderModel.load(CKPT)
    t = m.torso.merge_and_unload().eval().cuda().to(torch.bfloat16)
    head = m.head.cuda().float().eval()
    return t, head


class PlainReq:
    """hobson's own layout: [state][question] (1 q) or shared state + one branch per question (packed)"""
    def __init__(self, m, Ts, qs, dev='cuda'):
        self.m = m; self.Ts = Ts; Mq = len(qs); Kmax = max(len(q['opt']) for q in qs)
        self.temps = torch.tensor([q['temp'] for q in qs], device=dev); self.nsl = torch.tensor([q['n_slots'] for q in qs], device=dev)
        self.lay = Q.Lay('single', Ts + len(qs[0]['q'])) if Mq == 1 else Q.Lay('packed', Ts, qlens=[len(q['q']) for q in qs])
        tail = [t for q in qs for t in q['q']]
        pr = []; orow = []; s = Ts
        for q in qs:
            pr.append(s + len(q['q']) - 1); orow.append([s + j for j in q['opt']] + [s + q['opt'][0]] * (Kmax - len(q['opt']))); s += len(q['q'])
        self.pool_rows = torch.tensor(pr, device=dev); self.opt_rows = torch.tensor(orow, device=dev)
        self.ids = torch.cat([torch.randint(1000, 100000, (Ts,), device=dev), torch.tensor(tail, device=dev)])
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]

    def run(self):
        m = self.m
        hn = m.forward(self.ids, self.lay, None)
        dec = m.unrot(hn[self.pool_rows]).float()
        opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
        lg = m.head(dec, opts) / self.temps[:, None]
        return torch.softmax(lg.masked_fill(~self.kmask, float('-inf')), -1)


def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl: return 'attn'
    if 'cutlass' in nl or 'gemm' in nl or 'cublas' in nl or 'xmma' in nl or 'ampere_' in nl or 'sm80' in nl or '_sk' in nl: return 'gemm'
    if 'chunk' in nl or 'recompute_w_u' in nl or 'kkt' in nl or 'solve' in nl or 'fwd_kernel_h' in nl or 'fused_recurrent' in nl: return 'gdn'
    return 'other'


def kprof(g):
    from torch.profiler import profile, ProfilerActivity
    import collections
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); n = 0
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        agg[kcat(e.name)] += (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000; n += 1
    r = {k: round(v, 3) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 3)
    return r


def to_dev(qs):
    out = []
    for x in qs:
        comp = []
        for c in x['comp']:
            d = {}
            for k, v in c.items():
                d[k] = {kk: vv.cuda() for kk, vv in v.items()} if isinstance(v, dict) else v.cuda()
            comp.append(d)
        out.append(dict(x, comp=comp))
    return out


if __name__ == '__main__':
    mode = sys.argv[1]; cache = torch.load(os.path.expanduser(sys.argv[2]), map_location='cpu')
    torso, head = load_model()
    import j5rt
    ln = j5rt.slim(Lean2(torso, fuse='fold')); del torso; torch.cuda.empty_cache()
    if mode == 'fid':
        prec = sys.argv[3] if len(sys.argv) > 3 else 'bf16'
        m = RT.make(ln, head, SPEC[prec])
        rows = []
        for it in cache['fid']:
            qs = to_dev([it['qc']])
            for md in ('replay', 'affine'):
                req = RT.J14Req(m, len(it['s']), qs, mode=md)
                req.ids[:len(it['s'])] = torch.tensor(it['s'], device='cuda')
                p = req.run()[0, :it['qc']['n_slots']].float().cpu()
                pl = it['p_lib']
                rows.append(dict(iid=it['iid'], q=it['qn'], mode=md, T=len(it['s']), arg=int(p.argmax()) == int(pl.argmax()), dmax=float((p - pl).abs().max())))
                print(json.dumps(rows[-1]), flush=True)
        for md in ('replay', 'affine'):
            r = [x for x in rows if x['mode'] == md]
            print(prec, md, 'argmax agree', sum(x['arg'] for x in r), '/', len(r), 'max|dp| median', sorted(x['dmax'] for x in r)[len(r) // 2], 'max', max(x['dmax'] for x in r))
        json.dump(rows, open(os.path.expanduser(f'~/work/j14/fid_{prec}.json'), 'w'))
        sys.exit(0)
    precs = sys.argv[3].split(','); Ts = [int(x) for x in sys.argv[4].split(',')]; sets = sys.argv[5].split(','); modes = sys.argv[6].split(',')
    tag = sys.argv[7] if len(sys.argv) > 7 else ''
    outp = os.path.expanduser(f'~/work/j14/lat{tag}.jsonl'); done = set()
    if os.path.exists(outp):
        for l in open(outp): done.add(json.loads(l)['key'])
    dsets = {k: to_dev(cache['sets'][k]) for k in sets}
    for prec in precs:
        m = RT.make(ln, head, SPEC[prec])
        for T in Ts:
            for sn in sets:
                for md in modes:
                    key = f'{prec}|{T}|{sn}|{md}'
                    if key in done: continue
                    qs = dsets[sn]
                    req = PlainReq(m, T, qs) if md == 'plain' else RT.J14Req(m, T, qs, mode=md)
                    g, out = RT.capture(req.run)
                    w = RT.wall(req, g, out, reps=REPS)
                    kp = kprof(g)
                    rec = dict(kern=kp, key=key, prec=prec, T=T, set=sn, mode=md, nq=len(qs), rows=req.lay.T, qtok=sum(len(x['q']) for x in qs),
                               live=sum(len(x['lp']) for x in qs), wall=w, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                    print(json.dumps(rec), flush=True)
                    with open(outp, 'a') as f: f.write(json.dumps(rec) + '\n')
                    del g, out, req; torch.cuda.empty_cache()
        del m; torch.cuda.empty_cache()
