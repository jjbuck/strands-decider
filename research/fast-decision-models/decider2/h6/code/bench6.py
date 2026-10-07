"""H6 latency (A10G, exclusive GPU): H2's bench method (CUDA graph of the whole request incl. head + softmax; per rep: H2D of FRESH random state
ids from pinned memory + graph replay + D2H of the probabilities; 3 warm + REPS timed; median and p95) for the H6 layouts.
  schema     : compiled bundle (low-bit), state + slots low-bit (H2's schema)
  schemamix  : bundle compiled in bf16 once, state rows low-bit, answer-slot rows bf16 (row split r0 = T_state)
  schemapq   : bundle bf16, state + slots low-bit
  plain / plainqb : hobson layout (1 question), uniform / question rows bf16
Slots: H6 layout = '<answer>' = 3 tokens per question. Bundles = H2's real bundles (bundles.pt: 1q 125 tok, 4q 1613, 15q 3720), question
tokens minus the final 3 as the bundle, the final 3 as the slot.  Multi-q schema = ONE shared bundle, state once, 3 slot rows per question.
python bench6.py PREC TS BUNDLES LAYOUTS TAG [--codes ..] [--lrot ..]"""
import sys, os, json, time, statistics as st, collections, argparse, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import capture
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
from bench import load_model, wall, kprof
ap = argparse.ArgumentParser()
ap.add_argument('prec'); ap.add_argument('Ts'); ap.add_argument('bundles'); ap.add_argument('layouts'); ap.add_argument('tag')
ap.add_argument('--codes', default=''); ap.add_argument('--lrot', default=''); ap.add_argument('--reps', type=int, default=20)
a = ap.parse_args()


class Req6:
    def __init__(self, m, Ts, qs, mode, dev='cuda'):
        self.m = m; self.Ts = Ts; self.mode = mode; Mq = len(qs)
        Kmax = max(len(q['opt']) for q in qs)
        self.temps = torch.tensor([q['temp'] for q in qs], device=dev)
        self.nsl = torch.tensor([q['n_slots'] for q in qs], device=dev)
        self.r0 = None
        if mode.startswith('sets'):          # H7 slot sets: shared bundle, state once, per question [option-end tokens][<answer>] (masking not modelled)
            pre = [t for q in qs for t in q['ids'][:-3]]
            self.P = len(pre)
            tail = []; dec = []; orow = []
            for q in qs:
                st0 = Ts + len(tail); tail += [q['ids'][o] for o in q['opt']] + q['ids'][-3:]
                orow.append([st0 + j for j in range(len(q['opt']))] + [st0] * (Kmax - len(q['opt']))); dec.append(Ts + len(tail) - 1)
            self.cache = m.compile_prefix(torch.tensor(pre, device=dev), Ts + len(tail), r0=(0 if mode in ('setsmix', 'setspq') else None))
            self.lay = Lay('schema', Ts, nslots=len(tail), P=self.P)
            self.pool_rows = torch.tensor(dec, device=dev); self.opt_rows = torch.tensor(orow, device=dev); self.hP = None
            if mode == 'setsmix': self.r0 = Ts
        elif mode.startswith('schema'):
            pre = [t for q in qs for t in q['ids'][:-3]]
            self.P = len(pre)
            self.cache = m.compile_prefix(torch.tensor(pre, device=dev), Ts + 3 * Mq, r0=(0 if mode in ('schemamix', 'schemapq') else None))
            self.lay = Lay('schema', Ts, nslots=3 * Mq, P=self.P)
            tail = [t for q in qs for t in q['ids'][-3:]]
            self.pool_rows = torch.arange(Ts + 2, Ts + 3 * Mq, 3, device=dev)
            offs = []; o = 0
            for q in qs:
                offs.append([o + j for j in q['opt']] + [o + q['opt'][0]] * (Kmax - len(q['opt']))); o += len(q['ids']) - 3
            self.opt_rows = torch.tensor(offs, device=dev); self.hP = self.cache['hP']
            if mode == 'schemamix': self.r0 = Ts
        else:
            assert Mq == 1
            self.cache = None; self.P = 0
            self.lay = Lay('single', Ts + len(qs[0]['ids']))
            tail = list(qs[0]['ids'])
            self.pool_rows = torch.tensor([Ts + len(tail) - 1], device=dev)
            self.opt_rows = torch.tensor([[Ts + j for j in qs[0]['opt']]], device=dev)
            if mode == 'plainqb': self.r0 = Ts
        self.ids = torch.cat([torch.randint(1000, 100000, (Ts,), device=dev), torch.tensor(tail, device=dev)])
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]

    def run(self):
        m = self.m
        hn = m.forward(self.ids, self.lay, self.cache, r0=self.r0)
        dec = m.unrot(hn[self.pool_rows]).float()
        if self.mode.startswith('schema'): opts = self.hP[self.opt_rows].float()
        elif self.mode.startswith('sets'): opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
        else: opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
        lg = m.head(dec, opts) / self.temps[:, None]
        lg = lg.masked_fill(~self.kmask, float('-inf'))
        return torch.softmax(lg, -1)


out_path = os.path.expanduser(f'~/work/h6/res_bench_{a.tag}.jsonl')
done = set()
if os.path.exists(out_path):
    for l in open(out_path): done.add(json.loads(l)['key'])
bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
torso, head = load_model()
ln = Lean2(torso, fuse='fold'); del torso; torch.cuda.empty_cache()
if a.lrot:
    _sd = torch.load(os.path.expanduser(a.lrot), map_location='cuda')
    if '_head' in _sd:
        with torch.inference_mode(): head.load_state_dict(_sd['_head'])
lays = a.layouts.split(',')
m = Q6.QRT6(ln, head=head, prec=a.prec, wcodes=Q6.load_codes(a.codes), lrot=Q6.load_lrot(a.lrot), split=any(l in ('schemamix', 'schemapq', 'plainqb', 'setsmix', 'setspq') for l in lays),
            force_rot=bool(a.lrot))
if not m.fold: m.slim()
pname = a.tag
for T in [int(x) for x in a.Ts.split(',')]:
    for bn in a.bundles.split(','):
        for md in lays:
            key = f'{pname}|{T}|{bn}|{md}'
            if key in done or (md.startswith('plain') and bn != '1q'): continue
            req = Req6(m, T, bundles[bn], md)
            g, out = capture(req.run)
            w = wall(req, g, out, reps=a.reps)
            kp = kprof(g)
            rec = dict(key=key, prec=a.prec, T=T, bundle=bn, mode=md, rows=req.lay.T, P=req.P, nq=len(bundles[bn]), wall=w, kern=kp,
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            print(json.dumps(rec), flush=True)
            with open(out_path, 'a') as f: f.write(json.dumps(rec) + '\n')
            del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
