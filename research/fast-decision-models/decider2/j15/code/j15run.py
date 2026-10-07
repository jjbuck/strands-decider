"""J15 runner: questions through the deployed H2 kernels (hobson layout, one question per sequence, exact reference token ids).
Writes probs/logits jsonl (H2 evalrun format, resumable) and optionally residual taps for depth-exit heads / draft-uncertainty features.

python j15run.py OUT.jsonl --prec PREC [--codes w8=..,w4=..] [--suites JB-all,REAL-agree,...|DEV|EXIT] [--taps 4,8,12,24] [--limit n]
taps: layer boundaries L (residual after L layers; 24 = final residual). For each tapped question, saves the rows the pointer head reads
(answer row = last row, then option rows) of the residual, UNROTATED, bf16 -> OUT.taps.pt (dict (id,q) -> {L: [1+nopt, 2048]}), sharded."""
import sys, os, json, time, argparse, torch, triton
sys.path[:0] = [os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch.nn.functional as F
import qrt as Q
from qrt import K, QG


class QRTJ(Q.QRT):
    """H2 QRT + residual taps + layer-range execution (start from a given residual at layer i0, stop after layer i1)."""
    taps = (); taprows = None; tapped = None

    def _tap(self, L, x):
        if L in self.taps and self.taprows is not None:
            self.tapped[L] = x[self.taprows].clone()

    def _fwd_fold(self, ids, lay, cache, cos, sin, x0=None, i0=0, i1=None):
        import lean2 as L2
        T = lay.T; i1 = self.nL if i1 is None else i1
        x = F.embedding(ids, self.ln2.embed).contiguous() if x0 is None else x0
        ss = x.float().pow(2).sum(-1)
        for i in range(i0, i1):
            d = self.L[i]
            proj = L2.tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=2048)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, None, None, lay, cache, False)
                y = torch.empty(T, 2048, device=self.dev, dtype=torch.bfloat16)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], proj, proj, d['gn_w'], proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), self.eps,
                                                 DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, None, None, lay, cache, False, cos, sin)
                y = torch.empty(T, 2048, device=self.dev, dtype=torch.bfloat16)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, proj, proj, proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), o.stride(0), o.stride(1),
                                                 DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            L2.tgemm(y, d['Wo'], epi=3, res=x, ssout=ss)
            m = L2.tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=2048)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            L2.tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
            self._tap(i + 1, x)
        self.xres = x
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))

    def _fwd_q(self, ids, lay, cache, cos, sin, x0=None, i0=0, i1=None):
        T = lay.T; eps = self.eps; i1 = self.nL if i1 is None else i1
        x = F.embedding(ids, self.embed).contiguous() if x0 is None else x0
        nxt = self._next_in(i0, 'Win', T)
        K.addq(x, None, None, None, nxt['q'], nxt['s'], nxt['hb'], eps, dq=False, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
        cur = nxt
        for i in range(i0, i1):
            d = self.L[i]
            e = self.qw[i]['Win']
            proj, ra, cs, dq = self._run(cur, e)
            nxt = self._next_in(i, 'Wo', T)
            R2 = self.r2_for(i)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, ra, cs, lay, cache, dq)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], ra if dq else proj, cs if dq else proj, d['gn_w'], R2.sign, self.M['H32'], self.M['H64'], self.M['H128'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), eps, DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, ra if dq else proj, cs if dq else proj, R2.sign, self.M['H32'], self.M['H64'], self.M['H16'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), o.stride(0), o.stride(1), DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wo'])
            nxt = self._next_in(i, 'Wgu', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
            eg = self.qw[i]['Wgu']; nxt = self._next_in(i, 'Wd', T)
            if eg.get('evt'):
                GU = QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and T >= 2500) else 3)); dq = False; ra = cs = None; mi = True
            else:
                GU, ra, cs, dq = self._run(cur, eg); mi = False
            K._swiglu_k[(triton.cdiv(T, self.rows),)](GU, ra if dq else GU, cs if dq else GU, self.R4.sign, self.M['P12T'], self.M['H16'], self.M['H32'], nxt['q'], nxt['s'], nxt['hb'], T,
                              DQ=dq, OUTQ=nxt['outq'], QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, MIN=mi, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wd'])
            nxt = self._next_in(i + 1, 'Win', T) if i + 1 < self.nL else self._next_in(None, 'final', T)
            if i + 1 == i1 and i1 < self.nL:      # stopping early: residual add only (bf16 normed copy is cheap; codes not needed)
                nxt = self._next_in(None, 'final', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
            self._tap(i + 1, x)
        self.xres = x
        return cur['hb']

    def forward(self, ids, lay, cache=None, keep_hn=False, x0=None, i0=0, i1=None):
        T = lay.T
        self._ztail = torch.zeros(1, 6144, device=self.dev)
        fr = lay.pos[:, None] * self.ln2.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        if self.fold: return self._fwd_fold(ids, lay, cache, cos, sin, x0, i0, i1)
        return self._fwd_q(ids, lay, cache, cos, sin, x0, i0, i1)

    def unrot_raw(self, x):
        """residual rows -> unrotated basis (fp32)"""
        return x.float() if self.fold else self.R1.inv(x.float())


def load_codes(spec):
    if not spec: return None
    if '=' in spec:
        return {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in spec.split(',')}
    return torch.load(os.path.expanduser(spec))


def build(prec, codes=''):
    from kitrun import load_P
    from lean2 import Lean2
    P = load_P()
    fold = prec == 'bf16'
    ln = Lean2(P.tm, fuse='fold' if fold else ''); head = P.model.head.float().eval()
    m = QRTJ(ln, head=head, prec=prec, wcodes=load_codes(codes)); m.tune = False
    if not fold:   # free the bf16 fold copies the quantized path does not use (H6 slim)
        import gc
        for d in ln.layers:
            for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
            for k_ in ('Win', 'Wo', 'Wgu', 'Wd'):
                if all(m.pm[(i, k_)] != 'bf16' for i in range(m.nL)): pass
        gc.collect(); torch.cuda.empty_cache()
    return P, m, head


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--prec', default='bf16'); ap.add_argument('--codes', default='')
    ap.add_argument('--suites', default='all'); ap.add_argument('--taps', default=''); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    import evalkit as EK
    from kitrun import prep_question, probdict
    P, m, head = build(a.prec, a.codes)
    print('built', a.prec, 'fold' if m.fold else 'rot', 'mem', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    suites = None if a.suites == 'all' else a.suites.split(',')
    items = list(EK.all_question_items(suites))
    if a.limit: items = items[:a.limit]
    byid = {}
    for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
        for it in EK.load_suite(s): byid[it['id']] = it
    out = os.path.expanduser(a.out)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r['id'], r['q']))
    taps = tuple(int(x) for x in a.taps.split(',')) if a.taps else ()
    m.taps = taps
    shard = {}; nshard = len([f for f in os.listdir(os.path.dirname(out) or '.') if f.startswith(os.path.basename(out) + '.taps.')])
    print('questions', len(items), 'done', len(done), 'taps', taps, flush=True)
    t0 = time.time(); n = 0; buf = []

    def flush():
        global shard, nshard
        with open(out, 'a') as f:
            for l in buf: f.write(l + '\n')
        if taps and shard:
            torch.save(shard, f'{out}.taps.{nshard:03d}.pt'); nshard += 1; shard = {}
        buf.clear()

    with torch.inference_mode():
        for suite, iid, qn, st, spec in items:
            if (iid, qn) in done: continue
            pr = prep_question(P, byid[iid], qn)
            ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
            rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
            m.taprows = rows; m.tapped = {}
            hn = m.forward(ids, Q.Lay('single', T))
            h = m.unrot(hn[rows]).float()
            lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
            pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
            if taps:
                rec = {L: m.unrot_raw(v).to(torch.bfloat16).cpu() for L, v in m.tapped.items()}
                rec.update(kind=pr['rq'].kind, n=pr['rq'].n_slots, T=T, temp=P.temp_for(pr['rq'].kind), labels=list(pd))
                shard[(iid, qn)] = rec
            buf.append(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, kind=pr['rq'].kind, probs=pd, logits=[round(x, 5) for x in lg.tolist()]))); n += 1
            if n % 200 == 0:
                flush(); print(a.prec, n, f'{time.time() - t0:.0f}s', flush=True)
    flush()
    print('done', a.prec, n, f'{time.time() - t0:.0f}s', flush=True)
