"""Step 2: oracle routing test, untrained.  Per question: one dense forward+backward gives the oracle score of every state token
for every (k, r): s_t = d margin / d alpha_t, the first-order change in hobson's decision log-odds (argmax vs rest) when token t takes
the thin path in all layers >= k.  Chunk score = |sum over its 32 tokens|.  Top-f chunks (+ 4 sinks + all question rows) run full
width; the rest take the thin path in layers >= k.  Baselines: random chunks at the same f, tail (most recent) chunks, all-thin.
python route.py LIST OUT --mode out --cfgs A [--shard i/n]"""
import os, sys, json, time, math, random, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tw import TW, SINK, chunks_of, select_chunks
import evalkit as EK

PRESETS = {
    'B': (['thin@4r128', 'thin@4r256', 'thin@4r512', 'thin@8r256', 'thin@8r512']
          + ['orc30@4r128', 'orc50@4r128', 'orc10@4r256', 'orc30@4r256', 'orc50@4r256', 'orc30@4r512', 'orc50@4r512', 'orc30@8r128', 'orc20@8r256', 'orc10@8r512']
          + ['rnd30@4r256', 'rnd50@4r256', 'rnd30@4r512', 'tail30@4r256', 'oab30@4r256']),
    'A': (['thin@%dr%d' % (k, r) for k in (4, 8) for r in (128, 256, 512)]
          + ['orc%d@4r128' % f for f in (30, 50)] + ['orc%d@4r256' % f for f in (10, 20, 30, 50)] + ['orc%d@4r512' % f for f in (10, 20, 30, 50)]
          + ['orc%d@8r128' % f for f in (20, 30)] + ['orc%d@8r256' % f for f in (10, 20, 30)] + ['orc%d@8r512' % f for f in (10, 20)]
          + ['rnd%d@4r256' % f for f in (10, 20, 30, 50)] + ['rnd30@4r512', 'rnd20@8r256']
          + ['tail%d@4r256' % f for f in (10, 30, 50)] + ['tail30@4r512'] + ['oab30@4r256', 'oab30@4r512']),
}


def parse(c):
    """'orc20@8r256' or a depth schedule 'orc20@8r512+13r64' (rank 512 in layers 8-12, 64 from 13). Returns (router, f, k, r, segs)"""
    if c == 'dense': return None
    a, rest = c.split('@')
    segs = [tuple(int(v) for v in sg.split('r')) for sg in rest.split('+')]
    k, r = segs[0]
    if a == 'thin': return ('thin', 0.0, k, r, segs)
    for rt in ('orc', 'rnd', 'tail', 'oab', 'or2'):
        if a.startswith(rt): return (rt, int(a[len(rt):]) / 100.0, k, r, segs)
    raise ValueError(c)


def seg_rank(segs, l):
    rr = None
    for st, r in segs:
        if l >= st: rr = r
    return rr


def seg_score(sc, segs, NL=24):
    """oracle token score for a schedule: sum over segments of s(start_i, r_i) - s(start_{i+1}, r_i)"""
    tot = 0
    for i, (st, r) in enumerate(segs):
        tot = tot + sc[(st, r)]
        if i + 1 < len(segs): tot = tot - sc[(segs[i + 1][0], r)]
    return tot


def seed_of(*a):
    return int(hashlib.sha1('|'.join(map(str, a)).encode()).hexdigest()[:8], 16)


def g3(v):
    return [float('%.3g' % x) for x in v]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('lst'); ap.add_argument('out'); ap.add_argument('--mode', default='out')
    ap.add_argument('--cfgs', default='A'); ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--heal', default='')
    a = ap.parse_args()
    cfgs = PRESETS[a.cfgs] if a.cfgs in PRESETS else a.cfgs.split(',')
    P = {c: parse(c) for c in cfgs}
    ranks = sorted({r for p in P.values() if p for _, r in p[4]})
    ks = sorted({st for p in P.values() if p for st, _ in p[4]})
    L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lists.json')))[a.lst]
    si, sn = map(int, a.shard.split('/')); L = L[si::sn]
    if a.limit: L = L[:a.limit]
    done = set()
    if os.path.exists(a.out):
        good = []
        for l in open(a.out):
            try: j = json.loads(l); done.add((j['id'], j['q'])); good.append(l)
            except Exception: pass
        open(a.out, 'w').writelines(good)
    tw = TW(bases=os.path.expanduser('~/work/j13/bases.pt'))
    for m in [x for x in ('in', 'out') if x != a.mode]:
        tw.A.pop(m, None); tw.B.pop(m, None)
    if a.heal:   # healed thin factors overwrite the first r basis columns (all configs must then use that r)
        hz = torch.load(a.heal, map_location='cpu'); hr = hz['r']
        for (l, g), (A_, B_) in hz['fac'].items():
            tw.A[a.mode][l][g][:, :hr] = A_.to(tw.dev); tw.B[a.mode][l][g][:hr] = B_.to(tw.dev)
        assert all(p[3] == hr and p[2] >= hz['k'] for p in P.values() if p), 'healed run: every config must use the healed r and k >= heal k'
    torch.cuda.empty_cache()
    items = {}
    for s in {x[0] for x in L}:
        for it in EK.load_suite(s): items[it['id']] = it
    t0 = time.time(); ntok = 0
    with open(a.out, 'a') as fo:
        for n, (suite, iid, qn) in enumerate(L):
            if (iid, qn) in done: continue
            t1 = time.time()
            pr = tw.prep(items[iid], qn); q0 = pr['q0']; T = len(pr['ids'])
            if all(p is None or p[0] in ('thin', 'rnd', 'tail') for p in P.values()):   # no oracle needed: plain dense pass
                with torch.no_grad():
                    x0, cos, sin = tw.embed_rope(pr['ids'])
                    xf, hk = tw.run(x0, cos, sin, keep=set(ks))
                    lp = tw.head(xf, pr)
                sc = {}
            else:
                lp, sc, hk, (cos, sin) = tw.oracle_pass(pr, a.mode, ranks=ranks, k0=min(ks), ks=ks)
            rec = {'suite': suite, 'id': iid, 'q': qn, 'n_s': q0, 'T': T, 'dense': tw.probdict(pr, lp), 'cfg': {}, 'fl': {}, 'nfull': {}}
            ch = chunks_of(q0, 32)
            for c, p in P.items():   # schedule scores
                if p and len(p[4]) > 1: sc[tuple(p[4])] = seg_score(sc, p[4])
            csig = {kr: [float(v[s:e].sum()) for s, e in ch] for kr, v in sc.items()}
            cabs = {kr: [float(v[s:e].abs().sum()) for s, e in ch] for kr, v in sc.items()}
            rec['chunk'] = {str(kr): g3(v) for kr, v in csig.items()}
            rec['chunk_abs'] = {str(kr): g3(v) for kr, v in cabs.items()}
            with torch.no_grad():
                for c, p in P.items():
                    if p is None:
                        rec['cfg'][c] = rec['dense']; rec['fl'][c] = 1.0; continue
                    rt, f, k, r, segs = p
                    key = (k, r) if len(segs) == 1 else tuple(segs)
                    if rt == 'thin': sel = []
                    elif rt == 'orc': sel = select_chunks([abs(x) for x in csig[key]], f)
                    elif rt == 'oab': sel = select_chunks(cabs[key], f)
                    elif rt == 'or2':   # one refinement step at the routed point of 'orc'
                        sel0 = select_chunks([abs(x) for x in csig[(k, r)]], f)
                        full0 = set(range(min(SINK, q0)))
                        for j in sel0: full0.update(range(*ch[j]))
                        thin0 = [t for t in range(q0) if t not in full0]
                        g2, _ = tw.refine_pass(pr, a.mode, k, r, thin0, hk, cos, sin, lp)
                        sel = select_chunks([float(g2[s:e].sum()) for s, e in ch], f)
                    elif rt == 'rnd': sel = select_chunks([0.0] * len(ch), f, rng=random.Random(seed_of(iid, qn, c)))
                    elif rt == 'tail':
                        m = int(math.floor(f * len(ch) + 0.5)); sel = list(range(len(ch) - m, len(ch))) if m > 0 else []
                    full = set(range(min(SINK, q0)))
                    for j in sel: full.update(range(*ch[j]))
                    thin = [t for t in range(q0) if t not in full]
                    pls = {rr: tw.thin_plan(a.mode, rr, thin, T) for rr in {r_ for _, r_ in segs}}
                    xo, _ = tw.run(hk[k], cos, sin, k, plans={l: pls[seg_rank(segs, l)] for l in range(k, tw.NL)})
                    rec['cfg'][c] = tw.probdict(pr, tw.head(xo, pr))
                    rec['fl'][c] = round(tw.flops_ratio(T, len(thin), k, r, segs), 4)
                    rec['nfull'][c] = q0 - len(thin)
            del hk, sc
            rec['ms'] = round((time.time() - t1) * 1000)
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            ntok += T
            if n % 10 == 0:
                print(n, len(L), suite, T, '%.0fs' % (time.time() - t0), '%.0f tok/s' % (ntok / (time.time() - t0)),
                      'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
