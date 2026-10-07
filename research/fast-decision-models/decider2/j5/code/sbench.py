"""J5 short-length latency matrix on the A10G (exclusive GPU), h2's runtime/bench machinery with real short question bundles.
python sbench.py PRECS TS BUNDLES MODES [tag]
  PRECS: bf16 | w8a8 | w4a8 | w4a4 | map:<json>:<default> (';'-separated when maps are used), or j5 variants handled by j5rt (see RT env)
  TS: state tokens (exact), BUNDLES from ~/work/j5/bundles.pt (jb1, jb4, bk1, bk4), MODES plain (hobson layout, exact) | schema
Per config: CUDA graph of [embed .. 24 layers .. final norm .. pointer head .. softmax]; wall = H2D fresh ids + replay + D2H probs,
3 warm + N timed reps (default 30), median / p95; one profiled replay gives kernel categories, kernel count, busy and span (gaps)."""
import sys, os, json, time, statistics as st, collections, torch
sys.path[:0] = [os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import capture
from lean2 import Lean2
import qrt as Q
import bench as HB
from torch.profiler import profile, ProfilerActivity

REPS = int(os.environ.get('REPS', '30'))


def kprof2(g):
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); cnt = collections.defaultdict(int); byname = collections.defaultdict(float); ks = []
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        c = HB.kcat(e.name)
        if 'gdn' in e.name or 'fused_recurrent' in e.name or 'short_gdn' in e.name: c = 'gdn_core'
        if '_sk' in e.name and 'kernel' not in e.name.lower(): c = 'gemm'
        agg[c] += dt; cnt[c] += 1; byname[e.name[:60]] += dt
        if e.time_range.end > e.time_range.start: ks.append((e.time_range.start, e.time_range.end))
    ks.sort()
    r = {k: round(v, 3) for k, v in agg.items()}
    r['counts'] = dict(cnt); r['kernels'] = sum(cnt.values()); r['busy'] = round(sum(agg.values()), 3)
    r['span'] = round((ks[-1][1] - ks[0][0]) / 1000, 3) if ks else 0
    r['gaps'] = round(sum(max(0, ks[i + 1][0] - ks[i][1]) for i in range(len(ks) - 1)) / 1000, 3)
    r['top'] = [(k, round(v, 3)) for k, v in sorted(byname.items(), key=lambda kv: -kv[1])[:12]]
    r['nongemm'] = round(r['busy'] - r.get('gemm', 0.0), 3)
    return r


def make_model(ln, head, prec):
    if prec.startswith('j5:'):
        import j5rt
        return j5rt.make(ln, head, prec[3:])
    return Q.QRT(ln, head=head, prec=prec)


if __name__ == '__main__':
    precs = sys.argv[1].split(';') if ';' in sys.argv[1] else sys.argv[1].split(',')
    Tss = [int(x) for x in sys.argv[2].split(',')]; bnames = sys.argv[3].split(','); modes = sys.argv[4].split(',')
    tag = sys.argv[5] if len(sys.argv) > 5 else ''
    out_path = os.path.expanduser(f'~/work/j5/res_sbench{tag}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path): done.add(json.loads(l)['key'])
    bundles = torch.load(os.path.expanduser('~/work/j5/bundles.pt'))
    torso, head = HB.load_model()
    import j5rt; ln = j5rt.slim(Lean2(torso, fuse='fold')); del torso; torch.cuda.empty_cache()
    for prec in precs:
        pname = ('j5_' if prec.startswith('j5:') else '') + prec.replace('map:', '').split('/')[-1].replace('.json', '')
        if all(f'{pname}|{T}|{b}|{md}' in done for T in Tss for b in bnames for md in modes): continue
        m = make_model(ln, head, prec)
        for T in Tss:
            for bn in bnames:
                for md in modes:
                    key = f'{pname}|{T}|{bn}|{md}'
                    if key in done: continue
                    req = HB.Req(m, T, bundles[bn], md)
                    g, out = capture(req.run)
                    w = HB.wall(req, g, out, reps=REPS)
                    kp = kprof2(g)
                    rec = dict(key=key, prec=pname, T=T, bundle=bn, mode=md, rows=req.lay.T, P=req.P, nq=len(bundles[bn]), wall=w, kern=kp,
                               mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                    print(json.dumps({k: rec[k] for k in ('key', 'rows', 'wall')}), json.dumps({k: v for k, v in kp.items() if k != 'top'}), flush=True)
                    with open(out_path, 'a') as f: f.write(json.dumps(rec) + '\n')
                    del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        del m; torch.cuda.empty_cache()
