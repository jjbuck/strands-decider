"""Q4 B13 end to end (A10G, exclusive GPU): W8A8-b8 (H2 QRT, GPTQ codes) with and without the exact eliminations.
python bench_rt.py TS BUNDLES VARIANTS [reps]     e.g. 64,256,1000,4000 1q,15q base,l0,kv,l0+kv,read,l0+read
H2's bench machinery: CUDA graph of [embed .. 24 layers .. final norm .. pointer head .. softmax]; wall per request = H2D of fresh random state
ids (pinned) + graph replay + D2H of the probabilities; 3 warm-up + reps timed (default 30), median and p95; one profiled replay gives the
GEMM / non-GEMM split. hobson layout: 1 question = one sequence, 15 questions = shared state + one branch per question (packed).
-> ~/work/q4/res_b13_e2e.jsonl"""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import q4rt as R
import bench as HB
from prof_d1 import capture

OUT = os.path.expanduser(os.environ.get('Q4OUT', '~/work/q4/res_b13_e2e.jsonl'))
VAR = {'base': (False, 'none'), 'l0': (True, 'none'), 'kv': (False, 'kv'), 'l0+kv': (True, 'kv'), 'read': (False, 'read'), 'l0+read': (True, 'read')}
SPL = {'sp12': range(12, 23), 'spall': range(0, 23), 'ar12': range(12, 23), 'arall': range(0, 23)}   # B11: sp* = state rows through 2:4 int8
# weights in these layers (question rows dense, two launches); ar* = ALL rows through the 2:4 weights (one launch, one weight copy)
def parse(v):
    base = v.replace('_inv', ''); sp = ()
    for k_, L in SPL.items():
        if base.startswith(k_):
            sp = L; base = base[len(k_):].lstrip('+') or 'base'
    return VAR[base], sp


def main():
    Ts = [int(x) for x in sys.argv[1].split(',')]; bnames = sys.argv[2].split(','); variants = sys.argv[3].split(',')
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)['key'])
    bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
    P, m, head = R.build()
    if any(parse(v)[0][0] for v in variants): m.build_l0()
    if any(parse(v)[1] for v in variants): print('sparse GEMMs built', m.build_sparse(range(0, 23)), flush=True)
    for T in Ts:
        for bn in bnames:
            for v in variants:
                key = f'{v}|{T}|{bn}'
                if key in done: continue
                (m.l0, m.l23), spl = parse(v); m.set_inv23(v.endswith('_inv')); m.set_sparse_layers(spl); m.sp_all_rows = v.startswith('ar')
                req = HB.Req(m, T, bundles[bn], 'plain')
                m.q0 = T
                rows = req.pool_rows.tolist() + req.opt_rows.reshape(-1).tolist()
                m.set_read(req.lay, rows)
                g, out = capture(req.run)
                w = HB.wall(req, g, out, reps=reps)
                kp = HB.kprof(g)
                rec = dict(key=key, variant=v, T=T, bundle=bn, rows=req.lay.T, nread=int(m.read_rows.shape[0]), wall=w, kern=kp,
                           mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                print(json.dumps(rec), flush=True)
                with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
                del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m.l0, m.l23 = False, 'none'; m.set_sparse_layers(())


if __name__ == '__main__':
    main()
