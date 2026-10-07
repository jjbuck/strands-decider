"""Batch-1 decision latency at a 1000-token state + 64-token question, bf16, CUDA graph, fresh inputs every rep, sync, median/p95.
Also counts FLOPs with torch.utils.flop_counter.  usage: python bench.py OUT.json SCALE:ARM [SCALE:ARM ...]"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from torch.utils.flop_counter import FlopCounterMode

N, Q = 1000, 64
REPS, WARM = 40, 8
out = {}
torch.backends.cuda.matmul.allow_tf32 = True
for spec in sys.argv[2:]:
    sc, arm = spec.split(':')
    cfg = C.Cfg(sc, arm)
    m = C.build(cfg).cuda().to(torch.bfloat16).eval()
    tot = 2 * Q + N
    ids = torch.randint(0, C.V - 1, (1, tot), device='cuda')
    last = torch.tensor([tot - 1], device='cuda'); nctx = torch.tensor([Q + N], device='cuda'); nq = torch.tensor([Q], device='cuda')

    def f():
        if cfg.kind == 'slot': return m.dec_hidden(ids, nctx, nq, nrows=Q)
        return m.dec_hidden(ids, last)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as fc:
            f()
        flops = fc.get_total_flops()
        for _ in range(3): f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): f()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            o = f()
        ts = []
        for r in range(REPS):
            ids.copy_(torch.randint(0, C.V - 1, (1, tot), device='cuda'))
            torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts[WARM:])
        # eager (no graph) for reference
        te = []
        for r in range(20):
            ids.copy_(torch.randint(0, C.V - 1, (1, tot), device='cuda'))
            torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize(); te.append((time.perf_counter() - t0) * 1e3)
    im = C.infer_macs(cfg)
    out[spec] = dict(graph_median_ms=float(np.median(ts)), graph_p95_ms=float(np.percentile(ts, 95)), n_reps=len(ts),
                     eager_median_ms=float(np.median(te[5:])), counted_gflop=flops / 1e9, analytic_gflop=2 * im['total'] / 1e9,
                     per_state_token_mflop=2 * im['per_state_token'] / 1e6, nonemb_M=C.nonemb_params(cfg) / 1e6,
                     tflops_achieved=flops / 1e9 / np.median(ts))
    print(spec, json.dumps(out[spec]), flush=True)
    del m, g; torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], 'w'), indent=1)
