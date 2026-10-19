"""Raw dec_eval timing vs prompt length and context/ubatch sizing (no tokeniser, no head).

  python shim_bench.py GGUF THREADS  -> ms per forward for n in (275, 512, 1309) at n_ctx in (1536, 4224)
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, "/opt/lam/engine")
from lcpp_engine import Torso

gguf, threads = sys.argv[1], int(sys.argv[2])
lib = sys.argv[3] if len(sys.argv) > 3 else "/opt/lam/libdecshim.so"
rng = np.random.default_rng(0)
for n_ctx in [int(x) for x in os.environ.get("NCTX", "4224").split(",")]:
    t = Torso(lib, gguf, threads, n_ctx=n_ctx)
    for n in [int(x) for x in os.environ.get("LADDER", "275,512,1309").split(",")]:
        ids = rng.integers(1000, 200000, size=n).tolist()
        pos = [n - 1, n - 20, n - 40]
        t.hidden(ids, pos)
        ts = []
        for _ in range(3):
            s = time.perf_counter()
            t.hidden(ids, pos)
            ts.append(time.perf_counter() - s)
        print(f"n_ctx={n_ctx} n={n}: {min(ts)*1e3:.0f} ms ({n/min(ts):.0f} tok/s)", flush=True)
    t.lib.dec_free(t.h)
