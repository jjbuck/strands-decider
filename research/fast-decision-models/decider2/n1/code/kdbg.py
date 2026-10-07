"""kdbg.py (inf2): run gdn4_dbg on HW (or sim) at H=4, T=128 and compare each intermediate with alg_dbg."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import neuronxcc.nki as nki
import importlib
modname = sys.argv[1] if len(sys.argv) > 1 else 'gdn4'
mode = sys.argv[2] if len(sys.argv) > 2 else 'hw'
M = importlib.import_module(modname)
D = 128


def inputs(H, T, seed=0):
    rng = np.random.default_rng(seed)
    def l2(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    k = l2(rng.standard_normal((H, T, D)) + 2 * rng.standard_normal((H, 1, D))).astype(np.float32)
    q = (l2(rng.standard_normal((H, T, D))) * D ** -0.5).astype(np.float32)
    v = rng.standard_normal((H, T, D)).astype(np.float32)
    g = (-rng.random((H, T, 1)) * 0.5).astype(np.float32)
    beta = rng.random((H, T, 1)).astype(np.float32)
    S0 = np.zeros((H, D, D), np.float32)
    return q, k, v, g, beta, S0


q, k, v, g, beta, S0 = inputs(4, 128)
args = M.prep(q, k, v, g, beta, S0)
f = M.gdn4_dbg
o, Sf, dbg = (nki.simulate_kernel(f, *args) if mode == 'sim' else nki.baremetal(f)(*args))
ref = M.alg_dbg(q, k, v, g, beta, S0)
for i, nm in enumerate(M.DBG_NAMES):
    a = dbg[i]; r = ref[nm]
    err = np.abs(a - r); j = np.unravel_index(err.argmax(), err.shape)
    print(f'{nm:5s} maxerr {err.max():.3e} refmax {np.abs(r).max():.3e} at {j} got {a[j]:.5f} want {r[j]:.5f}', flush=True)
