"""mlpsim.py: simulate (or run on HW) mlp_kernel with numpy bf16 inputs.  python mlpsim.py T [sim|hw] [h|aT|out]"""
import os, sys
import numpy as np, ml_dtypes
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import neuronxcc.nki as nki
import mlp_nki as MN
T = int(sys.argv[1]); mode = sys.argv[2]; what = sys.argv[3] if len(sys.argv) > 3 else 'out'
K = {'out': MN.mlp_kernel, 'h': MN.mlp_kernel_h, 'aT': MN.mlp_kernel_aT}[what]
bf = ml_dtypes.bfloat16
rng = np.random.default_rng(0)
x = (rng.standard_normal((T, 2048)) * 0.5).astype(bf)
w1 = (1 + 0.1 * rng.standard_normal((1, 2048))).astype(np.float32)
WguT = (rng.standard_normal((2048, 12288)) * 0.02).astype(bf)
Wd = (rng.standard_normal((6144, 2048)) * 0.02).astype(bf)
eye = np.eye(128).astype(bf)
args = (x, w1, np.ascontiguousarray(MN.tile_wgu(WguT)), Wd, eye)
y = nki.simulate_kernel(K, *args) if mode == 'sim' else nki.baremetal(K)(*args)
y = y.astype(np.float32)
xf = x.astype(np.float32)
h = xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-6) * w1
gu = h @ WguT.astype(np.float32); g, u = gu[:, :6144], gu[:, 6144:]; a = g / (1 + np.exp(-g)) * u
r = {'h': h, 'aT': a[:, :2048], 'out': xf + a @ Wd.astype(np.float32)}[what]
e = np.abs(y - r); i = np.unravel_index(e.argmax(), e.shape)
print(what, 'max abs err', e.max(), 'ref absmax', np.abs(r).max(), 'at', i, y[i], r[i], 'mean err', e.mean())
