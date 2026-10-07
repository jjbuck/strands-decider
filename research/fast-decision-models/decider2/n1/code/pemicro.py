"""pemicro.py (inf2): TensorE cost calibration. Each kernel issues NREP x 32 independent matmuls (8 stationaries cycled,
outputs into 4 PSUM banks, evicted once per rep).  python pemicro.py"""
import os, sys, json, glob
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

NREP = 16


def make(dt, N, stat01=False, mov01=False, K=128, M=128):
    @nki.jit
    def k(a, b):
        out = nl.ndarray((M, 4, N), dtype=nl.float32, buffer=nl.shared_hbm)
        A = nl.ndarray((K, 8, M), dtype=dt, buffer=nl.sbuf)
        for i in nl.static_range(8):
            A[:, i, :] = nl.load(a[i, 0:K, 0:M], dtype=dt)
        Bm = nl.load(b[0:K, 0:N], dtype=dt)
        acc = nl.ndarray((M, 4, N), dtype=nl.float32, buffer=nl.sbuf)
        acc[...] = nisa.memset((M, 4, N), 0.0, dtype=nl.float32)
        for r in nl.static_range(NREP):
            ps = nl.ndarray((M, 4, N), dtype=nl.float32, buffer=nl.psum)
            for j in nl.static_range(32):
                if j < 4:
                    ps[:, j % 4, :] = nisa.nc_matmul(A[:, j % 8, :], Bm, is_stationary_onezero=stat01, is_moving_onezero=mov01)
                else:
                    ps[:, j % 4, :] += nisa.nc_matmul(A[:, j % 8, :], Bm, is_stationary_onezero=stat01, is_moving_onezero=mov01)
            acc[...] = nisa.tensor_tensor(ps, acc, op=nl.add)
        nl.store(out, value=acc)
        return out
    return k


res = []
cases = [('fp32', nl.float32, 128, False, False), ('fp32', nl.float32, 512, False, False), ('fp32_mov01', nl.float32, 128, False, True),
         ('fp32_stat01', nl.float32, 128, True, False), ('bf16', nl.bfloat16, 128, False, False), ('bf16', nl.bfloat16, 512, False, False),
         ('fp32', nl.float32, 64, False, False), ('fp32_K1', nl.float32, 512, True, False)]
for name, dt, N, s01, m01 in cases:
    K = 1 if 'K1' in name else 128
    a = np.random.rand(8, 128, 128).astype(np.float32); b = np.random.rand(128, 512).astype(np.float32)
    if s01: a = (a > 0.5).astype(np.float32)
    if m01: b = (b > 0.5).astype(np.float32)
    wd = os.path.expanduser(f'~/work/n1/prof/pemicro_{name}_{N}')
    os.makedirs(wd, exist_ok=True)
    for f in glob.glob(f'{wd}/*'):
        if os.path.isfile(f): os.remove(f)
    nki.profile(working_directory=wd, save_neff_name='file.neff', save_trace_name='profile.ntff', overwrite=True)(make(dt, N, s01, m01, K=K))(a, b)
    js = glob.glob(f'{wd}/json_reports/*.json')[0]
    d = json.load(open(js)); s = d['summary'][0] if isinstance(d['summary'], list) else d['summary']
    t = s['total_time'] * 1e6
    r = dict(case=name, N=N, K=K, total_us=round(t, 2), per_matmul_ns=round(t * 1e3 / (NREP * 32), 1),
             pe_active=round(s.get('tensor_engine_active_time_percent', 0), 3))
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(os.path.expanduser('~/work/n1/pemicro.json'), 'w'), indent=1)
