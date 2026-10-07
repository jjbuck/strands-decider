"""kbench.py (inf2): correctness + device time of an NKI GDN kernel, numpy only (no torch).
  python kbench.py MODULE KERNEL T [H] [mode]
    mode: sim   -> nki.simulate_kernel at (H, T), max abs err vs fp64 recurrence
          hw    -> nki.baremetal on the NeuronCore, max abs err vs fp64 recurrence
          prof  -> nki.profile, device time + engine summary (json written to ~/work/n1/prof/<tag>)
          all   -> hw then prof
Kernel signature: KER(q, k, v, g, beta, cst, S0) -> (o, Sf) with q,k,v [H,T,128] fp32, g,beta [H,T,1] fp32,
cst = module.consts_np(), S0 [H,128,128].  Modules may expose prep(q,k,v,g,beta,S0) -> tuple of kernel args (layout prep
that the XLA side would do) and consts_np().
"""
import os, sys, json, time, importlib, glob, subprocess
import numpy as np
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import neuronxcc.nki as nki

modname, kname, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
H = int(sys.argv[4]) if len(sys.argv) > 4 else 16
mode = sys.argv[5] if len(sys.argv) > 5 else 'all'
S0nz = os.environ.get('S0', '0') == '1'
M = importlib.import_module(modname)
KER = getattr(M, kname)
D = 128


def inputs(H, T, seed=0):
    rng = np.random.default_rng(seed)
    def l2(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    k = l2(rng.standard_normal((H, T, D)) + 2 * rng.standard_normal((H, 1, D))).astype(np.float32)
    q = (l2(rng.standard_normal((H, T, D))) * D ** -0.5).astype(np.float32)
    v = rng.standard_normal((H, T, D)).astype(np.float32)
    g = (-rng.random((H, T, 1)) * float(os.environ.get('GSCALE', '0.5'))).astype(np.float32)
    beta = rng.random((H, T, 1)).astype(np.float32)
    S0 = (rng.standard_normal((H, D, D)) * 0.05).astype(np.float32) if S0nz else np.zeros((H, D, D), np.float32)
    return q, k, v, g, beta, S0


def ref_np(q, k, v, g, beta, S0):
    H, T, Dd = q.shape
    S = S0.astype(np.float64).copy(); out = np.zeros((H, T, Dd))
    q, k, v, g, beta = (x.astype(np.float64) for x in (q, k, v, g, beta))
    for t in range(T):
        S = S * np.exp(g[:, t, 0])[:, None, None]
        kt = k[:, t]; pred = np.einsum('hk,hkv->hv', kt, S)
        S = S + np.einsum('hk,hv->hkv', kt, (v[:, t] - pred) * beta[:, t, 0][:, None])
        out[:, t] = np.einsum('hk,hkv->hv', q[:, t], S)
    return out, S


q, k, v, g, beta, S0 = inputs(H, T)
args = M.prep(q, k, v, g, beta, S0) if hasattr(M, 'prep') else (q, k, v, g, beta, M.consts_np(), S0)
post = getattr(M, 'post', lambda o, S: (o, S))
tag = f'{modname}_{kname}_H{H}_T{T}'
res = dict(tag=tag, H=H, T=T)


def check(o, Sf):
    o, Sf = post(o, Sf)
    r, Sr = ref_np(q, k, v, g, beta, S0)
    res.update(max_abs_err=float(np.abs(o - r).max()), ref_absmax=float(np.abs(r).max()),
               state_err=float(np.abs(Sf - Sr).max()), state_absmax=float(np.abs(Sr).max()))


grid = getattr(M, 'GRID', {}).get(kname, H)
call = (lambda f: f[grid]) if grid else (lambda f: f)
if mode == 'alg':
    o, Sf = M.alg_np(q, k, v, g, beta, S0)
    r, Sr = ref_np(q, k, v, g, beta, S0)
    res.update(max_abs_err=float(np.abs(o - r).max()), ref_absmax=float(np.abs(r).max()), state_err=float(np.abs(Sf - Sr).max()))
if mode == 'sim':
    t0 = time.time()
    o, Sf = nki.simulate_kernel(call(KER), *args)
    res['sim_s'] = round(time.time() - t0, 1); check(o, Sf)
if mode in ('hw', 'all'):
    t0 = time.time()
    o, Sf = call(nki.baremetal(KER))(*args)
    res['hw_s'] = round(time.time() - t0, 1); check(o, Sf)
if mode in ('prof', 'all'):
    wd = os.path.expanduser(f'~/work/n1/prof/{tag}' + os.environ.get('PTAG', ''))
    import shutil
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    call(nki.profile(working_directory=wd, save_neff_name='file.neff', save_trace_name='profile.ntff', overwrite=True, profile_nth=int(os.environ.get('PNTH', '3')))(KER))(*args)
    res['prof_s'] = round(time.time() - t0, 1)
    js = sorted(glob.glob(f'{wd}/*.json'))
    res['prof_json'] = js
    for j in js:
        try:
            d = json.load(open(j))
            s = d.get('summary', [d])[0] if isinstance(d.get('summary'), list) else d.get('summary', d)
            keep = {kk: s[kk] for kk in s if any(w in kk for w in ('total_time', 'active', 'busy', 'utilization', 'dma', 'spill', 'flops', 'instruction'))}
            res['summary'] = keep
        except Exception as e:
            res['summary_err'] = repr(e)[:200]
print(json.dumps(res), flush=True)
open(os.path.expanduser('~/work/n1/kbench.jsonl'), 'a').write(json.dumps(res) + '\n')
