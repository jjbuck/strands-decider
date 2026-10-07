import sys, os, time, json, torch, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import load_torso, capture
from lean2 import Lean2
import qrt as Q
torch.manual_seed(0); dev = 'cuda'
torso = load_torso()
ln = Lean2(torso, fuse='fold')
del torso; torch.cuda.empty_cache()
print('mem after load', torch.cuda.memory_allocated() / 1e9, flush=True)
T = 1000
ids = torch.randint(1000, 100000, (T,), device=dev)
with torch.inference_mode():
    href = ln.forward(ids[None])[0].float()                          # lean2 fold: rms_zc(x, norm_w)
def cosrows(a, b): return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
R = {}
for prec in ('bf16', 'w8a8', 'w4a8', 'w4a4'):
    t0 = time.time(); m = Q.QRT(ln, prec=prec); tb = time.time() - t0
    lay = Q.Lay('single', T)
    with torch.inference_mode():
        hn = m.forward(ids, lay); h = m.unrot(hn)
    c = cosrows(h, href)
    print(f'{prec}: build {tb:.0f}s  final-hidden cos vs lean2-fold mean {c.mean().item():.5f} min {c.min().item():.4f}  mem {torch.cuda.memory_allocated()/1e9:.1f}G', flush=True)
    # timing (graph)
    g, out = capture(lambda: m.forward(ids, lay))
    ts = []
    for r in range(15):
        torch.cuda.synchronize(); t1 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t1) * 1000)
    print(f'   T={T} graph replay median {st.median(ts[3:]):.2f} ms', flush=True)
    del g, out
    if prec not in ('bf16', 'w8a8'): del m
    else: keep = keep if 'keep' in dir() else {}; keep[prec] = m
    torch.cuda.empty_cache()
for prec, m in keep.items():
    print('== exactness', prec)
    # schema exactness (w4a4 model m): prefix P, state S, N slots
    P_ = 300; S_ = 500; N_ = 4
    allids = torch.randint(1000, 100000, (P_ + S_ + N_,), device=dev)
    with torch.inference_mode():
        full = m.forward(allids, Q.Lay('single', P_ + S_ + N_))
        cache = m.compile_prefix(allids[:P_], S_ + N_)
        hs = m.forward(allids[P_:], Q.Lay('schema', S_, nslots=N_, P=P_), cache)
    c = cosrows(hs, full[P_:])
    print(f'schema vs full single pass: cos mean {c.mean().item():.6f} min {c.min().item():.5f}; last-row max abs diff {(hs[-N_:].float()-full[-N_:].float()).abs().max().item():.4f}', flush=True)
    # packed exactness: state + 3 branches vs separate single passes
    Ls = 400; ql = [37, 120, 64]
    st_ids = torch.randint(1000, 100000, (Ls,), device=dev); br = [torch.randint(1000, 100000, (L,), device=dev) for L in ql]
    with torch.inference_mode():
        hp = m.forward(torch.cat([st_ids] + br), Q.Lay('packed', Ls, qlens=ql))
        s = Ls
        for L, b in zip(ql, br):
            hsg = m.forward(torch.cat([st_ids, b]), Q.Lay('single', Ls + L))
            c = cosrows(hp[s:s + L], hsg[Ls:])
            print(f'packed branch L={L}: cos mean {c.mean().item():.6f} min {c.min().item():.5f}', flush=True)
            s += L
    