"""J5: fidelity of the deployable short-M W8A8-b8 runtime. GPTQ int8 codes for H1 FORMAT v0 (rotated basis; Hessians R^T H R from wq.py's
calibration), then every evalkit question through (a) h2's QRT with its CUTLASS kernels and (b) j5rt.IntSK (small-M CUTLASS tiles, serial
split-K, sk kernels; selection tuned per power-of-two row bucket). Writes preds_w8a8b8_{h2,j5}.jsonl.  python w8eval.py codes | eval h2|j5"""
import os, sys, json, time, torch
sys.path[:0] = [os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import triton
J5 = os.path.expanduser('~/work/j5')
PREC = 'map:~/work/h1/precmap_w8a8_b8.json:w8a8'


def gptq(W, H, qmax, s, blocksize=128, percdamp=0.01):
    W = W.clone().float(); N, K = W.shape; H = H.clone().float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]
    H[range(K), range(K)] += percdamp * torch.mean(torch.diag(H))
    L = torch.linalg.cholesky(H); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]; q = torch.round(w / s).clamp(-qmax, qmax); Q[:, i1 + j] = q
            e = (w - q * s) / Hi[j, j]; W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]; E1[:, j] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)], s


def load():
    from kitrun import load_P
    from lean2 import Lean2
    import j5rt
    P = load_P(); ln = j5rt.slim(Lean2(P.tm, fuse='fold')); head = P.model.head.float().eval()
    return P, ln, head


def codes():
    import qrt as Q
    P, ln, head = load()
    m = Q.QRT(ln, head=head, prec=PREC); out = {}; t0 = time.time()
    for i in range(24):
        for k in Q.GEMMS:
            if m.pm[(i, k)] == 'bf16': continue
            W = m.wfold(i, k)
            H0 = torch.load(f'{J5}/hess/H_{i}_{k}.pt', map_location='cuda').float()
            R = m.R1 if k in ('Win', 'Wgu') else (m.R4 if k == 'Wd' else m.r2_for(i))
            Hr = R(R(H0).t().contiguous())
            s = Q.rtn_scales(W, 127.0, False)
            q, s = gptq(W, Hr, 127.0, s)
            out[(i, k)] = (q.to(torch.int8).cpu(), s.float().cpu())
        print('layer', i, f'{time.time() - t0:.0f}s', flush=True)
    torch.save(out, f'{J5}/codes_gptq_w8_rot.pt'); print('saved', flush=True)


def evaluate(which):
    import evalkit as EK, qrt as Q
    from kitrun import prep_question, probdict
    P, ln, head = load()
    wc = torch.load(f'{J5}/codes_gptq_w8_rot.pt')
    if which == 'h2':
        m = Q.QRT(ln, head=head, prec=PREC, wcodes=wc); m.tune = False
    else:
        import j5rt
        j5rt.patch_qg()
        m = j5rt.IntSK(ln, head=head, prec=PREC, wcodes=wc); m.tune = False
        # bucket the per-shape kernel choice by power-of-two rows (exactness does not depend on the choice)
        orig = m.qgemm
        def qg(A, e, _o=orig):
            M = A.shape[0]; Mb = min(triton.next_power_of_2(M), 4096)
            key = ('g', Mb, e['codes'].shape[0], A.shape[1])
            if key not in m.choice:
                Ab = torch.randint(-50, 50, (Mb, A.shape[1]), device=A.device, dtype=torch.int8)
                _o(Ab, e); m.choice[key] = m.choice[('g', Mb, e['codes'].shape[0], A.shape[1])]
            m.choice[('g', M, e['codes'].shape[0], A.shape[1])] = m.choice[key]
            return _o(A, e)
        m.qgemm = qg
    items = list(EK.all_question_items(None)); byid = {}
    for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        for it in EK.load_suite(s): byid[it['id']] = it
    out = f'{J5}/preds_w8a8b8_{which}.jsonl'; done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
    t0 = time.time(); n = 0
    with open(out, 'a') as f, torch.inference_mode():
        for suite, iid, qn, st, spec in items:
            if (iid, qn) in done: continue
            pr = prep_question(P, byid[iid], qn)
            ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
            hn = m.forward(ids, Q.Lay('single', T))
            rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
            h = m.unrot(hn[rows]).float()
            lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
            pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
            f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, probs=pd)) + '\n'); n += 1
            if n % 400 == 0: f.flush(); print(which, n, f'{time.time() - t0:.0f}s', flush=True)
    print('done', which, n, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'codes': codes()
    else: evaluate(sys.argv[2])
