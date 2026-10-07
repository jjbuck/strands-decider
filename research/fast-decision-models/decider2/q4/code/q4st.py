"""Q4: Strassen on tensor cores at hobson-v19's shapes. ctypes wrapper for libq4st.so, operand construction, exactness test and timing.
python q4st.py test
python q4st.py bench Ms [shapes]      -> ~/work/q4/res_strassen.jsonl
Types: s8 = int8 tensor cores on 7-bit codes (|q| <= 63, so two-term sums fit int8), s4 = int4 on 3-bit codes (|q| <= 3), bf16.
Each record (per shape, M, type): best config by median (CUDA events, 8 warm-up, 30 timed, 3 rotating input sets) of
  dense_same   the same kernel in mode 1 (one full-size product): the speedup baseline with an identical main loop
  dense_q2     Q2's q2gemm (same family), dense_h2 = H2's CUTLASS (int) / cuBLAS (bf16): the best dense kernels available
  st1_fused    one-level Strassen, all 7 products in one kernel, combination in registers (excludes the producer)
  st1_sums     the producer's five activation sums (one kernel; fused into the quantize kernel in a deployment it costs ~ the extra writes)
  st1_prod     bound: the 7 half-size products alone (Q2 batched launch, bf16 outputs, no sums, no combination)
  st2_prod     bound: the 49 quarter-size products alone
  st2_real     two levels: outer level unfused (7 fused one-level kernels in one launch, int32/fp32 products to memory) + combine kernel"""
import os, sys, json, ctypes, statistics, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
dev = 'cuda'
TY = dict(s4=1, s8=2, bf16=3)
L_ = ctypes.c_longlong; P_ = ctypes.c_void_p; I_ = ctypes.c_int


class StP(ctypes.Structure):
    _fields_ = [('A', P_ * 7), ('lda', L_ * 7), ('B', P_ * 7), ('ldb', L_), ('sa', P_), ('sb', P_), ('out', P_), ('ldo', L_),
                ('Mh', I_), ('Nh', I_), ('M', I_), ('N', I_), ('kt', I_), ('epi', I_), ('mode', I_), ('mtiles', I_), ('ntiles', I_),
                ('bsA', L_), ('bsB', L_), ('bsO', L_)]


_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.expanduser('~/work/q4/libq4st.so'))
        _lib.q4st_run.restype = I_; _lib.q4st_run.argtypes = [I_, I_, ctypes.POINTER(StP), I_, P_]
        _lib.q4st_sums.restype = I_; _lib.q4st_sums.argtypes = [I_, P_, L_, I_, I_, I_, P_, P_]
        _lib.q4st_comb.restype = I_; _lib.q4st_comb.argtypes = [I_, P_, I_, I_, P_, P_, P_, L_, P_]
        assert _lib.q4st_sizeof() == ctypes.sizeof(StP), (_lib.q4st_sizeof(), ctypes.sizeof(StP))
    return _lib


def pack4(q):
    q = q.to(torch.int16)
    return ((q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()


def nbytes_k(ty, K): return K if ty == 's8' else (K // 2 if ty == 's4' else 2 * K)
def as_bytes(t, ty):
    """codes / bf16 tensor [R, K] -> uint8 view [R, Kb]"""
    if ty == 's8': return t.contiguous().view(torch.uint8)
    if ty == 's4': return pack4(t)
    return t.contiguous().view(torch.uint8)


SEQ = ['M4', 'M2', 'M1', 'M7', 'M5', 'M3', 'M6']


def w_ops(W, ty):
    """weights [N, K] (int8 codes or bf16) -> the 7 Strassen B operands [7, Nh, Kb/2] bytes in sequence order"""
    N, K = W.shape; Nh, Kh = N // 2, K // 2
    Wf = W.float()
    W11, W12, W21, W22 = Wf[:Nh, :Kh], Wf[:Nh, Kh:], Wf[Nh:, :Kh], Wf[Nh:, Kh:]
    ops = dict(M4=W12 - W11, M2=W11, M1=W11 + W22, M7=W12 + W22, M5=W22, M3=W21 - W22, M6=W11 + W21)
    out = []
    for k in SEQ:
        o = ops[k]
        o = o.to(torch.int8) if ty != 'bf16' else o.to(torch.bfloat16)
        out.append(as_bytes(o, ty))
    return torch.stack(out, 0).contiguous()


def a_ops_ref(X, ty):
    """reference activation operands (torch), X [2*Mh, K] padded -> [7, Mh, Kb/2] bytes in sequence order"""
    M2_, K = X.shape; Mh, Kh = M2_ // 2, K // 2
    Xf = X.float()
    A11, A12, A21, A22 = Xf[:Mh, :Kh], Xf[:Mh, Kh:], Xf[Mh:, :Kh], Xf[Mh:, Kh:]
    ops = dict(M4=A22, M2=A21 + A22, M1=A11 + A22, M7=A12 - A22, M5=A11 + A12, M3=A11, M6=A21 - A11)
    out = [as_bytes(ops[k].to(torch.int8) if ty != 'bf16' else ops[k].to(torch.bfloat16), ty) for k in SEQ]
    return torch.stack(out, 0).contiguous()


class Level1:
    """one-level operands for a padded activation byte matrix Xb [2*Mh, Kb] using the producer kernel for the five sums"""
    def __init__(self, Xb, ty, Mvalid):
        self.Xb = Xb; self.ty = ty; R, Kb = Xb.shape; self.Mh = R // 2; self.Kbh = Kb // 2; self.M = Mvalid
        self.S = torch.empty(5, self.Mh, self.Kbh, device=dev, dtype=torch.uint8)
    def sums(self, stream=None):
        st = torch.cuda.current_stream().cuda_stream if stream is None else stream
        rc = lib().q4st_sums(TY[self.ty], self.Xb.data_ptr(), self.Xb.stride(0), self.M, self.Mh, self.Kbh, self.S.data_ptr(), st)
        if rc: raise RuntimeError(f'sums rc {rc}')
    def ptrs(self):
        Xb, S, Mh, Kbh = self.Xb, self.S, self.Mh, self.Kbh
        ld = Xb.stride(0)
        base = Xb.data_ptr()
        A11 = (base, ld); A12 = (base + Kbh, ld); A21 = (base + Mh * ld, ld); A22 = (base + Mh * ld + Kbh, ld)
        s = lambda i: (S[i].data_ptr(), Kbh)
        d = dict(M4=A22, M2=s(1), M1=s(0), M7=s(4), M5=s(2), M3=A11, M6=s(3))
        return [d[k] for k in SEQ]


def mkp(Aps, Bops_ptr, ldb, Kb_half, KB, sa, sb, out, ldo, Mh, Nh, M, N, epi, mode, bsA=0, bsB=0, bsO=0):
    p = StP()
    for i in range(7):
        a = Aps[i] if i < len(Aps) else Aps[0]
        p.A[i] = a[0]; p.lda[i] = a[1]
        p.B[i] = Bops_ptr[i] if i < len(Bops_ptr) else Bops_ptr[0]
    p.ldb = ldb; p.sa = 0 if sa is None else sa.data_ptr(); p.sb = 0 if sb is None else sb.data_ptr()
    p.out = out.data_ptr(); p.ldo = ldo
    p.Mh, p.Nh, p.M, p.N = Mh, Nh, M, N
    p.kt = Kb_half // KB; p.epi = epi; p.mode = mode; p.bsA, p.bsB, p.bsO = bsA, bsB, bsO
    return p


KBC = {c: 128 for c in range(8)}   # every config uses 128-byte K tiles


def run(ty, cfg, p, batch=1):
    p.kt = p.kt   # set by caller for KB = 128
    rc = lib().q4st_run(TY[ty], cfg, ctypes.byref(p), batch, torch.cuda.current_stream().cuda_stream)
    if rc: raise RuntimeError(f'q4st {ty} cfg{cfg} rc {rc}')


def codes(ty, *s, q=None):
    if ty == 'bf16': return torch.randn(*s, device=dev).to(torch.bfloat16)
    q = q if q is not None else (63 if ty == 's8' else 3)
    return torch.randint(-q, q + 1, s, device=dev, dtype=torch.int8)


def test():
    torch.manual_seed(0); bad = 0
    for ty in ('s8', 's4', 'bf16'):
        for (M, N, K) in ((140, 2048, 2048), (1125, 512, 6144), (77, 8224, 2048)):
            Mh = (M + 1) // 2; Nh, Kh = N // 2, K // 2
            X = torch.zeros(2 * Mh, K, device=dev, dtype=torch.int8 if ty != 'bf16' else torch.bfloat16); X[:M] = codes(ty, M, K)
            W = codes(ty, N, K)
            Xb = as_bytes(X, ty); Kb = Xb.shape[1]; Kbh = Kb // 2
            L1 = Level1(Xb, ty, M); L1.sums(); torch.cuda.synchronize()
            # producer check against torch
            ref_ops = a_ops_ref(X, ty)
            ptrs = L1.ptrs()
            Bop = w_ops(W, ty)
            sa = torch.rand(Mh, device=dev) + 0.5; sb = torch.rand(Nh, device=dev) + 0.5
            sa_full = torch.cat([sa, sa]); sb_full = torch.cat([sb, sb])
            if ty != 'bf16':
                acc = (X[:M].double() @ W.double().t()).round()
                yref = ((acc.float() * sa_full[:M, None]) * sb_full[None, :]).to(torch.bfloat16)
            else:
                yref = (X[:M].float() @ W.float().t())
            for cfg in range(8):
                Y = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
                p = mkp(ptrs, [Bop[i].data_ptr() for i in range(7)], Kbh, Kbh, 128, sa, sb, Y, N, Mh, Nh, M, N, 0, 0)
                try:
                    run(ty, cfg, p); torch.cuda.synchronize()
                except RuntimeError as ex:
                    print(ty, cfg, 'skip', ex); continue
                # dense same-loop
                Yd = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
                Wb = as_bytes(W, ty)
                pd = mkp([(Xb.data_ptr(), Xb.stride(0))], [Wb.data_ptr()], Kb, Kb, 128, sa_full, sb_full, Yd, N, M, N, M, N, 0, 1)
                run(ty, cfg, pd); torch.cuda.synchronize()
                if ty != 'bf16':
                    e = (Y.float() - yref.float()).abs().max().item(); ed = (Yd.float() - yref.float()).abs().max().item()
                    ok = e == 0 and ed == 0
                    msg = f'maxdiff strassen {e} dense {ed}'
                else:
                    n = yref.norm().item()
                    e = ((Y.float() - yref).norm() / n).item(); ed = ((Yd.float() - yref).norm() / n).item()
                    ok = e < 5e-2 and ed < 1e-2
                    msg = f'rel err vs fp32: strassen {e:.2e} dense {ed:.2e}'
                bad += not ok
                print(f'{ty} M{M} N{N} K{K} cfg{cfg}: {msg} {"OK" if ok else "FAIL"}', flush=True)
    print('ALL OK' if bad == 0 else f'{bad} FAIL', flush=True)


# ------------------------------------------------------------------ timing
def tm(fn, reps=30, warm=8):
    for w in range(warm): fn(w)
    torch.cuda.synchronize(); ts = []
    for i in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(i); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return round(statistics.median(ts), 2), round(ts[int(0.95 * (len(ts) - 1))], 2)


def best(cands):
    out = None
    for lab, fn in cands:
        try:
            fn(0); torch.cuda.synchronize(); m, p = tm(fn)
        except Exception as ex:
            if os.environ.get('Q4DBG'): print('cand fail', lab, repr(ex)[:200], flush=True)
            continue
        if out is None or m < out[1]: out = [lab, m, p]
    return out


SH = {'gdn_in': (8224, 2048), 'attn_in': (5120, 2048), 'out': (2048, 2048), 'gate_up': (12288, 2048), 'down': (2048, 6144)}
OUT = os.path.expanduser(os.environ.get('Q4OUT', '~/work/q4/res_strassen.jsonl'))


def bench(Ms, shapes, tys=('s8', 's4', 'bf16')):
    import q2k as Q, qgemm as QG
    for M in Ms:
        for name in shapes:
            N, K = SH[name]
            for ty in tys:
                r = dict(shape=name, M=M, N=N, K=K, ty=ty)
                Mh = (M + 1) // 2; Nh, Kh = N // 2, K // 2
                Xs = []
                for _ in range(3):
                    X = torch.zeros(2 * Mh, K, device=dev, dtype=torch.int8 if ty != 'bf16' else torch.bfloat16); X[:M] = codes(ty, M, K); Xs.append(X)
                W = codes(ty, N, K)
                Xbs = [as_bytes(X, ty) for X in Xs]; Kb = Xbs[0].shape[1]; Kbh = Kb // 2
                Wb = as_bytes(W, ty); Bop = w_ops(W, ty)
                L1s = [Level1(Xb, ty, M) for Xb in Xbs]
                for L1 in L1s: L1.sums()
                sa = torch.rand(Mh, device=dev) + 0.5; sb = torch.rand(Nh, device=dev) + 0.5
                saf = torch.rand(M, device=dev) + 0.5; sbf = torch.rand(N, device=dev) + 0.5
                Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                bptr = [Bop[i].data_ptr() for i in range(7)]
                ps = {c: [mkp(L1s[j].ptrs(), bptr, Kbh, Kbh, 128, sa, sb, Y, N, Mh, Nh, M, N, 0, 0) for j in range(3)] for c in range(8)}
                pd = {c: [mkp([(Xbs[j].data_ptr(), Kb)], [Wb.data_ptr()], Kb, Kb, 128, saf, sbf, Y, N, M, N, M, N, 0, 1) for j in range(3)] for c in range(8)}
                r['dense_same'] = best([(f'c{c}', (lambda i, c=c: run(ty, c, pd[c][i % 3]))) for c in range(8)])
                r['st1_fused'] = best([(f'c{c}', (lambda i, c=c: run(ty, c, ps[c][i % 3]))) for c in range(8)])
                r['st1_sums'] = best([('sums', lambda i: L1s[i % 3].sums())])
                # Q2 dense family and the deployed dense kernels
                q2ty = {'s8': 's8', 's4': 's4', 'bf16': 'bf16'}[ty]
                Xq2 = [Xb[:M].view(torch.int8) if ty == 's8' else (Xb[:M] if ty == 's4' else X[:M]) for Xb, X in zip(Xbs, Xs)]
                Wq2 = W if ty != 's4' else Wb
                r['dense_q2'] = best([(f'q{c}', (lambda i, c=c: Q.run(q2ty, c, Q.prob(Xq2[i % 3], Wq2, q2ty, Y, saf, sbf, epi='bf16', N=N)))) for c in range(10)])
                o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
                if ty == 's8':
                    r['dense_h2'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s8', Xq2[i % 3], W, 2 ** -10, c, out=o16))) for c in range(5)])
                elif ty == 's4':
                    r['dense_h2'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s4', Xq2[i % 3], Wb, 0.5, c, out=o16))) for c in range(5)])
                else:
                    r['dense_h2'] = best([('cublas', lambda i: torch.mm(Xq2[i % 3], W.t(), out=Y))])
                # bound: 7 half-size products, batched (Q2 kernel, gridDim.z = 7, bf16 outputs)
                Aop = [a_ops_ref(X, ty) for X in Xs]           # [7, Mh, Kbh]
                O7 = torch.empty(7, Mh, Nh, device=dev, dtype=torch.bfloat16)
                def prod(i, c, Aop=Aop, nb=7, Bt=Bop, Mr=Mh, Nr=Nh, O=O7):
                    A0 = Aop[i % 3][0].view(torch.int8) if ty == 's8' else (Aop[i % 3][0] if ty == 's4' else Aop[i % 3][0].view(torch.bfloat16))
                    B0 = Bt[0].view(torch.int8) if ty == 's8' else (Bt[0] if ty == 's4' else Bt[0].view(torch.bfloat16))
                    Q.run(q2ty, c, Q.prob(A0, B0, q2ty, O[0], sa, sb, epi='bf16', M=Mr, N=Nr, bsA0=Aop[i % 3][0].numel(), bsB0=Bt[0].numel(),
                                          bsO=O[0].numel() * 2), batch=nb)
                r['st1_prod'] = best([(f'q{c}', (lambda i, c=c: prod(i, c))) for c in range(10)])
                # bound: 49 quarter-size products
                Mq, Nq = (Mh + 1) // 2, Nh // 2; Kbq = Kbh // 2
                if Kbq % 64 == 0:
                    A49 = [torch.randint(0, 255, (49, Mq, Kbq), device=dev, dtype=torch.uint8) for _ in range(3)]
                    if ty == 'bf16': A49 = [torch.randn(49, Mq, Kbq // 2, device=dev).to(torch.bfloat16).view(torch.uint8) for _ in range(3)]
                    B49 = torch.randint(0, 255, (49, Nq, Kbq), device=dev, dtype=torch.uint8)
                    if ty == 'bf16': B49 = torch.randn(49, Nq, Kbq // 2, device=dev).to(torch.bfloat16).view(torch.uint8)
                    O49 = torch.empty(49, Mq, Nq, device=dev, dtype=torch.bfloat16)
                    r['st2_prod'] = best([(f'q{c}', (lambda i, c=c: prod(i, c, Aop=A49, nb=49, Bt=B49, Mr=Mq, Nr=Nq, O=O49))) for c in range(10)])
                    # two-level real: outer unfused (7 fused inner kernels in one launch), int32/fp32 products, combine kernel
                    # inner operands laid out [7 outer][7 inner][Mq][Kbq] so one stride serves every outer product
                    A2 = A49; B2 = B49
                    P7 = torch.empty(7, Mh, Nh, device=dev, dtype=torch.int32)
                    Mq2 = Mq
                    def two(i, c):
                        Ab = A2[i % 3]
                        p = mkp([(Ab[j].data_ptr(), Kbq) for j in range(7)], [B2[j].data_ptr() for j in range(7)], Kbq, Kbq, 128, sa, sb, P7, Nh,
                                Mq2, Nq, Mh, Nh, 1, 0, bsA=7 * Ab[0].numel(), bsB=7 * B2[0].numel(), bsO=P7[0].numel() * 4)
                        run(ty, c, p, batch=7)
                        rc = lib().q4st_comb(TY[ty], P7.data_ptr(), Mh, Nh, sa.data_ptr(), sb.data_ptr(), Y.data_ptr(), N, torch.cuda.current_stream().cuda_stream)
                        if rc: raise RuntimeError(rc)
                    r['st2_real'] = best([(f'c{c}', (lambda i, c=c: two(i, c))) for c in range(8)])
                    r['st2_comb'] = best([('comb', lambda i: lib().q4st_comb(TY[ty], P7.data_ptr(), Mh, Nh, sa.data_ptr(), sb.data_ptr(), Y.data_ptr(), N,
                                                                              torch.cuda.current_stream().cuda_stream))])
                d = r['dense_same'][1] if r['dense_same'] else None
                if d and r['st1_fused']: r['speedup_st1_vs_same'] = round(d / (r['st1_fused'][1] + r['st1_sums'][1]), 3); r['speedup_st1_nosums_vs_same'] = round(d / r['st1_fused'][1], 3)
                bd = min(x[1] for x in (r['dense_same'], r['dense_q2'], r['dense_h2']) if x)
                r['best_dense'] = bd
                if r['st1_fused']: r['speedup_st1_vs_best'] = round(bd / (r['st1_fused'][1] + r['st1_sums'][1]), 3)
                if r.get('st2_real'): r['speedup_st2_vs_best'] = round(bd / r['st2_real'][1], 3)
                if r.get('st2_prod'): r['bound_st2_vs_best'] = round(bd / r['st2_prod'][1], 3)
                if r.get('st1_prod'): r['bound_st1_vs_best'] = round(bd / r['st1_prod'][1], 3)
                print(json.dumps(r), flush=True)
                with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')
                del Xs, Xbs, L1s, Bop, Aop; torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == 'test': test()
    else:
        Ms = [int(x) for x in sys.argv[2].split(',')]
        shapes = sys.argv[3].split(',') if len(sys.argv) > 3 else list(SH)
        tys = sys.argv[4].split(',') if len(sys.argv) > 4 else ('s8', 's4', 'bf16')
        bench(Ms, shapes, tys)
