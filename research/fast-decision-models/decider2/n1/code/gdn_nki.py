"""gdn_nki.py -- NKI kernel for the chunked gated delta rule (C = D = 128, fp32), one SPMD program per head.

Math is identical to hob.gdn_chunk (bd inverse + sequential state): per chunk
  gcum = cumsum(g); decay_ij = exp(gcum_i - gcum_j) (i>=j); L = beta_i (k_i.k_j) decay_ij (i>j)
  P = (I + L)^-1 by recursive block inversion (two 128x128 matmuls per level, 6 levels)
  u = P (beta v); w = P (beta e^gcum k); then with state S:
  vn = u - w S; o = (q e^gcum) S + (q k^T o decay) vn; S <- e^glast S + (k e^(glast-gcum))^T vn
nc_matmul(a, b) = a^T @ b with the contraction on the partition dim.
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

C = 128
D = 128


def consts_np():
    """[10, 128, 128] fp32: 0 U (cumsum, U[j,i]=1 if j<=i), 1 incl (i>=j), 2 strict (i>j), 3 inclT (i<=j), 4 ones,
    5 identity, then block masks for s = 2,4,8,16,32,64 at 6..11 (s=1 mask at 12). -> returns [13,128,128]."""
    ar = np.arange(C)
    i, j = ar[:, None], ar[None, :]
    out = [(i <= j), (i >= j), (i > j), (i <= j), np.ones((C, C), bool), (i == j)]
    for s in (2, 4, 8, 16, 32, 64):
        out.append(((i // (2 * s)) == (j // (2 * s))) & ((i // s) % 2 == 1) & ((j // s) % 2 == 0))
    s = 1
    out.append(((i // (2 * s)) == (j // (2 * s))) & ((i // s) % 2 == 1) & ((j // s) % 2 == 0))
    return np.stack(out).astype(np.float32)


def _tr(x):
    """PE transpose of a [128,128] SBUF tile -> SBUF."""
    return nl.copy(nisa.nc_transpose(x), dtype=nl.float32)


def _mm(a, b):
    return nl.copy(nisa.nc_matmul(a, b), dtype=nl.float32)


def _level(L, masks, lv, Dinv, DinvT):
    """one block-inversion level, fresh tiles (no in-place reuse: in-place updates inside unrolled loops raced on hardware)."""
    Lb = nl.multiply(L, masks[lv])
    LbT = _tr(Lb)
    Z = _mm(LbT, Dinv)                                     # Lb @ Dinv
    Y = _mm(DinvT, Z)                                      # Dinv @ Lb @ Dinv
    Dn = nl.subtract(Dinv, Y)
    return Dn, _tr(Dn)


@nki.jit
def gdn_kernel(q, k, v, g, beta, cst, S0):
    """q,k,v: [H, T, 128] fp32 (q,k l2-normalised, q pre-scaled); g, beta: [H, T, 1] fp32; cst: [13,128,128]; S0 [H,128,128].
    -> o [H, T, 128], final state [H, 128, 128]."""
    H, T, _ = q.shape
    n = T // C
    o = nl.ndarray((H, T, D), dtype=nl.float32, buffer=nl.shared_hbm)
    Sf = nl.ndarray((H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    h = nl.program_id(0)
    U = nl.load(cst[0]); incl = nl.load(cst[1]); strict = nl.load(cst[2]); inclT = nl.load(cst[3])
    ones = nl.load(cst[4]); eye = nl.load(cst[5]); m1 = nl.load(cst[12])
    masks = nl.ndarray((6, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    for lv in nl.static_range(6):
        masks[lv] = nl.load(cst[6 + lv])
    S = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
    S[...] = nl.load(S0[h])
    for c in nl.sequential_range(n):
        kc = nl.load(k[h, c * C:(c + 1) * C, :])
        qc = nl.load(q[h, c * C:(c + 1) * C, :])
        vc = nl.load(v[h, c * C:(c + 1) * C, :])
        gc = nl.load(g[h, c * C:(c + 1) * C, :])           # [C,1]
        bc = nl.load(beta[h, c * C:(c + 1) * C, :])        # [C,1]
        gcum = _mm(U, gc)                                  # [C,1]  sum_{j<=i} g_j
        Gb = _mm(nl.multiply(ones, gc), U)                 # Gb[p,f] = gcum_f
        Gl = _mm(ones, gc)                                 # [C,1]  chunk total, every partition
        dg = nl.subtract(gcum, Gb)                         # gcum_p - gcum_f (broadcast [C,1] over free)
        decay = nl.multiply(nl.exp(nl.multiply(dg, incl)), incl)
        decayT = nl.multiply(nl.exp(nl.multiply(nl.negative(dg), inclT)), inclT)
        eg = nl.exp(gcum)
        kT = _tr(kc); qT = _tr(qc)
        KK = _mm(kT, kT)                                   # k_i . k_j
        L = nl.multiply(nl.multiply(nl.multiply(KK, decay), strict), bc)
        # (I + L)^-1, recursive block inversion
        D0 = nl.subtract(eye, nl.multiply(L, m1))
        D0T = _tr(D0)
        D1, D1T = _level(L, masks, 0, D0, D0T)
        D2, D2T = _level(L, masks, 1, D1, D1T)
        D3, D3T = _level(L, masks, 2, D2, D2T)
        D4, D4T = _level(L, masks, 3, D3, D3T)
        D5, D5T = _level(L, masks, 4, D4, D4T)
        Dinv, DinvT = _level(L, masks, 5, D5, D5T)
        vb = nl.multiply(vc, bc)
        kbe = nl.multiply(nl.multiply(kc, bc), eg)
        u = _mm(DinvT, vb)                                 # P @ vb          [C, D]
        wT = _mm(kbe, DinvT)                               # (P @ kbe)^T     [D, C]
        intraT = nl.multiply(_mm(kT, qT), decayT)          # [j, i] = k_j.q_i decay_ij
        qgT = _tr(nl.multiply(qc, eg))                     # [D, C]
        kdec = nl.multiply(kc, nl.exp(nl.subtract(Gl, gcum)))
        el = nl.exp(Gl)                                    # [D,1] (D == C partitions)
        # state-dependent part
        vn = nl.subtract(u, _mm(wT, S))                    # u - w @ S
        oc = nl.add(_mm(qgT, S), _mm(intraT, vn))
        S[...] = nl.add(nl.multiply(S, el), _mm(kdec, vn))
        nl.store(o[h, c * C:(c + 1) * C, :], value=oc)
    nl.store(Sf[h], value=S)
    return o, Sf


def ref_np(q, k, v, g, beta):
    """fp64 recurrent reference. q,k,v [H,T,D], g,beta [H,T,1]."""
    H, T, Dd = q.shape
    S = np.zeros((H, Dd, Dd)); out = np.zeros((H, T, Dd))
    for t in range(T):
        S = S * np.exp(g[:, t, 0])[:, None, None]
        kt = k[:, t]; pred = np.einsum('hk,hkv->hv', kt, S)
        S = S + np.einsum('hk,hv->hkv', kt, (v[:, t] - pred) * beta[:, t, 0][:, None])
        out[:, t] = np.einsum('hk,hkv->hv', q[:, t], S)
    return out


@nki.jit
def gdn_kernel2(q, k, v, g, beta, cst, S0):
    """Two-phase version: phase A (affine over chunks, pipelinable) computes every state-independent tile into SBUF;
    phase B (sequential) does only the 4 state matmuls per chunk. Same math and outputs as gdn_kernel."""
    H, T, _ = q.shape
    n = T // C
    o = nl.ndarray((H, T, D), dtype=nl.float32, buffer=nl.shared_hbm)
    Sf = nl.ndarray((H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    h = nl.program_id(0)
    U = nl.load(cst[0]); incl = nl.load(cst[1]); strict = nl.load(cst[2]); inclT = nl.load(cst[3])
    ones = nl.load(cst[4]); eye = nl.load(cst[5]); m1 = nl.load(cst[12])
    masks = nl.ndarray((6, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    for lv in nl.static_range(6):
        masks[lv] = nl.load(cst[6 + lv])
    ub = nl.ndarray((n, nl.par_dim(C), D), dtype=nl.float32, buffer=nl.sbuf)
    wTb = nl.ndarray((n, nl.par_dim(D), C), dtype=nl.float32, buffer=nl.sbuf)
    qgTb = nl.ndarray((n, nl.par_dim(D), C), dtype=nl.float32, buffer=nl.sbuf)
    iTb = nl.ndarray((n, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    kdb = nl.ndarray((n, nl.par_dim(C), D), dtype=nl.float32, buffer=nl.sbuf)
    elb = nl.ndarray((n, nl.par_dim(D), 1), dtype=nl.float32, buffer=nl.sbuf)
    for c in nl.affine_range(n):
        kc = nl.load(k[h, c * C:(c + 1) * C, :])
        qc = nl.load(q[h, c * C:(c + 1) * C, :])
        vc = nl.load(v[h, c * C:(c + 1) * C, :])
        gc = nl.load(g[h, c * C:(c + 1) * C, :])
        bc = nl.load(beta[h, c * C:(c + 1) * C, :])
        gcum = _mm(U, gc)
        Gb = _mm(nl.multiply(ones, gc), U)
        Gl = _mm(ones, gc)
        dg = nl.subtract(gcum, Gb)
        decay = nl.multiply(nl.exp(nl.multiply(dg, incl)), incl)
        decayT = nl.multiply(nl.exp(nl.multiply(nl.negative(dg), inclT)), inclT)
        eg = nl.exp(gcum)
        kT = _tr(kc); qT = _tr(qc)
        KK = _mm(kT, kT)
        L = nl.multiply(nl.multiply(nl.multiply(KK, decay), strict), bc)
        D0 = nl.subtract(eye, nl.multiply(L, m1))
        D0T = _tr(D0)
        D1, D1T = _level(L, masks, 0, D0, D0T)
        D2, D2T = _level(L, masks, 1, D1, D1T)
        D3, D3T = _level(L, masks, 2, D2, D2T)
        D4, D4T = _level(L, masks, 3, D3, D3T)
        D5, D5T = _level(L, masks, 4, D4, D4T)
        Dinv, DinvT = _level(L, masks, 5, D5, D5T)
        vb = nl.multiply(vc, bc)
        kbe = nl.multiply(nl.multiply(kc, bc), eg)
        ub[c] = _mm(DinvT, vb)
        wTb[c] = _mm(kbe, DinvT)
        iTb[c] = nl.multiply(_mm(kT, qT), decayT)
        qgTb[c] = _tr(nl.multiply(qc, eg))
        kdb[c] = nl.multiply(kc, nl.exp(nl.subtract(Gl, gcum)))
        elb[c] = nl.exp(Gl)
    S = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
    S[...] = nl.load(S0[h])
    for c in nl.sequential_range(n):
        vn = nl.subtract(ub[c], _mm(wTb[c], S))
        oc = nl.add(_mm(qgTb[c], S), _mm(iTb[c], vn))
        S[...] = nl.add(nl.multiply(S, elb[c]), _mm(kdb[c], vn))
        nl.store(o[h, c * C:(c + 1) * C, :], value=oc)
    nl.store(Sf[h], value=S)
    return o, Sf



BF = nl.bfloat16


def _b(x):
    return nl.copy(x, dtype=BF)


def _trb(x):
    """bf16 PE transpose -> bf16 SBUF tile."""
    return nl.copy(nisa.nc_transpose(_b(x)), dtype=BF)


def _mmb(a, b):
    """bf16 operands (cast if needed), fp32 PSUM accumulate -> fp32 SBUF."""
    return nl.copy(nisa.nc_matmul(a, b), dtype=nl.float32)


def _level_b(L, masks, lv, Dinv, DinvT16):
    Lb = nl.multiply(L, masks[lv])
    Z = _mmb(_trb(Lb), _b(Dinv))                           # Lb @ Dinv      (bf16 operands, fp32 acc)
    Y = _mmb(DinvT16, _b(Z))                               # Dinv @ Z
    Dn = nl.subtract(Dinv, Y)                              # running inverse kept in fp32
    return Dn, _trb(Dn)


@nki.jit
def gdn_kernel3(q, k, v, g, beta, cst, S0):
    """kernel2 with bf16 operands (fp32 accumulation) for every state-independent matmul and transpose, as FLA does;
    the running inverse, the state S and the four state matmuls stay fp32."""
    H, T, _ = q.shape
    n = T // C
    o = nl.ndarray((H, T, D), dtype=nl.float32, buffer=nl.shared_hbm)
    Sf = nl.ndarray((H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    h = nl.program_id(0)
    U = nl.load(cst[0]); incl = nl.load(cst[1]); strict = nl.load(cst[2]); inclT = nl.load(cst[3])
    ones = nl.load(cst[4]); eye = nl.load(cst[5]); m1 = nl.load(cst[12])
    masks = nl.ndarray((6, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    for lv in nl.static_range(6):
        masks[lv] = nl.load(cst[6 + lv])
    ub = nl.ndarray((n, nl.par_dim(C), D), dtype=nl.float32, buffer=nl.sbuf)
    wTb = nl.ndarray((n, nl.par_dim(D), C), dtype=nl.float32, buffer=nl.sbuf)
    qgTb = nl.ndarray((n, nl.par_dim(D), C), dtype=nl.float32, buffer=nl.sbuf)
    iTb = nl.ndarray((n, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    kdb = nl.ndarray((n, nl.par_dim(C), D), dtype=nl.float32, buffer=nl.sbuf)
    elb = nl.ndarray((n, nl.par_dim(D), 1), dtype=nl.float32, buffer=nl.sbuf)
    for c in nl.affine_range(n):
        kc = nl.load(k[h, c * C:(c + 1) * C, :])
        qc = nl.load(q[h, c * C:(c + 1) * C, :])
        vc = nl.load(v[h, c * C:(c + 1) * C, :])
        gc = nl.load(g[h, c * C:(c + 1) * C, :])
        bc = nl.load(beta[h, c * C:(c + 1) * C, :])
        gcum = _mm(U, gc)
        Gb = _mm(nl.multiply(ones, gc), U)
        Gl = _mm(ones, gc)
        dg = nl.subtract(gcum, Gb)
        decay = nl.multiply(nl.exp(nl.multiply(dg, incl)), incl)
        decayT = nl.multiply(nl.exp(nl.multiply(nl.negative(dg), inclT)), inclT)
        eg = nl.exp(gcum)
        kT = _trb(kc); qT = _trb(qc)
        KK = _mmb(kT, kT)
        L = nl.multiply(nl.multiply(nl.multiply(KK, decay), strict), bc)
        D0 = nl.subtract(eye, nl.multiply(L, m1))
        D1, D1T = _level_b(L, masks, 0, D0, _trb(D0))
        D2, D2T = _level_b(L, masks, 1, D1, D1T)
        D3, D3T = _level_b(L, masks, 2, D2, D2T)
        D4, D4T = _level_b(L, masks, 3, D3, D3T)
        D5, D5T = _level_b(L, masks, 4, D4, D4T)
        Dinv, DinvT16 = _level_b(L, masks, 5, D5, D5T)
        vb = _b(nl.multiply(vc, bc))
        kbe = _b(nl.multiply(nl.multiply(kc, bc), eg))
        ub[c] = _mmb(DinvT16, vb)
        wTb[c] = _mmb(kbe, DinvT16)
        iTb[c] = nl.multiply(_mmb(kT, qT), decayT)
        qgTb[c] = _tr(nl.multiply(qc, eg))
        kdb[c] = nl.multiply(kc, nl.exp(nl.subtract(Gl, gcum)))
        elb[c] = nl.exp(Gl)
    S = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
    S[...] = nl.load(S0[h])
    for c in nl.sequential_range(n):
        vn = nl.subtract(ub[c], _mm(wTb[c], S))
        oc = nl.add(_mm(qgTb[c], S), _mm(iTb[c], vn))
        S[...] = nl.add(nl.multiply(S, elb[c]), _mm(kdb[c], vn))
        nl.store(o[h, c * C:(c + 1) * C, :], value=oc)
    nl.store(Sf[h], value=S)
    return o, Sf


if __name__ == '__main__':
    import sys
    rng = np.random.default_rng(0)
    H, T = int(sys.argv[1]) if len(sys.argv) > 1 else 2, int(sys.argv[2]) if len(sys.argv) > 2 else 256
    def l2(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    k = l2(rng.standard_normal((H, T, D)) + 2 * rng.standard_normal((H, 1, D))).astype(np.float32)
    q = (l2(rng.standard_normal((H, T, D))) * D ** -0.5).astype(np.float32)
    v = rng.standard_normal((H, T, D)).astype(np.float32)
    g = (-rng.random((H, T, 1)) * 0.5).astype(np.float32); beta = rng.random((H, T, 1)).astype(np.float32)
    KER = {'2': gdn_kernel2, '3': gdn_kernel3}.get(sys.argv[3] if len(sys.argv) > 3 else '1', gdn_kernel)
    o, _ = nki.simulate_kernel(KER[H], q, k, v, g, beta, consts_np(), np.zeros((H, D, D), np.float32))
    r = ref_np(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), g.astype(np.float64), beta.astype(np.float64))
    print('sim max abs err', float(np.abs(o - r).max()), 'ref absmax', float(np.abs(r).max()))


