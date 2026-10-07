"""gdn4.py -- pipelined multi-head exact NKI kernel for the chunked gated delta rule (C = D = 128, fp32 math).

Why: J8's kernel ran one dependent chain per head (~35 TensorE ops per chunk, 14 transposes, each followed by a PSUM->SBUF
copy); engines were 19-36% busy.  This kernel:
  * runs ALL heads in one program and stacks G=4 heads along the free axis, so every PSUM->SBUF copy / mask / elementwise op
    covers 4 heads ([128, 512] = one PSUM bank) and the 4 per-head matmuls are back-to-back on TensorE;
  * takes q, k, v token-major [B, T, H, 128] (the projection's natural layout; no XLA transposes), only 2 PE transposes
    per head-chunk (k^T, q^T), everything else is arranged by operand order ((AB)^T = B^T A^T);
  * builds exp-decay tiles with one K=1 broadcast matmul + per-partition ACT bias (no gcum matmuls, no [C,C] subtracts);
  * inverts (I + L) by exact block doubling (s = 1..64) with the running inverse M and M^T held in PSUM accumulators
    (no transposes, the mask is applied once per level at the eviction of L M);
  * reformulates the serial part as S_{c+1} = A_c S_c + B_c, A_c = e^{gl} I - kdec^T w, B_c = kdec^T u, and
    o_c = Q'_c S_c + O0_c with Q' = qg - intra w, O0 = intra u.  All of A, B, Q', O0 are state-independent, so the only
    serial chain per chunk is ONE fp32 matmul (+ accumulate) and one copy per 4 heads.

Math per head and chunk (gc = in-chunk cumsum of g, gl = gc[-1]):
  decay_ij = exp(gc_i - gc_j) (i >= j);  L_ij = beta_i k_i.k_j decay_ij (i > j);  P = (I + L)^-1
  u = P (beta v);  w = P (beta e^gc k);  kdec = e^(gl - gc) k;  qg = e^gc q;  intra_ij = q_i.k_j decay_ij (i >= j)
  o = (qg - intra w) S + intra u;   S' = (e^gl I - kdec^T w) S + kdec^T u
nc_matmul(stat, mov) = stat^T @ mov, contraction over partitions.
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

C = 128
D = 128
G = 4
BIG = 16384.0
LOGB_FLOOR = -1.0e4
NCOL = 6          # per-token columns: beta, e^gc, -beta e^gc, e^(gl-gc), -gc, beta e^g (g not cumulated)
LEVELS = (1, 2, 4, 8, 16, 32, 64)
F32 = nl.float32
BF = nl.bfloat16
GRID = {'gdn4': 0, 'gdn5': 0, 'gdn6': 0, 'gdn7': 0, 'gdn8': 0}
STATIC_MAX = 0    # always sequential loops (measured as fast per chunk as unrolled, and compiles in seconds)
RAW_QK = True     # q, k arrive un-normalised (after conv+SiLU); the kernel does l2norm and the 1/sqrt(128) q scale


def consts_np():
    """[2 + 7, 128, 128] fp32: 0 identity; 1 decay mask add in [j, i] layout (0 if i >= j else -BIG);
    2..8 negated block masks per level s in L layout [i, c]: -1 where i is in the odd half and c in the even half of the same
    2s block."""
    ar = np.arange(C)
    i, j = ar[:, None], ar[None, :]
    out = [np.eye(C), np.where(j >= i, 0.0, -BIG)]          # rows = j (partition), cols = i
    for s in LEVELS:
        m = ((i // (2 * s)) == (j // (2 * s))) & ((i // s) % 2 == 1) & ((j // s) % 2 == 0)
        out.append(-m.astype(np.float64))
    out.append(out[2].T.copy())                                   # 9: level-1 mask transposed (for M_2^T from L^T)
    return np.stack(out).astype(np.float32)


def prep_np(q, k, v, g, beta, S0):
    """J8-style [H, T, D] / [H, T, 1] / [H, D, D] numpy inputs -> kernel args (what the XLA side computes)."""
    H, T, _ = q.shape
    n = T // C
    NG = H // G
    gc = g[..., 0].reshape(H, n, C).astype(np.float64).cumsum(-1)          # [H, n, C]
    gl = gc[..., -1:]
    b = beta[..., 0].reshape(H, n, C).astype(np.float64)
    eg = np.exp(gc)
    ge = np.exp(g[..., 0].reshape(H, n, C).astype(np.float64))
    cols = np.stack([b, eg, -b * eg, np.exp(gl - gc), -gc, b * ge], -1)       # [H, n, C, 6]
    cols = cols.reshape(H, T, NCOL).transpose(1, 0, 2)[None]                  # [1, T, H, 5]
    logb = np.maximum(np.log(np.maximum(b, 1e-38)), LOGB_FLOOR)
    el = np.broadcast_to(np.exp(gl), gc.shape)
    rows = np.stack([gc, gc + logb, el], 0)                                   # [3, H, n, C]
    rows = rows.reshape(3, NG, G, n, C).transpose(0, 1, 3, 2, 4)                             # [3, NG, n, G, C]
    tm = lambda x: np.ascontiguousarray(x.transpose(1, 0, 2)[None]).astype(np.float32)      # [1, T, H, D]
    S0d = np.ascontiguousarray(S0.transpose(1, 0, 2)[None]).astype(np.float32)              # [1, D, H, Dv]
    return (tm(q * np.sqrt(D)), tm(k), tm(v), np.ascontiguousarray(cols).astype(np.float32), np.ascontiguousarray(rows).astype(np.float32),
            consts_np(), S0d)


def post_np(o, Sf):
    return o[0].transpose(1, 0, 2), Sf[0].transpose(1, 0, 2)


prep = prep_np
post = post_np


def alg_np(q, k, v, g, beta, S0):
    """float64 numpy implementation of exactly the kernel's algorithm (debug reference)."""
    H, T, _ = q.shape
    n = T // C
    cst = consts_np().astype(np.float64)
    out = np.zeros((H, T, D)); Sf = np.zeros((H, D, D))
    for h in range(H):
        S = S0[h].astype(np.float64)
        for c in range(n):
            sl = slice(c * C, (c + 1) * C)
            qc, kc, vc = (x[h, sl].astype(np.float64) for x in (q, k, v))
            gcm = np.cumsum(g[h, sl, 0].astype(np.float64)); gl = gcm[-1]; b = beta[h, sl, 0].astype(np.float64)
            logb = np.maximum(np.log(np.maximum(b, 1e-38)), LOGB_FLOOR)
            decT = np.exp(gcm[None, :] - gcm[:, None] + cst[1])                   # [j, i]
            decBT = np.exp(gcm[None, :] + logb[None, :] - gcm[:, None] + cst[1])
            Lt = (kc @ kc.T) * decBT                                             # [j, i]
            iT = (kc @ qc.T) * decT
            L = Lt.T
            M = np.eye(C); MT = np.eye(C)
            for lv, s in enumerate(LEVELS):
                T1n = (L @ M) * cst[2 + lv]
                if s < 64:
                    M = M + M @ T1n
                MT = MT + T1n.T @ MT
            P = MT.T
            u = P @ (vc * b[:, None]); wn = P @ (-kc * (b * np.exp(gcm))[:, None])
            kd = kc * np.exp(gl - gcm)[:, None]; qg = qc * np.exp(gcm)[:, None]
            AT = wn.T @ kd + np.exp(gl) * np.eye(C)
            QT = wn.T @ iT + qg.T
            out[h, sl] = QT.T @ S + iT.T @ u
            S = AT.T @ S + kd.T @ u
        Sf[h] = S
    return out, Sf


def _rsqrt_nr(ss, sc, b, bias_tile):
    """rsqrt(ss * sc + b) on ScalarE, then one Newton step on VectorE (ScalarE's rsqrt is only ~1e-5 accurate)."""
    y = nisa.activation(op=nl.rsqrt, data=ss, scale=sc, bias=bias_tile)
    a = nisa.tensor_scalar(data=ss, op0=nl.multiply, operand0=sc, op1=nl.add, operand1=b, engine=nisa.vector_engine)
    y2 = nisa.tensor_tensor(y, y, op=nl.multiply)
    t = nisa.tensor_tensor(a, y2, op=nl.multiply)
    f = nisa.tensor_scalar(data=t, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5, engine=nisa.vector_engine)
    return nisa.tensor_tensor(y, f, op=nl.multiply)


def _rep4(t, dtype=F32):
    """[128,128] SBUF tile -> [128, G, 128] tile with G copies."""
    r = nl.ndarray((C, G, C), dtype=dtype, buffer=nl.sbuf)
    for h in nl.static_range(G):
        r[:, h, :] = nisa.tensor_copy(t, dtype=dtype)
    return r


DBG_NAMES = ['kT', 'Lt', 'iT', 'M2', 'MT2', 'PT', 'U', 'W', 'AT', 'QT', 'S1', 'o0', 'dT', 'dbT', 'T1_0']


def _dbg(dbg, name, t, first):
    if dbg is not None and first:
        nl.store(dbg[DBG_NAMES.index(name)], value=t)


@nki.jit
def gdn4_dbg(q, k, v, cols, rows, cst, S0):
    B, T, H, _ = q.shape
    o = nl.ndarray((B, T, H, D), dtype=F32, buffer=nl.shared_hbm)
    Sf = nl.ndarray((B, D, H, D), dtype=F32, buffer=nl.shared_hbm)
    dbg = nl.ndarray((len(DBG_NAMES), C, G, C), dtype=F32, buffer=nl.shared_hbm)
    _body(q, k, v, cols, rows, cst, S0, o, Sf, dbg)
    return o, Sf, dbg


@nki.jit
def gdn8(q, k, v, cols, rows, cst, S0):
    B, T, H, _ = q.shape
    o = nl.ndarray((B, T, H, D), dtype=F32, buffer=nl.shared_hbm)
    Sf = nl.ndarray((B, D, H, D), dtype=F32, buffer=nl.shared_hbm)
    _body(q, k, v, cols, rows, cst, S0, o, Sf, None)
    return o, Sf


def _body(q, k, v, cols, rows, cst, S0, o, Sf, dbg):
    B, T, H, _ = q.shape
    n = T // C
    NG = H // G

    # ---- constants ----
    I = nl.load(cst[0])
    Ib = nisa.tensor_copy(I, dtype=BF)
    madd4 = _rep4(nl.load(cst[1]), BF)                         # exact in bf16 (0 / -16384)
    negm4 = []
    for lv in nl.static_range(len(LEVELS)):
        negm4.append(_rep4(nl.load(cst[2 + lv])))
    negm1T4 = _rep4(nl.load(cst[9]))
    I4 = _rep4(I)
    ones1 = nl.ndarray((1, C), dtype=F32, buffer=nl.sbuf)
    ones1[...] = nisa.memset((1, C), 1.0, dtype=F32)
    epsk = nl.ndarray((C, 1), dtype=F32, buffer=nl.sbuf)
    epsk[...] = nisa.memset((C, 1), 1e-6, dtype=F32)
    epsq = nl.ndarray((C, 1), dtype=F32, buffer=nl.sbuf)
    epsq[...] = nisa.memset((C, 1), 1e-6 * D, dtype=F32)

    STATIC = B * n <= STATIC_MAX
    if not STATIC:
        S = []
        for gg in nl.static_range(NG):
            S.append(nl.ndarray((D, G, D), dtype=F32, buffer=nl.sbuf))
    for bb in (nl.static_range(B) if STATIC else nl.sequential_range(B)):
        if STATIC:
            S = []
            for gg in nl.static_range(NG):
                St = nl.ndarray((D, G, D), dtype=F32, buffer=nl.sbuf)
                St[...] = nl.load(S0[bb, :, gg * G:(gg + 1) * G, :])
                S.append(St)
        else:
            for gg in nl.static_range(NG):
                S[gg][...] = nl.load(S0[bb, :, gg * G:(gg + 1) * G, :])
        for c in (nl.static_range(n) if STATIC else nl.sequential_range(n)):
            for gg in nl.static_range(NG):
                h0 = gg * G
                first = False
                q4r = nl.load(q[bb, c * C:(c + 1) * C, h0:h0 + G, :])         # [t, G, d] raw
                k4r = nl.load(k[bb, c * C:(c + 1) * C, h0:h0 + G, :])
                # ---- l2norm over d per (token, head); q also scaled by 1/sqrt(128) ----
                ssq = nl.ndarray((C, G), dtype=F32, buffer=nl.sbuf)
                ssk = nl.ndarray((C, G), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    jq = nisa.activation_reduce(op=nl.square, data=q4r[:, h, :], reduce_op=nl.add, reduce_res=ssq[:, h:h + 1])
                    jk = nisa.activation_reduce(op=nl.square, data=k4r[:, h, :], reduce_op=nl.add, reduce_res=ssk[:, h:h + 1])
                rq = _rsqrt_nr(ssq, float(D), 1e-6 * D, epsq)                                # rsqrt((ss + 1e-6) * 128)
                rk = _rsqrt_nr(ssk, 1.0, 1e-6, epsk)                                         # rsqrt(ss + 1e-6)
                q4 = nl.ndarray((C, G, D), dtype=F32, buffer=nl.sbuf)
                k4 = nl.ndarray((C, G, D), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    q4[:, h, :] = nisa.tensor_scalar(data=q4r[:, h, :], op0=nl.multiply, operand0=rq[:, h:h + 1], engine=nisa.vector_engine)
                    k4[:, h, :] = nisa.tensor_scalar(data=k4r[:, h, :], op0=nl.multiply, operand0=rk[:, h:h + 1], engine=nisa.vector_engine)
                v4 = nl.load(v[bb, c * C:(c + 1) * C, h0:h0 + G, :])
                cl = nl.load(cols[bb, c * C:(c + 1) * C, h0:h0 + G, :])       # [t, G, 6]
                rgi = bb * NG + gg
                rg = nl.load(rows[0, rgi, c:c + 1, :, :])                      # [1, G, C] gc_i
                rgb = nl.load(rows[1, rgi, c:c + 1, :, :])                     # [1, G, C] gc_i + log beta_i
                rel = nl.load(rows[2, rgi, c:c + 1, :, :])                     # [1, G, C] e^gl

                # ---- k^T, q^T ----
                pkT = nl.ndarray((D, G, C), dtype=F32, buffer=nl.psum)
                pqT = nl.ndarray((D, G, C), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pkT[:, h, :] = nisa.nc_matmul(k4[:, h, :], I, is_moving_onezero=True)
                for h in nl.static_range(G):
                    pqT[:, h, :] = nisa.nc_matmul(q4[:, h, :], I, is_moving_onezero=True)
                kT4 = nisa.tensor_copy(pkT, engine=nisa.scalar_engine)
                qT4 = nisa.tensor_copy(pqT, engine=nisa.scalar_engine)

                # ---- KK = k k^T, KQ[j,i] = k_j.q_i ----
                pKK = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                pKQ = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pKK[:, h, :] = nisa.nc_matmul(kT4[:, h, :], kT4[:, h, :])
                for h in nl.static_range(G):
                    pKQ[:, h, :] = nisa.nc_matmul(kT4[:, h, :], qT4[:, h, :])

                # ---- decay tiles in [j, i] layout + e^gc_i broadcast ----
                pR = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                pR[...] = nisa.nc_matmul(ones1, rg, is_stationary_onezero=True)
                pR[...] += nisa.nc_matmul(Ib, madd4)
                pRb = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                pRb[...] = nisa.nc_matmul(ones1, rgb, is_stationary_onezero=True)
                pRb[...] += nisa.nc_matmul(Ib, madd4)
                pRg = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                pRg[...] = nisa.nc_matmul(ones1, rg, is_stationary_onezero=True)
                dT4 = nl.ndarray((C, G, C), dtype=F32, buffer=nl.sbuf)
                dbT4 = nl.ndarray((C, G, C), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    dT4[:, h, :] = nisa.activation(op=nl.exp, data=pR[:, h, :], bias=cl[:, h, 4:5])
                for h in nl.static_range(G):
                    dbT4[:, h, :] = nisa.activation(op=nl.exp, data=pRb[:, h, :], bias=cl[:, h, 4:5])
                EG4 = nisa.activation(op=nl.exp, data=pRg)                                   # [d, (h, i)] = e^gc_i
                Lt4 = nisa.tensor_tensor(pKK, dbT4, op=nl.multiply)           # L^T (finite above; diag unused)
                iT4 = nisa.tensor_tensor(pKQ, dT4, op=nl.multiply)            # intra^T [j, i]
                qgT4 = nisa.tensor_tensor(qT4, EG4, op=nl.multiply)           # qg^T [d, i]
                _dbg(dbg, 'kT', kT4, first)
                _dbg(dbg, 'Lt', Lt4, first); _dbg(dbg, 'iT', iT4, first); _dbg(dbg, 'dT', dT4, first); _dbg(dbg, 'dbT', dbT4, first)

                # ---- (I + L)^-1 by block doubling (M and M^T in SBUF, deltas in fresh PSUM) ----
                # s = 1:  M_2 = I - L o m1,  L_{2t+1,2t} = beta e^g (2t+1) * KK[2t+1, 2t]  (KK symmetric -> no transpose)
                X4 = nl.ndarray((C, G, C), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    X4[:, h, :] = nisa.scalar_tensor_tensor(data=pKK[:, h, :], op0=nl.multiply, operand0=cl[:, h, 5:6],
                                                            op1=nl.multiply, operand1=negm4[0][:, h, :])
                M4 = nisa.tensor_tensor(X4, I4, op=nl.add)
                XT4 = nisa.tensor_tensor(Lt4, negm1T4, op=nl.multiply)
                MT4 = nisa.tensor_tensor(XT4, I4, op=nl.add)
                _dbg(dbg, 'T1_0', X4, first); _dbg(dbg, 'M2', M4, first); _dbg(dbg, 'MT2', MT4, first)
                for lv in nl.static_range(1, len(LEVELS)):
                    last = lv == len(LEVELS) - 1
                    pL1 = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                    for h in nl.static_range(G):
                        pL1[:, h, :] = nisa.nc_matmul(Lt4[:, h, :], M4[:, h, :])           # L M_s
                    T1 = nisa.tensor_tensor(pL1, negm4[lv], op=nl.multiply)               # -(L M_s) o mask_s
                    if not last:
                        pDM = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                        for h in nl.static_range(G):
                            pDM[:, h, :] = nisa.nc_matmul(MT4[:, h, :], T1[:, h, :])      # -M_s T1
                    pDMT = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                    for h in nl.static_range(G):
                        pDMT[:, h, :] = nisa.nc_matmul(T1[:, h, :], MT4[:, h, :])         # -T1^T M_s^T
                    if not last:
                        M4 = nisa.tensor_tensor(pDM, M4, op=nl.add)
                    MT4 = nisa.tensor_tensor(pDMT, MT4, op=nl.add)
                PT4 = MT4                                                                  # P^T
                _dbg(dbg, 'PT', PT4, first)

                # ---- [u | w_neg] = P [beta v | -beta e^gc k] ----
                R4 = nl.ndarray((C, G, 2 * D), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    R4[:, h, 0:D] = nisa.activation(op=nl.copy, data=v4[:, h, :], scale=cl[:, h, 0:1])
                    R4[:, h, D:2 * D] = nisa.activation(op=nl.copy, data=k4[:, h, :], scale=cl[:, h, 2:3])
                kd4 = nl.ndarray((C, G, D), dtype=F32, buffer=nl.sbuf)
                for h in nl.static_range(G):
                    kd4[:, h, :] = nisa.activation(op=nl.copy, data=k4[:, h, :], scale=cl[:, h, 3:4])
                pUW = nl.ndarray((C, G, 2 * D), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pUW[:, h, :] = nisa.nc_matmul(PT4[:, h, :], R4[:, h, :])
                UW4 = nisa.tensor_copy(pUW, engine=nisa.scalar_engine)
                _dbg(dbg, 'U', UW4[:, :, 0:D], first); _dbg(dbg, 'W', UW4[:, :, D:2 * D], first)

                # ---- e^gl I for 4 heads ----
                pEL = nl.ndarray((C, G, C), dtype=F32, buffer=nl.psum)
                pEL[...] = nisa.nc_matmul(ones1, rel, is_stationary_onezero=True)
                elI4 = nisa.tensor_tensor(pEL, I4, op=nl.multiply)

                # ---- A^T = e^gl I + w_neg^T kdec ; Q'^T = qg^T + w_neg^T intra^T ----
                pA = nl.ndarray((D, G, D), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pA[:, h, :] = nisa.nc_matmul(UW4[:, h, D:2 * D], kd4[:, h, :])
                pQ = nl.ndarray((D, G, C), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pQ[:, h, :] = nisa.nc_matmul(UW4[:, h, D:2 * D], iT4[:, h, :])
                AT4 = nisa.tensor_tensor(pA, elI4, op=nl.add)
                QT4 = nisa.tensor_tensor(pQ, qgT4, op=nl.add)
                _dbg(dbg, 'AT', AT4, first); _dbg(dbg, 'QT', QT4, first)

                # ---- state step and output: S' = A S + kdec^T u ; o = Q' S + intra u ----
                pS = nl.ndarray((D, G, D), dtype=F32, buffer=nl.psum)
                pO = nl.ndarray((C, G, D), dtype=F32, buffer=nl.psum)
                for h in nl.static_range(G):
                    pS[:, h, :] = nisa.nc_matmul(kd4[:, h, :], UW4[:, h, 0:D])
                    pS[:, h, :] += nisa.nc_matmul(AT4[:, h, :], S[gg][:, h, :])
                for h in nl.static_range(G):
                    pO[:, h, :] = nisa.nc_matmul(iT4[:, h, :], UW4[:, h, 0:D])
                    pO[:, h, :] += nisa.nc_matmul(QT4[:, h, :], S[gg][:, h, :])
                if STATIC:
                    Sn = nl.ndarray((D, G, D), dtype=F32, buffer=nl.sbuf)
                    Sn[...] = nisa.tensor_copy(pS, engine=nisa.scalar_engine)
                    S[gg] = Sn
                else:
                    S[gg][...] = nisa.tensor_copy(pS, engine=nisa.scalar_engine)
                    Sn = S[gg]
                o4 = nisa.tensor_copy(pO, engine=nisa.scalar_engine)
                _dbg(dbg, 'S1', Sn, first); _dbg(dbg, 'o0', o4, first)
                nl.store(o[bb, c * C:(c + 1) * C, h0:h0 + G, :], value=o4)
        for gg in nl.static_range(NG):
            nl.store(Sf[bb, :, gg * G:(gg + 1) * G, :], value=S[gg])


def alg_dbg(q, k, v, g, beta, S0):
    """intermediates of chunk 0, heads 0..3, in the kernel's stacked layouts [name][128, 4, 128]."""
    cst = consts_np().astype(np.float64)
    out = {n_: np.zeros((C, G, C)) for n_ in DBG_NAMES}
    for h in range(G):
        sl = slice(0, C)
        qc, kc, vc = (x[h, sl].astype(np.float64) for x in (q, k, v))
        gcm = np.cumsum(g[h, sl, 0].astype(np.float64)); gl = gcm[-1]; b = beta[h, sl, 0].astype(np.float64)
        logb = np.maximum(np.log(np.maximum(b, 1e-38)), LOGB_FLOOR)
        decT = np.exp(gcm[None, :] - gcm[:, None] + cst[1]); decBT = np.exp(gcm[None, :] + logb[None, :] - gcm[:, None] + cst[1])
        Lt = (kc @ kc.T) * decBT; iT = (kc @ qc.T) * decT; L = Lt.T
        M = np.eye(C); MT = np.eye(C)
        for lv, s in enumerate(LEVELS):
            T1n = (L @ M) * cst[2 + lv]
            if lv == 0: out['T1_0'][:, h] = T1n
            if s < 64: M = M + M @ T1n
            MT = MT + T1n.T @ MT
            if lv == 0: out['M2'][:, h] = M; out['MT2'][:, h] = MT
        P = MT.T
        u = P @ (vc * b[:, None]); wn = P @ (-kc * (b * np.exp(gcm))[:, None])
        kd = kc * np.exp(gl - gcm)[:, None]; qg = qc * np.exp(gcm)[:, None]
        AT = wn.T @ kd + np.exp(gl) * np.eye(C); QT = wn.T @ iT + qg.T
        S = S0[h].astype(np.float64)
        for nm, val in [('kT', kc.T), ('Lt', Lt), ('iT', iT), ('PT', MT), ('U', u), ('W', wn), ('AT', AT), ('QT', QT),
                        ('S1', AT.T @ S + kd.T @ u), ('o0', QT.T @ S + iT.T @ u), ('dT', decT), ('dbT', decBT)]:
            out[nm][:, h] = val
    return out
