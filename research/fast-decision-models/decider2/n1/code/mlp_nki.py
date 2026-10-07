"""mlp_nki.py -- fused hobson MLP block on one NeuronCore:  out = x + down(silu(h Wg) * (h Wu)),  h = rmsnorm(x) * w1.

Layout ("transpose once"):  x token-major [T, 2048] bf16 -> RMSNorm on token-major tiles (free-axis reduction) ->
16 PE transposes per 128 tokens -> h^T [2048, T] (feature-major, SBUF-resident) -> gate/up GEMM with the WEIGHTS stationary and
h^T moving (N <= 512 tokens per instruction), output feature-major a^T = silu(g) * u [6144, T] in SBUF -> down GEMM with a^T
stationary and Wd moving (N = 512 output features), output token-major in PSUM -> + x -> bf16 -> HBM.
No transposes of activations other than the 16 per 128 tokens of h, no weight transposes (weights are pre-tiled on the host:
WguP [96, 128, 16, 128] = n-tile, k-row-in-tile, k-tile, n-col), each weight byte read from HBM once per token super-block."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

BF = nl.bfloat16
F32 = nl.float32
P = 128
TS_MAX = 768
EPS = 1e-6
GRID = {'mlp_kernel': 0}
OPT = {'w2d': True, 'hT_perk': False}


def blocks(T, size):
    out, t = [], 0
    while t < T:
        n = min(size, T - t); out.append((t, n)); t += n
    return out


def tile_wgu(WguT):
    """WguT [2048, 12288] (numpy or torch) -> [96*128, 16*128] (row = n_tile*128 + k_row, col = k_tile*128 + n_col).
    2-D on purpose: 4-D HBM inputs to an NKI kernel inside a torch_neuronx graph were read wrongly (measured)."""
    K, N = WguT.shape
    if isinstance(WguT, np.ndarray):
        return np.ascontiguousarray(WguT.reshape(K // P, P, N // P, P).transpose(2, 1, 0, 3).reshape(N, K))
    return WguT.reshape(K // P, P, N // P, P).permute(2, 1, 0, 3).reshape(N, K).contiguous()


def _mlp_body(x, w1, WguP, Wd, eye, out, stage):
    """x [T, 2048] bf16 (T % 128 == 0); w1 [1, 2048] fp32 (1 + norm weight); WguP [96, 128, 16, 128] bf16; Wd [6144, 2048] bf16;
    eye [128, 128] bf16 identity."""
    T, DM = x.shape
    FF = Wd.shape[0]
    KT = DM // P
    FT = FF // P

    ones1 = nl.ndarray((1, P), dtype=F32, buffer=nl.sbuf)
    ones1[...] = nisa.memset((1, P), 1.0, dtype=F32)
    epsc = nl.ndarray((P, 1), dtype=F32, buffer=nl.sbuf)
    epsc[...] = nisa.memset((P, 1), EPS, dtype=F32)
    w1r = nl.load(w1)
    w1b = nl.ndarray((P, DM), dtype=F32, buffer=nl.sbuf)
    for j in nl.static_range(DM // 512):
        pw = nl.ndarray((P, 512), dtype=F32, buffer=nl.psum)
        pw[...] = nisa.nc_matmul(ones1, w1r[:, j * 512:(j + 1) * 512], is_stationary_onezero=True)
        w1b[:, j * 512:(j + 1) * 512] = nisa.tensor_copy(pw)
    Ib = nl.load(eye)

    for (s0, SB) in blocks(T, TS_MAX):
        nt = SB // P
        hT = nl.ndarray((P, KT, SB), dtype=BF, buffer=nl.sbuf)
        # ---- RMSNorm (token-major) + transpose to h^T ----
        for tt in nl.static_range(nt):
            xt = nl.load(x[s0 + tt * P:s0 + (tt + 1) * P, :])
            ss = nl.ndarray((P, 1), dtype=F32, buffer=nl.sbuf)
            sq = nisa.activation_reduce(op=nl.square, data=xt, reduce_op=nl.add, reduce_res=ss)
            rs = nisa.activation(op=nl.rsqrt, data=ss, scale=1.0 / DM, bias=epsc)
            h = nisa.scalar_tensor_tensor(data=xt, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=w1b, dtype=BF)
            if stage == 'h':
                nl.store(out[s0 + tt * P:s0 + (tt + 1) * P, :], value=h)
            for k4 in nl.static_range(KT // 4):
                pT = nl.ndarray((P, 4, P), dtype=F32, buffer=nl.psum)
                for j in nl.static_range(4):
                    kk = k4 * 4 + j
                    pT[:, j, :] = nisa.nc_matmul(h[:, kk * P:(kk + 1) * P], Ib, is_moving_onezero=True)
                if OPT['hT_perk']:
                    for j in nl.static_range(4):
                        hT[:, k4 * 4 + j, tt * P:(tt + 1) * P] = nisa.tensor_copy(pT[:, j, :], dtype=BF)
                else:
                    hT[:, k4 * 4:(k4 + 1) * 4, tt * P:(tt + 1) * P] = nisa.tensor_copy(pT, dtype=BF)
        if stage == 'h':
            continue
        # ---- gate/up: weights stationary, h^T moving -> a^T [f, t] ----
        aT = nl.ndarray((P, FT, SB), dtype=BF, buffer=nl.sbuf)
        for ft in nl.affine_range(FT):
            if OPT['w2d']:                                           # WguP given as [96*128, 16*128]
                wg2 = nl.load(WguP[ft * P:(ft + 1) * P, :])
                wu2 = nl.load(WguP[(FT + ft) * P:(FT + ft + 1) * P, :])
            else:
                wg = nl.load(WguP[ft])                               # [P, KT, P]
                wu = nl.load(WguP[FT + ft])
            for (b0, BN) in blocks(SB, 512):
                pg = nl.zeros((P, BN), dtype=F32, buffer=nl.psum)
                pu = nl.zeros((P, BN), dtype=F32, buffer=nl.psum)
                for kk in nl.affine_range(KT):
                    if OPT['w2d']:
                        pg += nisa.nc_matmul(wg2[:, kk * P:(kk + 1) * P], hT[:, kk, b0:b0 + BN])
                    else:
                        pg += nisa.nc_matmul(wg[:, kk, :], hT[:, kk, b0:b0 + BN])
                for kk in nl.affine_range(KT):
                    if OPT['w2d']:
                        pu += nisa.nc_matmul(wu2[:, kk * P:(kk + 1) * P], hT[:, kk, b0:b0 + BN])
                    else:
                        pu += nisa.nc_matmul(wu[:, kk, :], hT[:, kk, b0:b0 + BN])
                sg = nisa.activation(op=nl.silu, data=pg)
                aT[:, ft, b0:b0 + BN] = nisa.tensor_tensor(sg, pu, op=nl.multiply, dtype=BF)
        if stage == 'aT':     # first 2048 features of a^T, back to token-major
            for kk in nl.static_range(KT):
                for tt in nl.static_range(nt):
                    pT = nl.ndarray((P, P), dtype=F32, buffer=nl.psum)
                    pT[...] = nisa.nc_matmul(aT[:, kk, tt * P:(tt + 1) * P], Ib, is_moving_onezero=True)
                    nl.store(out[s0 + tt * P:s0 + (tt + 1) * P, kk * P:(kk + 1) * P], value=nisa.tensor_copy(pT, dtype=BF))
        # ---- down: a^T stationary, Wd moving -> token-major, + x ----
        for ob in nl.static_range(DM // 512 if stage == 'out' else 0):
            pds = []
            for tt in nl.static_range(nt):
                pds.append(nl.zeros((P, 512), dtype=F32, buffer=nl.psum))
            for ft in nl.affine_range(FT):
                wd = nl.load(Wd[ft * P:(ft + 1) * P, ob * 512:(ob + 1) * 512])
                for tt in nl.static_range(nt):
                    pds[tt] += nisa.nc_matmul(aT[:, ft, tt * P:(tt + 1) * P], wd)
            for tt in nl.static_range(nt):
                xr = nl.load(x[s0 + tt * P:s0 + (tt + 1) * P, ob * 512:(ob + 1) * 512])
                ot = nisa.tensor_tensor(pds[tt], xr, op=nl.add, dtype=BF)
                nl.store(out[s0 + tt * P:s0 + (tt + 1) * P, ob * 512:(ob + 1) * 512], value=ot)


@nki.jit
def mlp_kernel(x, w1, WguP, Wd, eye):
    out = nl.ndarray(x.shape, dtype=BF, buffer=nl.shared_hbm)
    _mlp_body(x, w1, WguP, Wd, eye, out, 'out')
    return out


@nki.jit
def mlp_kernel_h(x, w1, WguP, Wd, eye):
    out = nl.ndarray(x.shape, dtype=BF, buffer=nl.shared_hbm)
    _mlp_body(x, w1, WguP, Wd, eye, out, 'h')
    return out


@nki.jit
def mlp_kernel_aT(x, w1, WguP, Wd, eye):
    out = nl.ndarray(x.shape, dtype=BF, buffer=nl.shared_hbm)
    _mlp_body(x, w1, WguP, Wd, eye, out, 'aT')
    return out


def ref_np(x, w1, WguT, Wd):
    xf = x.astype(np.float32)
    h = xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + EPS) * w1
    h = h.astype(np.float32)
    gu = h @ WguT.astype(np.float32)
    I = gu.shape[1] // 2
    g, u = gu[:, :I], gu[:, I:]
    a = g / (1 + np.exp(-g)) * u
    return xf + a @ Wd.astype(np.float32)
