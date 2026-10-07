"""W8A8 int8 GEMM with fused dequant epilogue (per-token act scale x per-out-channel weight scale -> fp16/bf16 out), Triton, plus
a fused per-token activation-quant kernel (what a norm kernel would emit). Benchmarks vs cuBLAS bf16 and torch._int_mm on 2B shapes."""
import torch, triton, triton.language as tl, statistics, sys, json, os

@triton.jit
def _i8mm(A, W, SA, SW, C, M, N, K, sam, sak, swn, swk, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN); npg = GROUP_M * npn
    gid = pid // npg; fm = gid * GROUP_M; gsm = min(npm - fm, GROUP_M)
    pm = fm + ((pid % npg) % gsm); pn = (pid % npg) // gsm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + rm[:, None] * sam + rk[None, :] * sak
    b_ptr = W + rk[:, None] * swk + rn[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0)
        b = tl.load(b_ptr, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptr += BK * sak; b_ptr += BK * swk
    sa = tl.load(SA + rm, mask=rm < M, other=0.)
    sw = tl.load(SW + rn, mask=rn < N, other=0.)
    out = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _quant_rows(X, Q, S, K, BK: tl.constexpr):
    r = tl.program_id(0); cols = tl.arange(0, BK)
    x = tl.load(X + r * K + cols, mask=cols < K, other=0.).to(tl.float32)
    s = tl.max(tl.abs(x), 0) / 127.0 + 1e-12
    q = tl.extra.cuda.libdevice.rint(x / s).to(tl.int8)
    tl.store(Q + r * K + cols, q, mask=cols < K); tl.store(S + r, s)

def quant_rows(x):
    M, K = x.shape; q = torch.empty((M, K), device=x.device, dtype=torch.int8); s = torch.empty(M, device=x.device, dtype=torch.float32)
    _quant_rows[(M,)](x, q, s, K, BK=triton.next_power_of_2(K), num_warps=8); return q, s

def i8mm(q, sa, wq, sw, cfg):
    M, K = q.shape; N = wq.shape[0]; c = torch.empty((M, N), device=q.device, dtype=torch.bfloat16)
    BM, BN, BK, nw, ns = cfg
    _i8mm[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](q, wq, sa, sw, c, M, N, K, q.stride(0), q.stride(1), wq.stride(0), wq.stride(1), c.stride(0), c.stride(1),
                                                     BM=BM, BN=BN, BK=BK, GROUP_M=8, num_warps=nw, num_stages=ns)
    return c

if __name__ == "__main__":
    def tm(fn, reps=40):
        for _ in range(5): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
        ts.sort(); return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]
    SH = {"gdn_in": (8224, 2048), "attn_in": (5120, 2048), "o/out": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
    CF = [(128, 128, 64, 4, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (64, 128, 128, 4, 3), (64, 64, 128, 4, 4), (32, 128, 128, 4, 4), (16, 128, 128, 4, 4), (256, 128, 64, 8, 3)]
    out = []
    for name, (N, K) in SH.items():
        w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        sw = (w.float().abs().amax(1) / 127).contiguous(); wq = torch.round(w.float() / sw[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
        wqt = wq.t().contiguous().t()  # for _int_mm (needs column-major B)
        for M in (64, 256, 1024, 4096):
            x = torch.randn(M, K, device="cuda").bfloat16()
            r = dict(shape=name, M=M)
            r["bf16"] = tm(lambda: x @ w.t())[0]
            q, sa = quant_rows(x)
            r["quant_kernel"] = tm(lambda: quant_rows(x))[0]
            r["int_mm_only"] = tm(lambda: torch._int_mm(q, wq.t()))[0]
            best = None
            for c in CF:
                try: t_ = tm(lambda: i8mm(q, sa, wq, sw, c))[0]
                except Exception: continue
                if best is None or t_ < best[0]: best = (t_, c)
            r["tri_i8mm_fused_dequant"] = best[0]; r["cfg"] = best[1]
            # accuracy check vs bf16
            ref = (x.float() @ w.float().t()); got = i8mm(q, sa, wq, sw, best[1]).float()
            r["relerr"] = ((got - ref).norm() / ref.norm()).item()
            out.append(r)
            print(f"{name:8s} M={M:5d} bf16 {r['bf16']:.0f}us | int8 GEMM+dequant (Triton) {r['tri_i8mm_fused_dequant']:.0f}us ({r['bf16']/r['tri_i8mm_fused_dequant']:.2f}x) | +quant kernel {r['quant_kernel']:.0f}us -> "
                  f"{r['bf16']/(r['tri_i8mm_fused_dequant']+r['quant_kernel']):.2f}x | torch._int_mm alone {r['int_mm_only']:.0f}us | relerr {r['relerr']:.4f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "int8mm.json"), "w"), indent=1)
