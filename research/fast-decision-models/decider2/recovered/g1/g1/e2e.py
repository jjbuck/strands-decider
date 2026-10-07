"""G1 end-to-end latency: hobson-shaped fused runtime (d1 lean2 glue kernels) with BTT projections.
  python e2e.py OUT.jsonl T1,T2 CFG [CFG ...]
  CFG = dense_fold (d1's full fusion, the 52.7 ms reference) | dense (same glue as the BTT runtime: addrms,gnorm,silu,prep,conv; cuBLAS GEMMs)
        | WHICH:b:frac   (WHICH = mlp | all; BTT b_in=b_out=b at FLOP fraction frac per matrix; random factors -- timing does not depend on values)
Timing: CUDA graph of one forward, fresh token ids copied in per call (pinned), 3 warm + 20 timed, median / p95 (d1 prof_d1.wall)."""
import os, sys, json, math, time, statistics as st, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/work/systems/g")); sys.path.insert(0, os.path.expanduser("~/work/d1"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean as LM
import lean2 as L2
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import btt as B
import slib as S

GLUE = "addrms,gnorm,silu,prep,conv"


def btt_into(x, R, L, out):
    """BTT with the stage-2 output written into a (possibly column-sliced) 2-D view `out`."""
    M = x.shape[0]; b_out, kp, N2 = L.shape
    z = B.btt1(x, R, b_out)
    b_in = R.shape[0]; r = R.shape[1] // b_out
    zv = z.view(M, b_out, b_in * r).transpose(0, 1)
    yv = out.as_strided((b_out, M, kp), (kp, out.stride(0), 1), out.storage_offset())
    torch.bmm(zv, L.transpose(1, 2), out=yv)
    return out


def mk(n_in, n_out, b, frac, bo_mult=1, dev="cuda"):
    r = S.btt_r(n_in, n_out // bo_mult, b, frac)
    r = max(8, int(round(r / 8)) * 8)
    return B.init_factors(n_in, n_out, b, b * bo_mult, r), r


class StructRT(L2.Lean2):
    def __init__(self, torso, which, b, frac):
        super().__init__(torso, fuse=GLUE)
        self.which, self.b, self.frac = which, b, frac
        mac = 0
        for d in self.layers:
            (d["Rgu"], d["Lgu"]), r1 = mk(2048, 12288, b, frac, bo_mult=2)
            (d["Rd"], d["Ld"]), r2 = mk(6144, 2048, b, frac)
            mac += r1 * (2 * b * 2048 + b * 12288) + r2 * (b * 6144 + b * 2048)
            if which == "all":
                (d["Ro"], d["Lo"]), r3 = mk(2048, 2048, b, frac); mac += r3 * 2 * b * 4096
                if d["type"] == "linear_attention":
                    (d["Rqkv"], d["Lqkv"]), r4 = mk(2048, 6144, b, frac)
                    (d["Rz"], d["Lz"]), r5 = mk(2048, 2048, b, frac)
                    d["Wba"] = d["Win"][8192:].contiguous()
                    mac += r4 * b * 8192 + r5 * b * 4096 + 2048 * 32
                else:
                    (d["Rq"], d["Lq"]), r6 = mk(2048, 4096, b, frac)
                    d["Wkv"] = d["Win"][4096:].contiguous()
                    mac += r6 * b * 6144 + 2048 * 1024
            else:
                mac += d["Win"].numel() + d["Wo"].numel()
        self.macs = mac

    @torch.no_grad()
    def forward(self, ids):
        B_, T = ids.shape
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        h = LM._rms_zc(x, self.layers[0]["in_norm"], self.eps)
        n = len(self.layers); al = self.which == "all"
        for i, d in enumerate(self.layers):
            lin = d["type"] == "linear_attention"
            if al:
                proj = torch.empty(T, 8224 if lin else 5120, device=x.device, dtype=x.dtype)
                if lin:
                    btt_into(h, d["Rqkv"], d["Lqkv"], proj[:, :6144]); btt_into(h, d["Rz"], d["Lz"], proj[:, 6144:8192])
                    proj[:, 8192:] = h @ d["Wba"].t()
                else:
                    btt_into(h, d["Rq"], d["Lq"], proj[:, :4096]); proj[:, 4096:] = h @ d["Wkv"].t()
            else:
                proj = h @ d["Win"].t()
            if lin:
                qkv3 = L2.conv_l2(proj, d["conv_w"])
                q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                a = proj[:, 8208:8224].reshape(1, T, 16); bb = proj[:, 8192:8208].reshape(1, T, 16)
                o, _ = chunk_gated_delta_rule(q, k, v, a, bb, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                              A_log=d["A_log"], dt_bias=d["dt_bias"], use_beta_sigmoid_in_kernel=True)
                o = L2.gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d["gn_w"], self.eps)
            else:
                q, k, gate = L2.attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
                o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            if al:
                d_out = btt_into(o.contiguous(), d["Ro"], d["Lo"], torch.empty(T, 2048, device=x.device, dtype=x.dtype))
            else:
                d_out = o @ d["Wo"].t()
            x, h2 = L2.add_rms(x, d_out, d["post1"], self.eps)
            gu = btt_into(h2, d["Rgu"], d["Lgu"], torch.empty(T, 12288, device=x.device, dtype=x.dtype))
            m = L2.silu_mul(gu, d["I"])
            d_mlp = btt_into(m, d["Rd"], d["Ld"], torch.empty(T, 2048, device=x.device, dtype=x.dtype))
            x, h = L2.add_rms(x, d_mlp, (self.layers[i + 1]["in1"] if i + 1 < n else self.norm1), self.eps)
        return h.reshape(1, T, -1)


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out


def wall(g, ids_static, T, reps=20):
    pool = [torch.randint(1000, 100000, (1, T), dtype=torch.long).pin_memory() for _ in range(reps + 3)]
    ts = []
    for i, x in enumerate(pool):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids_static.copy_(x, non_blocking=True); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[3:])
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), n=len(ts))


if __name__ == "__main__":
    out = open(sys.argv[1], "a"); Ts = [int(t) for t in sys.argv[2].split(",")]
    from prof_d1 import load_torso
    torso = load_torso()
    DENSE_MAC = None
    for cfg in sys.argv[3:]:
        if cfg in ("dense", "dense_fold"):
            rt = L2.Lean2(torso, fuse=(GLUE + ",gemm_swiglu,fold") if cfg == "dense_fold" else GLUE)
            mac = sum(d["Win"].numel() + d["Wo"].numel() + d["Wgu"].numel() + d["Wd"].numel() for d in rt.layers)
            DENSE_MAC = mac
        else:
            which, b, frac = cfg.split(":"); rt = StructRT(torso, which, int(b), float(frac)); mac = rt.macs
        for T in Ts:
            ids = torch.randint(1000, 100000, (1, T), device="cuda")
            g, o = capture(lambda: rt.forward(ids))
            r = wall(g, ids, T)
            rec = dict(cfg=cfg, T=T, **r, gflop_per_tok=round(2 * mac / 1e9, 3), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
            del g, o; torch.cuda.empty_cache()
        del rt; torch.cuda.empty_cache()
