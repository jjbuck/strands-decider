import sys, os, json, time, statistics as st, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/work/d1")); sys.path.insert(0, os.path.expanduser("~/work/systems/g"))
@triton.jit
def mm_k(A, B, C, M, N, K, ACC16: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0); npn = tl.cdiv(N, BN); GROUP: tl.constexpr = 8
    npm = tl.cdiv(M, BM); gsz = GROUP * npn; g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + rm[:, None] * K + rk[None, :]; b_ptr = B + rn[None, :] * K + rk[:, None]
    if ACC16:
        acc = tl.zeros([BM, BN], dtype=tl.float16)
    else:
        acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.); b = tl.load(b_ptr)
        if ACC16:
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=rm[:, None] < M)
def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return st.median(ts)
M, N, K = 4096, 12288, 2048
a = torch.randn(M, K, device="cuda", dtype=torch.float16); b = torch.randn(N, K, device="cuda", dtype=torch.float16) * .02
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
for acc16 in (False, True):
    best = None
    for (BM, BN, BK, nw, ns) in [(128, 128, 32, 4, 4), (128, 256, 32, 8, 3), (128, 128, 64, 4, 3), (256, 128, 32, 8, 3)]:
        try:
            t = bench(lambda: mm_k[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](a, b, c, M, N, K, ACC16=acc16, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns))
        except Exception as e:
            print("fail", e.__class__.__name__); continue
        if best is None or t < best[0]: best = (t, (BM, BN, BK, nw, ns))
    ref = (a.float() @ b.float().t())
    mm_k[(triton.cdiv(M, best[1][0]) * triton.cdiv(N, best[1][1]),)](a, b, c, M, N, K, ACC16=acc16, BM=best[1][0], BN=best[1][1], BK=best[1][2], num_warps=best[1][3], num_stages=best[1][4])
    err = ((c.float() - ref).norm() / ref.norm()).item()
    print(f"fp16 inputs, acc {'fp16' if acc16 else 'fp32'}: {best[0]*1000:.0f} us  {2*M*N*K/best[0]/1e9:.1f} TFLOPS  rel err {err:.2e}  cfg {best[1]}", flush=True)
tc = bench(lambda: a @ b.t()); print(f"cuBLAS fp16 (fp32 acc): {tc*1000:.0f} us {2*M*N*K/tc/1e9:.1f} TF", flush=True)
# ---- Qwen3.5-0.8B through lean2
import transformers
from prof_d1 import capture, wall
from lean import Lean
from lean2 import Lean2
name = "Qwen/Qwen3.5-0.8B"; cfg = transformers.AutoConfig.from_pretrained(name)
torso = transformers.Qwen3_5ForCausalLM.from_pretrained(name, config=cfg.get_text_config(), dtype=torch.bfloat16).cuda().eval().model
tc_ = torso.config
print("0.8B cfg", tc_.hidden_size, tc_.intermediate_size, tc_.num_hidden_layers, getattr(tc_, "linear_num_value_heads", None), getattr(tc_, "linear_value_head_dim", None), flush=True)
P = sum(p.numel() for n, p in torso.named_parameters() if "embed" not in n); print("0.8B body params", P / 1e6, flush=True)
ref = Lean(torso)
for nm, ln in (("lean_base", ref), ("lean2_swiglu", Lean2(torso, fuse="addrms,gnorm,silu,prep,conv,gemm_swiglu")), ("lean2_fold", Lean2(torso, fuse="gnorm,prep,conv,fold"))):
    r = {}
    idsv = torch.randint(1000, 100000, (1, 1000), device="cuda")
    with torch.inference_mode():
        h = ln.forward(idsv).float(); hr = ref.forward(idsv).float()
    r["rel"] = round(((h - hr).norm() / hr.norm()).item(), 5)
    for T in (256, 1000, 4000):
        ids = torch.randint(1000, 100000, (1, T), device="cuda")
        g, o = capture(lambda: ln.forward(ids)); r[T] = wall(g, ids, T); r[T]["tflops"] = round(2 * P * T / (r[T]["median"] / 1e3) / 1e12, 1)
        del g, o; torch.cuda.empty_cache()
    print("0.8B", nm, json.dumps(r), flush=True)
