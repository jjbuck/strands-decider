"""Latency model: lat(T) = host + floor(weights, kernels) + slope*T.   Calibrated on A10G kernel measurements; projected by datasheet ratios.
Usage: python3 model2.py [nonGEMM_factor] [host_ms]"""
import sys
NG = float(sys.argv[1]) if len(sys.argv) > 1 else 1.25   # total GPU time / pure-GEMM time at M~1000 (lean fused). UNMEASURED (pending exclusive run)
HOST = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5  # ms: render+tokenize (~0.7 ms @128 tok, ~1.5-2 @1000) + H2D/D2H + python
# gpu: (datasheet bf16 dense TFLOPS, datasheet GB/s, achieved GEMM TFLOPS at M~1000, achieved weight-stream GB/s)
#  A10G achieved from our kernel-only measurement (58 TFLOPS mean of 52-64; 400 GB/s of 600).  Others: ASSUMED fractions (A10G ratios: 0.47 of A10 dense peak, 0.67 of BW).
GPUS = {
 "A10G (measured)": (125, 600, 58, 400),
 "RTX3090":  (71.2, 936, 55, 730),
 "RTX4090":  (165.2, 1008, 100, 800),
 "L4":       (121, 300, 55, 240),
 "L40S":     (362, 864, 190, 690),
 "H100-SXM": (989.5, 3350, 500, 2700),
}
BODIES = {"2B (1.39B body, 24L)": (1.39, 24), "Qwen3.5-0.8B (0.50B, 24L)": (0.50, 24), "Qwen3-0.6B (0.44B, 28L)": (0.44, 28),
          "300M (20L)": (0.30, 20), "150M (16L)": (0.15, 16), "100M (12L)": (0.10, 12)}
KPL, TK = 14, 3.0e-3    # kernels/layer after fusion (speculative), ms per small kernel incl. graph gap
def lat(gpu, body, T, ng=NG):
    tf, bw, atf, abw = GPUS[gpu]; N, L = BODIES[body]
    floor = N * 2e9 / (abw * 1e9) * 1e3 + L * KPL * TK           # ms (weights streamed once + small kernels)
    slope = 2 * N * 1e9 / (atf * 1e12) * 1e3 * ng                # ms/token
    return HOST + floor + slope * T, floor, slope
print(f"non-GEMM factor {NG}, host {HOST} ms")
for gpu in GPUS:
    print(f"--- {gpu}: datasheet bf16 {GPUS[gpu][0]} TF, {GPUS[gpu][1]} GB/s; assumed achieved {GPUS[gpu][2]} TF / {GPUS[gpu][3]} GB/s")
    for b in BODIES:
        f = lat(gpu, b, 0)[1]; s = lat(gpu, b, 0)[2]
        print(f"  {b:28s} floor {f:5.1f}+host {HOST} ms  slope {s*1000:6.1f} us/tok | T=100 {lat(gpu,b,100)[0]:6.1f}  T=1000 {lat(gpu,b,1000)[0]:6.1f}  T=4000 {lat(gpu,b,4000)[0]:6.1f} ms")
# body params that give 10 ms @1000 tokens (solve floor+slope)
print("--- max body params (B) for 10 ms @1000 tok (layers scale ~ 24; floor from weights)")
for gpu in list(GPUS):
    best = None
    for N in [x/1000 for x in range(10, 3000, 5)]:
        BODIES["_x"] = (N, 24)
        if lat(gpu, "_x", 1000)[0] <= 10: best = N
    print(f"  {gpu:18s} {best} B  (NG={NG})")
print("--- consistency checks")
# Clef-flash: Qwen3.5-9B (~8.4B non-emb?) on an unnamed GPU, 38.8 ms median. On H100-SXM model:
BODIES["9B"] = (8.4, 32); 
for T in (300, 600, 1000): print(f"  9B on H100 T={T}: {lat('H100-SXM','9B',T)[0]:.1f} ms  (published Clef-flash median 38.8 ms, token count unreported)")
BODIES["8B-CLM"] = (6.95, 36)
for T in (100, 300, 600): print(f"  CLM 8B bf16 on RTX4090 T={T}: {lat('RTX4090','8B-CLM',T)[0]:.1f} ms  (published p50 ~28 ms for a new state)")
print("  weight-read floor alone: 8B bf16 = 16.4 GB / 1008 GB/s = %.1f ms raw" % (16.4/1008*1000))
