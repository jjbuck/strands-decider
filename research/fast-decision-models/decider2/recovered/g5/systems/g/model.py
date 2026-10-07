"""latency(T) = floor + slope*T, for candidate body sizes and GPUs. Datasheet peaks + efficiency factors from A10G measurement.
All inputs are assumptions printed in the header; swap in measured values."""
import sys, json
# GPU: (bf16 dense TFLOPS, GB/s)  -- sources: NVIDIA whitepapers/pages (see NOTES)
GPUS = {"RTX3090": (71.2, 936), "RTX4090": (165.2, 1008), "L4": (121, 300), "A10G*": (125, 600), "L40S": (362, 864), "H100-SXM": (989.5, 3350)}
# candidate bodies: (name, non-embedding params B, layers)
BODIES = [("2B(1.39B)", 1.39, 24), ("0.8B(0.50B)", 0.50, 24), ("0.6B(0.44B)", 0.44, 28), ("300M", 0.30, 20), ("150M", 0.15, 16), ("100M", 0.10, 12)]
MFU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45    # achieved GEMM fraction of bf16 dense peak at M~1000 (measure!)
BWE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80    # achievable fraction of datasheet bandwidth
KPL = 20          # kernels per layer in the fused graph
TK = 2.5e-3       # ms per tiny kernel in a graph (launch gap + exec)
HOST = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0   # ms host: tokenize+H2D+D2H+format (to be measured)
def lat(body, gpu, T):
    _, N, L = body; tf, bw = GPUS[gpu]
    floor_mem = N * 1e9 * 2 / (bw * BWE * 1e9) * 1e3
    floor_k = L * KPL * TK
    floor = max(floor_mem, floor_k) + HOST
    comp = 2 * N * 1e9 * T / (tf * MFU * 1e12) * 1e3
    return floor + comp, floor, comp
print(f"assumptions: MFU={MFU} BWeff={BWE} kernels/layer={KPL} t/kernel={TK*1000:.1f}us host={HOST} ms")
for gpu in GPUS:
    print(f"--- {gpu}")
    for b in BODIES:
        row = []
        for T in (100, 1000, 4000):
            t, fl, c = lat(b, gpu, T); row.append(f"T={T}: {t:6.1f}")
        print(f"  {b[0]:12s} floor={lat(b,gpu,0)[1]:5.1f} ms  " + "  ".join(row) + f"   body params for 10ms@1000: {(10-lat(b,gpu,0)[1])*1e-3*GPUS[gpu][0]*1e12*MFU/(2*1000)/1e9:.2f}B")
