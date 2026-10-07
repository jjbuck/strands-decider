# H2 report: an end-to-end low-bit runtime for hobson-v19 on the A10G (box g1)

The coordinator saved this from H2's final message, condensed. The full detail is in `NOTES.md`, `res_bench_v4.jsonl`, `scores.json` and `preds/`. The code is on g1 at `~/work/h2`.

## What was built

**GEMMs**
- CUTLASS kernels: s4×s4 and s8×s8 GEMMs (G2's), plus a new W4A8 mixed-input GEMM.
- A gate_up GEMM with dequantization and SwiGLU fused into its epilogue (EVT). It is bit-exact against torch and runs as fast as the plain GEMM.

**Fused Triton glue kernels (6)**
- residual add + RMSNorm + per-token quantization;
- GDN conv + SiLU + l2norm, including the prefix and branch tails;
- GDN gated norm + online R2 Hadamard + quantization;
- attention gate + R2 + quantization;
- online R4 Hadamard (6144 = 12·16·32) + quantization;
- q/k norm + RoPE, writing K/V into the prefix buffers.

**Format**
- Implements H1's FORMAT v0/v1 exactly, with per-GEMM precision maps (k24, k48, b8).
- GPTQ codes regenerated on g1.

**Layouts**
- *plain*: hobson's state-first layout. One shared pass over the state, then branches per question.
- *schema*: the question bundle is cached once (this-that-model style). A request computes the state plus one token per question.
- Compiling the bundle once costs 186 ms (bf16) or 81 ms (W4A4); the cache is 115 MB.

**Checks**
- **Kernels against the torch reference:** code differences only at rounding ties.
- **Cached layout against an uncached pass:** cosine 0.99999 in bf16.
- **The bf16 runtime against hobson:** 0.37% of REAL decisions flip (the runtime floor).

## Latency (measured, A10G)

Medians in ms, graph replay including the id and probability copies, excluding host tokenization. The state has T tokens. Question bundles: 1q = 125 tokens, 4q = 1613, 15q = 3720.

| T | Q | layout | bf16 | W8A8 | W4A8 | W4A4 | b8 (W8A8-GPTQ, 8 GEMMs in bf16) | k48 (48 GEMMs at W8A8) |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1 | plain | 57.0 | 35.6 | 38.6 | 23.5 | 36.3 | 29.2 |
| 1000 | 1 | schema | 52.8 | 32.9 | 35.6 | 21.7 | 33.5 | 27.0 |
| 1000 | 15 | plain | 237.4 | 163.6 | 166.5 | 103.4 | 166.4 | 131.5 |
| 1000 | 15 | schema | 56.6 | 36.8 | 39.6 | 25.6 | 37.4 | 30.9 |
| 4000 | 1 | schema | 199.3 | 135.3 | 135.9 | 82.8 | 138.8 | 107.8 |
| 4000 | 15 | schema | 212.2 | 148.2 | 148.9 | 95.9 | 151.7 | 120.3 |

**Observations**
- W4A8 is no faster than W8A8, because prefill is compute-bound.
- Every precision runs its GEMMs at about 81% of the tensor peak.

**Where the W4A4 time goes** (1 question, schema layout)

| T | GEMM | attention | glue | GDN | non-GEMM share |
|---|---|---|---|---|---|
| 1000 | 12.1 | 1.0 | 4.8 | 3.4 | 44% |
| 4000 | 43.7 | 8.5 | 17.8 | 12.3 | 47% |

## Projections (speculation: GEMMs scale with peak rate, memory-bound kernels with DRAM bandwidth)

| config, T = 1000 | 3090 | 4090 | 5090 (NVFP4) |
|---|---|---|---|
| bf16 1q, fp16 accumulation | 27.4 | 13.7 | 10.0 |
| W8A8-GPTQ b8, 1q schema | 18.3 | 10.7 | 7.3 |
| W4A4, 1q schema | 12.5 | 8.1 | 5.3 |
| W4A4, 15q schema | 16.3 | 9.8 | 6.6 |

## Suites (all 3227 evalkit questions, hobson layout, through H2's kernels)

| config | REAL flips | REAL sd | LONG sd | CF fgh | CF-probe fgh | JB-hard (McNemar p) | REAL-label |
|---|---|---|---|---|---|---|---|
| bf16 | 0.37% | .994 | .988 | 1.000 | .971 | .538 (.50) | .785 |
| W8A8 RTN | 1.20% | .988 | .970 | .991 | .924 | .531 (1.0) | .790 |
| W8A8 GPTQ | 0.83% | .991 | .970 | 1.000 | .990 | .531 (1.0) | .787 |
| **W8A8 GPTQ b8** | **0.18%** | **.997** | **.982** | **1.000** | **.971** | **.523 (1.0)** | **.785** |
| W4A8 RTN | 9.51% | .899 | .897 | .844 | .590 | .500 | .743 |
| W4A4 RTN | 13.5% | .855 | .879 | .670 | .514 | .508 | .728 |
| W4A4 GPTQ | 8.96% | .850 | .885 | .826 | .657 | .523 | .755 |
| k48 GPTQ | 2.59% | .965 | .939 | .972 | .848 | .538 | .780 |

## Verdict
- **W8A8-GPTQ b8 passes the low-bit kill criterion.** 0.18% flips, CF fgh 1.0, CF-probe 0.971, no JB-hard drop. It runs in 36.3 ms (plain, 1q) and 33.5 ms (schema, 1q) on the A10G, against 57.0 ms for bf16.
- **Every 4-bit config fails.** The best is k48 at 2.6% flips.
- **4-bit decisions are noise-dominated.** Two numerically equivalent runtimes disagree on 12% of W4A4-RTN decisions, so 4-bit accuracy must be measured in the deployed kernels.
- **The schema layout's accuracy is unmeasured.** hobson was trained state-first, so the layout needs a schema-first fine-tune.

## Next step
Run QAT in the deployed kernels, schema-first, starting from k48. Distil from bf16 hobson's state-first answers, and score through these kernels. In parallel, shrink the memory-bound floor: fused GDN chunk kernels, the residual epilogue, and prefix attention.
