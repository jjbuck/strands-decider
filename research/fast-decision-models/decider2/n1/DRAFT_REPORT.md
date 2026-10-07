# N1: hobson-v19 on Inferentia2: pipelined GDN kernel, layout cleanup, M1/M2 shapes

**Labels.**
- **[M]** measured on box n1: inf2.xlarge with one Inferentia2 chip, SDK identical to J8's (torch-neuronx 2.8.0.2.12, neuronx-cc 2.23).
- **[V]** verified from J8's files or the ARCH docs.
- **[A]** arithmetic.

**Latency:** one NeuronCore, weights resident, fresh inputs, median of 20 warm calls (p95 within 1 ms unless stated).

## Result

**Latency.** At T = 1000 with one question, hobson runs in **132.1 ms** (p95 132.5) on one NeuronCore [M]. J8 measured 152.0 ms, and his graph re-measured on this box gives 153.7 ms [M].

**Cost.** That is **$14.7 per million decisions** on inf2.xlarge with both cores. J8's figure was $16.0 and the A10G's is $17.7, so this is 0.83x the A10G [A on M].

**Target.** The ≤ 90 ms target is **not met**.

**Decisions.** They stay at the noise floor: 4 flips in 837 questions [M].

**Kernel against glue.** The new GDN kernel is 1.47x faster than J8's (both warm), but in situ the XLA glue around it dominates the GDN layer.

## What I built

### 1. Exact multi-head NKI GDN kernel (`code/gdn11.py`; variants gdn4–gdn12)

- **Heads in one program.** All 16 heads run in one program, with 4 heads stacked per [128, 512] tile. Each copy, mask and PSUM bank therefore covers 4 heads, and the per-head matmuls run back to back.
- **Layout.** q, k and v arrive token-major [T, 16, 128], which is the projection's own layout. Only 2 transposes are needed per head-chunk.
- **Triangular solve.** Exact block doubling with M and Mᵀ in SBUF; updates by operand order ((AB)ᵀ = BᵀAᵀ), so no transposes; the mask is applied at the eviction of L·M.
- **Serial part.** It is reformulated as S′ = A S + B and o = Q′S + O₀. A, B, Q′ and O₀ do not depend on the state, so the only serial chain per chunk is one fp32 matmul plus an accumulate.
- **Loop.** The chunk loop runs sequentially with the state updated in place. It is as fast as unrolling and compiles in 14 s instead of 60–90 s.
- **Exactness [M].** Max abs error against J8's fp64 reference:

  | case | error |
  |---|---|
  | T = 1152 | **4.5e-7** |
  | T = 4096 | 7.1e-7 |
  | nonzero initial state, 6x decay | 1.4e-6 |

  The bar is 1e-5.
- **Warm device time, 16 heads [M].**

  | T | new kernel | J8's kernel2 | speedup |
  |---|---|---|---|
  | 1152 | **1.373 ms** (TensorE 83% busy) | 2.022 ms | 1.47x |
  | 4096 | 4.747 ms | 6.778 ms | 1.43x |

- J8's 4.13 ms came from nki.profile's cold first run, which includes a one-time ~2.08 ms DMA wait [M].

### 2. Token-major layout (`code/hob5.py`), shape models, tooling

- GDN layers use plain [T, 2048] GEMMs (q|k|v|z, output projection), no head-major batched matmuls or per-head einsums, and a permute-free kernel prep.
- M1 (`m1.py`) and M2 (`m2.py`, k = 8, 256-token blocks) follow their ARCH v1 documents, with hobson's weights where shapes match.
- Non-inlined weights (`move_trace_to_device` at load) compile the full model in **111 s instead of 1059 s** at the same runtime (4-layer graph: 24.88 vs 24.77 ms) [M]. This made a 20-graph sweep possible on a 15 GB box.

## Latency (ms, one NeuronCore) [M]; A10G column [V]

| T | hobson 1 q | J8 1 q | hobson 4 q | J8 4 q | M1 1 q | M2 1 q | A10G |
|---|---|---|---|---|---|---|---|
| 64 | 32.6 | 30.9 | **82.9** | 94.8 | **23.0** | – | 9.4 |
| 256 | 42.0 | 43.1 | **103.1** | 110.0 | **28.3** | – | 15.7 |
| 1000 | **132.1** | 152.0 | **159.5** | 181.1 | 137.3 (4 q: 163.7) | 175.4 | 52.7 |
| 4000 | 653.6 | **537.5** | 506.3 | – | 588.4 | no compile | 200.5 |

**Variant per length.** The hobson figures use the best variant for each length; all variants compute the same function.
- Each variant either l2-normalises inside the kernel or in XLA.
- That choice moves the result in opposite directions at different lengths [M]:

  | l2-norm runs in | T = 256 | T = 1000 |
  |---|---|---|
  | the kernel | 42.2 ms | 155.8 ms |
  | XLA | 58.4 ms | 132.1 ms |

**T = 4000.** One question takes 654 ms against J8's 538, yet the packed 4-question graph takes 506 ms. Unresolved.

**M2 builds.** T = 4000 and 4 questions exceed the compiler's 5M-instruction limit (104.7M, 26.3M) [M]. T = 1000 compiled after padding U's 4 keys to 128 (a 260-key block attention failed SBUF allocation).

## Where the time goes at T = 1000 [M]

**Full model: 132.1 ms.** Removing the GDN core and its q/k glue leaves 72.2 ms, so **GDN costs 59.9 ms**, or 3.33 ms per layer:
- **Kernel:** about 1.37 ms per layer, the same as its standalone time (about 25 ms in all).
- **XLA glue:** about 1.95 ms per layer (about 35 ms in all). This covers the conv and SiLU on [T, 6144], the l2-norm, the β/g/decay columns, and the copies around the call.

**Decay-column prep.** Removing only the [T, 16] prep saved about 2 ms per GDN layer (4-layer graph). Moving it into the kernel (gdn10) cost the kernel 0.06 ms standalone, yet the graph got slower: the cost is at the XLA↔NKI boundary (layouts, spills), not in the arithmetic.

**Non-GDN part.** The token-major layout cut it from J8's 80.2 to 72.2 ms (about 48 effective TFLOPS). A fused NKI MLP (`mlp_nki.py`, exact) reached 55 TFLOPS against XLA's 52 (1.58 vs 1.69 ms); not adopted. The measured back-to-back bf16 PE rate is about 55 TFLOPS.

**Two cores at once.** Each core takes 139.1 ms, against 132.1 ms alone (+5%), giving 14.4 decisions/s per chip.

## Fidelity [M]

J8's subset: 837 of 931 questions (all ≤ 4096 tokens), final 4-bucket graphs.

**Agreement with hobson:** **.9952** (4 flips, all at hobson p_max .36–.50); TV .0028; max |dp| .031.

**Task accuracy.**

| measure | this port | hobson-v19 |
|---|---|---|
| JB-all | .7273 | .7229 |
| REAL-label, on the 110 covered items | .7455 | .7364 |

JB-hard McNemar 1/0 (p 1.0); CF flip-given-hobson 1.000 (90 pairs); CF-probe .966 (76); REAL state-dependent agreement .987.

**Noise floor.** An earlier build of the same function gave .9940 (5 flips); J8's port 6 flips; the bf16 floor is 0.37–0.46%. Both builds sit at it within Poisson noise. Against the strict bar: no significant JB drop, CF/CF-probe retention 1.00/.97; full-kit REAL-label not run. The T = 256 and T = 4000 graphs were verified at kernel level only (4.5e-7).

## Cost per million decisions [A on M]

inf2.xlarge at $0.76/h with both cores; A10G at $1.212/h; J8's method.

| T | hobson inf2 | J8 inf2 | A10G | M1 inf2 |
|---|---|---|---|---|
| 64 | $3.44 | $3.26 | $3.16 | **$2.43** |
| 256 | $4.43 | $4.55 | $5.29 | **$2.99** |
| 1000 | **$14.7** (measured with two cores) | $16.0 | $17.7 | $14.5 |
| 4000 | $69.0 | $56.7 | $67.5 | $62.1 |

**Four-question requests (hobson):** $8.75, $10.9, $16.8, $53.4 per million requests at T = 64, 256, 1000, 4000 (J8: $10.0, $11.6, $19.1).

**M1 against its own A10G timing** (ARCH: 15.1, 25.3, 56.5, 211.0 ms): 0.48x, 0.35x, 0.76x, 0.87x the A10G per decision, for a function M1 has yet to train.

## Projections [A, assuming the decomposition adds]

**hobson, one fix at a time.**

| change | T = 1000 latency |
|---|---|
| fuse the GDN glue down to about 0.5 ms per layer (an elementwise floor for conv and norms on [T, 6144]) | about 106 ms |
| also bring the kernel to 1.0 ms per layer | about 99 ms |
| also run the non-GDN GEMMs at 60 rather than 48 TFLOPS | about 85 ms ($9.0 per million, 0.5x the A10G) |

So ≤ 90 ms needs all three changes.

**M1** with an NKI flash-attention mixer projects to about 72 + 18 × 0.5 ≈ 81 ms at T = 1000 (0.5 ms per mixer layer, not measured). It is already the fastest shape on inf2 at T ≤ 256.

**M2** costs more on inf2 than on the GPU: three GDN calls per live GDN layer and a full scan per deep GDN layer, each call carrying about 2 ms of in-situ overhead here.

## The single most valuable next step

**Fuse each GDN layer's glue into the kernel.** The layer would then run as:
1. the q|k|v|z GEMM;
2. one NKI kernel covering conv and SiLU, l2-norm, β/g/decay prep, the delta rule, and the gated RMSNorm;
3. the output projection.

Then re-measure at T = 1000.

**Why:** the glue is 35 of the 132 ms, more than the kernel's 25, and every partial move between XLA and NKI shifted 10–25 ms either way, which points at the boundary rather than the work.

**Kill test:** if a fused GDN layer cannot get below about 2 ms in situ at T = 1000, hobson on inf2 stays at about 0.8x the A10G's cost, and M1 with an NKI flash mixer is the better inf2 architecture.

## Files

`~/decider2/n1/`: `code/` (kernels gdn4–12, hob5, m1, m2, mlp_nki, scripts), `results/` (latencies, profiles, predictions, scores), `NOTES.md`.

Box n1 is left running and terminates at its TTL; I did not reset its timer, touched no other AWS resource, and no permission was denied.
