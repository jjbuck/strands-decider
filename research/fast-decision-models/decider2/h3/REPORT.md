# H3 report (coordinator's condensed copy of H3's final message)

## Error propagation in hobson (measured)
- **No amplification through depth.** Relative residual error from one W4A4 layer decays to 0.2–0.36× (int4) or 0.11–0.35× (NVFP4) by layer 23.
- **No amplification inside submodules.** The paper reports about 30× at the attention output; in hobson the relative error at each position is:

  | position | relative error |
  |---|---|
  | GDN in-proj | 0.50 |
  | GDN core | 0.45 |
  | GDN out | 0.98 |
  | MLP out | 1.21 |
  | attention core | 0.47 |

  hobson's mixers are already bounded: L2-normalised GDN q/k and gated RMSNorm, attention q/k norm and an output gate.
- **The 4-bit damage is local, set by the activation crest factor.**
  - The median max/rms at GEMM inputs is 15–40, so per-token int4 loses 50–70% of each activation.
  - Hadamard rotation brings the crest to 3.2–3.8 (13–16% error); NVFP4 gives 7–9% error.
  - The crest is scale-invariant, so no normalisation, hyperspherical included, can lower it.
- **The decision-critical GEMMs** are the layer-0 out-proj, the attention in-proj at layers 7 and 11, and layer 23. Layers 12–22 are almost inert.

## Retrofits (v1 hyperspherical head, v2 L2Norm) against QAT, equal budget (all 1083 REAL questions; floor 0.55%)

| config | REAL flips | agree_sd | CF fgh | CF-probe fgh | JB-hard |
|---|---|---|---|---|---|
| hobson NVFP4 | 9.14% | .884 | .697 | .705 | .492 |
| v1 bf16 | 4.99% | .902 | .817 | .810 | .515 |
| v1 NVFP4 | 8.96% | .847 | .651 | .695 | .515 |
| v2 NVFP4 | 10.3% | .801 | .459 | .648 | .492 |
| QAT NVFP4 | 7.20% | .899 | .596 | .733 | .469 |
| **hobson NVFP4, question rows bf16** | **4.16%** | **.954** | .807 | .781 | .508 |

- **The retrofits cut each arm's own quantisation flips** (NVFP4: 9.3% → 5.4% for v1, 4.3% for v2). But the retrofit itself costs 5–7.6% flips in bf16, and doubling the budget did not close that.
- **The paper's "retrofit ≈ QAT" claim is false here.** QAT beats v1 at equal budget: int4 p 3e-4, NVFP4 p .045.
- **Tiny pretraining** (35M parameters, 33M tokens): the nGPT-style model loses less in relative terms under quantisation, but is not better in absolute terms. At that scale there are no outliers.

## Verdict
- Every 4-bit configuration fails the low-bit bar, by 8–120×.
- Hyperspherical co-design is killed for hobson.

## Next step
The lever is precision assigned by row role and by GEMM. On a 5090, test:
- NVFP4 on the state rows;
- bf16 question and readout rows;
- 8-bit on about 8 critical GEMMs;
- KL-QAT at ≥10× the budget.

On Ampere, ship W8A8-GPTQ with bf16 question rows.
