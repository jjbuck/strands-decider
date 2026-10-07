# H5 report: learned-rotation W4A4 on full-depth hobson (coordinator's condensed copy of H5's final message)

## Approach
- **Rotations.** SpinQuant's R2 cannot be folded into hobson's weights, because the attention output gate and the GDN gated RMSNorm sit between V and the out-projection. So H5 learned two things:
  - the dense R1 residual rotation, folded offline at zero runtime cost;
  - per-GEMM orthogonal Kronecker rotations plus a learned clip.
- **The R1 objective.** A local Gram-weighted A4 error worked; end-to-end KL was too noisy. Gradient only flows once the absmax scale is not detached.
- **LoRA QAT.** Dev KL went .0372 → .0369, i.e. no gain.

## All suites (3227 questions, against bf16 hobson; dense runtime floor 0.46% flips)

| config | REAL flips | REAL sd | CF fgh | CF-probe fgh | JB-hard (p) | REAL-label |
|---|---|---|---|---|---|---|
| W4A4 Hadamard GPTQ | 9.23% | .847 | .725 | .629 | .462 (.13) | .777 |
| W4A4 learned R1 GPTQ | 8.59% | .815 | .697 | .705 | .523 (1.0) | .762 |
| k48, learned R1 | 3.14% | .945 | .936 | .914 | .546 (.38) | .772 |
| k48, R1 + Kronecker | **2.22%** | .951 | .936 | .905 | .538 (.69) | .787 |
| W8A8 GPTQ8, learned R1, all 96 GEMMs | 0.83% | .994 | .991 | .952 | .538 (.5) | .782 |
| NVFP4 W4A4, no rotation, emulated | 5.17% | .910 | .835 | .781 | .515 (1.0) | .767 |

## Findings
- **Learned rotations are indistinguishable from Hadamard on the suites.** McNemar 51 vs 58, p .57. They cut KL by only 14–25%.
- **4-bit decisions are noise-dominated.** Two 4-bit models disagree with each other on about 10% of questions.
- **Block scaling beats rotation.** NVFP4 (16-element block scales) halves int4 KL, and works best with no rotation at all. NVFP4 vs int4: p .0003.
- **Kronecker helps mixed maps directionally.** 2.22% vs 3.14% flips, p .05.
- **Sensitivity by GEMM class at W4A4** (KL, highest first): gate_up .028, GDN out .019, GDN in .014, down .009, attention .006. The GDN b/a gates are insensitive. A8 is nearly lossless.

## Verdict
- Every 4-bit-majority configuration fails the bar. The only passing low-bit format is W8A8-GPTQ8-b8 (H1/H2): 0.18% flips, 33.5 ms on the A10G with a 1-question schema, about 18 ms projected on a 3090.
- **The decisive test for 4-bit:** NVFP4 plus a k48-style map with FP8 on the sensitive GEMMs, plus full-weight KL distillation on all train_pool states, measured on real Blackwell kernels. If that cannot reach under 1% flips with CF-probe fgh ≥ .95, 4-bit hobson needs base retraining.
