# H5 -> H2: learned-rotation format for hobson-v19 (24 layers, unchanged size)

Version v1 (2026-10-06 00:40 PDT). It is a strict superset of H1 FORMAT v0/v1 (~/decider2/h1/FORMAT.md), so H2's runtime runs it unchanged:
**the GEMM list, the online rotations (R2, R4), the quantizers, the epilogue and the precision-map syntax are all H1's. Only R1 and the weight
codes differ.** The reference emulation is `~/decider2/h5/code/h5lib.py` (class Q5, mode 'q'). Artifacts live on box g5 only (no weights on the laptop).

## 1. GEMMs
The same as H1 section 1: Win GDN [8224, 2048], Win attention [5120, 2048], Wo [2048, 2048], Wgu [12288, 2048], Wd [2048, 6144].

## 2. Rotations
- **R1 (2048 x 2048) is learned.** It is a dense orthogonal fp32 matrix, initialized at H1's R1 (sign seed 1234, H32 (x) H64).
  - Parametrization: R1 = R0 expm(A - A^T), with Adam on A (lr 3e-4, 300 full-batch steps).
  - Objective: the output-space error of per-token int4 activation quantization (clip 0.9) at all 48 R1-rotated GEMM inputs (Win and Wgu of every layer). It is weighted by the 16-bit weights through G = diag(f) W^T W diag(f).
  - Data: 64 train-split states x 128 tokens (token 0 always kept). Weights are frozen and 16-bit, as in SpinQuant (arXiv 2405.16406).
  - Files on g5: `~/work/h5/rot/R1_l1.pt`. Reproduce with `h5rotl.py --steps 300 --lr 3e-4 --tag l1` (seeded, about 11 min on the A10G).
  - R1 is folded offline exactly like H1's R1:
    - E' = E R1;
    - Win' = Win diag(1 + in_norm) R1 and Wgu' = Wgu diag(1 + post_norm) R1;
    - Wo' = R1^T Wo R2 and Wd' = R1^T Wd R4.
  - The head reads (RMSNorm(x R1) R1^T)(1 + norm_w) on the last-token and option rows only.
  - It has **zero runtime cost**. Because R1 is dense (not Kronecker), it must never be applied online: the residual stays in the R1 basis end to end.
- **R2 and R4 are H1's** (seeds 1235 and 1236), applied online before Wo and before Wd.
  - SpinQuant's absorbable head-wise R2 does not exist in hobson. Attention has an elementwise sigmoid output gate, and GDN has a gated RMSNorm, between V and the out-projection.

## 3. Quantization (same as H1)
- **Activations:** per-token symmetric absmax. int4 uses clip 0.9 and codes [-7, 7]; int8 uses clip 1.0.
- **Weights:** per-output-channel symmetric. Scales come from an MSE clip search over {1.0, 0.95, ..., 0.6} x absmax, then GPTQ:
  - act-order, damp 0.01, block 128;
  - codes un-permuted;
  - sequential over layers and sites, with inputs taken from the already-quantized prefix;
  - calibration on 128 train-split sequences of up to 2048 tokens (`h5build.py`, seed 21).
- **Epilogue:** acc_s32 * s_t[row] * s_w[col] in fp32, then cast to bf16.
- **GDN in_proj_b / in_proj_a rows:** a separate bf16 GEMM from the unquantized normed input (32 rows). Measured: quantizing them too makes no difference.

## 4. bf16 / fp32 parts
The same as H1 section 4.

## 5. Precision maps (H1's JSON syntax `{"<layer>.<Win|Wo|Wgu|Wd>": "w8a8"}`; anything not listed is W4A4)
- `all-W4A4`: dev (240 train-split questions) flips 7.5%, KL 0.036.
- `k48` = H1's `precmap_w4a4_k48.json`: dev flips 1.25%, KL 0.0045. The suite results are in REPORT.md.

Note that 8-bit sites under the learned R1 use the same R1; R1 was learned for the all-4-bit objective.

## 6. Results of each map (all evalkit suites, emulation, vs bf16 hobson refs; details in NOTES.md)
| map | REAL flips | CF fgh | CF-probe fgh | JB-hard |
|---|---|---|---|---|
| all-W4A4, learned R1 + GPTQ | 8.59% | .697 | .705 | .523 |
| k48, learned R1 + GPTQ | 3.14% | .936 | .914 | .546 |
| k48, learned R1 + per-GEMM learned Kronecker P + GPTQ | 2.22% | .936 | .905 | .538 |
| all-W8A8, learned R1 + GPTQ8 | 0.83% | .991 | .952 | .538 |
None passes the low-bit bar; H1's W8A8-GPTQ8 + b8 (8 GEMMs bf16) does (H2 runtime, 0.18%).
Per-GEMM Kronecker option (k48+kron row): every GEMM input gets x -> x (P1 (x) P2) after its R1 / R2 / R4 rotation, P orthogonal, factors
32x64 (K=2048) or 96x64 (K=6144), file rot/kron_k1.pt on g5 ({(layer, site): (P1, P2)}, plus a per-GEMM activation clip). For Wo
(2048) it merges into R2's factors: (H32 P1) (x) (H64 P2); for Wd and the Win/Wgu inputs it is one extra pair of small dots in the
existing fused glue kernels (not built in H2's runtime).

## 7. NVFP4 variant (next hardware generation; emulated only)
FP4 E2M1 values, FP8 E4M3 scale per 16 consecutive K-elements, fp32 per-tensor scale, weights by block-scale-aware GPTQ
(fixed block scales, E2M1 grid). **No rotations**: R1 / R2 / R4 all hurt with block scales (dev KL .0447 Hadamard vs .0325 none, RTN).
Dev: all 96 GEMMs NVFP4 W4A4 GPTQ = 3.3% flips, KL .0185 (int4 best .035).
