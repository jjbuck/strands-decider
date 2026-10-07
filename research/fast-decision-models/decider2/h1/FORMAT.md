# H1 -> H2: quantization format for hobson-v19 on int8 / int4 tensor cores

Version v1 (2026-10-06 00:45 PDT). Sections 1-4 are FIXED. Section 5 (the precision map) is v1, and QAT may still change the weights, but not the format.
The emulation that defines the numerics is `~/decider2/h1/code/h1lib.py` (class H1, method `lin`). It is integer-exact: torch._int_mm, s8 x s8 -> s32, with 4-bit codes held in int8.

## 1. GEMMs (4 per layer, 24 layers, 96 total). The names follow G2's lean runtime.

| name | weight [N, K] | input to the GEMM | rotation of the input |
|---|---|---|---|
| Win (GDN) | in_proj qkv+z+b+a, [8224, 2048] | RMSNorm(x) without gain | R1 (offline residual rotation) |
| Win (attn, layers 3, 7, 11, 15, 19, 23) | q+gate, k, v: [5120, 2048] | RMSNorm(x) without gain | R1 |
| Wo | out_proj / o_proj, [2048, 2048] | GDN: gated-norm output; attn: attn_out * sigmoid(gate) | R2, online Hadamard |
| Wgu | gate+up, [12288, 2048] | RMSNorm(x) without gain | R1 |
| Wd | down, [2048, 6144] | silu(g) * u | R4, online Hadamard |

## 2. Rotations. All are orthogonal: a random sign vector times a Kronecker Hadamard, normalized.

- R1 (2048): `diag(s1) (H32 (x) H64) / sqrt(2048)`. s1 comes from `torch.Generator('cpu').manual_seed(1234)`, `randint(0, 2, (2048,)) * 2 - 1`.
- R2 (2048, seed 1235): the same construction.
- R4 (6144, seed 1236): `diag(s4) (P12 (x) H16 (x) H32) / sqrt(6144)`. P12 is the Paley-I Hadamard of order 12 (`g2lib.paley12`).
- The exact code is `h1lib.Rot(K, seed, 'cuda')`. Use `Rot.__call__` (x -> (x*s) @ H) for activations and weights, and `Rot.inv` (x -> x R^T).

The residual stream is stored rotated (x R1), QuaRot-style. Offline:
- embedding: E' = E R1;
- readers: Win' = Win diag(1 + in_norm) R1 and Wgu' = Wgu diag(1 + post_norm) R1, with the norm gain folded first;
- writers: Wo' = R1^T Wo R2 and Wd' = R1^T Wd R4;
- the RMSNorm kernels compute the gain-free norm of the rotated residual (it commutes with R1), and quantize per token in the same kernel;
- pointer head: un-rotate only the rows the head reads (the last token and the option tokens): h = (RMSNorm(xR1) R1^T) * (1 + norm_w).

Online:
- before Wo: o -> o R2 (2048-point; G2's rq.py Triton kernel does H32 (x) H64 with two tl.dot calls);
- before Wd: m -> m R4 (6144 = 12 x 512).

## 3. Quantization

- **Activations.** Dynamic, symmetric, one scale per token (row): s_t = clip * max|x_t| / qmax.
  - int8: qmax 127, clip 1.0.
  - int4: qmax 7, clip 0.9.
  - codes = round(x / s_t), clamped to [-qmax, qmax]; round half to even, as torch.round does.
- **Weights.** Static, symmetric, one scale per output channel (column of the output), codes in [-7, 7] (4-bit) or [-127, 127] (8-bit). Scales come from an MSE clip search (4-bit) or absmax (8-bit), or from GPTQ (act-order, with the codes un-permuted). All of this is offline; H2 just loads codes and scales.
- **Epilogue.** y = acc_s32 * s_t[row] * s_w[col], in fp32, cast to bf16, then the residual add.
- The 4-bit kernels need even code packing: low nibble first, as in g2s4.cu.

## 4. What stays in bf16 or fp32

Embedding lookup, all norms, the GDN short conv + SiLU, q/k l2norm, beta/decay glue, the delta rule (fla chunk kernel), attention softmax, the GDN gated norm, the pointer head and the residual adds.

Under test (answer in v1):
- the GDN b/a rows of Win (32 of 8224 rows) as a separate tiny bf16 GEMM, so the low-bit GEMM is [T, 2048] x [2048, 8192];
- a per-head Hadamard (128 for GDN, 256 for attention) for R2 instead of the full 2048-point one, because it can be fused into the gated-norm epilogue.

## 5. Weights, and the precision map (v1)

### Weights: use GPTQ for both 8 and 4 bits, not RTN

GPTQ halved W8A8's REAL flips: RTN 1.29% vs GPTQ8 0.65% against hobson (MEASURED, main subset, numbers in NOTES).

The GPTQ settings:
- input-Hessian GPTQ on the rotated, gain-folded W'' (output side included for Wo and Wd);
- act-order, damp 0.01 * mean(diag H), block 128;
- per-channel scale fixed BEFORE GPTQ:
  - 8-bit: absmax / 127;
  - 4-bit: the MSE clip search over {1.0, 0.95, ..., 0.6} x absmax / 7.

Reproduce exactly. No weight files: the brief forbids models on the laptop, and I cannot reach g1.
1. Copy ~/decider2/h1/code/ (h1lib.py, h1calib.py) to your box next to G2's g2lib.py (~/work/g2) and evalkit (~/work/evalkit). h1lib imports g2lib from ~/work/g2.
2. Run `python h1calib.py --n 64`. It takes about 75 s and writes the unrotated input Hessians to ~/work/h1/hess/H_{i}_{k}.pt, from 64 train-split real questions (h1lib.cal_items(64, seed=0), truncated to 3000 tokens).
3. Get the codes and scales: `g = h1lib.H1(); g.opt['wq'] = 'gptq'; g.opt['wq8'] = 'gptq'; q, s = g.qweight(i, k, 'w8a8' or 'w4a4')`.
   - q is int8 [N, K] (4-bit codes in [-7, 7]), already in the rotated layout R1^T W R_in (Wo and Wd) or W diag(1+g) R1 (Win and Wgu).
   - s is fp32 [N].
   - If a QAT LoRA is given, call `g.load_lrot(path)` first. The codes then include it.

### W8A8 precision map, candidate v1

Every GEMM is W8A8, except these 8, which stay bf16:
23.Wd, 0.Wo, 7.Wo, 11.Wo, 10.Wo, 23.Wo, 12.Wo, 9.Wo (file h1/res/precmap_w8a8_b8.json).
- Measured cost: +0.71 ms per 1k tokens of GEMM on the A10G, against 25.2 ms for all-W8A8.
- Measured effect: it removes 38% of the W8A8 decision KL. Ranked by single-GEMM KL per microsecond in an fp32 model, so without the bf16 floor.
- Layers 14-21 contribute almost nothing at 8 bits; 23.Wd alone is 23% of the KL.
- QAT run A (LoRA in the rotated space, merged into these codes) is training on this map now.

### W4 maps

- W4A4 with GPTQ is far from the bar: calibration KL 0.065 against 0.0004 for W8A8.
- Candidate mixed map: h1/res/precmap_w4a4_k48.json, with 48 GEMMs at W8A8, chosen by W4A4 KL per microsecond. Measured on calibration questions, it removes 90% of the 4-bit KL at 19.3 ms per 1k tokens of GEMM, against 14.1 ms for all-int4.
- Wo GEMMs are picked first because they are cheap. Early layers (0-12) are the sensitive ones at 4 bits.

## 6. v1.1 (02:35 PDT): question rows in bf16 ("qb16"), the strongest W8A8 lever so far (MEASURED in H1 emulation)

- **The format.** For each GEMM, rows t >= q0 (the question tokens, in the hobson layout `[state][question]`) use the bf16 weights. Rows t < q0 (the state) use the W8A8 codes. Everything else is unchanged; per-token scales make the row split exact.
- **REAL, 1083 questions, all 96 GEMMs W8A8-GPTQ8 on state rows:**

  | config | TV vs hobson | flips vs hobson |
  |---|---|---|
  | W8A8-GPTQ8 + qb16 | 0.0036 | 7/1083 |
  | W8A8-GPTQ8, all rows | 0.0055 | 7/1083 |
  | b8 map | 0.0046 | 8/1083 |
  | bf16 dense floor | 0.0031 | 5/1083 |
  | fp32-path floor | 0.0031 | 4/1067 |

  Flip counts at this level are Poisson noise around the floor; TV is the metric with power.
- **Cost.** The question rows' share of tokens runs at bf16 instead of the int8 rate. Questions are 110 tokens median and 1014 at p90, so about +5-10% GEMM time at T = 1000.
- **Schema layout.** In your schema layout the question bundle is precomputed once, so qb16 is free there: compile the bundle in bf16.
- **Your b8 result.** Your 2/1083 for GPTQ8 + b8 and my 8/1083 for the same recipe in emulation are both within noise of the floors (yours 4/1083, mine 5/1083). Treat both as "at the floor".
