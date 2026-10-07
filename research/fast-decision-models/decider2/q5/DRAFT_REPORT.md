# Q5 report: decision-aware structure for hobson-v19 (B11, B12, B13 checks, B7)

**Labels and baselines.**
- [M] measured, [V] verified, [A] arithmetic.
- "bf16" is the in-runtime bf16 model; "b8" is W8A8-GPTQ with 8 GEMMs in bf16.
- "Kit" is all 3,227 evalkit questions.
- "Dev" is 256 train-split requests from tau tasks held out of calibration. Dev KL is against bf16: b8 has 9.5e-5; w4q8, which fails the kit, has 3.7e-3.

## What I built (`~/decider2/q5/code/`; copies of Q1's `q1lib`/`q1fmt` and H1's GPTQ, unedited)

- **`q5cal.py`: a decision-gradient pass per deployment** (160 banking or retail train-split requests). Per GEMM and row role it accumulates:
  - plain and decision-weighted input Hessians, H_d = Σ_t w_t x_tᵀx_t, with w_t the Fisher-weighted squared decision gradient at row t;
  - the decision Fisher diagonal;
  - per-neuron saliency of replacing a SwiGLU neuron by its mean.
- **`q5lib.S5`: structure over any Q1 format.**
  - **B11, 2:4 along K,** on all rows or on state rows only (question rows dense). Masks by magnitude, Wanda, SparseGPT with the layer Hessian or with H_d, or the decision Fisher; then joint GPTQ with the mask fixed.
    - int4 uses Ampere's pair-wise pattern (Q4).
    - int8 is in the natural basis with smoothing, because a rotation destroys 2:4.
  - **B12, neuron removal.** Neurons are ranked per layer or globally by decision saliency. The kept down-projection columns get a decision-weighted least-squares update, and the removed means become a bias.
- **Evaluation:** dev screening, then kit runs with a banking subset. A layer-12 residual cache, bit-exact against full forwards [V], halves kit runs.

## Results against the bar (kit) [M]

**Reference points.**
- bf16: 5/1083 REAL flips, TV .0031, CF 1.000, CF-probe .971.
- b8: 5/1083 (paired vs bf16 2/2), TV .0047, CF .982 (one pair short in this GPTQ draw), CF-probe .962.

**Threshold items decide the outcome.** Every lost REAL question or CF pair, in every config, has a bf16 probability within ~0.02 of the boundary.

| config | GEMM work changed | REAL flips (paired vs bf16) | TV | CF | CF-probe | JB-hard (p) | REAL-label | bar |
|---|---|---|---|---|---|---|---|---|
| **b8 + B12 60% of L12-23 neurons (global)** | 19.8% of MACs removed | 3 = 0.28% (1/3) | .0054 | 1.000 | .952 | .538 (.5) | .788 | **meets** |
| **b8 + int4 2:4 (pair-wise), state rows L14-23** | 40.2% of state-row MACs | 3 = 0.28% (1/3) | .0054 | .991 | .962 | .531 (1.0) | .790 | **meets** |
| **b8 + int8 2:4, state rows L12-22** | 45.9% of state-row MACs | 6 = 0.55% (5/4) | .0069 | .991 | .952 | .531 (1.0) | .793 | **meets** |
| b8 + B12 60% + int4 2:4 on mixer state rows L14-23 | 19.8% removed + 13.6% of state-row MACs | 5 (2/2) | .0059 | .982 | .952 | .538 | .785 | misses CF by b8's own pair |
| B12 60%, bf16 | 19.8% removed | 6 (1/0) | .0039 | 1.000 | .971 | .546 | .785 | meets |
| B12 75%, bf16, 640 / 320 calibration requests | 24.7% removed | 5 (1/1) / 8 (3/0) | .0050 / .0052 | 1.000 | .962 / .981 | .546 | .783 / .785 | meets / misses by 1 |
| bf16 2:4, state rows L12-22 (generic / banking-only masks) | 46.2% of state-row MACs | 7 / 7 | .0062 / .0059 | 1.000 | .952 / .971 | .523 | .790 / .788 | meets / meets |
| b8 + int4 2:4, state rows L13-23 | 44.5% of state-row MACs | 8 (4/1) | .0066 | .972 | .962 | .538 | .788 | misses |
| b8 + B12 75% (alone or with int4 mixers) | 24.7% removed | 10 (8/3) | .0064-.0075 | .982-.991 | .943-.962 | .538-.546 | .783-.788 | fails |

The 60% stack loses exactly the two airline CF pairs b8 alone loses. Paired vs bf16 and JB-hard McNemar are not significant in any row.

**Banking subset, deployable passes** (B12 / int4 / int8):
- REAL: 3/883, 3/883, 6/883;
- LONG: 3, 2 and 4 of 471;
- CF-probe: 33, 34 and 34 of 36.

hobson tracks 1 of 161 banking CF pairs, so banking CF retention is not measurable.

**B11** [M], dev, layers 12-22.
- All rows:
  - magnitude: 4.5e-2;
  - Wanda: 2.5e-2;
  - SparseGPT with the layer Hessian: 2.3e-3;
  - SparseGPT with H_d: 9.6e-4;
  - decision Fisher: 9.8e-4.

  Decision weighting beats per-layer error 2.2-2.4x, in bf16 and int8.
- State rows only: 1.0e-4, because question rows carry the late-layer sensitivity.
- Per-layer scan:
  - any one of layers 8-11 costs 1.5-2.0e-3;
  - each of layers 14-23 costs under 1e-5 on state rows.
- Question rows in int4 2:4 fail.

**B12** [M]:
- **Exactly dead.** None: all 147,456 neurons exceed |m| = 0.074 on both domains, so exact removal saves 0 FLOPs.
- **Ranked removal**, 25% per layer of L12-23 (dev):
  - variance ranking: 1.8e-3;
  - decision saliency: 3.7e-4;
  - with decision-weighted compensation: 4.2e-5.
- **Global allocation across layers** cuts KL another 10x.
- **The 60% setting keeps** layers 12-13 whole and 488-1,229 neurons per layer in 17-22: late MLPs barely matter, as R1 found.

FLOPs against flips, kit (banking):

| removal | GEMM MACs removed | REAL flips, kit | REAL flips, banking |
|---|---|---|---|
| uniform 50% | 16.5% | 8 | 8 |
| global 60% | 19.8% | 6 | 6 |
| global 75% | 24.7% | 8 (5 with 640 calibration requests) | 8 (5) |

On dev, global 50 / 60 / 75 / 85 / 90% gives 1.4e-5 / 2.5e-5 / 5.4e-5 / 2.5e-4 / 9.5e-4. Layers 8-11 fail.

**B13** [V+M].
- **Layer 23.** The state rows compute only K/V, and everything else they would compute is NaN. No NaN reached any of 3,227 logit vectors, so those values are never read.
- **Layer 0.** The per-token table of layer 0's input projection is a function of the token id.
- **In bf16, row-subset GEMMs change cuBLAS rounding:**
  - layer-23 check: max |Δlogit| 0.031, 1 flip in 3,227;
  - table check: bit-identical on 93.8% of questions, 1 flip.
- **In b8 (600 questions, both changes):** 0 flips, bit-identical on 583. The other 17 are short requests where the emulator's int8 GEMM falls back to a non-exact path for 16 or fewer rows.
- **Table size.** Banking train traffic's 13,376 token ids cover 99.75% of banking eval tokens: a 220 MB bf16 table.

**B7** [M], dev:
- **Neuron removal (global 75%), 320 calibration requests:** banking-only gives banking KL 4.8e-5 and retail-only gives retail 1.6e-5, 12% and 30% better than mixed. Retail-only gives banking 2.1e-3, 44x worse: mismatch is what costs.
- **2:4 masks.** Banking-only and generic masks differ by 1-2 kit items.
- **GPTQ precision.**
  - b8: no gain from per-deployment calibration (banking KL 1.2e-4 banking-calibrated vs 9.5e-5 pool).
  - Q1's w4q8: matched calibration helps only the domain the pool under-represents (retail flips 5 → 2 of 96).

## Speed against W8A8-b8

- **B12** [M], this box, GPU idle; cuBLAS int8/bf16; the 96-GEMM sequence; median of 30 runs. GEMM time vs b8 at M = 381 / 1,125 / 4,125:
  - 60%: 0.81 / 0.82 / 0.82x;
  - 60% + B13: 0.79x at every size;
  - 75% + B13: 0.75 / 0.75 / 0.74x;
  - B13 alone: 0.95x.
- **2:4** [M], Q4's kernels, 1000+125 rows: int4 2:4 state rows L12-22 0.83x; int8 0.92x; int8 plus B13 end to end 0.89x.
- **End to end at 1000 tokens** [A], from Q4's 36.35 ms (27.0 ms GEMMs):
  - b8 + B12 60% + B13 ≈ 30.7 ms (0.84x);
  - the 60% stack is 0.69x of GEMM time nominally;
  - the failing 75% stack is 0.64x.
- **Against the bar** (≤ 0.65x GEMM, ≤ 27 ms): no accurate setting reaches it. Layers 0-11 hold 54% of the MACs and 92% of the MLP decision saliency, and no training-free structure survived there.

## Projections [A]

b8 at 1000 tokens: 3090 19.9 ms, 4090 11.6 ms, 5090 7.9 ms.
- **B12 60% + B13** (GEMM share 74%, ratio 0.79): ≈ 16.8, 9.8 and 6.7 ms. B12 also removes 20% of weight bytes, which helps short, weight-bound 4090 requests.
- **int8 2:4 state rows + B13** (sparse tensor cores exist on all three GeForce generations): Q4's 17.7, 10.3 and 7.0 ms.

## Most valuable next step

Integrate **b8 + B12 60% + B13** into QRT and time it end to end. It needs only narrower GEMMs and a Hadamard at each down projection's kept width. Then add int4 2:4 state rows on layers 14-23's mixers, with a b8 GPTQ draw that keeps CF. Beyond that, structure must reach layers 0-11, which needs training: Q3's distillation at the B12 widths.
