# J3: the depth-split decision transformer (DT)

Labels: [M] measured, [V] verified, [A] arithmetic, [S] speculation. Box j3 (A10G). Design: `DESIGN.md`; timeline:
`NOTES.md`; code: `code/`; numbers: `scores.json`, `lat_table.json`, `order/`, `holdout/`, `tf_scores.json`.

## 1. Hypothesis
A decoder gives every token 24 layers because any position may emit the next token. A decision model reads out only
the option-end and `<answer>` rows. A state row's deep outputs feed only later layers' attention and recurrences, which
only question rows consume, and hobson's state is question-independent. So about 89% of a 1000-token request's FLOPs is
generic deep processing that no output needs directly.

**DT design.**
- **State rows:** hobson's first L_s layers at full width, every token kept.
- **Question rows:** all 24 layers.
- **The bridge:** each deep layer reads the frozen layer-L_s state residual through its own K/V projections plus a
  trained memory adapter (the SwiftKV/YOCO pattern). Deep GDN layers recur over question rows only.
- **Cost:** state FLOPs per token fall to 0.18 / 0.35 / 0.51 / 0.67x at L_s = 4 / 8 / 12 / 16 [A], with the same 2B
  parameters.

**Why it is not marginal.** It changes the cost class of the dominant term: state cost scales with L_s/24.

**Why it is not a failed idea.** Selection, routing, pooling and small readers cut tokens or width. DT cuts only depth:
every deep attention layer still reads all T × 2048 state rows by softmax attention.

## 2. Built
- **`dtlib.py`:** DT on H3/H7's forward. [V] At L_s = 24 it equals hobson (56/56, max |Δp| 0).
- **`dtset.py` (DT-set):** option-order invariance by construction, as the engineer asked. Each question is a tree:
  a stem off the state, each unnumbered option its own branch at the same position, and an `<answer>` tail that reads
  its options as a set.
- **`train_dt.py`:**
  - hobson is the in-process teacher, with the F7 recipe: CE + KL on gold rows, KL on real train-split states, option
    shuffling and ordinal smoothing.
  - Counterfactual augmentation: H7's `cf_aug`, built on train-split states.
  - Dense distillation of the deep question rows.
  - LoRA r16 everywhere, plus rank-32 memory adapters.
  - The split is elastic: L_s = 8 with probability 2/3, else 12.
- **The two runs.** DT: 700 updates (11.2k sequences, 26M tokens, 2.45 h). DT-set: 400 updates, 1.4 h.
- **`dt_lat.py`:** a fused runtime: TTL kernels, one GEMM for all deep memory K/V, CUDA graph, exact T. [V] It matches
  the references on 40/40 decisions per configuration (max |Δp| ≤ .0085).

## 3. Results [M; every evalkit suite, absolute accuracy]

| | hobson | DT-A8 | DT-A12 | DT-set8 | **DT-set12** | DT-A16, untrained |
|---|---|---|---|---|---|---|
| state FLOPs | 1 | .35 | .51 | .35 | .51 | .67 |
| JB-all (McNemar p) | .723 | .684 (.20) | .693 (.14) | .654 (**.026**) | .688 (.12) | .736 (.25) |
| JB-hard (p) | .523 | .477 (.41) | .477 (.21) | .431 (.09) | .462 (.12) | .546 (.25) |
| REAL-label (bar ≥ .78) | .785 | **.782** | .770 | .760 | **.787** | .782 |
| CF pair accuracy | .268 | .719 | .722 | .732 | .732 | .271 |
| CF-probe pair accuracy | .328 | .988 | .997 | .994 | .997 | .319 |
| REAL / LONG agree_sd | 1 / 1 | .83 / .88 | .89 / .91 | .76 / .87 | .86 / .91 | 1.00 / .98 |
| Brier, JB-all / REAL-label | .348 / .347 | .413 / .347 | .370 / .351 | .419 / .362 | .360 / .328 | .347 / .349 |
| ECE, JB-all / REAL-label | .048 / .116 | .059 / .088 | .075 / .087 | .081 / .053 | .070 / .074 | .045 / .112 |
| decisions unchanged under option rotation | .919 | .902 | .928 | **.9985** | **.999** | – |
| mean abs Δp under rotation | .037 | .035 | .035 | **.0009** | **.0009** | – |
| new-architecture bar | – | **pass** | miss (REAL-label, by 4 items) | fail | **pass** | (fidelity) |

Rotations: 1,969 of REAL-agree and JB-all questions; DT-set's residual flips are bf16 near-ties.

**Two models pass the brief's bar.**
- **DT-A8:** 0.35x state FLOPs.
- **DT-set12:** 0.51x, invariant to option order, and better calibrated than hobson on REAL-label.

**Caveats.**
- **The CF gains belong to the recipe.** H7's hobson-layout fine-tune gets similar gains.
- **CF-probe templates are in-distribution.** The never-augmented identity and procedure CF kinds stay at 0, as for
  hobson.
- **Differences between the DT variants are within noise.** JB has n = 231 and REAL-label n = 400.
- **DT-set8's miss is the training budget, not the layout.** DT-set8 trained 400 updates. At the same 400 updates, DT-A8
  also misses (REAL-label .777, JB-all .667, p .06) [M].

**Untrained DT-A16 is a function-preserving trick to within the noise of the bf16 floor.**
- It flips 0.4% of REAL decisions (the bf16 floor) and 0.6% of LONG decisions.
- CF fgh is 1.000. CF-probe fgh is .943, against a floor of .971 (3 pairs).
- Untrained DT-G12 (0.57x, with a GDN memory scan) is nearly lossless: 0.7% REAL flips, 1.1% LONG flips, CF fgh 1.000,
  CF-probe fgh .952. It is 1.20x faster at 1000 tokens and 1.43x at 4000 [M].

[M] Training-free on the subset, REAL / LONG agree_sd: 12G 1.00 / 1.00, 12A .95 / .96, 8A .78 / .74, 4A .23 / .22.

**Held-out capability probe** [M; v19's `train_v5.holdout`, n = 150 per task]. Training contains ruletaker depths 0–2
only, so depths 3 and 5 test whether deduction over the state generalizes in depth.

| | hobson | DT-A8 | + 38 min of ruletaker-heavy data | DT-set12 | untrained 8 / 12 / 16 |
|---|---|---|---|---|---|
| ruletaker d3 | .860 | .647 | .687 | .800 | .653 / .833 / .860 |
| ruletaker d5 | .720 | .560 | .700 | .673 | .500 / .713 / .720 |
| ruletaker natlang | .780 | .660 | .773 | .767 | .593 / .780 / .780 |

The other four tasks (emotion, intent, hate, sarcasm) are within ±.06 of hobson.

- **The measured limit.** evalkit does not probe multi-hop composition inside the state, and at L_s = 8 that capability
  drops 12–21 points.
- **Mostly data, partly structure.** Targeted data recovers d5 and natlang to within 2 points, and the model still
  passes the evalkit bar (REAL-label .780, JB-all .675, p .13). d3 stays 17 points short.

## 4. Latency [M; A10G exclusive, bf16 fused, CUDA graph, exact T, fresh states, 20 reps]
Values are median ms with 1 question (93 question tokens); p95 is within 0.1 ms.

| T | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| hobson, one sequence (d1 layout) | 15.3 | 15.6 | 22.9 | 28.2 | 57.1 | 200.8 |
| hobson, two-pass (DT's harness) | 25.5 | 25.9 | 30.6 | 46.8 | 67.4 | 216.0 |
| DT-A4 | 16.4 | 16.4 | 17.3 | 20.2 | 24.2 | 52.5 |
| **DT-A8** | 18.2 | 18.3 | 20.0 | 25.5 | **33.0** | **85.3** |
| DT-A12 | 20.0 | 20.2 | 22.7 | 30.9 | 41.5 | 117.9 |
| **DT-set12** | 21.2 | 21.4 | 23.8 | 32.0 | **42.7** | **119.1** |
| DT-A16 | 21.9 | 22.1 | 25.3 | 36.2 | 50.2 | 150.6 |

- **Four questions (364 question tokens), 1000 / 4000 tokens.** hobson 95.1 / 245.5, DT-A8 60.4 / 114.5, DT-set12
  71.8 / 149.9. DT-set8 is within 1.1 ms of DT-A8.
- **Speedup against single-sequence hobson:** DT-A8 1.73x at 1000 and 2.36x at 4000; DT-set12 1.34x and 1.69x;
  DT-A4 3.8x at 4000.
- **Short states.** Below 256 tokens DT is 1–6 ms slower. The cause is unfused question-pass glue (gather-conv, masked
  SDPA, a second pass): the same-harness hobson is slower than DT everywhere. [A] Fused, DT-A8 would be about 26 ms
  at 1000.

**Projections** [A; 1000 / 4000 tokens, ms]. GEMMs are taken at 2x on GeForce (fp16 accumulation); other work scales
with bandwidth.

| card | hobson | DT-A8 | DT-set12 |
|---|---|---|---|
| 3090 | 29.7 / 104.9 | 17.9 / 45.1 | 22.7 / 62.6 |
| 4090 | 15.1 / 54.6 | 10.8 / 24.8 | 12.8 / 33.5 |
| 5090 | 11.0 / 39.1 | 7.2 / 17.3 | 8.8 / 23.7 |

[S] DT is orthogonal to W8A8. DT-A8 with W8A8 would be about 11–12 ms on a 3090 at 1000 tokens.

## 5. Verdict
**The bold claim, that uniform depth per token is wasted in a decision model, survives on real-traffic decisions and is
partly falsified for deep composition.**
- **Survives [M].** With about 26M tokens of training, DT-A8 matches hobson's absolute accuracy on every evalkit suite
  at 0.35x state FLOPs, which is 1.7–2.4x faster at 1–4k tokens on the same 2B. DT-set12 does the same at 0.51x, is
  1.3–1.7x faster, and makes option order irrelevant (unchanged .999 against hobson's .919).
  Untrained, state depth past layer 16 is redundant, and nearly so past 12.
- **Partly falsified [M].** Deeper deduction over the state needs state depth: at 8 layers, ruletaker d3 is −17 points
  even after targeted data.
- **Short requests [M].** Below 256 tokens there is no win.

So DT is a real cost-class change for long states, with a measured capability boundary. It is not a free 3x.

## 6. Decisive next step
Train **DT-set8 at F7 scale**: the full 108k-row v19 mix, with ruletaker d3 and d5 still held out, plus all 46k
train-pool pairs. That is about 4x this run's data.
- **Pre-register three gates:** the evalkit bar; held-out ruletaker d3 and d5 within 3 points of hobson; option
  invariance ≥ .995.
- **Fuse the question pass** into a single-sequence shallow path, to remove the short-length penalty.

If d3 still trails by more than 5 points, depth-on-state is structural for composition, and L_s = 12 is the floor.
