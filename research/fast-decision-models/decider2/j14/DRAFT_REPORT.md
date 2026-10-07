# J14: compiling the question bundle, state first (box j14, A10G)

Labels: [M] measured, [V] verified, [A] arithmetic from measured anchors, [S] speculation. Everything is in `~/decider2/j14/`: `code/`, `preds/`, `score_*.txt`, `lat_*.jsonl`, `lat_table.txt` and `NOTES.md`.

## Hypothesis
A row costs about 2.745 GFLOP, and every hobson question row depends on the state. Yet question text is a deployment constant, 72–93% of the rows in short requests. Compile each question once per deployment and recompute only a few live rows: a request then pays for T + Σlive rows instead of T + Σ|q|. That is exact work elimination, and it would put multi-question short requests below the int8 knee.

The layout I tested is new: state first, in place.
- **State rows** are exactly hobson's, since in hobson they never see the question.
- **Question rows keep hobson's positions.**
- **Compiled rows** are hobson's own no-state computation, cached as pre-RoPE K/V plus GDN k/v/β/g. At runtime the K/V is re-rotated, and the GDN values are replayed from the state's final GDN state (exact delta-rule composition).
- **Live rows** run hobson's weights.

This leaves exactly one approximation: compiled rows do not see the state. [V] When the state is the compile context, the result is identical to hobson (|dp| ≤ 3e-4); with every row live it equals plain hobson (≤ .003). Unlike H7's schema, there is no reorder and the state rows are untouched.

## What I built
- **`j14lib.py`.** A differentiable lean hobson with a no-grad cached state prefix, so training costs one hobson prefix plus small suffix passes. It has a live-set dial and a policy `H8` that leaves questions with more than 8 options uncompiled.
- **`j14train.py`.** KL to state-first hobson plus relative MSE on the live rows (layers 5/11/17/23). Data is train_pool and train_v5 (no eval tasks, no CF augmentation). Selection is on 120 held-out train-split requests.

| arm | adaptation | live rows per question |
|---|---|---|
| A | compile adapter (LoRA r16 on compile rows, as in J9) + live-row LoRA r64 on the question rows only | option ends + suffix |
| B | **full FT**, one weight set (stochastic-rounding Adam, L2-SP λ 300, drift 0.87%) | as A |
| C | as A, procedure questions weighted 2x | option block (≤ 8 options) or the last 4 rows of each option line (option definitions as candidates) + suffix |
| E | compile adapter only; the live path is exactly hobson | as C |
| G | compile adapter only | last 32 instruction rows + last 4 rows of each option line + suffix |
| F | compile adapter only | option block + last 32 instruction rows + suffix |

- **`j14rt.py`.** H2's QRT (bf16 and W8A8-b8) on J5's kernels, with compiled span buffers, GDN replay or an exact affine transfer, and masked SDPA. [V] Against the library: argmax 11/11, max |dp| median .0014.

## Results, all 3,227 questions [M]
p is an exact McNemar test against hobson; fgh is the share of hobson-tracked pairs the model also tracks.

| | hobson | A | B | C | E | G | **F** | bar |
|---|---|---|---|---|---|---|---|---|
| question rows compiled | 0 | 92% | 92% | 69% | 69% | 68% | **12%** | |
| REAL agree_sd | 1 | .720 | .613 | .827 | .717 | .766 | **.951** | ≥ .95 |
| LONG agree_sd | 1 | .764 | .727 | .855 | .794 | .782 | **.952** | ≥ .95 |
| CF pair acc / fgh | .268 / 1 | .170 / .19 | .089 / .30 | .140 / .50 | .118 / .41 | .200 / .70 | **.276 / 1.00** | fgh ≥ .90 |
| CF-probe pair acc / fgh | .328 / 1 | .019 / .03 | .006 / .01 | .150 / .33 | .134 / .30 | .263 / .63 | **.319 / .91** | fgh ≥ .90 |
| JB-hard (p) | .523 | .508 (.84) | .415 (.04) | .515 (1.0) | .538 (.69) | .515 (1.0) | .515 (1.0) | no drop |
| REAL-label (p) | .785 | .703 (<.001) | .728 (.005) | .785 (1.0) | .752 (.06) | .752 (.06) | **.782** (1.0) | ≥ .78 |
| JB-all | .723 | .693 | .567 | .714 | .727 | .710 | .719 | |

Brier on REAL-label: hobson .347, F .345, C .362, A .429.

With H8, agreement rises but CF is unchanged:

| arm + H8 | REAL agree_sd | LONG agree_sd |
|---|---|---|
| C | .922 | .927 |
| E | .864 | .958 |
| G | .902 | .933 |
| F | .968 | .970 |

- **F passes every bar.** It is J9's compile adapter on the question header and instruction head, with hobson's live path intact. But it compiles 12% of question rows (banking 15-question bundle 27%, median JevBench question 0%).
- **Compiling most of a question destroys detail reading.**
  - A answers identically on both items of 195 of 320 CF-probe pairs.
  - Agreement on real traffic comes from correlates across the 37 training specs.
- **Full FT (B) is worse than LoRA everywhere, and it forgets.** In hobson's own layout its weights score JB-all .619.
- **The 23-way procedure questions are the largest single loss**, as J9 found. Leaving them uncompiled (H8) fixes them by construction.

## Mechanism: the question's content rows are the readers [M]
Untrained hobson, with more question rows made live:

| live rows / 204 | REAL agree_sd | CF fgh | CF-probe fgh |
|---|---|---|---|
| 17: option ends + suffix | .055 | .00 | .01 |
| 49: + last 32 instruction rows | .254 | .26 | .47 |
| 148: option block | .555 | .35 | .11 |
| 180: option block + 32 instruction rows | .832 | .98 | .85 |
| 200: option block + 64 instruction rows | .974 | 1.00 | .98 |

hobson gathers evidence at the rows that carry the question's content: the statement and the option criteria. The `<answer>` and option rows only read those rows.

A compile adapter can make the remaining boilerplate transparent: F goes from .832 to .951 in 150 updates. It cannot teach compiled content rows to read a state they never see. Live-row LoRA and full FT learn shortcuts instead.

This inverts J9's case. There, compiled documents were read by live question rows; here the compiled rows are the readers.

## Latency, A10G [M]
Exclusive GPU, exact shapes, CUDA graph, fresh state ids, 15 reps, median in ms (p95 within ≤ 2 ms).
- `oa` is A's live set (17 rows per question); F is F's live set.
- jb1/jb4 are real JevBench questions; bk4/bk15 are banking (bk15 = 1,733 question tokens).

| W8A8-b8, T = | 32 | 64 | 128 | 256 | 400 | 1000 |
|---|---|---|---|---|---|---|
| jb1 plain / oa | 7.98 / 7.52 | 9.36 / 8.61 | 10.79 / 10.12 | 15.34 / 14.74 | 19.70 / 19.35 | 41.83 / 40.50 |
| jb4 plain / oa | 18.96 / 9.68 | 20.73 / 10.24 | 23.59 / 12.82 | 28.20 / 17.00 | 33.24 / 22.83 | 55.34 / 43.92 |
| bk4 plain / F | 18.45 / 15.59 | 18.90 / 17.80 | 21.22 / 19.68 | 25.07 / 24.65 | 31.02 / 29.35 | 53.73 / 51.78 |
| bk15 plain / oa / F | 68.98 / 20.27 / 58.42 | 71.12 / 22.02 / 59.08 | 72.63 / 24.75 / 60.99 | 78.39 / 28.39 / 66.89 | 84.04 / 32.66 / 73.30 | 107.65 / 53.70 / 95.50 |

bf16, plain → oa, T = 32: jb1 10.57 → 9.74, jb4 25.07 → 11.84, bk15 92.92 → 23.90 (T = 1000: 146.09 → 71.53).

**Where the time goes:**
- One question already sits at the int8 knee, so even 92% compilation saves 6%.
- At 15 questions the GEMMs fall from 48.4 to 8.0 ms, but GDN replay of the compiled rows still costs 7.45 ms.
- The affine transfer halves the GDN kernels, but an fp32 bmm gives the saving back.
- About 400 extra small kernels add about 1 ms. Fusion would recover about 1–4 ms [S].

**Projections** (J5's method) [A], W8A8-b8, T = 32, 3090 / 4090 / 5090:

| | plain | oa | F |
|---|---|---|---|
| bk4 | 11.5 / 7.8 / 5.8 | 6.8 / 5.4 / 4.0 | 10.2 / 7.2 / 5.5 |
| bk15 | 41.4 / 25.1 / 19.9 | 14.6 / 11.0 / 8.7 | 36.8 / 23.7 / 18.8 |

## Verdict [M]
- **The transformative version fails the bar.** Compiling 92% of question rows (A's live set) gives 2.0x at 4 questions and 3.4x at 15 (W8A8, T = 32); the 68–69% sets give 1.2–1.7x. All of them lose hobson's detail reading: CF-probe fgh .01–.63, CF fgh .19–.70. That holds for a compile adapter, live-row LoRA and full FT with L2-SP alike.
- **The faithful version passes every bar (F).** But it compiles 12% of rows and buys 1.04–1.18x at 4–15 questions and nothing at 1 (slower in this runtime), which makes it a marginal lever.
- **The boundary is structural.** In a state-first decider, the question's content tokens are what reads the state.

## Decisive next step
Train, not distill, a decider whose reading lives outside the question text:
- a question-agnostic state index (a masked bidirectional state, as in e1b);
- 8–16 learned query slots per question, which see the compiled question text only through K/V;
- CF-style detail objectives on far more than 37 specs.

Gate it on the ladder above: CF-probe fgh ≥ .90 with ≤ 16 live rows per question. At that point the measured 2–3.4x multi-question gain becomes usable. Until then, ship F's lossless part: compiled headers, H8 and packed W8A8.
