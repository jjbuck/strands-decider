# J2: a bidirectional GDN-hybrid encoder from hobson's own torso (box j2, A10G)

Labels: **[M]** measured, **[V]** verified, **[S]** speculation or arithmetic.

Files: `~/decider2/j2/` (code/, NOTES.md, score_final.*, lat.json, preds/); checkpoints on the box.

## 1. Hypothesis
hobson's state rows are causal: row t encodes only s1..st. The question reads the state through two paths:
- 6 attention layers;
- each GDN layer's fixed-size forward state, which is weighted toward recent tokens.

So any comparison between an early detail and later context has to happen at the question rows. hobson tracks 17% of CF edits in the first half of the state and 34% in the second [M].

A bidirectional state lets every row absorb what follows it, at the same cost class:
- a reverse delta-rule scan reuses q, k, v, β and g, so it adds only GDN-kernel time, which is linear in T;
- non-causal attention touches only the 6 attention layers;
- the GEMMs (86–91% of time) are unchanged.

A full-attention encoder pays 4·T²·d in every layer [S].

Precedents (fetched): BiMamba 2403.03234 (flipped pass, shared projections, summed), Vision Mamba 2401.09417, Hydra 2407.09941, LION 2502.16249, Gated DeltaNet 2412.06464, LLM2Vec 2404.05961.

## 2. Why it is not marginal
It changes what the state rows can know, at the same size and cost class. On identical weights it also tests Brooker's open question: should the state see the question?

## 3. What I built
- **Function-preserving bidirectionalisation** (`j2lib.py`, packed varlen, fla kernels):
  - GDN: o = o_fwd + γ⊙o_rev.
  - Attention: o = o_c + λ_h(o_nc − o_c).
  - γ and λ start at 0, so the untrained model is hobson: max |Δlogit| 0.03, and the eval pipeline agrees with hobson on .996 of JB-all and .988 of CF-probe decisions [V].
- **qag (masked state cache, e1b-like).**
  - Forward stream s1..sT, then q1..qL. Reverse stream sT..s1, then qL..q1.
  - Each question's reverse scan therefore starts from the state's reverse final state.
  - The state attends to itself; questions attend to the state and themselves.
  - State rows are bit-identical under different questions [V].
- **qa (question-aware, e1a-like):** a plain bidirectional pass over [S;Q].
- **early:** reverse scans in GDN layers 0–10 only.
- **Adaptation.**
  - MNTP: 200 steps × 8k tokens of real train-split states, 20% masked, each token predicted from i−1 through the tied embedding.
  - Then F7-recipe fine-tuning: LoRA r16/α32 on all projections, hobson's pointer head, CE + KL to calibrated hobson, 280 steps × 32 rows (4.6M tokens).
  - Every arm sees identical rows.
  - The budget is about 1/7 of F7's epoch, the same for every arm.
- **Controls, same fine-tuning:** `causal` (no MNTP) and `causal_mntp` (the identical MNTP, run causally). `causal_mntp` is the matched control.
- **Fused runtime** (`j2lat.py`: d1's lean2 kernels, reverse scans, state shared across questions): answer-row cosine 0.99997 against j2lib [V].

## 4. Accuracy on every suite [M]
LONG was not run for causal_mntp or early (time). The CF and CF-probe columns give item accuracy / pair accuracy.

| model | JB-all | JB-hard | REAL-label | REAL agree_sd | LONG agree / sd | CF | CF-probe | SHUF | Brier / ECE |
|---|---|---|---|---|---|---|---|---|---|
| hobson | .723 | .523 | .785 | 1 | 1 / 1 | .594 / .268 | .653 / .328 | .245 | .436 / .037 |
| causal | .714 | .508 | .772 | .954 | .977 / .964 | .579 / .254 | .642 / .303 | .230 | .449 / .064 |
| causal_mntp | .740 | .562 | .790 | .899 | – | .599 / .271 | .661 / .331 | .253 | .425 / .044 |
| qag_all | .745 | .569 | .785 | .919 | .932 / .855 | .599 / .286 | .659 / .331 | .266 | .422 / .047 |
| qa_all | .740 | .562 | .792 | .919 | .936 / .867 | .607 / .308 | .655 / .325 | .292 | .422 / .048 |
| qag_early | .753 | .585 | .780 | .916 | – | .595 / .281 | .659 / .331 | .269 | .422 / .059 |

**Paired exact McNemar tests**, as (model-only right / other-only right, p):

| model | vs hobson | vs causal_mntp |
|---|---|---|
| qag_all | JB-hard 9/3 (p .15), CF pairs 17/10 (.25), CF-probe 30/29, REAL-label 10/10 | JB-hard 8/7, CF 12/6 (.24), CF-probe 20/20 |
| qa_all | JB-hard 8/3 (.23), **CF pairs 23/7 (.005)**, CF-probe 29/30, REAL-label 12/9 | JB 7/7, **CF pairs 25/10 (.017)**, CF items 26/19 (.37), CF-probe 19/21 |
| qag_early | JB-hard 11/3 (.057), CF 18/13 | all p > .5 |
| causal_mntp | JB-hard 6/1 (.13), CF 13/12 | – |

- **Against the plain causal fine-tune**, every bidirectional arm looks significantly better: CF 20/7 (p .019) for qag and 28/6 (p .0002) for qa; JB-hard 12/2 (p .013) for early. The matched control removes most of this, so the gain came from the LM-style adaptation on real states, not from bidirectionality [M].
- **qa against qag:** CF 15/6 (p .078); the two agree on .98 of REAL decisions.
- **early against all:** mutual agreement .979; CF 10/12. The two are indistinguishable.
- **No arm changes:**
  - CF on states of at least 4k tokens (.01–.03);
  - CF-probe date_order and the `_distract` kinds.
- **The reverse paths do carry information:** at MNTP step 200 the bidirectional loss is 1.17 nats against 1.67 for the causal run. The gates end at mean |γ| .04 and mean λ .17.
- **The bar:** all three bidirectional arms pass it (no JB drop, REAL-label ≥ .78, CF and CF-probe ≥ hobson). So does causal_mntp.
- **J1's full-attention encoder** (T5Gemma e1b, full F7 epoch; from `j1/NOTES.md`) scores JB-hard .492, REAL-label .770, CF pair .170 and CF-probe pair .222. Every hybrid arm beats it.

## 5. Latency [M]
A10G, fused runtime, CUDA graph, exclusive GPU, fresh ids, 20 reps. T is the number of state tokens, plus 100 question tokens. Values are median ms (difference from hobson in brackets); p95 is within 0.1 ms of the median (`lat.json`).

| 1 question | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| hobson | 15.6 | 15.9 | 25.4 | 32.9 | 57.1 | 205.2 |
| bi_all (λ-mix; qa identical) | 17.1 (+1.5) | 17.9 | 28.0 | 36.4 | 64.8 (+13%) | 243.6 (+19%) |
| bi_all, pure non-causal | 16.9 | 17.5 | 27.6 | 35.9 | 63.3 (+11%) | 233.4 (+14%) |
| bi_early | 16.4 (+0.8) | 17.0 | 26.9 | 34.9 | 61.8 (+8%) | 232.7 (+13%) |

**Four questions**, in the order hobson / bi_all / bi_early / qa:
- 1000 tokens: 83.3 / 94.4 / 90.1 / 240.1 ms;
- 4000 tokens: 222.8 / 265.4 / 253.2 / 939.7 ms.

qa cannot share the state across questions, so it costs 2.9–4.2x hobson.

**Where the time goes:**
- the reverse scans cost about 5.5 ms per 1000 tokens because the flips are unfused; fusing them into the conv and gated-norm kernels should bring this to about 3.4 ms [S];
- the λ-mix attention adds about 1.5 ms at 1000 tokens and 10–17 ms at 4000.

**Against J1's encoder** (J1's harness): J1 measured its encoder at 1.17x, 1.51x and 1.66x hobson at 64, 1000 and 4000 tokens. bi_all is 1.10x, 1.13x and 1.19x. The hybrid is faster than the full-attention encoder at every length, and the gap widens with T [M, two harnesses].

## 6. Projections, 1 question [S]
Method: hobson's part is scaled by the documented card ratio; the extra time is scaled by DRAM bandwidth.

| model, 1000 tokens | 3090 | 4090 | 5090 |
|---|---|---|---|
| hobson | 29.7 | 14.8 | 10.8 |
| bi_all | 34.6 | 19.4 | 13.4 |
| bi_early, pure non-causal | ≈32 | ≈17 | ≈12 |

At 4000 tokens, bi_all projects to 131, 76 and 52 ms, against hobson's 107, 53 and 39 ms. W8A8 stacks on top, because it touches only the GEMMs.

## 7. Verdict
**The architecture survives; the transformative claim fails.**
- **What holds [M].** A function-preserving bidirectional GDN hybrid built from hobson:
  - passes every accuracy bar;
  - keeps linear scaling at +8–19% latency;
  - beats the full-attention encoder on accuracy and on speed at every length.
- **What fails [M].** At equal data and budget, bidirectional state reading does not make the masked-cache model read details better than a causal model given the same LM adaptation. qag ties causal_mntp everywhere, and distractor, date and long-state reading are unchanged.
- **The one residual effect: question-aware reading.** qa is the only arm better than hobson on CF pairs (23/7, p .005) and better than the matched control (25/10, p .017). These p-values are uncorrected over about 20 tests, so the effect is suggestive.
- **Brooker's question.** Letting the state see the question is not a weakness here; it is the only variant that helped. It costs one pass per question.
- **Side result [M].** A causal MNTP on real states alone lifts JB-hard by 3.9 points and REAL-label to .790. Neither gain is significant against hobson.

## 8. Decisive next step
Train qa_all and causal_mntp for the full F7 epoch (about 4 h each on one A10G), adding reading augmentation built from held-out templates. Then repeat the paired CF, CF-probe and REAL-label tests:
- if qa keeps a significant edge over the matched control, ship qa for single-question requests (9 in 10 JF100 requests) and qag_early for multi-question requests;
- if it does not, bidirectionality is not the lever for detail reading, and the effort belongs in training data.
