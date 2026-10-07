# J6: the question in the weights

[M] measured, [V] verified, [A] arithmetic, [S] speculation. Box j6 (A10G). Files: `~/decider2/j6/`; checkpoints on the box.

## 1. Hypothesis
- **Where the time goes.** hobson re-reads every deployed question on every request: |q| = 78–1,201 tokens of full 2B compute, and Σ|q| ≈ 1.5k tokens for the deployed 4- and 15-question sets.
- **The idea.** The question text is a per-deployment constant, so compile it into weights. Each question gets:
  - a low-rank delta (shared r16 + per-question r8 LoRA on Win/Wo/Wd, all 24 layers);
  - K+1 slot rows (one per option, plus the answer row).
- **Cost.** T + Σ(K+1) rows instead of T + Σ|q|, a 27–42x smaller question term [A].
- **Two placements:**
  - **late:** the delta touches only the slot rows. The state is read question-blind, so the state rows are exactly hobson's and are shared by all questions.
  - **early:** the delta also touches the state rows (question-aware reading). This costs n·(T+K+1). Comparing the two answers Brooker's question.
- **Why it is not marginal.** Cost goes from O(T + Σ|q|) to O(T + nK), and hobson's state pass is untouched. h7's compiled schema reached the same row count by retraining the state reading and stalled at REAL agree_sd .685.

## 2. Built
- **Model.** A differentiable hobson split into a question-blind state pass (its cache holds GDN states, conv tails and K/V) and branch segments that continue it. Per-question deltas are batched by padded bmm (Punica/S-LoRA style).
- **Checks.**
  - [V] The teacher matches hobson's references: 49/49, max |dp| .0056.
  - [V] The fused runtime matches the library: in context 59/60, late 60/60, max |dp| ≤ .009.
- **(a) Per-question adapters** for all 38 deployed questions. Distilled from hobson (KL plus h7's dense row term) for 120 min on train_pool. Then 704 updates distilling hobson's own answers on 50% counterfactually edited train states (h7's human/amount/wrap-up edits; construction labels unused).
- **Brooker test.** Two arms from the same checkpoint on identical samples: late, and early, which adds a zero-initialised per-question r8 delta on the state rows. At init they agree to |dp| ≤ .004 [V].
- **(b) Hypernetwork.** hobson's no-state reading of the question → MLP → delta U·C(q)·V (r24) plus slot inputs. Trained on 31 deployed questions and 9.6k train_v5 rows; 7 deployed questions held out.
- **Latency arms.** d1/H4 fused kernels; deltas enter at GEMM outputs or residual pre-adds.

## 3. Accuracy, absolute [M]
McNemar counts are (model right, hobson wrong) / (hobson right, model wrong).

| suite | hobson | (a) late, real traffic | late + CF-state distil | early | hypernet |
|---|---|---|---|---|---|
| JB-all / JB-hard | .723 / .523 | n/a | n/a | n/a | .524 (19/65, p 5e-7) / .423 (18/31, p .09) |
| REAL agree / agree_sd | 1 / 1 | .934 / .864 | .930 / .858 | .934 / .870 | .790 / .514 |
| LONG agree / agree_sd | 1 / 1 | .951 / .891 | .924 / .873 | .913 / .848 | .843 / .618 |
| agree_sd without the 2 procedure questions (REAL / LONG) | 1 / 1 | .900 / .957 | .908 / .966 | .916 / .957 | – |
| REAL-label (Brier) | .785 (.347) | .785 (.360), p 1 | .777 (.365), p .69 | .790 (.366), p .84 | .728 (.394), p .009 |
| CF acc / flip / fgh | .594 / .268 / 1 | .514 / .126 / .43 (7/72, p 1e-14) | .624 / .352 / .86 (52/27, p .007) | .633 / .367 / .90 (54/22, p 3e-4) | .475 / .049 / .17 |
| CF-probe acc / flip | .653 / .328 | n/a | n/a | n/a | .503 / .006 |
| SHUF both_right | .245 | .114 | .331 | .349 | .044 |

- **CF on real traffic alone.** (a) misses human_insert (.02 against .35): real traffic has almost no such cases.
- **CF after edited-state distillation.** human_insert rises to .60 and wrap-up to .96, both above hobson. These gains are template-in-distribution: amount_insert falls to .03 (hobson .10), and the identity and procedure kinds stay at 0, as they do for hobson.
- **The agreement gap is the two many-way procedure questions.** 87–93% of those misses are near-ties for hobson itself (top-2 margin < .2). A capacity probe (+r32 on those questions, 30 min, same samples) does not help: REAL/LONG agree_sd .771/.755 against .760/.776 for the r8 control, and the extra rank had worse training KL in 32 of 32 windows.
- **Brooker's question (early vs late, paired).**
  - early is better on CF ground truth (8 pairs vs 1, p .04).
  - It is no different on REAL-label (p .13) or REAL agree_sd (p .39).
  - It is worse on LONG agree_sd (0 vs 4, p .13), and its training KL was worse in 66 of 70 windows.
- **The hypernet does not generalise.** On held-out questions, REAL agree_sd is .26 and LONG .17, below no-state. REAL-label is .672 against hobson's .844 on the same items (p .0005).

## 4. Latency, A10G, bf16, median ms [M]
Exclusive GPU, n = 20 fresh real states. p95 is within 1.5 ms except early Q15 at T = 4000 (+50). ctx = the best of the plain, packed and sequential in-context layouts.

| set | arm | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|---|
| Q1 procedure (1,014 tok) | ctx | 56.8 | 57.3 | 61.6 | 80.7 | 103.9 | 246.8 |
| | **late** | **10.5** | **16.0** | **23.1** | **28.6** | **54.0** | **201.2** |
| Q1 states_amount (84 tok) | ctx | 15.3 | 15.6 | 23.0 | 28.1 | 56.8 | 200.6 |
| | late | 12.0 | 17.2 | 24.8 | 29.8 | 55.6 | 202.8 |
| Q4 deployed | ctx | 101.0 | 101.4 | 106.4 | 113.4 | 146.3 | 294.7 |
| | **late** | **18.8** | **23.0** | **32.6** | **40.3** | **64.8** | **211.1** |
| | early | 39.7 | 54.2 | 77.2 | 118.7 | 251.4 | 971.8 |
| Q15 deployed | ctx | 97.6 | 98.0 | 102.6 | 107.6 | 132.8 | 292.3 |
| | **late** | **19.8** | **24.0** | **33.5** | **41.2** | **65.7** | **212.1** |
| | early | 67 | 128 | 240 | 360 | 876 | 3,632 |

- **Multi-question:** late is 4.9–5.4x faster at 64 tokens, 2.0–2.3x at 1000 and 1.4x at 4000.
- **Long single question:** 5.4x / 1.9x / 1.2x at the same lengths.
- **One short question:** no gain beyond 64 tokens; the unfused delta kernels cost about 2.5 ms.
- **train_pool traffic:** row ratio 1.32x on average, 2.2–2.7x for states under 1k tokens [A].
- **Projections at T = 1000 (GEMMs scaled by tensor rate, the rest by bandwidth) [A/S]:**

  | arm | 3090 | 4090 | 5090 |
  |---|---|---|---|
  | Q15 ctx | 69.5 | 34.7 | 25.4 |
  | Q15 late | 35.9 | 22.0 | 14.6 |
  | Q1 procedure ctx | 54.9 | 29.5 | 20.8 |
  | Q1 procedure late | 28.6 | 15.4 | 10.8 |

  At T = 64, Q4 late projects to 11.3 / 9.2 / 5.4 ms.

## 5. Verdict
- **The late-bound compiled question survives as a cost-class change** for multi-question and long-question requests: 2–5x at T ≤ 1000 [M].
  - On the deployed questions there is no significant REAL-label drop, and CF is significantly above hobson, with the in-distribution caveat [M].
  - It **fails the fidelity bar**: agree_sd .86–.87 against .95. The misses are near-tie decisions on the two procedure questions, and extra capacity does not fix them.
- **The hypernetwork fails on arbitrary questions:** JB-all drops 20 points and CF-probe is at chance. Unseen questions must therefore stay in context. Because late keeps hobson's weights and state rows, those questions share the same state cache, so JB for the combined system equals hobson's [A].
- **Brooker's question:** at equal training, question-aware reading neither clearly helps nor hurts, and it costs n× the state. Question-blind reading plus a compiled question is the right factorisation.
- **Not measured:** late on W8A8 state rows; LONG REAL-label (no long labelled items).

## 6. Decisive next step
Run the state rows in the W8A8-GPTQ kernels and keep the K+1 compiled rows in bf16. H6 measured this layout's cost at 40.6 ms for 15 questions at T = 1000 [M], so late should land near 43 ms on the A10G and 22 ms on a 3090 [A/S]. In the same run, distil on all 46k train_pool pairs plus edited states, and gate on agree_sd ≥ .95 measured on decisions where hobson's margin is ≥ .2, and REAL-label ≥ .78.

Fetched: T2L 2506.06105; S-LoRA 2311.03285; Punica 2310.18547; context distillation 2209.15189; Prompt Injection 2206.11349; HyperTuning 2211.12485; DnD 2506.16406; Gisting 2304.08467; LoRA 2106.09685; HyperNetworks 1609.09106.
