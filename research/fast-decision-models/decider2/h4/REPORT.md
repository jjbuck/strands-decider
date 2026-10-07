# H4 report: this-that-model-1.0 against hobson-v19 (coordinator's condensed copy of H4's final message)

## Model
- **Same torso as hobson.** this-that-1.0's config is identical to decider-2b's and to Qwen3.5-2B-Base's text config, the base hobson is built on.
- **Same cost.** In the fused runtime both models take 52.7 ms at 1000 tokens.
- **Weights.**
  - this-that ≈ decider-2b plus a 0.2–0.5% full-rank change; decider-2b ≈ the base plus 1–3%.
  - hobson is the base plus a rank-16 LoRA, so the two models are siblings.
- **Head.** A softmax over the tied-embedding rows of the letter labels, read at "Answer: (".

## Release gaps
- No prefix cache is shipped.
- The default truncation (1536 tokens) keeps the HEAD of the state and drops the newest turns. This costs 7 points of REAL-label.

## The 30.9 ms claim
- The input was 176–183-token prompts, one question per pass, HF eager, on an RTX 5080 laptop GPU.
- On the A10G, the repo's own script gives 47.6 ms; our fused runtime gives 16.0 ms.
- The cohort result reproduces: 0.941 accuracy, Brier 0.041.

## Suites (state-first, against bf16 hobson)

| | this-that | hobson |
|---|---|---|
| JB-hard | .477 (McNemar p .41) | .523 |
| REAL agree_sd | .806 | 1 |
| LONG agree_sd | .636 | 1 |
| REAL-label | .698 | .785 |
| CF accuracy / flip | **.812 / .633** | .594 / .268 |
| CF-probe accuracy / flip | .669 / .338 | .653 / .328 |
| SHUF both right | .636 | .245 |

- **CF-probe in the schema-first layout:** flip .553; date_order .84 (hobson .06).
- **Where hobson wins:** it wins clearly on question meanings it was trained on:
  - cc_can_still_help .94 vs .22;
  - UserAskedForHuman .95 vs .58;
  - needed_procedure .62 vs .19.
- **JB-hard by family:** this-that loses on ambiguous and tradeoff, and wins on judge_hard and policy.
- **Calibration:** this-that is overconfident out of distribution (ECE .38).
- **Complementary errors:**
  - JB-hard: at least one model is right on 83 of 130;
  - REAL-label: at least one is right on 338 of 400.

## Latency (A10G, fused runtime, cached = schema prefix compiled once)

| ms | 256 | 1000 | 4000 |
|---|---|---|---|
| 1 question, state-first / cached | 22.9 / 22.7 | 56.8 / 53.7 | 201 / 204 |
| 15 questions, state-first / cached | 97.2 / 24.0 | 135.5 / 59.8 | 273.5 / 209.7 |

## Layout probes
- **State-first and schema-first agree on only 0.760 of decisions,** so the schema layout must be trained for.
- **All questions in one shared prefix interferes:** agreement 0.786. Use one slot set per question.
- **Option-order sensitivity is similar for both heads:** decisions unchanged on 0.918 (this-that) and 0.884 (hobson) of questions.

## Low-bit sensitivity
Simple rounding, no rotation, each model against its own bf16. Sensitivity is similar for both models:
- W8A8: 3.4% flips for hobson, 4.2% for this-that.
- W4 g64: 5.8% for hobson, 8.9% for this-that.

## Verdict
this-that fails as a hobson replacement:
- agree_sd .81 / .64;
- fgh .79 / .56;
- JB-hard −4.6.

## Adopt
- **Schema-first layout with the prefix compiled once.** It is exact up to bf16 noise. 15 questions over 1000 tokens: 135 → 60 ms in bf16. It must be trained for, with one slot set per question.
- **Letter-label head:** acceptable, with permutation augmentation.
- **Outcome supervision with the U-statistic Brier estimator:** worth testing, because it lets a decider learn from logged gate outcomes.

## Do not adopt
- Interleaved multi-question state-first passes.
- Head-keeping truncation.

## Next step
Train hobson's recipe schema-first with a compiled prefix, adding CF-style reading data or distilling from both models.
- **Pass:** REAL-label ≥ .78 and CF-probe flip ≥ .5.
- Then apply the rotated low-bit kernels to the state pass.
