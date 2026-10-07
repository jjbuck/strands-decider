# H7 report: training the compiled question schema into the full 24-layer hobson-v19 (box g4)

Labels: [V] verified, [M] measured, [S] speculation. Model: full hobson-v19 (24 layers, d 2048), no shrinking. Code is in `~/decider2/h7/code/`; the timeline is in `NOTES.md`.

## Idea
Compile the question bundle once per deployment, so a request runs only the state plus a few rows per question. hobson was trained state-first, so the layout has to be trained in: a LoRA fine-tune distilled from bf16 state-first hobson.

## What was built
- **h7lib.py.** H3's differentiable lean hobson forward plus branches: one main causal sequence, and branches that start from its final GDN state and conv tail, each with its own attention mask over the main rows.
  - [V] The teacher branch layout equals hobson's plain state-first forward (logit max diff ≤ 0.017, argmax 12/12).
  - [V] Each schema layout with one question equals the matching plain sequence (max diff ≤ 0.02).
  - [V] LoRA gradients through the branch path match the plain path (cosine ≥ 0.9994).
- **Two layouts.** The bundle is [Q1' … Qn'], where Qk' is question k's tokens minus its final `<answer>`. It is causal and compiled once.
  - **v1, single slot (H6's layout):** one `<answer>` row per question after the state. It attends to its own question's span and the state. Options are read from the compiled bundle rows.
  - **v2, slot sets (the delivered model):** after the state, each question gets a slot set: its option-end tokens plus `<answer>` (K+1 rows: 3 for yes/no, 24 for a 23-way choice). A set is causal, attends to its own question's bundle span and the state, and branches from the state's final GDN state (varlen), so sets never see each other. The pointer head reads options from the set rows, which see the state, as hobson was trained to.
- **Training (train_h7.py).** LoRA r16/α32 on every projection of all 24 layers, plus a trainable pointer head. The teacher is frozen bf16 hobson, state-first, in the same process.
  - Data mix (train split only, 0 eval tasks):
    - 70% train_pool requests, KL to the teacher. The bundle is the full question set (45%), a random 2–4 subset in random order (30%), or one question (25%).
    - 20% CF-style pairs from gen_cf.py: F0's CF-probe generator (8 kinds × 180 pairs) plus human/amount/wrap-up edits on my own templates. Loss: CE to the construction label plus 0.3·KL.
    - 10% train_v5 rows: 0.3·CE to gold plus KL.
  - Option-order permutation with p 0.5.
  - **Dense term (the step that made it learn):** relative MSE between the slot-set rows and hobson's own option-end and `<answer>` rows, at layer outputs 5, 11, 17 and 23, weight 1. It was added at update 150.
  - AdamW, lr 2e-4, cosine. 900 updates (1.5 h), then continued to 1300 at decayed lr.
- **Runtime (h7lat.py).** H4's TTL fused runtime with LoRA merged, plus slot sets (varlen GDN branch, masked SDPA).
  - [V] Against h7lib on 18 items (50 questions): argmax 50/50, max |dp| ≤ 0.006.

## Results: every suite next to the bf16 hobson references [M]
All 3,227 evalkit questions are scored. The bundle is each item's question set.

| | hobson | untrained schema | v1 single slot s300 | v2 sets s300 | s600 | **s900** | s1300 |
|---|---|---|---|---|---|---|---|
| REAL agree / agree_sd | 1 / 1 | .400 / .350 | .746 / .393 | .743 / .616 | .821 / .647 | **.881 / .685** | .883 / .691 |
| LONG agree / agree_sd | 1 / 1 | .261 / .206 | .762 / .424 | .777 / .642 | .860 / .770 | **.917 / .836** | .921 / .818 |
| CF fgh | 1 | .009 | .413 | .835 | .972 | **.991** | .991 |
| CF-probe fgh | 1 | 0 | 0 | .010 | .914 | **.981** | 1.000 |
| CF acc / flip | .594 / .268 | .440 / .015 | .512 / .123 | .754 / .606 | .784 / .667 | **.817 / .732** | .824 / .741 |
| CF-probe acc / flip | .653 / .328 | .502 / .003 | .500 / 0 | .505 / .009 | .944 / .887 | **.977 / .953** | .995 / .991 |
| JB-hard (McNemar p) | .523 | .362 (.01) | .346 (.001) | .431 (.16) | .423 (.09) | **.408 (.04)** | .423 (.09) |
| JB-long agree | 1 | .338 | .390 | .364 | .338 | .403 | .338 |
| SHUF both_right | .245 | .031 | .116 | .589 | .664 | .726 | .729 |
| REAL-label acc | .785 | .412 | .675 | .647 | .720 | **.785** | .757 |

- **Untrained, the layout is unusable.** Its REAL agreement (.40) is below the no-state baseline (.68).
- **v1 (single slot) learns only the no-state prior.** Agreement rises, but agree_sd stays near .39 and the CF-probe flip is 0. H6 independently saw no learning in this layout.
- **v2 plus the dense term reads the state.**
- **s900 against the bar:**
  - CF fgh .991 (pass) and CF-probe fgh .981 (pass).
  - REAL agree_sd .685 (fail; 11.9% of REAL decisions flip).
  - LONG agree_sd .836 (fail).
  - JB-hard −11.5 points (flagged: policy, long_policy, trap and adversarial; temporal_numeric improves from 3 to 6).
- **Where agreement breaks [M].** The state-dependent misses concentrate in the two 23-way procedure questions (`procedure` and `needed_procedure`, whose question text is 1.0–1.2k tokens):
  - on those questions: REAL agree_sd .396 (n 96), LONG .571 (n 49);
  - on all other questions: REAL .796 (n 250), LONG .948 (n 116).
- **Bundling costs nothing [M].** On multi-question items, the bundle and one-question-per-sequence layouts give the same decision on .965 of questions (H4's shared prefix for this-that: .786). One-question layout: REAL agree_sd .682, LONG .812.
- **Plateau [M].** From s300 to s600 to s900, REAL agree_sd went .616 → .647 → .685 and LONG .642 → .770 → .836. Continuing to s1300 at decayed lr (7e-5 → 2e-5) gave REAL .691 and LONG .818, with churn between questions (REAL procedure .396 → .625, other questions .796 → .716). With this recipe, real-traffic agreement is flat at about .69 / .82–.84. s900 is the delivered checkpoint (best LONG and REAL-label); s1300 is also kept.
- **Absolute accuracy against ground truth [M].**
  - CF-probe .977 against hobson's .653: date_order 1.0 against .06; status_equal_distract 1.0 against .25; amount_vs_limit .81 against .33.
  - CF .817 against .594: human_insert 1.0 against .35; amount_insert 1.0 against .10. Identity kinds and procedure_intent stay at 0 for both models.
  - REAL-label .785, which equals hobson. Where the two disagree (46 questions), each is right on 22.
  - **Caveat:** the CF-probe augmentation runs the suite's own generator code (on train-split states with fresh records), and the CF augmentation covers three CF edit kinds with other templates. The CF and CF-probe gains are therefore partly template-in-distribution. REAL and LONG agree_sd are the clean metrics, and they fail.

## Latency [M]
A10G g4 (exclusive GPU), bf16, TTL fused runtime, LoRA merged, CUDA graph, fresh real states, n = 20. Values are median / p95 in ms.

| T | questions | v2 sets | v1 single slot | hobson plain (state cached once, each question a branch) | bundle compiled once |
|---|---|---|---|---|---|
| 1000 | 1 | 56.7 / 56.7 | 56.1 | 64.0 / 64.0 | 92 tok, 22.5 ms |
| 1000 | 4 | 57.8 / 57.8 | 57.1 | 94.5 / 94.5 | 360 tok, 27.3 ms |
| 1000 | 15 | **66.1 / 66.1** | 61.2 | **226.3 / 226.4** | 1718 tok, 97.2 ms |
| 4000 | 1 | 210.0 / 210.1 | 209.6 | 215.2 / 215.3 | |
| 4000 | 4 | 211.9 / 212.0 | 211.2 | 247.1 / 247.2 | |
| 4000 | 15 | **219.9 / 219.9** | 218.5 | **385.6 / 385.8** | |

- With 15 questions the slot sets cost 31 rows (+4.9 ms) over the single slot, and save 3.4× (at 1000 tokens) and 1.75× (at 4000) over hobson's layout.
- **Low-bit in this layout ([M] by H6 on g1, in H2/H6's deployed kernels; H6 adopted the slot-set layout from these notes).** W8A8-GPTQ b8 sets: 33.6 ms (1q) and 40.6 ms (15q) at T = 1000; 139.0 / 155.9 ms at 4000. The trained model's accuracy has not been measured in those kernels.
- **Projections [S]** (H6's proj6, b8 sets, 1q, T = 1000): 3090 18.4 ms, 4090 10.7 ms, 5090 7.3 ms.

## Verdict against the kill criterion (architecture co-design bar)
**Fail.**
- REAL agree_sd .685 and LONG .836 (s1300: .691 / .818), against ≥ .95.
- JB-hard is flagged at −11.5 points.
- CF and CF-probe fgh pass (.991 / .981), and the model reads details far better than hobson.

The schema layout is not yet accuracy-preserving for hobson.

[M] v1 trained on KL alone did not learn to read the state; v2 with the dense row term did. The two changes are confounded in this run. [S] Both probably matter: state-independent option keys cannot carry evidence, and the dense term gives a 2048-dim target per row instead of a 2–5-way KL.

Off the procedure questions, LONG sits at .948 but REAL only at .80 (s900), so the gap is not only the procedure questions.

## Decisive next step
Keep the layout; change the recipe, because LoRA r16 with KL + row distillation flattens at REAL agree_sd ≈ .69.
- [S] Use more capacity and signal: higher-rank or full fine-tuning (this-that was full-rank), distillation of the state rows as well, and slot sets that carry each option's last few tokens. The 23-way procedure choices are the hardest case.
- Re-score against the same bar.
- H6's QAT should start from `ckpt/h7_sets_s900.pt` in the slot-set layout (described in `NOTES.md`), not the single-slot layout.

## Artifacts
`ckpt/h7_sets_s{300,600,900,1300}.pt` (LoRA plus head, 60 MB; copied to the laptop as the task instructed, although the brief says "no models on the laptop"), `preds/`, `scores.json`, `lat_h7.json`, `box_logs/`.
