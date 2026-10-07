# Brief 9: the combined stack and its ablations

You are one of two agents building and measuring one decision model that combines every speedup that has worked so far, plus a pre-registered ablation matrix that attributes its latency and accuracy to each part.
- **K1** owns the model library, training and accuracy.
- **K2** owns the fused runtime and latency.

Read this brief in full, then `~/code/jit-eval/docs/IDEAS_EXPLORED.md`, the table of every idea tested so far and the arithmetic of the latency floor. IDs below refer to that table.

## The components

All start from hobson-v19, the 2B Qwen3.5 GDN hybrid with a pointer head. The source code is local. None of the trained weights survived, so everything is retrained here.

| flag | ID | what it does | measured alone | code to reuse |
|---|---|---|---|---|
| **D** | R2 | **Depth split.** State tokens run only layers 0–7; question rows run all 24 and read the state's layer-8 outputs through each deep layer's own K/V projections plus rank-32 memory adapters (DT-A8). | 1.73x at 1000 tokens, 2.36x at 4000; meets the accuracy bar; RuleTaker depth-3 −17 points | `~/decider2/j3/code/` (`dtlib.py`, `train_dt.py`, args in `a4_args.txt`) |
| **C** | A6 | **Precompiled deployment documents.** Constant blocks (KB pages, hook notes, tool schemas) are compiled once per deployment as K/V re-rotated to their runtime position (K/V-only library), with a compile adapter (LoRA r16 acting only on compile rows). | 1.90x on banking; REAL-label .767 | `~/decider2/j9/` (`j9lib.py`, `train_j9.py`, `j9lat.py`) |
| **V** | A7 | **Domain vocabulary, 16k super-tokens.** BPE merges over Qwen ids; embeddings start from the mean of their constituents; RoPE stays at original positions. | 1.97x fewer tokens on REAL; agreement plateaued with LoRA r16 | `~/decider2/j7/code/` (`superbpe.py`, `j7lib.py`, `train_j7.py`) |
| **Q** | A4 | **Deployed questions compiled into weights, "late" placement.** Each deployed question gets a weight delta on its K+1 slot rows only (shared r16 + per-question r8). The state is read question-blind; questions not in the deployment stay in context. | 2.3x for 4 questions at 1000 tokens; agreement .86 | `~/decider2/j6/code/` (`j6lib.py`, `train_a.py`, `lat_j6.py`) |
| **P** | P1 | **W8A8:** Hadamard rotation, GPTQ act-order codes from 64 train-split states, the 8 GEMMs of `precmap_w8a8_b8.json` left in bf16, per-token absmax activations. Inference only; recalibrated per trained model. | 1.57x; at the noise floor | `~/decider2/h1/code/`, `~/decider2/h2/code/` (`qrt.py`, kernels), `~/decider2/j5/code/` (short-input kernels), `~/decider2/j15/code/j15gptq.py` |
| **X** | R1 | **Early exit at layer 16.** An exit head (the pointer head plus a rank-512 adapter on the layer-16 residual), trained by KL to the model's own final answer. It answers when its margin exceeds τ; otherwise the request runs all 24 layers. τ is fit on a train-split DEV set for zero residual changes. Inference-time cascade; one head per trained model. | 0.688x of W8A8 end to end; 0 of 6,216 real decisions changed | `~/decider2/j15/code/` (`j15learn.py`, `casc.py`, `mkdev.py`, `j15run.py`) |

## The ablation matrix (pre-registered; do not change without telling the coordinator)

Training factors D, C, V and Q change the weights, so each combination below is trained separately with the identical recipe. Inference factors P and X are applied to every trained model.

| run | flags | why |
|---|---|---|
| H | none, untrained | hobson-v19 itself: the baseline. It only needs an exit head and a GPTQ calibration. |
| R0, R0s2 | none, trained | the recipe control: the same data, steps and LoRA with no structural change. Two seeds. |
| D, C, V, Q | one each | each component alone against R0 |
| DCVQ, DCVQs2 | all four | the full stack. Two seeds. |
| CVQ, DVQ, DCQ, DCV | all but one | each component's contribution inside the stack against DCVQ |

**Evaluated configurations:** every trained run × {bf16, W8A8} × {no exit, exit cascade at 16}, so 13 runs × 4 = 52.

**Pre-registered analysis:**
- **A component alone:** single against R0.
- **A component in the stack:** DCVQ against the leave-one-out run without it.
- **Interaction:** the difference between those two.
- **Noise:** the two seeds of R0 and of DCVQ.
- **Tests:** paired McNemar for accuracy; latency medians and p95 from the runtime.

## The training recipe (identical for every trained run)

- **Student:** hobson-v19 weights + LoRA r32 on every projection of all 24 layers + the pointer head, plus each enabled component's own parameters at its original size: D memory adapters r32; C compile adapter r16; V new embedding rows plus per-position gains; Q per-question r8 + shared r16. Zero-initialize every adapter where the original code did.
- **Teacher:** frozen hobson-v19 in its native layout, in process, with calibrated logits as in the source scripts.
- **Data:** train split only; never evalkit eval items. Per update the mix is:
  - 60% real states from `evalkit/train_pool.jsonl`, loss KL(teacher‖student);
  - 25% `training/data/train_v5.jsonl` gold rows, loss CE + KL;
  - 15% detail augmentation: H7's edits of train-split states (`h7/data/cf_aug.jsonl`, made by `h7/code/gen_cf.py`) and J7's out-of-template pairs (`j7/code/gen_aug.py`), loss CE + 0.3·KL.

  Option order is permuted with p .5.
- **Dense row loss:** relative MSE between the student's and the teacher's hidden states at layers 5, 11, 17 and 23, on the rows that produce the answer (option ends and `<answer>`), aligned to the teacher's native rows. The weight is 1.0. With Q these are the slot rows; with V they are the rows ending at the same original token offsets.
- **Budget:** 1,000 updates of 8 sequences (≤ 6,144 tokens each), AdamW, cosine schedule with 50 warm-up steps, the same learning rates and seed stream for every run. If the pilot shows more than 13 s per update, lower the update count so every run finishes training in under 3.6 h. Use the same count for all runs and record the change.

## Accuracy evaluation (K1)

All 3,227 evalkit questions, untruncated, for each of the 52 configurations:
- JB-all, JB-hard, REAL-label;
- REAL and LONG state-dependent agreement;
- CF and CF-probe pair accuracy and retention (fgh);
- SHUF;
- Brier and ECE;
- agreement on the two 23-way procedure questions, reported separately;
- the held-out RuleTaker depth-3 and depth-5 probe from J3 (`training/data/train_v5.holdout.jsonl`; J3's `dt_holdout.py`), because D is known to lose there.

Paired McNemar against hobson-v19, against R0, and, for the leave-one-out runs, against DCVQ.

**W8A8 accuracy** is computed in a faithful emulation of the deployed format (same rotation, GPTQ codes, bf16 GEMMs, per-token activation rounding). K2 checks emulation against the deployed kernels on H and DCVQ; if they disagree on more than 0.3% of decisions, K1 re-scores W8A8 in K2's runtime.

## Latency (K2)

1. **Runtime.** Build one fused runtime, `stackrt.py`, on H2's QRT runtime and J5's short-input kernels, supporting every flag combination in bf16 and W8A8, with a CUDA graph per exact shape.

   **Fidelity:** for every combination of D, C, V and Q, match K1's library (`~/decider2/k1/code/stacklib.py`) on 40 real requests with random-initialized adapters. Argmax must agree on 40/40 and max |dp| must be ≤ .01. Fidelity is a property of the computation graph, so trained weights are not needed.
2. **Grid.** Exclusive GPU, fresh inputs, ≥ 12 warm reps, median and p95, for all 64 combinations of D, C, V, Q, P and X:
   - state lengths T = 32, 64, 128, 256, 400, 1000, 2000, 4000;
   - three question sets: one JevBench question, the 4 deployed banking questions, the 15 banking questions.

   The C and V settings for the grid:
   - **C:** run at compiled shares 0 and 0.46 (the mean over gate states).
   - **V:** use the measured median token ratio of the 16k vocabulary.
   - **X:** report both "always exit at 16" and the cascade cost at the exit share K1 measures.
3. **Real requests.** Use a fixed, stratified set of 240 eval requests (80 JB-all, 120 REAL-agree, 40 LONG), drawn with seed 0 and saved to `k2/requests.json`. For each request use its own token counts, compiled share, vocabulary tokenization and deployed questions. Time every trained run's four inference configurations.

   **Exit decisions:** read them per request from `~/decider2/k1/runs/<run>/exit_decisions.json` (`{request_id: {question: exited}}`). Until K1 writes them, use 93%.
4. **Projections** to RTX 3090, 4090 and 5090 with the method in `docs/FAST_DECISION_MODEL.md` section 9.

## Bars (pre-registered)

- **Strict:**
  - no significant drop on JB-all or JB-hard (p < .05);
  - REAL-label ≥ .78;
  - CF and CF-probe pair accuracy ≥ hobson's (.268 / .328).
- **Relaxed**, the user's "a marginal accuracy loss is acceptable for an outsized speedup":
  - JB-all ≥ .693, at most 3 points below hobson;
  - REAL-label ≥ .765;
  - REAL state-dependent agreement ≥ .85;
  - CF and CF-probe pair accuracy ≥ hobson's.

  Report which configurations pass each bar.
- **Primary latency outcomes:**
  - A10G median at a 1000-token state with 1 question;
  - median and p95 over the 240 real requests;
  - the 3090 projection at 1000 tokens. The target is about 10 ms.

## Infrastructure and rules

- **Boxes.** Use only `~/decider2/box.sh <BOX> run "<cmd>" | put <local> <remote dir under ~/work> | get <remote path under ~/work> <local> | status`.
  - The coordinator creates boxes; wait for `~/decider2/boxes/<BOX>.ready`, checking every ~5 minutes with `sleep 290`.
  - Each box has `~/work/sd` (strands-decider), `~/venv`, hobson-v19 and Qwen3.5-2B-Base in the HF cache, and the bundle (systems, tokens, shrink, training, d1, evalkit, h2/code, h4, h7/code).
  - Put any other code you need (j3, j5, j6, j7, j9, j15, h1) with `box.sh put`.
- **Box lifetime.** Each box auto-terminates 10 hours after launch. You may reset your own box's timer once, with `sudo shutdown -c; sudo shutdown -h +N`, if a run needs it; note it in NOTES.md.
- **Commands.** One command must finish in under 30 minutes. Run long jobs with nohup and make them resumable. GPU memory must stay under 22 GB.
- **No models on the laptop.** Don't run torch or any model locally, and keep weights and checkpoints on the boxes. Small JSON outputs (predictions, scores, decisions, timings) come back to the laptop.
- **No other AWS resources.** Don't touch any and don't launch instances.
- **Writing.** Write only under `~/decider2/<you>/`:
  - `NOTES.md` with timestamps every ~15 minutes;
  - `DRAFT_REPORT.md` (the harness blocks files named REPORT.md);
  - results JSON.

  Copy results off the boxes at least every 30 minutes. The laptop may sleep; box jobs keep running.
- **Permissions.** If a permission is denied, don't work around it; report it.
- **Language in reports.** Plain and specific. Give every number its metric, baseline and unit. No verdict labels such as "survives", "dominated" or "cost-class"; say what was measured and what it means.

## K1: library, training and accuracy

You own `k1` (development) and, once the trainer is validated, run boxes `k1a` to `k1k`.

1. **Build `~/decider2/k1/code/stacklib.py`:** one differentiable forward with flags D, C, V and Q that composes the reused code, and `stacktrain.py` for the recipe above.
   - **How the parts compose:**
     - C's compiled blocks are read by the state path, so under D they pass through layers 0–7 like other state tokens.
     - V tokenizes the dynamic state and the compiled blocks alike.
     - Q's slot rows are question rows, so under D they run all 24 layers.
     - Document any other composition choice in NOTES.md.
   - **Checks:**
     - With all flags off the library must reproduce hobson-v19's references: argmax on 60/60 real questions, max |dp| ≤ .003.
     - Each flag must run alone and in combination.
   - **Publish early.** Put `stacklib.py` and a `VERSION` file in `k1/code/` within the first ~2 hours, so K2 can build against it, and keep them updated.
2. **Pilot:** DCVQ for 150 updates. Confirm the loss falls, check memory, and measure seconds per update.
3. **Signal the coordinator.** Write `~/decider2/k1/READY_FOR_MATRIX` listing the 12 trained runs and the exact command for each. The coordinator launches `k1a`–`k1k` (11 boxes) within about 15 minutes. Run one job per box, using `k1` for the twelfth run and for H.
4. **Per run:**
   - train;
   - evaluate bf16;
   - calibrate GPTQ and evaluate W8A8;
   - train the layer-16 exit head on train-split DEV and EXIT sets (as in J15's `mkdev.py`), fit τ, and evaluate the cascade in bf16 and W8A8;
   - write per-request exit decisions.

   **Outputs:** `~/decider2/k1/runs/<run>/{config.json, train_log.jsonl, preds_{bf16,w8a8}_{full,exit}.json, scores.json, exit_decisions.json}`.
5. **Report:** the full matrix table and the pre-registered contrasts, with paired tests.

## K2: runtime and latency

You own box `k2`.

1. Build `~/decider2/k2/code/stackrt.py` as specified above. Until K1 publishes `stacklib.py`, build the parts that don't depend on it: QRT plus J5 kernels, W8A8, the exit at 16, J9's compiled-block cache and J6's per-question deltas.
2. Check fidelity against K1's library for every combination of D, C, V and Q, and check deployed W8A8 against K1's emulation on H and DCVQ. K1's weights stay on K1's boxes, so use random-initialized adapters plus hobson's weights. Put K1's code on your box and run both on your box.
3. Measure the grid, the 240 real requests and the projections.

   **Outputs:**
   - `~/decider2/k2/lat_grid.jsonl`;
   - `~/decider2/k2/lat_requests.jsonl`, one row per run, configuration and request, with ms_median and ms_p95;
   - `~/decider2/k2/fidelity.json`;
   - `~/decider2/k2/proj.json`;
   - `DRAFT_REPORT.md`.
4. **Timing:** time each request configuration at least 10 times. When K1's exit decisions appear, re-time the exit configurations with them.
5. **Report** the latency table for all 64 combinations and the real-request distributions, and say which components' savings multiply and which overlap (D and X both cut late layers for state tokens).

## Report (both agents)

At most 1,500 words. Return it as your final message and also write it to `DRAFT_REPORT.md`. Mark each claim measured, verified or arithmetic.
