# J11 notes: reinvented decision architectures, from-scratch scaling vs a dense control

## 2026-10-06 09:14 PDT
- Read BRIEF8, FAST_DECISION_MODEL.md (sec 5, 7 = failed ideas), evalkit README, F7_REPORT. J3 just started (2B pretrained).
- Box j11 not launched yet (no boxes/j11.txt; retry_j11.log empty). Doing design work on laptop meanwhile.

## 09:25
- Found reusable hobson teacher logits from the earlier g3 round on the laptop: recovered/g3/g3/rows_corpus_q.pt (102,547 rows of train_v5 + v19 synth docs,
  21.5M Qwen tokens, hobson t_logits) and rows_real_q.pt (6,000 train-split real states, 1 question each, 18.3M tokens, t_logits). Saves ~1 h of teacher time.
- Read Brooker posts (hobson build; encoder post). e1a (unmasked bidirectional) generalizes worse on benchmarks; e1b masked cache ties v19.
- Writing code/synth.py (exact-label synthetic decision families, train + held-out families) while the box launches.

## 09:55
- Box j11 ready 09:35 (A10G, torch 2.14.1). Started: (1) hobson teacher on 6,747 further train-split real states (rows_real2, 18.4M tokens,
  1 random question each, same render/fit as g3) -> doubles the KL data; (2) synth.py 400k train + 9.1k test (8 train families iid, 4 held-out
  families); (3) prep.py: Qwen ids + exact-value payloads (typed literal hashes, values, key/record bindings) + position-free option
  decomposition + 3 option rotations for kit and synth test (order-invariance metric requested by coordinator).
- Coordinator addition: option-order invariance. Design decision: ALL slot-family arms are position-free by construction (each option is
  encoded separately with no numbering; slots carry no position; memory is a set) -> exact permutation equivariance. DEC/DECV keep hobson's
  layout with shuffle augmentation. Metrics for every arm: decision-unchanged rate + mean |dp| over 3 rotations (score: reversal), ECE, latency to 8k.
- Code: models.py (dec, decv, slot, vslot, belief), train.py (FLOP-budgeted; all arms walk the same deterministic step stream), evalrun.py, lat.py.

## 10:20
- Box j11 became unreachable at ~10:02 (SSM TargetNotConnected) while the 250k-row synth prep (8 workers, all rows held in the parent)
  ran next to the hobson teacher: most likely RAM exhaustion (30 GB). Rewrote synth prep to write 25k-row chunks with 7 workers. Waiting
  for the box to come back. Teacher was at 5887/6747 rows at the last check.
- Smoke test (GPU shared with teacher, so throughput not clean) at d=512 L=12: dec 38.5M params, slot 39.5M, vslot 42.3M, belief 39.5M;
  all train; vslot NaN fixed (pointer retrieval now a softmax restricted to valued tokens, computed in fp32 with autocast off).

## 10:26
- j11 still TargetNotConnected (since ~10:02). Messaged the coordinator (main) asking for a reboot/relaunch; I cannot touch AWS myself.
- Meanwhile: wrote score.py (laptop: evalkit suites, ECE/Brier vs ground truth, McNemar vs hobson, order invariance from rotated preds).
- If the box returns with its disk intact: kit.pt, synth_test.pt, corpus.pt, real.pt, lceval.pt were already written; rows_real2 teacher
  labels may or may not have finished (5887/6747 at last check; the teacher saves only at the end).

## 10:45
- Coordinator rebooted j11 at ~10:30 (auto-shutdown ~19:13). Disk intact: kit/synth_test/corpus/real/lceval prep files and the finished
  real2 teacher labels (6,747 rows, 18.4M tokens) survived; the 250k synth train prep had died in torch.save (141M tokens!). Re-ran real2 prep and
  synth train prep chunked (6 x 25k rows, 85M tokens).
- Clean throughput (A10G, d512 L12, short synth rows): dec 87k tok/s = 22 TFLOPS analytic (16k-token micro-batches); slot/belief 120k tok/s =
  12 TFLOPS, vslot 124-134k tok/s = 12-13 TFLOPS with 48k-token micro-batches and the pointer ops on alternate slot layers. Equal FLOPs therefore costs
  the slot arms ~1.8x the wall clock of dec (lower MFU: small slot GEMMs, many short option sequences).
- Params at S (d512, L12): dec 37.6M non-emb, slot 38.7M, belief 38.7M (weight-tied block, 3 rounds), vslot 41.4M (pointer ops +2.7M).
- Queue started: S runs at 12 PF each (dec, slot, vslot, belief, decv), each followed by evalrun.

## 11:15
- TRAP HIT: S_dec at lr 1e-3, decision loss only: loss 1.69 -> 1.60 -> 1.70 (unstable, gn 5) by step 400; synthetic accuracy at chance
  (mean iid 0.40, count 0.13). Diagnostic at lr 3e-4: stable but flat (mean iid 0.40 at step 300 = 9.8M tokens). One scalar of supervision per
  example is too sparse to learn token-level structure from scratch at this budget (this is how the earlier attempt died).
- Fix (applied to every arm, training only, inference unchanged): dense auxiliary objective with a tied embedding head at 10% of positions:
  next-token CE for the causal decoder; masked-token CE on a separate masked state pass for the bidirectional encoders (GPT- vs BERT-style).
  Its FLOPs (head + the encoders' extra masked pass) are counted in each arm's training budget.
- Running diag_dec_aux (lr 6e-4, 4 PF, eval every 150 steps) to check the control now learns before re-launching the queue.

## 11:52
- Even with the aux objective, diag_dec_aux showed no decision learning by step 300 (mean iid 0.42). Overfit tests on 64 examples:
  dec memorizes in 50 steps; slot could not memorize at all (loss pinned at ln 2, grad norm collapsing to 0.002 at lr 1e-3, spikes to 30 at 1e-4).
  Causes found and fixed for EVERY arm: (1) no qk-norm -> unbounded attention logits; added RMS qk-norm (hobson's attention has it) to token and
  slot attention; (2) scratch slots initialised at 0.02 scale -> RMSNorm backward amplifies their gradients 50x; now unit scale. After the fixes
  slot/vslot memorize at lr 5e-4 (slot and belief still die at lr 1e-3; vslot does not). Using lr 5e-4 for all arms at S.
- Relaunched S queue: dec, slot (10 PF each) then evals, then vslot. Will decide belief/decv/M from the S learning curves.

## 12:45
- Synthetic-only diag (dec, 4 PF, lr 5e-4): decision loss 0.99 -> 0.93, mean iid 0.37 -> 0.45 (count 0.18 -> 0.57, cmp_num ~0.55); retrieval
  families (status, multihop) still at chance. Learning starts only after ~10M synthetic tokens. With the 45/25/30 mix (corpus+real dilute the
  synthetic signal) dec had learned nothing by 16M tokens.
- Changes: compact synthetic states (<=1 filler turn, no neutral padding in train; H_long keeps its padding), 353 tok/row vs 566 -> 1.6x more
  labelled decisions per FLOP; stream shares 70% synth / 15% corpus / 15% real. Old (long) synth prep moved to data/prep_old.
- Latency S (d512 L12, A10G, CUDA graph, batch 1, median ms; p95 within 0.1 ms) saved to lat_S.json:
  T=64/256/1000/4000/8000, Q=1: dec 2.91/3.59/6.94/23.85/50.4; slot 1.47/1.82/3.17/11.39/25.4; vslot 2.37/2.77/4.37/13.35/28.7;
  belief 3.28/3.88/6.17/18.65/38.4. Q=4 at 1000: dec 8.26, slot 3.49, vslot 5.07, belief 7.18. FLOPs at 1000 (Q=1): dec 99.6 GF, slot 36.4.
- Queue: S_dec, S_vslot, evals, S_slot.

## 13:35
- S_dec (d512 L12, 10 PF, 1067 steps, 16 min): synthetic mean iid 0.466 (count 0.70, status 0.42, cmp_num 0.50, cmp_date 0.49, multihop2 0.27),
  held 0.394. Learning curve 2.8 PF 0.423 -> 5.6 PF 0.460 -> 10 PF 0.466. evalkit: JB-all .316, JB-hard .369, REAL agree_sd .457, LONG agree_sd .582,
  CF acc .452 / flip .002, CF-probe acc .498 / flip .019, REAL-label .710 (nostate .635, hobson .785); kit order-invariance: decision unchanged .893,
  mean |dp| .011 (hobson .884 under reversal); synth unchanged only .39 (near-uniform outputs flip on tiny perturbations); synth ECE .026.
- S_vslot first OOM at 48k-token micro-batches (fp32 pointer tensors on 4k-token real rows); retry at 20k running (1582 steps for 10 PF).
  At 1.9 PF: mean iid 0.41 (dec at 2.8 PF: 0.42).
- NOTE: an rsync of the whole S_dec run dir started copying final.pt to the laptop; I killed it within ~3 min and removed the partial file.
  From now on only JSON results are fetched (code/fetch.sh).

## 14:35
- S_vslot (v1, operators without binding priors), 10 PF, 1582 steps: synth iid 0.471 / held 0.397 (dec 0.466 / 0.394) -> tie at equal
  training FLOPs with 0.37x the inference FLOPs. evalkit: JB-all .338 (dec .316), JB-hard .431 (.369), REAL agree_sd .303 (.457),
  LONG agree_sd .406 (.582), CF acc .451 flip .044 (.452/.002), CF-probe .498/.028 (.498/.019), REAL-label .640 (.710; nostate .635).
  So vslot reads synthetic/JB slightly better and real-traffic agreement much worse at this budget.
- Invariance: decision-unchanged is dominated by near-tied outputs (both models are barely above uniform): dec .893 / vslot .556 overall, but
  on decisions with top-2 margin > 0.1 both are .997; mean |dp| dec .011, vslot .0028 (bf16 batch-padding noise; slot arms are permutation
  equivariant by construction). invcheck.py (fp32, batch 1) will verify exactness on S_slot.
- Added exact binding priors to vslot (v2): each slot's own literal hashes -> pointer-logit bonus on state tokens with an equal canonical literal,
  and on every token of that literal's record (verified on status rows: the bonus lands exactly on the target record). Queued S_vslot2 + M runs
  (M budget cut to 20 PF to fit the box lifetime).

## 15:20
- S_slot (10 PF, 1590 steps, 38 min): synth iid .459 / held .386; JB-all .359, JB-hard .415, REAL agree_sd .390, LONG .582, CF .457/.012,
  CF-probe .494/.016, REAL-label .675; rotation mean |dp| .0008, confident decisions unchanged 1.000 (overall .971).
- S summary at equal training FLOPs (10 PF): dec .466, slot .459, vslot(v1) .471 synth iid -> indistinguishable (SE ~1 pt); slot arms need 0.37x
  the inference FLOPs and run 2.2x (slot) / 1.6x (vslot) faster at 1000 tokens. Real-traffic agreement favours dec (agree_sd .457 vs .390/.303).
- Queue rewritten for the box lifetime: S_vslot2 (binding priors) -> M_dec (16 PF) -> M_slot (16 PF) -> lat_M -> M_vslot2 -> invcheck. Belief
  training dropped (latency measured only), decv dropped.

## 16:15
- S_vslot2 (binding priors) 10 PF: synth iid .496 (best; dec .466) / held .406 (dec .394). Gain concentrated where the operator is the task:
  argmax .67 vs dec .43 (n=300; vslot2 already .60 at 1.9 PF, i.e. >5x training-compute efficiency on that family), multihop2 .34 vs .27,
  id_match .50 vs .43; but status .31 vs .42, count .60 vs .70. CF flip .074 (dec .002), CF-probe flip .056 (.019) but tiny absolutes;
  REAL agree_sd .327 (dec .457), REAL-label .650 (.710). ECE(CF) .144 (dec .209).
- M_dec (d768 L16, 16 PF) running.

## 14:05 (clock check: my earlier timestamps from ~12:45 on were estimates running ~3 h fast; laptop `date` now 14:01 PDT)
- M_dec (d768 L16, 113.7M non-emb, 16 PF = 615 steps, only ~20M tokens): synth iid .442 / held .394 -- WORSE than S_dec (.466) at 1.6x the compute:
  at this budget the larger model is token-starved. REAL agree_sd .382, LONG .624, REAL-label .645.
- invcheck [V]: S_slot fp32 batch-1, 322 rotation pairs: max |dp| 6.3e-7, decisions unchanged 1.000 -> order invariance is exact by construction.
- lat_M measured (A10G, ms @T=1000 Q=1): dec 13.28, slot 5.53, vslot 7.37, belief 9.93; @4000: dec 48.9, slot 20.4.
- More box time than I thought: added M_vslot2, an XS scale (d448 L8, 5 PF: dec/slot/vslot2), S_belief, S_decv, lat_XS.

## 14:50
- M_vslot2 (d768 L16, 114.7M non-emb, 16 PF, 1013 steps, 30 min): synth iid .452 / held .399 (M_dec .442/.394); JB-all .368 (.299),
  JB-hard .446 (.362), REAL agree_sd .384 (.382), LONG .630 (.624), CF-probe flip .069 (.025), REAL-label .677 (.645). At M, vslot2 >= dec on
  every suite at 0.33x inference FLOPs (93.6 vs 285.9 GF @1000) and 1.8x lower latency (7.37 vs 13.28 ms).
- Equal inference compute pair M_vslot2 (93.6 GF) vs S_dec (99.6 GF): synth -1.4, JB-all +5.2, JB-hard +7.7, REAL agree_sd -7.3, REAL-label -3.3.
- M_slot retry (micro 12288) running; then XS (3 arms), S_belief, S_decv, lat_XS.

## 15:15
- M_slot (111.9M non-emb, 16 PF, 1017 steps vs M_dec 615): synth iid .450 / held .395; JB-all .333, JB-hard .408, REAL agree_sd .358, LONG .582,
  CF-probe .505/.019, REAL-label .672; rotation |dp| .0008.
- Margins (arm - dec, same scale, equal training FLOPs):
  synth iid: slot S -0.7 / M +0.8; vslot2 S +3.0 / M +1.0.  JB-hard: slot +4.6 / +4.6; vslot2 +2.3 / +8.4.
  REAL agree_sd: slot -6.7 / -2.4; vslot2 -13.0 / +0.2.  REAL-label: slot -3.5 / +2.7; vslot2 -6.0 / +3.2.
  -> on real traffic the slot arms' deficit closes with scale, but M_dec itself regressed vs S_dec (token-starved: 615 steps), while the slot arms
  get 1.5x (S) / 1.65x (M) more decisions for the same training FLOPs. That is the measured mechanism behind any "growing margin" here.

## 15:25 (box clock 22:25 UTC)
- XS_dec (d448 L8, 19.7M non-emb, 5 PF, 950 steps): synth iid .461 / held .374; JB-all .342, JB-hard .415, REAL agree_sd .350, LONG .558,
  REAL-label .675. Decoder synthetic accuracy across XS/S/M = .461/.466/.442: flat -> the regime is limited by decisions seen, not parameters.
- Draft report written; waiting for XS_slot, XS_vslot2, S_belief, S_decv, lat_XS.

## 16:02
- S_belief (weight-tied 9-layer slot block x 3 rounds, 38.7M non-emb, 10 PF, 1524 steps): synth iid .449 / held .388; JB-all .325, JB-hard .400,
  REAL agree_sd .442 (slot .390, dec .457), LONG .582, CF-probe flip .031, REAL-label .695 (slot .675, dec .710). Iterative belief refinement recovers
  most of the slot arms' real-traffic deficit, but its 27 sequential slot layers make it latency-bound (6.17 ms vs slot 3.17, dec 6.94 @1000).
- Queue: XS_slot, XS_vslot2 (GQA fix for 7 heads: MQA), S_decv, lat_XS.

## 16:22
- XS_slot (18.2M non-emb with MQA, 5 PF, 1766 steps vs XS_dec 950): synth iid .463 (dec .461) / held .411 (.374); JB-all .372 (.342),
  JB-hard .431 (.415), REAL agree_sd .410 (.350), LONG .630 (.558), CF-probe flip .034 (.003), REAL-label .698 (.675).
- slot - dec margin on REAL agree_sd across XS/S/M: +6.0 / -6.7 / -2.4; synth iid +0.2 / -0.7 / +0.8 -> no monotone trend with scale; I cannot claim
  a growing margin for slot. vslot2's argmax capability holds at S and M (+24 / +22).

## 16:45
- XS_vslot2 (19.1M, 5 PF, 1750 steps): synth iid .475 / held .391; argmax .59 (dec .41); JB-all .342, JB-hard .415, REAL agree_sd .344,
  LONG .485, CF flip .042, CF-probe flip .044, REAL-label .625.
- vslot2 - dec (synth iid) across XS/S/M: +1.4 / +3.0 / +1.0; argmax +18 / +24 / +22. Margin is real on the operator family, but it does not grow
  with scale. Decoder itself flat across scales (.461/.466/.442).
- Remaining: S_decv (running), lat_XS. Then final report.

## 17:00
- S_decv (decoder + input value channels, no operators; 37.7M, 10 PF): synth iid .475 / held .401 (dec .466/.394); argmax .45 (dec .43,
  vslot2 .67) -> the argmax capability comes from the slot-side binding operator, not from the channels. JB-all .303, JB-hard .338,
  REAL agree_sd .382, REAL-label .677.
- lat_XS: @1000 dec 3.94 ms (53 GF), slot 1.71 (15 GF), vslot 2.58; @4000 13.81 / 5.51 / 6.96.
- Equal-accuracy view: XS_slot at 15 GF >= XS_dec at 53 GF on every suite; slot arms match the decoder's (flat) accuracy at 0.28-0.37x inference
  FLOPs -> a 2.7-3.6x inference-compute factor that is a cost-class factor, not a capability factor.
- All compute done. Final report in DRAFT_REPORT.md. Weights stay on the box (runs/*/final.pt).

## 17:10
- Final report written to DRAFT_REPORT.md (results.json, lat_{XS,S,M}.json, lat_proj.json, runs/*/score.json on the laptop; weights only on the box).
- Stopped the box job queue (idle). Box auto-shutdown ~19:13 PDT.
