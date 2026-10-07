# H4 notes (box g4): this-that-model-1.0 vs hobson-v19

## 21:50 PDT start
- Read BRIEF7, evalkit README, F7 report. Fetched arXiv abs 2609.23886 (v1, 20 Sep 2026, Cheng/Dai/Sun, FLock) and HTML.
- Paper: 1.88B hybrid delta-rule backbone "not designed by us" (decider-2b, Apache-2.0). Head = softmax over unembedding rows of single-token labels (A)..(J), temp tau.
  Loss = convex CE + Brier; outcome supervision w/ unbiased Brier reward U-stat (M=32 draws); abstention option in a fraction of questions; mixed schema-first/state-first.
  30.9 ms = Table 1 median per question at client, on 16GB laptop GPU; input length NOT stated in the first pass of the HTML.

## 22:05 PDT
- VERIFIED config.json of this-that-1.0 == Qwen3.5-2B-Base text_config (24 layers, 18 GDN + attn at 3,7,..,23, d=2048, FFN 6144, tied emb, vocab 248320). hobson-v19 = Qwen3.5-2B-Base + LoRA r16 + pointer head(256). Same torso shapes, different weights.
- decider-2b = Mapika/decider-2b (Apache-2.0, full FT of Qwen3.5-2B-Base, CE+proper scoring, two layouts; hub main is v11 dated 09-24, after this-that's 09-21 release; tags v8, v10).
- HF repo files: model.safetensors 3.76 GB bf16, chat_template, tokenizer. Code: github FLock-io/this-that-model (pkg thisthat 1.2.0; default model 1.0).
- Repro on g4 (HF torch, bf16, A10G): recorded_68 acc 0.941, Brier 0.041, NLL 0.123 (paper 0.941/0.042/0.126) at T=1.3. Model loads fine.
- IMPORTANT prompt facts (thisthat/prompt.py): state_first = "Context:\n<state>\n\nQuestion: ..\nOptions:\n(A) ..\nAnswer: (" ; slot = last token "(" ; head = lm_head rows of 'A'..'J' (+ 2-letter single tokens to 255).
  DEFAULT_MAX_STATE_TOKENS = 1536, truncation keeps the HEAD of the state (drops the end: the newest turns / proposed message).
  Multi-question state_first interleaves Q_k + Answer_k slot, so question k sees questions <k (not independent like hobson).
  30.9 ms: README says RTX 5080 laptop GPU, median per-question over internal suite (30.9 +- 16.6 ms); 68 recorded states; input length not given in paper -> measure.
- Adapter: thisthat.systemone_protocol._typed (its own JevBench adapter): noul -> options (no, yes) with legend of criteria; choice -> criteria keys; score -> '0'..'n-1'.
- Launched eval queue on g4 (sf, schema, multi, ship) -> ~/work/h4/preds/*.jsonl. HF path ~0.5 s at 6.4k tokens.

## 22:20 PDT
- VERIFIED: released code has NO prefix cache. prompt.build returns prefix_len but nothing consumes it (no past_key_values anywhere). The paper's schema-first caching (eq. 5) is described, not shipped.
- 30.9 ms claim: Table 1 = median per-question latency at the client on the 68 recorded questions; README: RTX 5080 laptop GPU; repo docstring says 30.9 +- 16.6 ms (median 27.3) from an internal suite. MEASURED recorded_68 prompt lengths: 176-183 tokens (median 181), one question per pass, HF eager, padded to multiple of 64 (=192).
- Built tt_lean.py: this-that in d1 lean2 fused runtime (+ typed head over tied-embedding label rows) + schema-first prefix cache (GDN state, conv tail, attn K/V, RoPE offset).
  MEASURED check: lean vs HF on 40 random eval items (95-6256 tok): argmax 40/40, max|dp| median 0.0000, max 0.0217. cached vs full schema pass (12 multi-q items, prefix 260-1863): argmax 12/12, max|dp| 0.0001-0.038 = same size as lean-vs-HF bf16 noise on long inputs.
- Partial sf eval (2465/3227): CF flip 0.61 vs hobson 0.27 (covered pairs); JB-hard 0.477 vs 0.523 (McNemar p=0.41); REAL-label 0.698 vs 0.785; REAL agree_sd 0.81, LONG agree_sd 0.64; SHUF both_right 0.61 vs 0.245.

## 22:35 PDT
- Lineage (MEASURED, wdiff.json): this-that config.json blob == decider-2b config blob. ||tt - decider-2b v8|| / ||W|| = 0.2-0.5% per family (v10 0.2-0.5% too, v8 slightly closer); decider-2b vs Qwen3.5-2B-Base 1-3%; hobson LoRA vs base 2.5-4%; tt vs hobson 3-5%.
  Delta (tt - decider) is full-rank: top-64 singular values hold only 10-32% of its energy. => siblings from the same base, not parent/child.
- sf eval complete (3227 q): JB-hard .477 (hob .523; McNemar 15 vs 21, p=.41), REAL-label .698 (.785), REAL agree_sd .806, LONG agree_sd .636,
  CF acc .812 (.594) flip .633 (.268), CF-probe acc .669 (.653) flip .338 (.328), SHUF both_right .636 (.245).
  CF wins: amount_insert .83 vs .10, human_insert .69 vs .35, procedure_intent .78 vs 0, state>=4k .59 vs .013; identity kinds still 0.
  CF-probe: status_equal_distract .70 vs .25, date_order_distract .22 vs .03, but amount_vs_limit 0.00 vs .24-.33.
  REAL-label losses concentrated in hobson-trained spec semantics: cc_can_still_help .22 vs .94, UserAskedForHuman .58 vs .95, needed_procedure .19 vs .62, procedure .18 vs .41.
- Latency (A10G g4 exclusive, fused lean2 + graph, n=20, fresh real states): fwd-only 15.75/52.69/200.48 at 256/1000/4000 (= hobson F7 16.19/52.88/200.95).
  Tile staircase: 256 -> 15.7 ms but 260 -> 22.3 ms (short path).
  Q15@1000: state-first 135.5 ms (2567 tok) -> schema-cached 59.8 ms (1096 tok). Q15@256: 97.2 -> 24.0. Q15@4000: 273.5 -> 209.7. Q1@1000 56.8 -> 53.7.
  First cached attempt used a dense mask + repeat_interleave: +32 ms at 4000; fixed with causal_lower_right (flash) + tail-aware conv kernel.

## 22:42 PDT
- hobson torso in the same harness: 15.74/52.71/200.64 ms (tt 15.75/52.69/200.48). VERIFIED identical cost.
- this-that's own package on A10G (HF eager, thisthat.decide, n=12): Q1 48.5/98.1/363 ms at 256/1000/4000; Q15 170.8/245.5/501; Q1proc 167/200/487.
- Repo's scripts/measure_latency.py on A10G (68 recorded items, ~180 tok, 1 q/pass): mean 47.62 ms, p50 47.62, p99 48.50 (paper: 30.9 ms on RTX 5080 laptop).
- Calibration (MEASURED, labelled suites, T=1 API default): tt ECE 0.38 JB-hard / 0.20 REAL-label / 0.12 CF / 0.25 CF-probe vs hobson 0.10-0.13 / 0.10-0.12 / 0.10-0.11 / 0.08. NLL-oracle temperature for tt is 2.3-6. Proper-scoring calibration does not transfer OOD.
- Ensemble avg(tt@T2.5, hobson): JB-hard .515, REAL-label .762, CF flip .552. Either-right oracle: JB-hard 83/130, REAL-label 338/400 => complementary errors.
- Queued on g4: eval schema/multi/ship, then rec68 fused timing, rev (option-order reversal) eval, quant (w8a8, w4g64 RTN fake-quant) for tt and hobson.

## 22:48 PDT
- schema-first eval (one q/seq, untruncated) MEASURED: JB-hard .454, REAL-label .708, REAL agree_sd .699, LONG agree_sd .673, CF acc .771 flip .544, CF-probe acc .759 flip .553 (!), SHUF both_right .553.
  Decision agreement state-first vs schema-first: 0.760 over 3227 q => "one checkpoint serves both layouts" holds only loosely.
  CF-probe schema-first by kind: date_order .84 (state-first .04, hobson .06), id_match_distract .67 (.36/.38), amount_vs_limit .26/.37 (0/0). Question-first reading lets the GDN layers read the state knowing what to look for.
  Costs: human_insert .47 (sf .69), procedure_intent .55 (.78), state>=4k CF .40 (.59), JB-hard .454.

## 22:58 PDT (earlier headers corrected to laptop clock)
- multi (state-first, all of an item's questions in ONE pass, native this-that) MEASURED: on multi-question items decisions agree with one-q-per-seq only 0.907 (REAL-agree, 696 q) / 0.904 (LONG, 426 q). REAL-label .685 (vs .698), REAL agree_sd .806, LONG agree_sd .703. Questions in one pass interfere (q_k sees q_<k).
- ship (as shipped: max_state_tokens=1536, keeps the HEAD of the state) MEASURED: REAL-label .627, REAL agree_sd .639, CF flip .458, CF-probe flip .406, JB-hard .469.
- rec68 fused timing crashed with a device-side assert; debugging after the queue.
- rec68 crash cause: the graph read index tensors (slot, n_opt) that were freed when rebound in the next loop iteration (multiple graphs kept alive). Fixed (keep them referenced). run_tt was not affected (each graph deleted before rebinding).

## 23:20 PDT
- rev (option order reversed: labels AND legend lines) MEASURED vs sf: same decision 0.918 overall; choice 0.725-0.85 (REAL-agree choice 0.752), noul 0.94-0.97, score 0.70-0.94.
  Accuracy moves too: REAL-label .735 (sf .698), JB-hard .500 (.477), CF-probe flip .406 (.338). The letter head carries order/position bias.
- Queued: quant (hob w8a8 running, then tt w8a8, tt w4g64, hob w4g64) -> rec68 fused -> this-that-1.2 sf+schema -> hobson rev probe (hob_rev.py).
- Results copied to ~/decider2/h4/preds and box_logs.

## 23:38 PDT
- quant RTN, no rotation, own-bf16 reference (MEASURED): W8A8 hobson: REAL flips 3.42% (vs merged 3.23%), CF flips 2.5%, CF-probe 5.3%, CF fg_own .936, CF-probe fg_own .848.
  W8A8 this-that: REAL flips 4.16%, CF 4.1%, CF-probe 3.6%, CF fg_own .938, CF-probe fg_own .889. => same order of sensitivity; neither passes without rotation (G2 rotated hobson: 1.20%).

## 23:54 PDT
- W4 g64 affine weight-only RTN (mlx-4bit analogue) MEASURED: hobson REAL flips 5.8%, CF 7.0%, CF-probe 16.9%, CF fg_own .908, CF-probe fg_own .676.
  this-that REAL 8.9%, CF 10.6%, CF-probe 9.4%, CF fg_own .837, CF-probe fg_own .907. Mixed; neither close to the criterion.
- rec68 (the paper's 30.9 ms workload, 176-183 tok, 1 q/pass) in the fused runtime on A10G: median 16.04 ms, p95 16.05 (n=204). Their package on A10G: 47.6 ms.
- REPORT.md: the harness blocked writing a report .md file from this subagent ("return findings as text"); per brief rule on denials, not worked around. Report is returned as the final message.
- Remaining on g4: this-that-1.2 sf+schema, hobson option-reversal probe, mschema (multi-question schema-first one pass).

## 00:08 PDT
- this-that-1.2 state-first (MEASURED): JB-hard .500, JB-long .390, REAL agree_sd .786, LONG agree_sd .491, REAL-label .682, CF acc .733 flip .473 fgh .550, CF-probe acc .706 flip .412, SHUF both_right .470. Newer checkpoint does not close the real-traffic gap and tracks fewer CF flips than 1.0.
- this-that-1.2 schema-first (MEASURED): JB-hard .423 (McNemar 11 vs 24, p=.04), REAL-label .685, REAL agree_sd .697, CF flip .468, CF-probe flip .425 (1.0 schema: .553).

## 00:33 PDT
- hobson option-order reversal (deployed engine path, render_question option_order reversed) MEASURED: same decision 0.884 overall (this-that 0.918); choice 0.475 (CF) / 0.796 (REAL) / 0.863 (JB); noul 0.84-0.95; score 1.0.
  hobson under reversal: REAL-label .757 (.785), JB-hard .523, CF flip .288. => the zero-parameter letter head is no more order-sensitive than hobson's pointer head.

## 00:42 PDT
- mschema (all of an item's questions in one schema-first pass = the cacheable layout) MEASURED: REAL-label .720, REAL agree_sd .717, LONG agree_sd .655; on multi-q items decisions match single-q schema only .786 (REAL) / .836 (LONG).
- All queued jobs done; GPU idle. Everything copied to ~/decider2/h4 (preds/, box_logs/, lat_*.json, check.json, wdiff.json, calib.json, quant_scores.json, scores.json).
