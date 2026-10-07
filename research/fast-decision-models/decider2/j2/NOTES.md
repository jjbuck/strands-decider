# J2 notes: bidirectional GDN hybrid encoder from hobson (box j2)

## 09:05 start
- Read BRIEF8, FAST_DECISION_MODEL.md, F7_REPORT, evalkit README, Brooker's two posts (fetched), h7 code (h3lib/h7lib differentiable lean hobson forward, branch layouts), d1/lean2.py fused runtime.
- Fetched precedents: LLM2Vec 2404.05961 (bidirectional attn + MNTP, i-1 predicts masked token, 20% masking, 1000 steps x 32, LoRA r16);
  Caduceus 2403.03234 (BiMamba: forward + flipped-sequence pass, in/out projections SHARED, outputs summed); Vision Mamba 2401.09417 (bidirectional SSM backbone);
  Hydra 2407.09941 (quasiseparable bidirectional matrix mixer, +0.8 GLUE over BERT); LION 2502.16249 (bidirectional linear attention = forward + backward RNN, equivalent to full linear attention);
  Gated DeltaNet 2412.06464.
- Box j2 launched 08:56 (pending). Auto-terminates ~18:56.

## Design (decided before the box is up)
- Function-preserving init: every new path is gated at 0, so the untrained bidirectional model IS hobson (testable exactly).
  - GDN layer: o = o_fwd + gamma (.) o_rev, gamma per channel [2048] init 0; reverse scan shares q,k,v,beta,g and the causal conv output (Caduceus-style shared projections).
  - Attention layer: o = o_causal + lambda_h (o_noncausal - o_causal), lambda per head init 0.
- Masked state cache (QAg, e1b-like): both scan directions process the state before the question. Forward stream s1..sT, q1..qL; reverse stream sT..s1, qL..q1.
  So state rows never see the question (cacheable, shareable across questions), and each question's reverse scan starts from the state's reverse final state
  (an early-weighted summary of the state, which hobson's forward stream has mostly decayed away). Attention: state<->state full; question -> state + whole question.
- Question-aware (QA, e1a-like): plain bidirectional over [S;Q]: reverse stream qL..q1, sT..s1; attention fully non-causal over [S;Q].
- Early: reverse scans only in GDN layers 0-10 (9 of 18).
- Control: causal hobson + the same fresh LoRA + same data/steps (separates "more fine-tuning" from "bidirectionality").
- Adaptation: MNTP (LLM2Vec) on real train-split states with the base LM head (tied embedding), then F7-recipe decider fine-tuning.

## 09:50 lib verified on box j2 (A10G)
- j2lib.py (packed varlen rows; built on H3's differentiable lean hobson): with gates 0, causal/qag/qa all equal H3 hobson within bf16 noise
  (max |dlogit| 0.03 single, 0.02 packed; H3 vs evalkit hobson refs max |dp| <= 0.0044). [verified]
- Masked state cache is exact in 'qag': same state under two different questions -> state rows bit-identical (max |dh| 0.0000);
  d(state rows)/d(question inputs) = 1.2e-4 vs 1.8e4 for state inputs (kernel residue), while 'qa' gives 2.1e3. Answer row depends on first 5 state tokens in both. [verified]
- Train throughput (LoRA r16 all projections + gates + head, checkpointing, 8k-token packs): first version 1.9k tok/s (masked mem-efficient SDPA over the pack
  was 37% of time). Fixed: per-row flash SDPA, fla conv with cu_seqlens, fla fused gated norm -> causal 3.4k tok/s, qag (18 reverse scans + 2 extra attn) 2.9k tok/s. [measured]
- Data (j2data.py build, running): 12.8k train_v5 + 1.5k v19 synthetic (generated_v16/v18, adequacy_gen) + 2.6k real train-split states (one question each, KL only);
  option order shuffled p .5; hobson teacher distribution computed on the same ids. MNTP states: 3k real train-split states <= 2048 tokens.
- Plan: 500 FT steps x 32 rows per arm (~6M tokens, ~35 min), MNTP 200 steps x 8k tokens. Queue: qag_all -> causal control -> qa_all (reuses qag MNTP: identical on state-only rows) -> qag_early.

## 10:12 data built; 10:47 first run; 11:07 budget cut and restart
- Teacher pass: 16,900 rows (12.8k v5, 1.5k synthetic, 2.6k real), hobson argmax == gold on 0.886 of labelled rows (F7 reported 0.887). [measured]
- MNTP (qag_all, 200 steps x 8k tokens, 20% masked with '_', predicted from i-1 through the tied embedding): loss 7.07 (step 10) -> 1.25 (step 180);
  gates open: lam (non-causal attention mix) 0.12-0.22 by layer, mean |gam| (reverse GDN) 0.03-0.05. [measured]
  Right after MNTP, hobson agreement on real train rows was 0.70 (KL 1.13): the adaptation moves the decision function a lot; FT must recover it.
- First FT attempt was too slow for the budget (11M tokens at ~1.7k tok/s: allocator fragmentation after MNTP + 6k-token rows). Restarted (queue2.sh):
  280 steps x 32 rows, rows <= 4096 tokens (dropped, not truncated, so teacher targets stay exact): 8,960 rows, 4.6M tokens per arm, ~2.2k tok/s.
  All bidirectional arms start from the SAME MNTP checkpoint (qa and qag are identical on state-only rows; early simply ignores the late gates).
- Queue: qag_all -> causal control -> qa_all -> qag_early; evals overlap the next training run.

## 12:35 first result: qag_all (MNTP 200 + FT 280 steps, 4.6M tokens; 28 min train, eval 33 min shared GPU)
| | JB-all | JB-hard | REAL-label | REAL agree_sd | LONG agree / sd | CF acc / flip | CF-probe acc / flip | SHUF both | Brier / ECE |
| qag_all | .745 | .569 | .785 | .919 | .932 / .855 | .599 / .286 | .659 / .331 | .266 | .422 / .047 |
| hobson | .723 | .523 | .785 | 1 | 1 / 1 | .594 / .268 | .653 / .328 | .245 | .436 / .037 |
- McNemar vs hobson: JB-hard 9 vs 3 (p .15), JB-all 9/4 (p .27), REAL-label 10/10, CF pairs 17/10 (p .25), CF-probe pairs 30/29. [measured]
- Passes the new-architecture bar but CF / CF-probe essentially tie hobson. Need the causal control (same FT) to attribute anything.
- Gates after FT: mean |gam| .042, mean lam .174 (FT barely moved them from MNTP values).
- Fused bidirectional runtime check (j2lat.py check vs j2lib, gates 0.5): min row cosine 0.998, answer row 0.99997. [verified]
- 12:55 pipeline validation: untrained J2 (gates 0) through j2eval on JB-all + CF-probe: JB-all agree .996 (tv .003), JB-hard agree .992, CF-probe agree .988
  (tv .0035, flip .319 vs .328, flip_given_hobson .962). Same as evalkit's merged_full noise floor (JB-all .992). [verified]
- Prepared a reading-augmentation phase (j2cf.py: H7's train-split CF-style pairs, CE only; j2train --init --cf): to test whether bidirectional
  arms LEARN detail reading faster than the causal control when the data demands it.

## 13:55 causal control scored -> first attribution
| | JB-all | JB-hard | REAL-label | REAL agree_sd | LONG agree / sd | CF acc / flip | CF-probe acc / flip | SHUF both | Brier / ECE |
| causal (same FT) | .714 | .508 | .772 | .954 | .977 / .964 | .579 / .254 | .642 / .303 | .230 | .449 / .064 |
| qag_all | .745 | .569 | .785 | .919 | .932 / .855 | .599 / .286 | .659 / .331 | .266 | .422 / .047 |
| hobson | .723 | .523 | .785 | 1 | 1 / 1 | .594 / .268 | .653 / .328 | .245 | .436 / .037 |
- qag_all vs causal control, paired exact McNemar (qag-only right / causal-only right, p): JB-hard 11/3 p .057; JB-all 11/4 p .12; REAL-label 13/8 p .38;
  CF pairs 20/7 p .019; CF items 24/8 p .007; CF-probe pairs 34/25 p .30. [measured]
- The extra FT alone (causal) drifts slightly BELOW hobson everywhere; the bidirectional arm sits above hobson, but vs hobson itself nothing is significant
  (CF pairs 17/10 p .25, JB-hard 9/3 p .15).
- CONFOUND: qag_all also had the MNTP adaptation (1.6M tokens of LM-objective training on real states); the causal control did not. Added an
  MNTP-matched causal control (causal_mntp: identical MNTP rows/steps, causal) and dropped the CF-augmentation phase for time.
- qa_all training slower than planned (1.3k tok/s while sharing the GPU with evals). Queue3 now: qa eval || qag_early train -> exclusive latency ->
  qag_early eval (no LONG) || causal_mntp -> causal_mntp eval || zero-gate ablation of qag_all.

## 15:05 qa_all (question-aware, e1a-like) scored
| | JB-all | JB-hard | REAL-label | REAL agree_sd | LONG agree / sd | CF acc / flip | CF-probe acc / flip | SHUF both | Brier / ECE |
| qa_all | .740 | .562 | .792 | .919 | .936 / .867 | .607 / .308 | .655 / .325 | .292 | .422 / .048 |
- vs causal control: CF pairs 28/6 p .0002, CF items 30/7 p .0002, JB-hard 10/3 p .09, REAL-label 15/7 p .13. [measured]
- vs hobson: CF pairs 23/7 p .005 (the only significant gain over hobson itself so far), JB-hard 8/3 p .23, REAL-label 12/9.
- vs qag_all: CF pairs 15/6 p .078, REAL-label 6/3, JB identical, mutual agreement .98. Brooker's question: attending to the question while reading
  the state is NOT a weakness here; if anything a small strength on CF. Cost: no state sharing across questions.
- CF by position (edit in first half / second half): hobson .169/.342, causal .151/.329, qag .174/.368, qa .203/.385. CF on >=4k states: all ~0.01-0.03.
- 14:52 switched to queue4 (causal_mntp trains alongside qag_early; latency exclusive at the end).

## 16:35 qag_early scored (eval without LONG for time)
| qag_early | JB-all .753 | JB-hard .585 | REAL-label .780 | REAL agree_sd .916 | CF .595 / .281 | CF-probe .659 / .331 | SHUF .269 | Brier .422 / ECE .059 |
- vs qag_all: mutual agreement .979 on REAL; CF pairs 10/12, JB-hard 2/0: indistinguishable. Late-layer reverse scans add nothing measurable. [measured]
- vs causal control: JB-hard 12/2 p .013, JB-all 12/3 p .035, CF pairs 23/12 p .09.
- MNTP diagnostic: at step 200 the bidirectional model's masked-token loss is 1.17 vs 1.67 for the identical causal MNTP (same rows):
  the reverse scans + non-causal attention carry ~0.5 nats of right context. [measured]

## 17:20 MNTP-matched causal control scored -> the attribution changes
| causal_mntp | JB-all .740 | JB-hard .562 | REAL-label .790 | REAL agree_sd .899 | CF .599 / .271 | CF-probe .661 / .331 | SHUF .253 | Brier .425 / ECE .044 |
- vs causal_mntp (identical MNTP rows/steps, causal; identical FT): qag_all ties everywhere (JB-hard 8/7, CF pairs 12/6 p .24, CF-probe 20/20);
  qag_early ties (CF 17/13); qa_all ties except CF pairs 25/10 p .017 (CF items 26/19 p .37). [measured]
- So most of the gain over the plain causal FT came from the LM-style adaptation on real states, not from bidirectionality.
  The one residual bidirectional effect is question-aware reading on CF pairs.
- Latency (exclusive, 16:45): bi_all +1.5 ms at 64, +7.7 at 1000 (+13.5%), +38.4 at 4000 (+18.7%); early +0.8 / +4.7 / +27.5; qa = bi_all for 1 question,
  2.9x hobson for 4 questions at 1000 tokens (240 vs 83 ms), 4.2x at 4000 (940 vs 223). [measured]

## 17:35 final
- DRAFT_REPORT.md written. Box idle (all queues done); checkpoints (ck/*/final.pt, mntp.pt) remain on the box under ~/work/j2/ck.
- Laptop copies: preds/*.json, lat.json, score_final.json/.txt, box_logs/.
