# J1 notes: 2B-scale bidirectional encoder decider (T5Gemma-2B encoder, pointer head, masked state cache)

## 09:10 PDT start (box j1 launched 08:56, not ready yet; hard TTL ~18:56)
- Read BRIEF8, FAST_DECISION_MODEL.md, F7_REPORT, evalkit README, Brooker's two posts (WebFetch), d1/lean2.py, systems/g/lean.py+packed.py,
  h7 lat harness (h7lat.py: hobson 'plain' = state once + each question as a branch on the fused TTL runtime), g3's F7-replica trainer
  (recovered/g3/g3/train_g3.py, prep_g3.py, teacher.py). g3 has hobson teacher logits already computed on the F7 rows
  (rows_corpus_q.pt 102,547 rows, rows_real_q.pt 6,000 KL-only rows): re-running prep_g3 with the same seeds + a T5Gemma student
  tokenizer gives the same draws -> copy t_logits (saves ~65 min of teacher labelling).
- Checkpoint access: google/t5gemma-* are gated ('manual'); the laptop token's request is "awaiting a review" (403). Ungated encoder-only
  extraction Minthy/t5gemma-2b-2b-ul2-encoder-only (Gemma terms, 2,614,341,888 params, 288 tensors; byte-identical shards to
  PhatcatDK/t5gemma-2b-2b-ul2-encoder-only). Using it; flagged as a caveat (cannot hash-verify against Google's gated files).
- Arithmetic (first principles), per state token, GEMM FLOPs:
  - T5Gemma-2B encoder: 26 layers x (attn 2304x4096 + o 2048x2304 + MLP 3x2304x9216) = 2.025B weights -> 4.05 GFLOP/token (+ embedding gather).
  - hobson: ~2.77 GFLOP/token (F7: q08 0.996 = 0.36x). Encoder = 1.46x hobson's GEMM FLOPs AND 1.46x its weight bytes (4.05 GB vs ~2.8 GB bf16).
  - Bidirectional attention: 4*T^2*2048 per layer, x26: 5% of FLOPs at 1000 tokens, 17% at 4000 (no causal halving).
  - Prediction: in a fused runtime the encoder is slower than hobson at every length >= ~128 tokens (1.4-1.6x); at 64 tokens both are
    near their weight-streaming floors (encoder ~7.4 ms at 550 GB/s vs hobson 9.4 measured, which includes GDN fixed cost).
    Brooker's 30-35 vs 60-65 ms is then serving overhead, not architecture. To be measured.

## 09:45 box j1 up (setup ran 08:56-09:15); data, checks, training launched
- Data: prep_j1.py re-ran g3's F7 prep with the same seeds, Qwen twin == g3 rows id-for-id on all 102,547 + 2,086 + 6,000 rows (0 mismatches)
  -> hobson t_logits copied to the T5Gemma twin. Student rows: 108,547 (6,000 KL-only real states), 40.50M T5Gemma tokens (Qwen: 39.8M).
- Implementation check (check_layer.py, real REAL-agree state, 1,501 tokens): my EncTorso layer i vs HF T5GemmaEncoderLayer i on the same input:
  delta cos >= 0.9998 every layer. End to end, though, the T5Gemma encoder is chaotic in bf16: HF's own bf16 forward vs HF fp32 has 70/1501
  rows < 0.99 cos and 15 < 0.9 (min -0.007; low-information tokens ' the', ' $', spaces). Mine with softcap: 58 rows < 0.99, mean 0.9978 vs fp32
  (closer than HF bf16's 0.9942); mine without softcap: 103 rows < 0.99, mean 0.9947. => dropping the tanh softcap costs about the bf16 noise
  itself; DECISION: no softcap (lets training and serving use flash attention). Also: all 26 layers global (the checkpoint's sliding-4096 on
  even layers is ignored; identical below 4096 tokens).
- Attention for training: SDPA backends at hdim 256 on A10G (fwd+bwd, T=3072): flash 5.5 ms (49 TF-eq), mem-efficient 24 ms, math 22 ms;
  flex_attention backward needs 114 KB smem > 101 KB (fails). => packed rows + varlen flash (aten._flash_attention_forward/backward in an
  autograd.Function): masked mode = 2 calls/layer (state rows over state keys; question rows over their whole row), no padding at all.
  Verified vs the padded masked-mask reference: per-layer delta cos >= 0.99991 (masked, full, and windowed), LoRA grad cos median 0.9995.
- Train-step speed (compiled norms/GeGLU/rope, LoRA delta fused into the base GEMM via addmm with block-diagonal B): 5,140 tok/s short rows
  no-ckpt, 3,150 tok/s 3k-token rows with ckpt, 2,375 tok/s 9k rows. Activation memory 2.6 MB/token without ckpt -> ckpt above 4,600 tokens.
- MAIN RUN ck_main: masked (e1b) mode, F7 recipe, 3,392 steps x 32 rows, 1 epoch. Started 09:22, paused at step 52 for the LoRA fusion,
  resumed 09:40 at ~4,000 tok/s; ETA ~13:30.
- Coordinator addition (09:43): also report option-order flips (decisions unchanged + mean |dp| under 3 rotations; hobson 0.884 under reversal)
  and ECE besides Brier, for encoder and hobson. rot_hob.py (deployed engine, shared-prefix path) + ev_j1.py --rot (same orders, rotlib.py).

## 10:05 first exclusive latency window (training paused 09:54-10:01 at step ~260, resumed)
- Fused EncRT verified vs the training-path reference (LoRA unmerged, packed varlen) on 47 real questions (REAL-agree 4-q items, LONG, JB):
  argmax 46/47, median |dp| 0.0026, max 0.015 (bf16 noise of this chaotic encoder; LSE-merged multi-question path identical to single).
- A10G, 1 question (state T + 92-token question), median ms (p95 within 0.05 except where noted), n=20, fresh states:
  | T | 64 | 128 | 256 | 400 | 1000 | 2000 | 4000 |
  | encoder (EncRT) | 15.5 | 19.1 | 27.7 | 36.7 | 86.0 | 159.9 | 330.2 |
  | hobson fused (TTL/lean2, same layout) | 15.2 | 15.6 | 22.9 | 28.2 | 57.2 | 108.4 | 201.7 |
  | ratio | 1.02 | 1.22 | 1.21 | 1.30 | 1.50 | 1.48 | 1.64 |
  Encoder at 1000: 4.66 TFLOP in 86 ms = 54 TFLOPS; at 4000 60.6 TFLOPS: the runtime is near the A10G's GEMM rate; the gap is the 1.46x
  FLOPs/token (+ bidirectional attention, 17% at 4000). No crossover in the encoder's favour above 64 tokens; tie at 64.
- 4 / 15 questions (encoder: state once + all questions packed in ONE pass, LSE-merged attention). hobson: 'plain' = existing fused path
  (state once, each question its own branch = weights streamed M+1 times); 'single' = one causal pass over state+all questions = LOWER BOUND
  for a packed hobson:
  Q4:  enc 34.2/37.2/45.2/55.2/99.5/184.2/353.5; hob plain 48.6/49.0/55.6/67.2/93.4/145.8/243.7; hob LB 28.1(128)/37.5/44.3/79.9/117.4/215.1
  Q15: enc 130.1/135.9/144.7/156.2/202.3/285.4/464.0; hob plain 176.2/176.8/184.1/196.5/224.0/278.8/-; hob LB 97.2/97.7/104.3/108.7/138.8/183.3/-
- Fork plan: e1a (full attention: state reads the question) branch from ck_main's step-1700 resume state, same data order, to 3392.

## 10:35 waiting on ck_main (step ~600/3392, 4,050 tok/s, ETA ~13:10)
- post_main.sh queued on the box: [eval A (+3 option rotations) || hobson rotations] -> fork B = e1a/full-attention from the step-1700
  resume state (fork_watch.sh copies it) to 3392 -> [eval B || Q3 zero-shot window eval of A (21 local layers, window 512; global at 4,9,14,19,25)].
- Runtime additions (verified on the 2-layer debug model, both GEMM paths): cuBLAS + side kernels (row-scale, GeGLU) as autotune candidates
  for the folded/fused GEMMs. hobson fairness: AUTOTUNE=1 gives lean2's Triton GEMMs the same per-shape tile search (its fixed tile at
  M<=384 is BM=128, which wastes ~60% of a second M tile at 157 rows). Both re-timed in the final exclusive window.
- scorer score_j1.py: evalkit + McNemar (exact) + Brier/ECE (10-bin top-label) + option-rotation stats + REAL-label by state length.
  Sanity: hobson refs as preds reproduce JB-hard .523, REAL-label .785, CF flip .268, CF-probe .328; hobson Brier/ECE: JB-all .348/.048,
  JB-hard .557/.135, REAL-label .347/.116, CF .521/.111, CF-probe .410/.083.

## 11:00 ck_main step 1460/3392 (ETA ~12:40)
- LC held-out (train-distribution, gold) at step 848 (25%): val_v5 .816 (hobson .873), generated_v16_eval .777 (.854), generated_v18_eval .668
  (.773), adequacy_gen_eval .520 (.808). The encoder is behind hobson on its own training distribution at 25%; adequacy is near chance.

## 11:15 step 1760; fork1700.pt captured (11:11); LC at 50% (step 1696): val_v5 .833 (hobson .873), adequacy .768 (.808), gen_v16 .789 (.854),
  gen_v18 .749 (.773). Still below hobson on every held-out train-distribution set, gap closing.

## 12:00 step 2640; LC at 75% (step 2544): val_v5 .854 (hobson .873), adequacy .791 (.808), gen_v16 .817 (.854), gen_v18 .749 (.773).
- post_main.sh relaunched 11:17 with final_lat.sh first (exclusive window right after the main run: runtime check with the final ck,
  gemm_bench small-M floor, encoder v2 timings incl. cuBLAS candidates, Q3 local-512 timings, hobson with autotuned tiles).

## 12:35 ck_main DONE (3,392 steps, 9,713 s train time, 4,170 tok/s). LC final: val_v5 .866 (hobson .873), adequacy .818 (.808),
  gen_v16 .817 (.854), gen_v18 .761 (.773). Final latency window running.

## 13:10 RESULTS A (e1b masked encoder, F7 recipe, 3,392 steps) + final latency
- Final runtime check (trained ck): fused EncRT vs training path 47/47 argmax, median |dp| .0005, max .004.
- Latency v2 (A10G, Q1, state T + 92-tok question): enc 15.46/19.0/27.8/36.7/85.8/159.8/330.2 ms; hobson with autotuned GEMM tiles
  (fair) 13.19/14.66/20.8/28.1/56.7/105.2/199.2 -> enc/hob 1.17/1.30/1.34/1.30/1.51/1.52/1.66. No crossover at any length.
  GEMM-only floor (gemm_bench, best tile per shape, x26): 14.4 ms at 156 rows (43.9 TFLOPS) = 93% of the measured 15.5 ms; 74.5 of 85.8 at
  1,092 rows; 247.6 of 330 at 4,092 (rest mostly bidirectional attention). Weight stream floor ~8 ms.
- Q3 local-512 on 21/26 layers (global 4,9,14,19,25): Q1 85.6 (1000, -0.3%), 151.8 (2000, -5%), 294.4 (4000, -11%); still 1.48x hobson at 4000.
- Q4/Q15 (enc one packed pass with state cache) vs hobson autotuned single-pass LB / per-question branches:
  Q4 1000: 99.4 vs 69.9 / 89.4; Q15 1000: 202.1 vs 138.6 / 210.1; Q15 64: 130.9 vs 90.0 / 163.1; Q15 4000: 467.8 vs 279.5 / 365.9.
- Suites (calibrated temps fit on LC rows: noul .76, choice 1.0, score .58):
  JB-all .688 vs .723 (McNemar 21/29 p .32); JB-hard .492 vs .523 (20/24 p .65); REAL-label .770 vs .785 (12/18 p .36; consensus .807/.818);
  REAL agree_sd .809; LONG agree_sd .842; CF acc .536 vs .594, flip .170 vs .268 (fgh .532), dir .904; CF-probe acc .598 vs .653, flip .222 vs
  .328 (fgh .448); SHUF change .222 vs .339 (reads the state LESS). Brier JB-all .397 vs .348, REAL-label .359 vs .347, CF .591 vs .521;
  ECE REAL-label .066 vs .116 (better), CF .224 vs .111 (worse). CF by kind: human_insert .051 vs .346, amount_insert .167 vs .100,
  wrapup .939 vs .918. For scale: F7's q08 (0.8B decoder, same recipe) REAL agree_sd .824, hob12 .884 -> the 2.6B encoder is below both.
- hobson option rotations (fused TTL, 1,997 rotated renderings of REAL-agree + JB-all): decisions unchanged .922, mean |dp| .058, TV .070.

## 13:55 (resumed after two Bedrock 503s; box untouched, chain kept running)
- Encoder option rotations (same 1,997 renderings): unchanged .949 vs hobson .922, mean |dp| .036 vs .058, TV .046 vs .070.
- Fork B (e1a, full attention) started 13:19 PDT from step 1700; at step 2380, ETA ~14:40; then eval B || Q3 zero-shot local-512 eval of A.
- 14:06 DRAFT_REPORT.md written (A results + latency; B and Q3-accuracy placeholders). B LC at step 2544: val_v5 .864 (A .854 at same step),
  adequacy .788 (.791), gen_v16 .823 (.817), gen_v18 .765 (.749). B ETA ~14:45, then eval B || Q3 eval (~30 min).

## 15:22 Q3 zero-shot (ck_main, 21/26 layers local-512 on state->state, question rows global), vs full-attention A:
  LONG agree_sd .842 -> .758; REAL agree_sd .809 -> .783; CF flip .170 -> .126 (fgh .532 -> .394); CF-probe flip .222 -> .244;
  REAL-label .770 -> .765; JB-all .688 -> .693; JB-hard .492 -> .500; SHUF change .222 -> .189. Cost concentrated on long/detail suites.

## 15:28 RESULTS B (e1a fork: full attention from step 1700 to 3392; the state reads the question)
  JB-all .688 (McN vs hob 19/27 p .30); JB-hard .477 (18/24 p .44); REAL-label .762 (12/21 p .16); REAL/LONG agree_sd .812/.842;
  CF flip .180 (fgh .550); CF-probe flip .316 vs hobson .328 (item McN 120/124 p .85) vs e1b .222: paired pairs only-e1b 14 / only-e1a 44, p 1e-4.
  CF-probe by kind e1a/hobson: date_order .143/.061, date_order_distract .156/.031, status_distract .477/.250, amount_vs_limit .047/.326.
  CF pairs e1b vs e1a 4/8 (p .39). Brier CF-probe .417 vs .410, ECE .031 vs .083. SHUF change .227.

## 15:45 e1a option rotations: unchanged .951, mean |dp| .039 (e1b .949/.036, hobson .922/.058). All evals done; post chain finished 15:26.
- DRAFT_REPORT.md final (prose ~1,480 words). Results JSON in results/: scores_A.json, scores_B.json, scores_A_loc512.json, lat_enc_v2.json,
  lat_enc_loc512.json, lat_hob_auto.json, lat_hob.json, gemm_bench.json, latcheck_final.json, rot_hob.json, preds_*.json, lat_curves.png.
- Box left idle (nothing running); checkpoints stay on the box (ck_main/, ck_full/). No other AWS resources touched.
