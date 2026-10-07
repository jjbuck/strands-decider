# J9 notes: compile the deployment (box j9)

## 09:05 start
- Read BRIEF8, FAST_DECISION_MODEL.md, F7_REPORT, h7 DRAFT_REPORT, evalkit README.
- Box j9 (i-REDACTED) pending; no .ready yet.
- State structure (REAL-agree sample): framing sentence (per hook) + section headers + conversation; the steering hook injects
  KB documents ("injected 7 documents for choose_credit_card") as a system note after the first user turn, each doc truncated
  ("...[truncated]"); "Not shown: ..." lists; hook system notes (COMPLETION CHECK etc.); tool results.
- Constants are NOT a prefix in hobson's layout: only the frame + header is. Docs sit after the first user message.
- Plan: (1) measure constant share on suites + train_pool + raw logs (char level locally, tokens on the box);
  (2) layouts: exact prefix cache (function-preserving), relayout untrained, per-document compiled blocks with
  affine GDN state composition; (3) train LoRA distilled from hobson original layout; (4) evalkit + latency.

## 09:28 measurement + library
- const_analysis.py (laptop, chars) and seg_check.py (box, hobson tokenizer): token-level constant share (frame + compiled blocks >= 32 tok).
  0 tokenization mismatches: concat(piece tokens) == whole-state tokens on every state of REAL/LONG/CF/CF-probe/JB.
  - REAL-agree 37.2% (banking hooks 36-68%, retail 12-15%), LONG 54.9%, CF 33.9%, CF-probe 24.5%, airline ~2%, JevBench 0%.
  - Blocks are KB document bodies (dominant), hook notes, tool schemas; frame only ~1-2% (~45 tokens).
- j9lib.py: segmentation -> pieces (frame/blk/dyn); per-block compile in universal context U='<state>\n'; GDN affine composition
  S <- A_i (S - S_U) + E_i (identity verified: rel err 1.3e-4 vs direct); attention K stored pre-RoPE and re-rotated at runtime.
  Layouts: R (constants first, one live segment), S (in place splice), Rx (reorder exact). comp affine|last|skip.
- test: all-dyn plans reproduce native (0.327/0.331 vs 0.330 on one item; numerics of gconv/SDPA path).
- 09:28 launched untrained diagnostics queue (diag.sh): R-affine (REAL,LONG,CF,CF-probe), S-affine, Rx, R-skip.

## 09:58 untrained diagnostics (every 3rd item of REAL-agree, LONG, CF; CF pairs broken by the stride)
- R-affine (constants first, blocks compiled in U context, affine GDN composition), UNTRAINED hobson:
  REAL agree .956 / agree_sd .921 (n~360 q), LONG agree .899 / agree_sd .774, REAL-label .756 (hobson same items .756).
  For scale: drop@7 .41, rand50@7 .87, qattn10@7 .85 agree_sd; h7 trained schema-first .685.
- train_j9.py smoke OK (R layout, LoRA r16, KL + dense rows at layers 5/11/17/23 + CE on cf/v5). Untrained hid relMSE .054, KL .012.
- j9lat.py written: TTL fused runtime + compiled frame/blocks cache (A,E,tail,K,V), per-request composition in the graph.

## 10:12
- FULL untrained R-affine eval (all of REAL-agree/LONG/CF/CF-probe; JB identical to hobson by construction: JevBench has no constants):
  REAL agree .936 / agree_sd .899 (n_sd 346), LONG .911 / .800, REAL-label .775 (hobson .785, McNemar p .50),
  CF pair acc .256 (hobson .268, p .36) fgh .890, CF-probe pair acc .344 (hobson .328, p .51) fgh .848. SHUF both_right .240 (hobson .245).
- S-affine (in place) untrained, stride-3 subset: REAL agree_sd .868 vs R .921 on same items; LONG .792 vs .774 -> R chosen (also simplest runtime).
- First training run (h7 mix with CF-aug CE) restarted after 110 updates: switched to the F7 recipe (no CF aug, which is the CF-probe
  generator = in-distribution leak), lr 1e-4, head 3e-4, accum 6, 1200 updates, fixed 48-request held-out probe (train split).
  Untrained probe KL .0128, agree 48/48. Training-sample KL (~.05-.1) is dominated by the option-permutation floor.
- Fused runtime check (TTL + compiled frame/blocks cache, composition inside the graph) vs j9lib R: argmax 10/10, max|dp| med .004, max .007.
- LIBRARY SIZE PROBLEM (raw logs, non-strict segmentation): banking 9,542 distinct blocks (3.0M tokens; K/V 37 GB; GDN E 90 GB bf16).
  Cause: 604 distinct KB docs but 3,620 doc-body variants (truncation points; omission markers merged), plus 7,222 'other' recurring runs.
  Added strict segmentation (J9_STRICT=1: only doc bodies, hook notes, tool schemas). Measuring.

## 10:45
- Raw gate logs (31,013 records -> 24,399 unique states; hobson tokens; non-strict segmentation): mean 2143 tok, median 1768, p90 4505;
  constant share 46.5% (banking 52.1%: after_tool_call 67%, after_model_call 52%, before_invocation 41%, before_model_call 34%,
  before_tool_call 8%; retail 20%). Live tokens: median 856, p90 2538.
- Strict library (J9_STRICT=1: KB doc bodies, hook notes, tool schemas only): banking 2,781 blocks / 1.16M tokens (389 blocks = 90% of uses),
  K/V 14.2 GB total (12 KB/token), GDN E 26 GB bf16 (9.4 MB/block); retail 11 blocks. Frames: 10 banking, 6 retail (~45 tok).
- Untrained ablation (stride 3): comp=skip (blocks invisible to GDN, attention only) REAL agree_sd .912 / LONG .811 vs affine .921 / .774.
  => the GDN composition buys little; attention access to compiled K/V carries the documents. A K/V-only library suffices (no 9.4 MB/block states).
- Training at upd 90 (slowed 2x by concurrent evals).

## 10:55 (real clock; earlier stamps after 09:58 were ~20 min fast)
- Untrained ablations, stride-3 REAL-agree / LONG agree_sd (SE ~2.6 / ~3.5 pts):
  R-affine .921/.774 | R-last .912/.774 | R-skip .912/.811 | R-affine strict segmentation .930/.792 | R-affine primed (doc header as compile ctx) .904/.792
  | S-affine (in place) .868/.792 | Rx = constants-first ORDER but exact full recompute .947/.830.
  => the loss decomposes into the reorder (~5 pts REAL, ~17 LONG) and the context-free compile (~3 / ~6); composition variants are within noise.
- Training restarts: v2 (lr 1e-4 + CF CE + perm) and v3 (pure KL, lr 5e-5, accum 8) both made the probe WORSE early
  (v3: probe KL .0128 -> .0278, agree 1.0 -> .854 at upd 50): Adam noise on an already-close model (weak signal) dominates.
  v4 (running): lr 3e-5, head 3e-5, accum 12, 450 updates, pure KL + dense rows, 96-request probe (untrained KL .0258, agree .927).
- Strict raw-log share: 34.0% overall (banking 40.6%, retail 3.2%); strict suites: REAL 24.9%, LONG 48.0%, CF 27.0%.

## 11:34 COMPILE ADAPTER works
- v4 (all-weights LoRA, lr 3e-5, accum 12) probe at upd 50: KL .0258 -> .0289, agree .927 -> .906 (worse again). Killed.
- New design: the LoRA is a COMPILE ADAPTER acting only on the compile rows (U + blocks); the live path (frame, dynamic state,
  question, head) is exactly hobson. The adapter only shapes the cached constants (K/V, GDN E/A). States without constants
  (JevBench, airline) are bit-identical to hobson by construction.
  Run ckC: lr 5e-5, accum 8, 400 updates, pure KL + dense rows (layers 5/11/17/23), real states with blocks only.
  Probe (96 held-out train-split requests): upd 0 KL .0258 / agree .927; upd 50 .0259 / .958; upd 100 .0117 / .969.

## 12:15 latency (A10G, TTL fused runtime, bf16, CUDA graph per exact shape, n=15 warm reps, fresh inputs, exclusive GPU)
- Grid (state T + 125-token question; compiled share c=.37 / .55 of the state, 4 / 11 blocks), 1 q, skip composition:
  64: 15.2 -> 15.0/9.7 | 128: 15.5 -> 16.3/15.8 | 256: 22.8 -> 16.4/17.3 | 400: 28.1 -> 23.7/24.2 | 1000: 57.1 -> 45.0/39.2 | 4000: 201 -> 140/103.
  4 q: 1000: 93.3 -> 79.2/70.9; 4000: 243.5 -> 168.8/144.5. Affine composition costs +1..4 ms (fp32 state matmuls + elementwise on 19 MB states).
- Speedup < token ratio at <=1000 tokens: profile at 1000 (c=.55, 543 live rows): GEMMs 33.3 ms vs 46.0 ms for 1000 rows (small-M tile/wave
  efficiency 75%), + ~5 ms composition / re-rotation / prefix attention.
- Real distribution (84 eval-split requests: 60 REAL-agree + 24 LONG, first question, exact shapes): banking mean 180.5 -> 94.9 ms (1.90x),
  retail 116.5 -> 101.7 (1.14x), REAL-agree 1.47x, LONG 1.98x (skip); affine 1.85x banking.
- Running: full eval of ckC/s200 (compile adapter) on all suites.

## 12:45 trained compile adapter s200, all suites (3,227 q)
- JB-all .727 (hobson .723), JB-hard .531 (.523): live path = hobson; JB has no constants.
- REAL-label .767 (hobson .785, McNemar p .17); CF pair .259 (.268, p .42) fgh .917; CF-probe pair .341 (.328, p .59) fgh .876.
- REAL agree .947 / agree_sd .893 / tv .039 (untrained .936/.899/.050); LONG .928 / .836 / .060 (untrained .911/.800/.081).
- Misses concentrate in the 23-way procedure questions: REAL 22 of 37 sd misses (needed_procedure 43/58, procedure 31/38);
  LONG 23 of 27 (needed_procedure 23/46). All other questions: REAL .94, LONG .97 agree_sd.
- Same adapter evaluated with skip composition (K/V-only library): REAL .934/.884/.046, LONG .930/.842/.066.
- 12:40 resumed from s200 -> 400 with 40% focus sampling on needed_procedure/procedure (ckC2).

## 13:30
- ckC2 (resume s200 -> s400, 40% focus on procedure questions): probe KL .0102 (250), .0098 (300), .0095 (400); agree .948.
  Learning curve of the compile adapter on the 96-request probe: .0258 (0) -> .0117 (100) -> .0115 (200) -> .0095 (400).
- Running: full eval of ckC2/s400 (affine), then skip composition on REAL/LONG/CF/CF-probe.

## 13:46 s400 full eval (compile adapter, affine)
- JB-all .727, JB-hard .531 (= s200); REAL-label .767 (p .17); CF pair .256 (p .30) fgh .908; CF-probe pair .344 (p .46) fgh .886.
- REAL agree .952 / agree_sd .908 / tv .035; LONG .936 / .861 / .054. SHUF both_right .240.
- Procedure questions (needed_procedure + procedure) agree_sd: REAL .750 -> .771 -> .792, LONG .408 -> .531 -> .612 (untrained, s200, s400);
  all other questions REAL .956/.940/.952, LONG .966 throughout. Still rising at s400.

## 14:00 final
- s400 with K/V-only (GDN-invisible blocks) composition: REAL agree_sd .887 / LONG .861; REAL-label .765 (p .13); CF pair .264 (p .77) fgh .936;
  CF-probe pair .344 (p .44) fgh .895; SHUF both_right .245.
- All scores in final_scores.json (final_score.py). Report in DRAFT_REPORT.md. Checkpoints stay on the box (ckC/s200.pt, ckC2/s400.pt).
