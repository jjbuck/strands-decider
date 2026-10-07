# J15 notes: speculative precision (W4 draft -> W8A8-b8 verifier only when the draft's margin is small)

## 13:10 PDT start
- Read BRIEF8, FAST_DECISION_MODEL.md, H1 FORMAT/NOTES, H2 NOTES/REPORT, H6 NOTES/REPORT, J5 DRAFT_REPORT/NOTES, evalkit README.
- Box j15 launched 13:07 (i-REDACTED, pending). Waiting for .ready; meanwhile offline analysis of existing deployed-kernel preds
  (H2/H6 preds on all 3227 eval questions: full prob dists for W4A4-GPTQ, k48, k64+qb, b8, bf16) - pure numpy scoring on the laptop.
- FIRST-PRINCIPLES COST ARITHMETIC (from measured anchors, before any experiment):
  cascade mean = D + f*V (full re-run verifier). Measured A10G 1q plain: W4A4 23.5 / b8 36.3 ms at T=1000 (D/V = 0.647);
  84.7 / 142.1 at 4000 (0.596); 5.58 / 7.38 at 32+1q (0.756); 11.5 / 17.0 at 400+1q (0.68).
  => the bar "mean <= 0.65x W8A8-b8" needs f <= 0.3% at T=1000, f <= 5.4% at 4000, and is IMPOSSIBLE at short lengths
     even at f = 0 with H2/J5's W4A4 kernels. The draft's own non-GEMM floor (44-47% of W4A4 time) is the binding constraint.
  => the cascade can only meet the bar if (a) verification is much cheaper than a full re-run, or (b) the draft gets faster.

## 13:35 PDT box j15 ready (13:27); setup
- A10G 1710 MHz, torch 2.14.1+cu130, triton 3.8, fla 0.5.2. CUTLASS v3.5.1 (g2s4) + CUTLASS main (h2mix, h2evt) built rc 0.
- H1 recipe: h1calib 64 train-split states (145k tokens) running, then act-order GPTQ8 / GPTQ4 codes (h2gptq.py).
- EXPLORATORY (not used for fitting; eval-split, existing H2/H6 deployed-kernel preds, casc.py): draft margin vs flips against b8:
  | draft | flips vs b8 | f to catch 90% | f to catch ~97-100% |
  | W4A4-GPTQ all 96 | 10.1% | 36.6% (tau .3) | 49.7% (tau .4, 97.5%) / 75.9% (100%) |
  | k48 all rows | 3.35% | 11.1% (tau .1) | 16.3% (tau .15, 97.2%) |
  | k48+qb+ba16 | 1.46% | 10.3% (95.7%) | 15.7% (100%) |
  | k64+qb+ba16 | 0.87% | 5.4% (96.4%) | 10.3% (100%) |
  Flips DO concentrate at small margin (median flipped margin .10 vs .44 non-flipped for W4A4), but the tail is long: W4A4 flipped
  decisions have margins up to .60. A flip needs |margin| < |4-bit perturbation|, so the deferral band scales with the draft's error.
- => cost: W4A4 draft at f ~ 50% = 0.65 + 0.50 = 1.15x of b8 (slower than b8 alone). k48 at f=16%: 0.80 + 0.16 = 0.96x.
- ORACLE BOUND (perfect uncertainty: defer exactly the draft's flips): mean >= D/V + flip_rate. W4A4 at T=1000: 0.647 + 0.10 = 0.75x;
  at 4000: 0.596 + 0.10 = 0.70x. Even a perfect uncertainty signal misses the 0.65x bar. k48: 0.80 + 0.034 = 0.84x.
- Why the speculative-decoding analogy breaks: decode verification is parallel (k drafted tokens checked in ~1 memory-bound step);
  prefill verification of a decision is a full compute-bound re-run. Reuse of the draft's partial products (W8 = 16 W4hi + W4lo) is only
  possible while draft and verifier see identical inputs, i.e. layer 0; after one layer the residual streams differ.
- The only verifier that wastes none of the draft's work is one that RESUMES it: a depth draft (the verifier's own first k layers at b8
  + an exit head), verified by continuing layers k..23. Cost = k/24 + f_k (24-k)/24 of b8. Queued as the bold extension (EXIT/DEV sets
  built from the train pool, disjoint by tau task: DEV 1736 q / 710 requests / 44 tasks; EXIT 4205 q / 2826 requests / 104 tasks).

## 13:47 PDT first deployed-kernel result: my regenerated W8A8-b8 (the verifier) is NOT at the floor by flip count
- Setup done 13:36 (H1 recipe: calib 64 states / 145k tokens, act-order GPTQ8+GPTQ4). Runner j15run.py = H2 QRT + residual taps
  (smoke: b8 probs match H2's b8 preds to ~0.005). chain1.sh running: b8/W4A4/k48/bf16/k64qb on eval + DEV (+ EXIT for b8/W4A4/k48).
- b8 (my codes), all 3227 eval questions [MEASURED]: REAL flips 12/1083 = 1.11% vs hobson (H2's codes: 2/1083; J5's regenerated codes:
  8/1083); TV .0048 (H2 b8 .0047, bf16 runtime .0033); paired vs H2 bf16 runtime 1 gained / 9 lost (p .02); LONG 5/471;
  CF fgh 1.000, CF-probe fgh .962, JB-hard .538 (McNemar vs hobson 2/0), REAL-label .793, Brier .349/.348.
  All 17 REAL+LONG flips have hobson margin < 0.025 (boundary noise), and TV equals H2's. => 'b8 passes at 0.18%' was a lucky GPTQ
  draw; the code-draw distribution is ~0.7-1.1% flips. Queued chain2: two more calibration draws (seeds 1, 2) to measure it.
- arXiv fetched for citation: 2211.17192 (speculative decoding), 2207.07061 (CALM early exit), 2404.16710 (LayerSkip self-speculation).
  Brooker's encoder post fetched (no mention of precision, cascades or early exit).

## 13:57 PDT RESULT 1 [MEASURED, deployed kernels, my codes]: W4A4 -> b8 margin cascade, tau fitted on train-split DEV only
- W4A4-GPTQ draft alone (eval 3227 q): flips vs b8 10.0%, REAL flips vs hobson 8.7%, CF fgh .771, CF-probe fgh .581 (H2: 8.96/.826/.657).
  Flipped-decision margins: median .099, p95 .349 (non-flipped median ~.44). DEV (1736 q): draft flips vs b8 6.8%.
- tau from DEV (residual draft-vs-b8 disagreement among accepted <= eps of DEV), applied unchanged to eval:
  | eps (DEV) | tau | DEV f | EVAL f | REAL flips vs hob | paired vs b8 | TV | CF fgh | CF-probe fgh | JB-hard |
  | 0     | .559 | .636 | .714 | 1.11% | 0/0 | .018 | 1.000 | .962 | .538 |
  | 0.1%  | .405 | .423 | .518 | 1.57% | 1/6 p .13 | .032 | .991 | .962 | .531 |
  | 0.2%  | .346 | .358 | .442 | 1.75% | 1/8 p .04 | .038 | .991 | .971 | .531 |
  | 0.3%  | .313 | .317 | .399 | 2.03% | 1/11 p .006 | .041 | .982 | .971 | .531 |
  Deferral by suite at eps 0: JB .61, REAL .66, LONG .65, CF .67, CF-probe .95. DEV under-predicts eval deferral (DEV flips 6.8% vs 10%).
- Cost at A10G 1000 tokens: 0.647 + 0.714 = 1.36x b8 (equal decisions) ... 0.647 + 0.40 = 1.05x (and 2% flips). FAILS by a mile.

## 14:10 PDT cheaper-verification checks (from measured preds)
- Per-question-name tau (fitted on DEV) overfits: eval f .47 but REAL flips 2.6% (paired vs b8 2/18). Global margin tau stays.
- Re-run only the uncertain questions of a multi-question request does not help: the state rows carry the 4-bit error, so any deferred
  question needs the state re-run at W8. REAL-agree 4-question requests (174): P(any question deferred) = .98 at tau .559, .87 at .346.
- ORACLE (defer exactly the draft's flips vs b8, eval): f = 10.0% for W4A4 -> cost >= 0.647 + 0.100 = 0.75x of b8 at T=1000.
- Depth-exit smoke (layer-12 exit head on b8 residual, trained on EXIT, early-stopped on DEV): DEV agreement with the full b8 decision .945.
  Full exit-head training for layers 4..20 running on the box (exit.log).

## 14:22 PDT RESULT 2 [MEASURED, eval suites; tau from DEV]: depth-speculative cascade, single exit at layer 12 (head trained on EXIT)
  | DEV eps | tau | EVAL exit share | REAL flips | paired vs b8 | TV | CF fgh | CF-probe fgh | JB-hard |
  | 0     | .440 | .484 | 1.39% | 0/3 p .25 | .018 | .991 | .924 | .538 |
  | 0.2%  | .318 | .672 | 1.48% | 0/4 p .13 | .023 | .917 | .819 | .500 |
  | 0.4%  | .258 | .745 | 1.66% | 0/6 p .03 | .026 | .844 | .752 | .492 |
  Cost (layers computed / 24, before measuring): eps 0 -> .484*.5 + .516 = 0.76x b8, already = the W4A4 oracle bound, and 0.56x better
  than the W4A4 margin cascade at equal decisions. But exits are confidently wrong exactly where hobson reads details late:
  CF-probe fgh .924 (b8 .962) at eps 0; DEV (real traffic) has no JSON-probe items, so tau cannot see that shift.

## 14:33 PDT RESULT 3 [MEASURED]: a learned uncertainty from the draft's hidden state does not beat its margin
- MLP (final residual answer row + margin/entropy/logit-gap/kind/T/n-options) -> P(W4A4 decision != b8), trained on EXIT (4205 q,
  W4A4 and b8 both run), early-stopped on DEV. Residual flips at equal deferral: DEV identical to margin (f .4: .0012 both);
  eval worse than margin (f .4: .0112 vs .0068; f .6: .0019 vs .0003). DEV-fitted eps 0: eval f .604, REAL 1.20%, CF-probe .962.
- Interpretation: the 4-bit perturbation is noise the draft cannot see (if it could, it could be corrected), so the only observable is
  how close the draft is to the boundary. f >= P(|verifier margin| < ~2-3 sigma_draft-noise) is set by the draft's noise scale.

## 14:45 PDT RESULT 4 [MEASURED decisions; latency not yet measured]: hobson's last 8 layers barely move decisions -> depth speculation works
- Exit heads (hobson pointer head as init + LayerNorm/gain + rank-512 residual adapter; KL to the b8 FINAL distribution on EXIT;
  early-stopped on DEV). DEV agreement with the full b8 decision: L8 .908, L12 .951, L16 .994 (DEV KL .0004).
- Head alone, no deferral (eval 3227 q): agreement with b8 L8 .800, L12 .895, L16 .988. L16 alone: REAL flips vs hob 0.74%, sd .994,
  CF fgh .991, CF-probe fgh .943, JB-hard .538, REAL-label .787 (a truncated 16-layer model nearly passes; fails CF-probe).
- Cascade, tau fitted on DEV at zero residual (eps 0), applied to eval:
  | exits S | exit share eval (dev) | REAL flips | paired vs b8 | TV | CF fgh | CF-probe fgh | JB-hard | layer-cost |
  | {16}    | 16: .928 (.952)                 | 1.11% | 0/0 p 1 | .0086 | 1.000 | .962 | .538 (=b8) | .928*16/24+.072 = 0.69 |
  | {8,16}  | 8: .203, 16: .727 (.286/.666)    | 1.20% | 0/1 p 1 | .018 | 1.000 | .962 | .538 | .203/3+.727*2/3+.07 = 0.62 |
  | {12,16} | 12: .484, 16: .449               | 1.39% | 0/3 | .020 | .991 | .924 | .538 | 0.62 |
  | {8,12,16} | .203/.288/.443                 | 1.48% | 0/4 | .026 | .991 | .924 | .538 | 0.58 |
  Exits at L12 are where CF-probe breaks (confidently wrong on deep-context JSON reads); L8 (very conservative tau .62) and L16 are safe.
- {8,16} decisions = b8's except 1 REAL question, CF/CF-probe/JB-hard identical -> passes fidelity against its verifier; vs hobson it
  inherits b8's (my codes) 1.1% flips. Next: measure prefix latencies (pre:L) and e2e; repeat on the bf16 base (floor-level hobson
  fidelity, independent of the GPTQ draw); add L6/10/14/18/20 heads when trained.

## 14:58 PDT all exit heads; DEV-only exit-set selection; latency grid running (exclusive GPU)
- DEV agreement with full b8 / DEV KL(b8||exit): L4 .866/.087, L6 .891/.061, L8 .908/.049, L10 .925/.029, L12 .951/.013,
  L14 .981/.0016, L16 .994/.0004, L18 .9965/.0002, L20 1.000/.0001. For scale: KL(bf16||b8) on DEV = .00011 (agree .9977).
  => the decision is formed by layer ~14-16; layers 20-23 change it as little as 8-bit rounding does.
- DEV-optimal exit sets (all subsets of <=4 exits, tau at zero DEV residual, DEV layer-cost): (4,10,14,18) .504, (4,8,14,18) .505 ...
  On eval they cost .54-.56 but fail CF-probe (.895-.933) and add 4-7 REAL flips vs b8. Exits at L <= 14 are confidently wrong on the
  JSON-probe reads; DEV (real traffic) contains no such items, so a DEV-fitted tau cannot see the shift.
- Exit sets restricted to {16} or {8,16} keep CF 1.000 / CF-probe .962 (= b8) / JB-hard = b8 (eval layer-cost .691 / .622).
  Chosen after seeing L12/L16 eval singles => post hoc; a DEV-only rule that selects them: exit only where the head's DEV KL to the full
  model is within ~4x the verifier's own quantization KL (L >= 16).
- Timestamps of the 13:57-14:45 entries were corrected (first written ~10 min ahead of the clock).

## 15:03 PDT RESULT 5 [MEASURED latency, A10G exclusive, CUDA graph, fresh ids, 20 reps after 3 warm; p95 within 0.2 ms of median]
- 1 question (details_match, 125 tok), state T tokens (rows = T+125), median ms:
  | T | 16 | 64 | 256 | 400 | 1000 | 2000 | 4000 | 9000 |
  | bf16 (H2 fold) | 15.5 | 15.7 | 25.5 | 36.9 | 57.1 | 107.6 | 203.5 | 458.2 |
  | W8A8-b8 | 8.20 | 8.42 | 14.36 | 18.67 | 36.32 | 67.23 | 142.1 | 337.1 |
  | W4A4 | 5.78 | 5.95 | 9.65 | 12.21 | 23.42 | 42.81 | 84.57 | 220.0 |
  | k48 | 6.96 | 7.12 | 11.90 | 15.09 | 29.18 | 53.59 | 109.8 | 272.1 |
  W4A4/b8 = 0.70 (short) / 0.645 (1000) / 0.595 (4000) / 0.65 (9000). b8 reproduces the doc's 36.3 / 142.1.
- b8 prefix (layers [0,L) + exit head) / full: L4 .17, L8 .33, L12 .50, L14 .58, L16 .665, L18 .75, L20 .83 at every length;
  resume post:8 .67, post:12 .50 -> pre + post = full within 0.5%: resuming the draft wastes nothing.
- Real length distributions (per question, latency interpolated at each question's exact row count; REAL+LONG+JB n=1785, b8 mean 101 ms):
  | system | mean x b8 | REAL-agree | LONG | JevBench |
  | W4A4 draft alone | .626 | .616 | .634 | .638 |
  | W4A4 -> b8, eps 0 (decisions = b8) | 1.284 | 1.28 | 1.28 | 1.38 |
  | W4A4 -> b8, eps .3% (2.0% flips) | .969 | .98 | .95 | 1.11 |
  | W4A4 -> b8, ORACLE deferral | .707 | .71 | .70 | .79 |
  | k48 -> b8, eps 0 (decisions = b8, f .12) | .920 | .91 | .92 | 1.01 |
  | depth {16} | .694 | .684 | .704 | .684 |
  | depth {8,16} | .604 | .601 | .603 | .669 |
  | depth DEV-optimal {4,10,14,18} (fails CF-probe) | .519 | .517 | .516 | .588 |
- k64+qb (my codes) on its own: REAL 0.55% flips, paired vs bf16 2/3 p 1, TV .0041, CF 1.000, CF-probe .981 - better than my b8; as a
  draft it is pointless (slower than b8, 40.4 ms H6).
- Projections (H2 model): W4A4/b8 D/V 3090 .68, 4090 .76, 5090 (NVFP4) .73 at T=1000 -> the precision cascade gets WORSE on newer cards
  (non-GEMM floor); depth savings are a layer fraction and carry over unchanged: {8,16} at T=1000: 3090 12.0, 4090 7.0, 5090 4.8 ms.

## 15:19 PDT RESULT 6 [MEASURED]: depth cascade on the bf16 base (b8-trained heads applied to bf16 residual taps; tau re-fitted on bf16 DEV)
  | exits | exit share eval | REAL flips vs hob | paired vs bf16 base | LONG | TV | CF fgh | CF-probe fgh | JB-hard | REAL-label |
  | bf16 alone | - | 0.46% | - | 0.42% | .0033 | 1.000 | .971 | .538 | .782 |
  | {16}   | .932 | 0.46% | 0/0 | 0.42% | .0081 | 1.000 | .971 | .538 | .785 |
  | {8,16} | .207+.727 | 0.55% | 0/1 | 0.42% | .0183 | 1.000 | .971 | .538 | .782 |
  | {14,16} | .821+.115 | 0.46% | 0/0 | 0.42% | .0137 | 1.000 | .943 | .538 | .785 |
  | {12,16} | .482+.454 | 0.74% | 0/3 | 0.42% | .0197 | 1.000 | .924 | .538 | .787 |
  => on a floor-level base the {16} and {8,16} cascades pass the full function-preserving bar (flips at the floor, CF 1.000,
  CF-probe .971, JB-hard identical). Speed: 0.6-0.69x of the BASE, so with bf16 it is ~0.94x of b8 on the A10G
  (4090 with fp16 accumulation: 0.6*14.74 = 8.8 ms vs b8 11.6 = 0.76x). Hitting both bars needs a floor-level 8-bit base.
- e2e.sh (segment-graph cascades on real requests) queued after chain2; chain2 now on GPTQ draws s1/s2 (DEV-selectable).

## 15:31 PDT RESULT 7 [MEASURED]: b8 GPTQ code-draw variance; depth cascade replicated on a second draw
- Draw s1 (calibration states seed 1): REAL 1.02% flips vs hobson, TV .0049, paired vs bf16 2/8 (p .11), LONG 0.42%, CF 1.000,
  CF-probe .943, JB-hard .538. Draw s0 (mine): 1.11% / .0048 / CF-probe .962. H2's: 0.18% / .0047 / .971. On DEV the draws are
  indistinguishable (flips vs bf16 4 vs 6 of 1736, KL .000110 vs .000114) -> draw selection on train-split data is impossible;
  b8's eval flip rate (~1%) and CF-probe (.94-.97) vary with the draw, TV does not.
- Depth cascade on s1 (heads trained on s0 residuals, tau re-fitted on s1 DEV): {16} 0/0 vs its base (CF 1.000, CF-probe = base,
  JB-hard .546); {8,16} 0/5 vs base (p .06). The L8 exit is fragile across bases; {16} is not (0/0 on s0, s1, bf16).
- Calibration: Brier JB-all base .349 -> {16} .354 -> {8,16} .374 (JevBench is out of the exit heads' training distribution);
  REAL-label Brier unchanged (.347). JB-all acc {8,16} .723 vs base .732 (3/3 vs hobson, n.s.).

## 15:44 PDT RESULT 8 [MEASURED]: third draw + pooled replication of the depth cascade across four bases
- Draw s2: REAL 0.65% flips, TV .0048, paired vs bf16 3/5 (p .73), LONG 0.64%, CF 1.000, CF-probe .971, JB-hard .538 -> this draw
  passes the floor bar by itself (like H2's). b8 across draws: REAL 1.11 / 1.02 / 0.65% (H2 0.18%), CF-probe .962 / .943 / .971.
- Depth cascade, tau re-fitted on each base's DEV, heads from s0:
  | base | {16}: REAL flips, paired vs bf16, CF-probe (base) | {8,16}: REAL flips, paired vs bf16, CF-probe |
  | b8 s0 | 1.11%, 2/9 p .07, .962 (.962) | 1.20%, 2/10 p .04, .962 |
  | b8 s1 | 1.02%, 2/8, .943 (.943) | 1.48%, 2/13, .943 |
  | b8 s2 | 0.65%, 3/5 p .73, .962 (.971) | 0.83%, 3/7 p .34, .962 |
  | bf16  | 0.46%, 2/2, .971 (.971) | 0.55%, 2/3, .971 |
- POOLED over the 4 bases, REAL+LONG (6216 decisions): {16} changes 0 decisions of its base; {8,16} changes 9 (all losses vs hobson,
  sign test p .004). CF fgh 1.000 everywhere. => {16} is decision-identical to its base on real traffic at 0.694x of its cost;
  the L8 exit buys 0.09x more at a measurable 0.15% drift.

## 15:48 PDT RESULT 9 [MEASURED]: 'refine only the later layers' is not a verifier
- W4A4 layers [0,12) + b8 layers [12,24) (= the decisions of re-running layers 12..23 at b8 from the W4A4 draft's residual):
  9.1% flips vs b8 (W4A4 alone 10.0%), REAL 7.3% vs hobson, CF fgh .761, CF-probe .610. The 4-bit error is made in the early
  layers and survives an exact late half; refinement fixes 9% of the draft's flips for 50% of the verifier's cost.

## 16:00 PDT RESULT 10 [MEASURED e2e]: W4A4 -> b8 cascade on 120 real requests at exact lengths (draft graph -> host margin -> b8 graph)
- REAL-agree 54: cascade mean 109.3 / p50 91.9 / p95 231.3 ms vs b8 88.0 / 76.7 / 156.8 (1.24x); LONG 18: 246.9 vs 200.4 (1.23x);
  JB 48: 37.4 vs 25.0 (1.50x); all: 101.2 vs 79.7 ms = 1.27x (interpolation predicted 1.28x). Deferral 0.62.
- First dcasc attempt crashed (P.model.head moved to CPU by an in-place .cpu() when building exit heads); fixed (deepcopy), rerunning
  {16} and {8,16} on 120 requests each (e2e2.sh).

## 16:04 PDT RESULT 11 [MEASURED e2e]: depth cascade {16} on 120 real requests (segment graphs [0,16)+exit head -> host -> [16,24)+head)
- REAL-agree 54: 58.7 / p50 51.1 / p95 104.6 ms vs b8 88.1 / 76.8 / 156.8 (0.667x); LONG 18: 141.7 vs 200.4 (0.707x);
  JB 48: 17.7 / p50 5.56 vs 24.9 / 8.10 (0.710x); all: 54.8 vs 79.7 ms = 0.688x (interpolation predicted 0.694x). Exit share .97.
  Every e2e exit decision matches the offline exit prediction (0 mismatches).

## 16:08 PDT RESULT 12 [MEASURED e2e]: depth cascade {8,16} on the same 120 requests
- REAL-agree 50.8 / p50 45.3 / p95 91.0 ms vs b8 88.1 (0.577x); LONG 121.3 vs 200.3 (0.606x); JB 17.3 / p50 5.74 vs 24.9 / 8.11
  (0.695x); all 48.0 vs 79.7 ms = 0.603x. Exits: 24 at L8, 92 at L16, 4 full; 0 mismatches vs offline predictions.
- All results copied to ~/decider2/j15/{preds,res,box_logs}; exit-head weights and taps stay on the box (no models on the laptop).

## 16:12 PDT wrap-up
- DRAFT_REPORT.md written (final). Verdict: speculative precision refuted (1.27x b8 e2e at equal decisions; oracle 0.71x; worse on
  newer cards). Depth speculation (exit at layer 16, resume otherwise) preserves its base's decisions on 6216 real decisions across
  4 bases at 0.688x of b8 e2e; {8,16} 0.603x with a 0.15% drift. Combined bar missed narrowly (0.688 vs 0.65; base b8 draws ~1% flips).
- Box j15 left idle (no jobs); exit heads, taps and codes remain on the box (~/work/j15). It auto-terminates at its TTL.
