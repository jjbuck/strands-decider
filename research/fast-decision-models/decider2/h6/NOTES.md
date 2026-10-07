# H6 notes (box g1): full 24-layer hobson-v19, ~4-bit compute, compiled schema, mixed-row precision, QAT in deployed numerics

## 00:30 PDT start
- Read BRIEF7, H2 REPORT+NOTES, H1 NOTES/FORMAT (qb16 finding: question rows carry a large share of low-bit error), H3 NOTES
  (layer-0 Wo, attn Win at 7/11 most sensitive; layers 12-22 ~insensitive), H4 NOTES (this-that schema-first), H5 FORMAT v1 (learned R1,
  k48 dev flips 1.25%; suite numbers for learned-R1 k48 not yet available; artifacts only on g5).
- g1 free (GPU idle). H2 runtime at ~/work/h2 (qrt.py/qk.py identical to ~/decider2/h2/code). My dir: ~/work/h6.
- Plan: (1) QRT6 = H2 runtime + row-split precision (rows < r0 low-bit, rows >= r0 bf16, same forward, no kernel changes: bf16 rows write
  into the int GEMM's fp16 'acc*alpha' output domain with their own per-row scale) + H1 rotated-space LoRA rule on GPTQ codes.
  (2) schema-first layout for hobson = [question minus final '<answer>' token][state][one '<answer>' slot]; options read from the compiled prefix.
  (3) QAT with a custom autograd GEMM (exact H1/H2 quantizers in forward, STE, low-rank LoRA grads without materializing dW), codes cached per update.
  (4) bf16 schema-first control: same data/steps/LoRA. Teacher = H2 bf16 runtime, hobson state-first, precomputed.

## 00:52 PDT (corrected; first written as 01:15)  SCHEMA-FIRST LAYOUT USED BY H6 (for H7 to match)
- Token ids: hobson's own, exactly as the evalkit references: `kitrun.prep_question(P, item, q)` / `plib.P.prep(state, spec)` ->
  s = tokens of render_state(state) (add_special_tokens=True, starts '<state>\n', ends '</state>\n'), q = tokens of render_question(spec)
  (starts '<question type=..>', ends '</options>\n</question>\n<answer>'); the '<answer>' marker is the LAST 3 tokens of q ('<','answer','>', last id 29).
- Schema-first sequence, ONE question per sequence (questions independent, as SystemOneEngine/evalkit):
  [ q[:-3]  = question bundle, positions 0..P-1, compiled once per question spec (GDN states, conv tails, attention K/V) ]
  [ s       = state, positions P..P+Ls-1 ]
  [ q[-3:]  = '<answer>' slot (3 tokens), positions P+Ls..P+Ls+2 ]
  decision row = the last slot token ('>'); option rows = q0-relative option offsets pr['opt'] INSIDE the compiled bundle (they do not see the state).
  Head = hobson's PointerHead (frozen) on final RMSNorm rows, temperature P.temp_for(kind), as hobson.
- Per-question masking: none needed because each question gets its own sequence (its slot sees only its own question + the state).
  NB for multi-question latency: H2's 4q/15q schema bench put all questions in ONE bundle and ran the state once (slots see all questions;
  GDN layers cannot be masked per question). That is a different function from what is trained/evaluated here; I report 1q numbers as the
  evaluated function and the shared-bundle multi-q numbers as a speed bound only.
- Training data (h6teach.py): train_pool.jsonl requests (eval tasks excluded, n_state_tok >= 200), one random question per request, state truncated
  to 1536 tokens (first 384 + last 1152, as h1qat); 20% train_v5 rows (every 10th row, choice questions, gold label, weight 0.1 CE).
  Teacher = H2 bf16 runtime, hobson STATE-FIRST probs on the same truncated tokens (KL(teacher||student)).
- LoRA: r32 on all 96 GEMMs in H1's rotated input basis (y += (x R_in) A^T B^T), head frozen.

## 01:08 PDT (corrected; first written as 01:50)  built + verified + launched
- qrt6.QRT6 (row-split runtime) smoke on 12 REAL questions (MEASURED, deployed kernels): r0=0 split path (all rows through the bf16 split
  route) vs H2 fold bf16: final-hidden cos 0.99999, probs equal to 1e-3. Uniform k48-GPTQ: cos 0.9995. **Question rows bf16 / state rows k48
  (plain layout, r0=q0): cos 1.0000 to 4 decimals** (k48 0.9995). Only the last row bf16: 0.9996 (no help) -> the question rows as a group
  carry the readout error, consistent with H1's qb16.
  e.g. cc_refuses: bf16 .503, k48 .381 (flip), qb16 .505.
- h6teach.py: 5500 examples (4400 train-pool real requests, 1100 train_v5 gold rows), teacher = H2 bf16 runtime state-first. 495 s.
- h6qat.py run a launched 00:57: ctl (bf16 schema-first) + qat (k48 GPTQ, mixed rows: bundle+slot bf16, state low-bit) on the SAME example
  each step; LoRA r32 all GEMMs, lr 1.5e-4 cosine, accum 4, 2700 micro-steps planned, ckpt every 300. 2.0 s/step (both arms), 10.1 GB.
  Train KL (moving, 50 steps): step 250 ctl .71 / qat .70; 300 .51/.50; 350 .41/.44; train flips vs teacher ~0.35 (layout change dominates).
- Eval queue (run_queue.sh): plainqb_k48, schema_bf16_s0 (untrained), schemamix_k48_s0, then every checkpoint (qat_sN schemamix, ctl_sN schema bf16).

## 01:17 PDT  run 'a' stopped, run 'b' launched
- Run a (head frozen, lr 1.5e-4): no learning visible in 400 micro-steps (50-step mean train KL .70 .21 .59 .62 .71 .51 .41 .83; flips vs
  teacher .45 .18 .58 .46 .36 .32 .38 .46, both arms alike). Untrained schema-first hobson starts near chance on many questions.
  Diagnosis: in schema-first the option rows are computed BEFORE the state (fixed per question), so the frozen pointer head (trained on
  state-aware option rows) sees nearly identical option keys; all state evidence must move into the decision row.
- Run b: same data/order, pointer head TRAINABLE (one copy per arm, lr 1e-3; saved in the ckpt as _head and loaded by h6eval/bench6),
  LoRA lr 3e-4, 3000 micro-steps (750 updates), ckpt every 300. Queue b evaluates every ckpt (qat_b_sN schemamix k48, ctl_b_sN schema bf16).

## 01:25 PDT (corrected; first written as 01:40)  RESULT 1 (MEASURED, deployed kernels, all 3227 questions, no training): mixed rows in hobson's own layout
| metric | hobson | H2 bf16 | k48 GPTQ (H2) | k48 GPTQ, question rows bf16 (plainqb) | W8A8-GPTQ b8 (H2) |
| REAL flips vs hobson | 0 | 0.37% | 2.59% | **1.29%** | 0.18% |
| REAL agree_sd | 1 | .994 | .965 | .986 | .997 |
| LONG agree_sd | 1 | .988 | .939 | .976 | .982 |
| CF fgh | 1 | 1.000 | .972 | .991 | 1.000 |
| CF-probe fgh | 1 | .971 | .848 | .914 | .971 |
| JB-hard acc (McNemar p) | .523 | .538 | .538 (.62) | .546 (.38) | .523 (1.0) |
| REAL flips vs H2 bf16 | - | - | 2.95% | 1.11% | 0.55% |
- Question rows in bf16 HALVE the k48 error (flips 2.59 -> 1.29%, CF-probe fgh .848 -> .914) but k48 still fails the low-bit bar:
  the state rows' 4-bit error remains. (Passes the co-design bar: sd .986/.976, fgh .991/.914.)
- 01:36 run b (head trainable) looked the same as run a over its first 150 steps (same example order: KL .68/.21/.57 vs .70/.21/.59).
  Restarted as run c with a THIRD arm 'qatp' = QAT in hobson's own state-first layout with question rows bf16 (step 0 = the plainqb_k48
  model above; bf16 reference = hobson itself, no layout change) so the low-bit question is answered even if the schema-first layout
  does not train in time. Run c: arms ctl, qat (schema-first k48 mixed rows), qatp; 1800 micro-steps, lr 3e-4, head lr 1e-3 (ctl/qat only),
  ckpt every 300; eval at 600/1200 (subset: JB-all, REAL-agree, CF, CF-probe) and 1800 (all suites).
- qrt6: optional OVL=1 = bf16 rows on a side CUDA stream, overlapped with the int GEMMs (for latency; tested later).

## 01:59 PDT (corrected; first written as 02:35)  untrained schema-first, aggressive map, run d stopped, H1's best config queued (coordinator priority)
- schema_bf16_s0 (untrained hobson in my schema-first layout, bf16, 2756 q = all but LONG) MEASURED: REAL flips 39.3% vs hobson,
  **agree_sd 0.000, CF flip 0.000, CF-probe flip 0.000** (= the no-state baseline: decisions vary only ~0.01 in p across states), JB-hard .477,
  REAL-label .655. The frozen pointer head reads the bundle's state-blind option rows: the layout must be learned essentially from scratch.
- plainqb_a16 (aggressive map: W4A4 everywhere except layer 0 (4 GEMMs) and attention Win+Wo (12) at W8A8; question rows bf16), all suites:
  REAL flips 2.03%, agree_sd .962, LONG sd .897, CF fgh .899, CF-probe fgh .800, JB-hard .538 -> fails (W4A4-GPTQ all-row: 8.96%).
- Run d (3 arms, lr 3e-4 schema / 1e-4 qatp) stopped at step ~330 (ckpt s300 kept) to give the GPU to the coordinator's priority config.
  50-step train windows (same example order in every run): ctl KL .69 .23 .. .63 .72 .53, flips .51 .26 .. .50 .32 .40 (no clear learning
  in 75 updates); qat (schema, k48 mixed) tracks ctl; qatp (plain-layout QAT from plainqb_k48) KL .011 .018 .. .033 .034 .032, flips
  .04 .04 .. .10 .14 .10 -> rising, i.e. LoRA-QAT at lr 1e-4 is moving AWAY from the teacher (H1 found the same: no eval gain).
  (run c, lr 3e-4 on qatp: first window already .042.)
- qrt6 BA16=1: H1's 'ba16' = GDN b/a gate rows (Win rows 8192..8223) from a bf16 GEMM on the unquantized normed input. New Triton kernel
  _addq2_k (= H2's addq + also stores the bf16 normed row when quantizing), b/a rows of the conv-kernel output overwritten for the low-bit rows.
  Uniform layouts with BA16 go through the split forward with r0 = T. Smoke: k48 -> k48+ba16 moves e.g. cc_refuses .381 -> .502 (bf16 .501).
- run_prio.sh: plainqb_k48_ba16 (clip .9), plainqb_k48_ba16_c85 (clip .85 = H1's exact best), then qatp_d_s300 / ctl_d_s300 (subset).

## 02:14 PDT (corrected; first written as 03:05)  RESULT 2 (MEASURED, deployed kernels, ALL suites, hobson layout): H1's best 4-bit point fails in the real runtime
| metric | hobson | H2 bf16 | k48 GPTQ | +qb16 | +qb16+ba16 (clip .9) | +qb16+ba16 clip .85 (= H1 best) | W8A8-GPTQ b8 |
| REAL flips vs hobson | 0 | 0.37% | 2.59% | 1.29% | 1.20% | 1.11% | 0.18% |
| REAL TV to hobson | - | .0033 | .0246 | .0097 | .0096 | .0093 | .0047 |
| REAL paired vs H2 bf16 (agree-w-hobson lost/gained, p) | - | - | 28/4 (.00) | 11/1 (.01) | 10/1 (.01) | 9/1 (.02) | 2/4 (.69) |
| REAL agree_sd | 1 | .994 | .965 | .986 | .986 | .983 | .997 |
| LONG agree_sd | 1 | .988 | .939 | .976 | .982 | .958 | .982 |
| CF fgh | 1 | 1.000 | .972 | .991 | 1.000 | 1.000 | 1.000 |
| CF-probe fgh | 1 | .971 | .848 | .914 | .924 | .914 | .971 |
| JB-hard (McNemar vs hobson) | .523 | .538 | .538 | .546 | .538 (0/2) | .538 (1/3) | .523 |
- qb16 cuts the 4-bit excess TV over the bf16 floor by ~70% (.0213 -> .0064) but the remaining state-row error still flips 1.1-1.2% of REAL
  (vs 0.5% bar) and CF-probe fgh stays at .91-.92 (bar .97). ba16 and clip .85 are within noise of each other. H1's subset estimate
  (REAL-SD 0.58%, CF-probe .933) does not hold on all suites in the deployed kernels: REAL agree_sd .986 = 1.4% SD flips.
- run_prio's qatp/ctl s300 evals failed (head state_dict copy into inference tensors); fixed, re-queued after the latency bench.
- 02:12 latency matrix running (exclusive GPU): bf16 / k48+ba16 (plain, plainqb, schema, schemamix, sets, setsmix; OVL on/off) / k48 / W4A4.

## 02:55 PDT  latency matrix done; sets layout; QAT checks
- Writing ~/decider2/h6/REPORT.md was DENIED by the harness ("subagents should return findings as text"); not worked around. The report
  is returned as the final message (as H1/H2/H4).
- Latency (MEASURED, exclusive, res_bench_{bf16,k48,k48ba,k48baovl,w4a4ovl}.jsonl; medians ms, T=1000/4000, 1q):
  bf16 plain 56.9/203.6, bf16 sets 52.8/199.4; k48+ba16 plain 29.5/111.6, plainqb(OVL) 38.5/121.0, sets uniform 27.5/108.9,
  **setsmix (bundle bf16, state k48, slot-set rows bf16) serial 35.7/116.8, OVL 33.0/116.1**; W4A4 all schema 21.5/82.6, setsmix OVL 27.9/89.5.
  15q T=1000: sets 34.1, setsmix OVL 44.8, schemamix OVL 39.0. bf16 slot rows cost +8.2 ms serial / +5.5 ms OVL at 1q T=1000:
  a bf16 weight copy (2.8 GB) read for 5 rows + ~1100 extra small kernels (kernel count 477 -> 1557).
  => k48 mixed-row schema (33.0) is no faster than W8A8-GPTQ-b8 schema (33.5, H2), which passes.
- Untrained sets layout (H7's slot sets, 1 question per sequence) bf16: REAL flips 56.8%, agree_sd .197, CF fgh .037; k48+ba16 mixed in the
  same layout vs its bf16 version: 3.9% REAL flips (low-margin untrained model).
- Run d qatp s300 (all-row LoRA lr 1e-4, KL to runtime teacher): REAL flips 1.29% -> 5.63% vs hobson, TV .0097 -> .0449, CF fgh .972,
  CF-probe fgh .918 => QAT damaged the model.
- Run e launched 02:28: arm qatq = plain layout k48+qb16, LoRA ONLY in the low-bit state rows (question rows keep hobson's exact weights),
  loss = KL to the in-emulation dense teacher + 1.0 x residual relMSE (layers 5/11/17/23), lr 5e-5, 2000 steps. Train windows stable:
  KL 8e-4 .. 5e-4 .. 1e-3, flips 0-2%, hid .0099 -> ~.009 (flat). Evals at 600/1200 (subset), 2000 (all).

## 03:48 PDT  RESULT 3 (MEASURED, deployed kernels, ALL suites): k64 + qb16 + ba16 reaches the bf16 runtime's own accuracy
- k64 = H1's sensitivity ranking top-64 GEMMs at W8A8, the other 32 (layers 13-22 mostly; 37.9% of GEMM MACs) W4A4; GPTQ codes; GDN b/a bf16;
  question rows bf16 (hobson layout):
  REAL flips 0.46% vs hobson (H2 bf16 runtime 0.37%), REAL TV .0040 (bf16 runtime .0033; W8A8-b8 .0047; k48+qb .0096),
  paired vs bf16 runtime 2 lost / 1 gained (p 1.0); REAL agree_sd .997, LONG agree .998 / sd .994, JB-long agree 1.000;
  CF fgh 1.000, CF-probe fgh .962 (bf16 runtime .971; 101 vs 102 of 105 tracked pairs); JB-hard .523 (McNemar 0/0);
  REAL-label .782; SHUF .245. vs the in-runtime bf16: REAL flips 0.28%, CF fg_ref .973, CF-probe fg_ref .971 -> PASSES the low-bit bar
  against the bf16 runtime; against hobson's references fails only CF-probe fgh by one pair (.962 < .97, at the runtime floor .971).
- k48 + qb16 + ba16 (53.1% MACs at 4-bit): 1.20% / .924 -> fails. So the pass/fail boundary lies between 38% and 53% 4-bit MACs.
- Run e (qatq: LoRA only in the low-bit state rows, KL + residual MSE, lr 5e-5) evals: s600 REAL flips 1.57% (init 1.29%), TV .0110
  (init .0097), CF-probe fgh .914; s1200 0.92% flips but TV .0110, CF-probe .886, paired vs init 6 lost / 10 gained (p .45):
  decisions reshuffle within noise, TV does not improve -> no QAT gain (4th QAT recipe without eval gain, with H1/G2/H5).
- Run d ctl s300 (schema-first single slot, trainable head): REAL flips 28% (untrained 39%), agree_sd .358, CF fgh 0, CF-probe .012.
- Queued: plainq8_k48_ba16 (question rows W8A8 instead of bf16), qatq_e_s2000 (all), plainqbr_k64_ba16 (question rows bf16 EXCEPT the
  readout rows = option rows + '<answer>', which stay low-bit: the hobson-layout analogue of 'bundle bf16, slots low-bit' = zero
  per-request bf16 cost in the schema layout); then latency batch 2 (k64 plain/sets/setspq/setsmix, b8 sets, SLOT8).

## 04:05 PDT  RESULT 4 (MEASURED, deployed kernels)
- Readout rows matter: k64, question rows bf16 EXCEPT option rows + '<answer>' (low-bit) [plainqbr, all suites]: REAL flips 0.65%,
  TV .0110 (vs .0040 with them bf16), CF fgh .982, CF-probe .933 -> the slot rows must stay high precision (a schema layout cannot
  run its answer slots in 4-bit for free); with 'only the last row bf16' (smoke) there was no gain either -> both the question body and
  the readout rows need it.
- Question rows at W8A8-GPTQ8 instead of bf16 (SLOT8, k48+ba16, subset): REAL flips 1.39%, TV .0104 (bf16 question rows .0096,
  all-row k48 .0246) -> 8-bit high-precision rows keep ~95% of the qb16 gain, with int8 weights (no bf16 weight copy).
  (First SLOT8 run had a bug: Wo/Wd int8 rows were fed unrotated activations; fixed by rotating with R2/R4 first; the buggy preds were deleted.)
- Run e final s2000 (all suites): REAL flips 1.11% (init 1.29%), TV .0103 (init .0097), CF fgh .982 (.991), CF-probe .895 (.914),
  paired vs init 4/6 (p .75) -> no QAT gain.
- 04:00 latency batch 2 running (exclusive): k64 (plain, sets, setspq, plainqb/setsmix OVL), k48 setspq, SLOT8 OVL, W8A8-b8 sets.
  Then plainq8_k64_ba16 (all suites).

## 04:18 PDT  final measurements
- plainq8_k64_ba16 (k64, question rows W8A8 instead of bf16, all suites): REAL 0.65%, TV .0057, CF fgh .991, CF-probe .943 -> fails narrowly;
  k64 needs bf16 question/slot rows to reach the floor.
- Latency batch 2 (MEASURED, exclusive; medians ms T=1000 / 4000, 1q): k64+ba16 plain 31.4/119.6; plainqb OVL 40.4/128.9;
  sets uniform 29.2/116.7; **setsmix OVL (bundle bf16, state k64, slots bf16) 34.2/123.9**; W8A8-b8 plain 36.2/142.3, sets 33.6/139.0;
  SLOT8 (slots W8A8) k48 setsmix OVL 33.5 (no gain over bf16 slots 33.0 in this torch-op implementation).
  15q T=1000/4000: k64 setsmix OVL 46.6/144.4, b8 sets 40.6/155.9. Projections (proj6.py): k64 setsmix OVL T=1000 3090/4090/5090
  22.2/13.0/8.8, T=4000 73.6/42.6/29.0; b8 sets 18.4/10.7/7.3 and 77.4/43.2/29.9.
- Final scores in ~/decider2/h6/scores.json (h6score.py); preds in ~/decider2/h6/preds; bench json + train logs in ~/decider2/h6/box.
- Box g1 (~/work/h6) keeps code, teach.pt, LoRA checkpoints (not copied: no models on the laptop).
- 04:22 k56 (+qb16+ba16, 46% of GEMM MACs at 4-bit), all suites: REAL flips 1.02%, TV .0056, CF fgh .982, CF-probe .943, paired vs bf16
  runtime 8/1 (p .04) -> fails. Frontier with question rows bf16: k48 (53% 4-bit) TV .0096 fail; k56 (46%) .0056 fail; k64 (38%) .0040
  passes vs the bf16 runtime (floor .0033). Report returned as final message (REPORT.md write denied).
