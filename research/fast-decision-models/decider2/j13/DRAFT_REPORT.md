# J13: token-adaptive width on full 24-layer hobson-v19, tested oracle-first

**Verdict: dead by its own kill criterion.** No configuration at ≤ 0.6x FLOPs reaches the bar, even with an oracle router that knows hobson's answer. CF-probe fgh tops out at .867 (and that needs 0.63x; runtime floor .971, bar .90); LONG agree_sd tops out at .945. The only near-pass routes nothing: every state row thin after layer 13, a role-based depth cut in the eviction class (1.40x at 1000 tokens), and it still misses LONG (.933).

Labels: **[M]** measured, **[V]** verified two ways, **[S]** speculation/arithmetic.

## 1. Hypothesis
hobson spends 2.75 GFLOP on every token, yet half the state tokens carry under 0.5 bit. If background state tokens take a low-rank path through the same weights in layers ≥ k, while high-signal tokens and all question/answer rows stay full width, cost should fall about 2x at every length with the same 24-layer 2B, composable with W8A8.

**Arithmetic correction [S].** A rank-r GEMM costs r(K+N)/(KN), because outputs (N up to 12288) must be rebuilt at full width for conv/GDN/SwiGLU/residual: ρ = .08/.16/.32 of a layer at r = 128/256/512. Quarter rank is 0.32, not 1/8; 70% thin from layer 4 is 0.60x, not 0.5x.

## 2. What I built (`j13/code/`)
- **`tw.py`**: plain-torch hobson runtime (merged LoRA), TV .002 / 0 flips vs hobson refs on 24 questions [M]. Thin-path bases fitted on 128 train-split states (274k tokens): input PCA (`in`) and data-aware SVD y = U_r U_rᵀ W x (`out`, MSE-optimal).
- **Oracle**: one forward+backward per question; gradient of hobson's decision log-odds w.r.t. a per-token interpolation α_t between full and thin GEMM outputs in layers ≥ k. Chunk score |Σ| over 32 tokens; top-f chunks + 4 sinks + question rows run full. Baselines: a routed-point refined oracle, random chunks at equal f, all-thin.
- **`mixed.py`/`lat.py`**: mixed-width layer in the d1 lean2 fused runtime. Residual permuted once to [full | thin]; one grouped Triton GEMM per projection (full rows + thin first stage), then the thin second stage, with row gather/scatter, norm scales and SwiGLU fused. Matches `tw.py` on 12 real questions: TV ≤ .0016, 12/12 argmax [V].
- **`heal.py`**: train-split distillation of the thin factors only.

## 3. Step 1: rank spectrum [M]
Rank for 90/95/99% energy over state tokens (`spectra.json`):

| layers | normed residual (of 2048) | SwiGLU intermediate (of 6144) | MLP write to residual (data-aware) | its energy kept at r=512 |
|---|---|---|---|---|
| 4–12 | 400–750 / 750–1160 / 1500–1770 | 2200–3000 / 3200–4100 / 5000–5500 | 880–970 / 1180–1250 / 1660–1720 | .75–.79 |
| 13–17 | 500–755 / 880–1170 / 1590–1770 | 1300–2650 / 2200–3800 / 4100–5400 | 710–910 / 1020–1200 / 1560–1670 | .78–.85 |
| 18–23 | 110–410 / 330–760 / 1100–1510 | 2–860 / 2–1640 / 14–3490 | 2–500 / 2–810 / 7–1400 | .90–1.00 |

Mid-depth tokens do **not** share a low-rank subspace. Decision sensitivity of projecting one layer's state-token GEMM inputs (30 state-dependent questions, vs hobson): layers 0–11 TV .045–.31 / 7–53% flips at r=64 and .023–.069 / 0–20% at r=512; layers 13–23 sit at the floor (TV .004–.012, ≤ 1 flip in 30) even at r=64. Out-mode beats in-mode on TV for every cumulative (k, r) (e.g. .074 vs .095 at k=4, r=512), so step 2 uses out-mode.

## 4. Step 2: oracle routing, untrained, full suites [M]
n: REAL sd 346, LONG sd 165, CF 406 pairs (109 hobson-tracked), CF-probe 320 (105), JB-hard 130, REAL-label 400. FLOPs token-weighted, question rows at full width.

| config | FLOPs | REAL sd | LONG sd | CF fgh | CF-probe fgh | CF acc/flip | CF-probe acc/flip | JB-hard (lost/gained, p) | REAL-label |
|---|---|---|---|---|---|---|---|---|---|
| hobson | 1 | 1 | 1 | 1 | 1 | .594/.268 | .653/.328 | .523 | .785 |
| dense, this runtime | 1.00 | .991 | .988 | 1.000 | .971 | .595/.271 | .647/.322 | .531 (0/1) | .787 |
| oracle 30% @4 r256 | .55 | .942 | .927 | .936 | .857 | .611/.298 | .655/.334 | .469 (19/12, .28) | .772 |
| oracle 20% @8 r256 | .59 | .965 | .945 | .972 | .848 | .600/.281 | .648/.319 | .492 (12/8, .50) | .787 |
| random 20% @8 r256 | .59 | .936 | .891 | .982 | .724 | .672/.431 | .628/.281 | .515 (12/11) | .780 |
| oracle 10% @8 r512 | .63 | .971 | .945 | 1.000 | .867 | .607/.296 | .647/.312 | .515 (11/10) | .782 |
| all-thin @8 r512 | .58 | .951 | .921 | .982 | .800 | .638/.362 | .645/.306 | .500 (12/9, .66) | .777 |
| depth cut @11 r64 | .52 | .853 | .830 | .743 | .562 | .607/.313 | .609/.234 | .508 (10/8) | .790 |
| depth cut @12 r64 | .56 | .957 | .933 | .991 | .952 | .611/.298 | .662/.338 | .515 (6/5) | .780 |
| depth cut @13 r64 | .60 | .991 | .933 | 1.000 | .952 | .607/.288 | .652/.322 | .531 (2/3, 1.0) | .792 |
| depth cut @13 r128 | .61 | .997 | .945 | 1.000 | .971 | .602/.281 | .652/.322 | .523 (1/1) | .787 |

- The oracle does real work (vs random at equal FLOPs: CF-probe .848 vs .724, LONG .945 vs .891) and finds the probe record (in its top 10% for 89% of items), yet routing it full width does not save the decision.
- CF fgh is easy and misleading: thin paths over-flip (CF flip .458 vs hobson .268), the "drop the conservative context" effect, not capacity.
- **Stronger oracles do not rescue layers 4–12** (all 105 tracked CF-probe pairs): the routed-point refinement is worse or no better (.657/.752/.867 vs .867/.848/.867); 50% full at r=512 in layers 4–12 plus r=64 from 13 (0.67x) gives .876; thinning nothing before layer 13 gives .952. Dev subsets agree: any thinning in layers 4–12 loses CF-probe and JB-hard, routed or not.

**Kill or continue.** No configuration at ≤ 0.6x meets agree_sd ≥ .95 on REAL and LONG with CF-probe fgh ≥ .90. Token-adaptive width's best (oracle 20% @8 r256) reaches REAL .965 but LONG .945 and CF-probe .848. **Dead before any router is trained**, so the step-3 router was not built. The role-based depth cut @13 r64 passes REAL, CF, CF-probe and JB but misses LONG (.933 vs .988 floor).

**The step-3 heal, run anyway as a rescue test, does not transfer** [M]: thin from layer 10 at r=128, 1200 train-split items, 300 updates. Online (held-out-before-update) train KL fell .027 → .009; eval REAL sd .912 → .923, CF-probe fgh .638 → .638, TV unchanged.

## 5. Step 4: latency [M]
A10G, lean2 fused, 24 layers, CUDA graph, T state tokens + one 125-token question, fresh ids, 20 reps, median ms (p95 within 0.3 ms):

| config | T=256 | T=1000 | T=4000 |
|---|---|---|---|
| dense | 25.56 | 57.02 | 204.16 |
| depth cut @13 r64 | 20.46 (1.25x) | 40.66 (1.40x) | 135.67 (1.50x) |
| 30% full @4 r256 | 18.36 (1.39x) | 39.11 (1.46x) | 125.46 (1.63x) |
| all-thin @8 r512, r64 from 13 | 18.67 (1.37x) | 35.51 (1.61x) | 113.03 (1.81x) |

Realized speedup trails FLOPs (0.61x FLOPs buys 1.40x, not 1.64x): mixers and norms are unchanged and small-K thin GEMMs run below peak.

**Projections [S]** (GEMM share at 2x fp16-accumulate rate, the rest at 1.56x bandwidth, about 7.4 ms of non-GEMM time at 1000 tokens): RTX 3090 at T=1000, dense about 29.5 ms, depth cut about 21 ms, all-thin @8 r512/r64 about 19 ms; 4090 and 5090 roughly the same 1.4–1.6x. Per-question routing also forfeits prefix sharing: 4 questions at 1000 tokens ≈ 4 × 39 = 156 ms vs hobson's packed 83 ms [S].

## 6. Why it fails
The FLOPs a thin path could remove sit in layers 4–12, where state activations are high-rank and the MLP writes span about 1000+ dimensions. Projection errors there corrupt the K/V that question rows read, breaking amount comparisons and id/status bindings. Keeping the oracle-chosen "relevant" chunks full width does not help: the errors on the rest of the context still flip the decision. Late layers are nearly free for state rows (the known layers 13–23 result).

## 7. Next step
None for token-adaptive width. The salvage is a role-based late-layer cut (about 1.4x at 1000 tokens, question-agnostic, composes with W8A8): eviction-class, not transformative. The single test that settles it: a LONG-aware heal of the depth cut @13 r64 on long train-split states, passing only if LONG agree_sd ≥ .95 with paired tests; otherwise drop it.

**Caveats.** Configs were pre-selected on eval-split dev subsets (bias favours the idea, so the negative stands); the oracle is first-order; the heal was short. About 3.2 A10G-hours.

**Files**: `~/decider2/j13/` (`NOTES.md`, `code/`, `box/`, `*_score.json`).
