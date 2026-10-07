# J13 notes: token-adaptive width on full 24-layer hobson-v19 (oracle-first falsification)

## 2026-10-06 10:20
- Read BRIEF8, FAST_DECISION_MODEL.md, evalkit README, F7, lean.py/lean2.py, plib.py, kitrun.py.
- Box j13 launched (j13.txt present), not ready yet. Writing code meanwhile; no model runs on the laptop.
- Plan:
  1. fit.py: second moments of all 96 GEMM inputs over state tokens of 128 train-pool states; per-layer input spectra
     (normed residual = Win in, mixer out = Wo in, post-norm residual = Wgu in, SwiGLU intermediate = Wd in) and
     output-weighted spectra (W Sigma^1/2); bases for two thin-path modes:
       'in'  : y = (W P_r)(P_r^T x)   (input PCA, the literal "project activations onto top-r")
       'out' : y = U_r (U_r^T W x)    (data-aware SVD; MSE-optimal rank-r map on the data)
     thin GEMM cost r(K+N) vs KN: per layer rho(r) = 0.080 / 0.161 / 0.321 for r = 128/256/512.
  2. sens.py: per-layer projection sensitivity (24 layers x r in {64,128,256,512}) + cumulative all-thin from k=4/8.
  3. route.py: oracle = first-order effect on hobson's decision margin of thinning each token in layers >= k
     (one forward+backward per question, alpha-interpolation trick gives all (k, r) at once), aggregated |sum| per
     32-token chunk; top-f chunks full width; vs random chunks at same f, vs all-thin, (vs tail/recency).
  4. lat.py: mixed-width layer in lean2 fused runtime (sorted residual, Triton GEMMs with gather/scatter rows).

## 10:47
- Box j13 up since 10:14 but the launcher's setup was stuck: its first `box.sh run "nvidia-smi -L"` SSM session hung 20 min
  (and that check always fails on a fresh box because box.sh does `cd ~/work` first). Killed the hung ssh child and created
  ~/work via `box.sh put` of my code dir; the launcher loop then proceeds to its puts + setup_box.sh.
- Code written: tw.py (runtime+thin path+oracle), fit.py, sens.py, route.py, score.py, sscore.py, mixed.py + lat.py (latency).

## 11:09
- Box ready 10:55. Smoke: my plain-torch runtime vs hobson refs on 24 REAL questions: mean TV 0.0020, max 0.0074, 0 flips (measured).
- fit.py (128 train-split states, 274k state tokens, sinks excluded): spectra.json. Measured, per layer, rank for 90/95/99% energy:
  normed residual (Win in) 400-760 / 750-1170 / 1500-1770 of 2048 in layers 4-15; SwiGLU intermediate (Wd in) 2200-3000 /
  3200-4100 / 5000-5500 of 6144; data-aware output spectra: Wd output (the MLP write to the residual) 90% at 860-970, 99% at
  ~1670 in layers 4-15; only layers 18-23 are genuinely low rank (Wd out 90% at 2-500).
  => mid-depth state tokens do NOT share a low-rank subspace; r=512 keeps only ~0.76 of the MLP-update energy in layers 4-13.
- sens_cum (100 items: 70 REAL sd + 15 CF pairs), ALL state tokens thin in layers >= k (measured, vs hobson):
  thin@4r512 out: TV .074, 8% flips, CF pairs tracked 8/15; thin@8r512 out: TV .041, 2% flips, CF 7/15 (FLOPs .63 incl. q rows)
  'out' (data-aware SVD) beats 'in' (input PCA) on TV at every (k, r) -> use 'out' for step 2.
- dev route (preset B, 20 configs, 480 questions) launched 11:06, ~680 tok/s -> ~30 min.

## 12:01
- dev_B (480 q, 20 configs, out mode) done ~11:31; table in dev_B_score.json. Highlights (dev subsets, FLOPs token-weighted incl. question rows):
  orc30@4r256 .555: REAL sd .95, LONG .87, CF fgh .925, CF-probe fgh .886, JB-hard 12 lost/6 gained (partial)
  orc20@8r256 .593: .96 / 1.00 / .975 / .886;  thin@8r512 (no router) .591: .96 / .967 / .95 / .914
  random routing at the same f is far worse (rnd30@4r256: .85 / .83 / .65 / .63): the oracle is doing real work.
- Per-layer sensitivity (sens30: 22 REAL sd + 4 CF pairs; one layer's state-token GEMM inputs projected, all 4 GEMMs):
  layers 13-23 are at the TV floor (.004-.009) even at r=64; layers 0-11 are sensitive (layer 0 worst; attention layers 7, 11
  at r=64). Matches the old stale-KV sweep (stale@15 TV .006, stale@11 .058, stale@7 .24).
- full_D (2475 q: JB-all, CF-probe, CF, LONG sd, REAL sd, REAL-label extras; 6 configs) running ~1240 tok/s, ETA ~13:05.
  Partial (JB + half of CF-probe): runtime floor CF-probe fgh .971; best configs thin@8r512 / orc10@8r512 .882 (< .90 bar);
  JB-hard: thin@8r512 .500 (12 lost / 9 gained, p .66), orc30@4r256 .469 (19/12, p .28).
- The oracle does find the probe record: its chunk window is in the oracle's top 10% for 89% of CF-probe items (8,512),
  yet fgh is no better than all-thin -> the loss is not the record itself.
- dev_E (depth-scheduled ranks, r=64 from layer 13) OOMed when run concurrently (2 processes > 22 GB); rerun after full_D.

## 12:52
- full_D essentially complete (2260/2475; REAL-label extras finishing). Full-suite results (measured; FLOPs token-weighted, question rows full):
  | config | FLOPs | REAL sd | LONG sd | CF fgh | CF-probe fgh | CF acc | CF-probe acc | JB-hard (lost/gained) |
  | dense (my runtime = floor) | 1 | .991 | .988 | 1.000 | .971 | .595 | .647 | .531 (0/1) |
  | orc20@8r256 | .586 | .965 | .945 | .972 | .848 | .600 | .648 | .492 (12/8) |
  | orc10@8r512 | .625 | .971 | .945 | 1.000 | .867 | .607 | .647 | .515 (11/10) |
  | orc30@4r256 | .548 | .942 | .927 | .936 | .857 | .611 | .655 | .469 (19/12) |
  | thin@8r512 (no router) | .584 | .951 | .921 | .982 | .800 | .638 | .645 | .500 (12/9) |
  | thin@8r256 | .485 | .925 | .879 | .991 | .714 | .685 | .631 | .500 (14/11) |
  | rnd20@8r256 | .588 | .936 | .891 | .982 | .724 | .672 | .628 | .515 (12/11) |
  => KILL criterion met: no config <= 0.6x reaches LONG sd .95 or CF-probe fgh .90 (best .945 / .867, the latter even at .625x).
  CF fgh is easy (thin paths over-flip: CF flip .46 vs .27 for hobson = the qattn-style "drop the conservative context" effect).
  CF-probe losses concentrate in amount_vs_limit and *_distract kinds (exact values, bindings).
- Queue on box: prT (all 105 hobson-tracked CF-probe pairs: refined oracle 'or2' + depth schedules), dev_E (schedules), lat check, lat bench.

## 13:35
- prT (all 105 hobson-tracked CF-probe pairs; floor .971):
  thin@13r64 (.582x) .952 | orc30@4r256 .867 | orc20@8r256 .848 | orc10@8r512 .867 | orc50@4r512+13r64 (.671x) .876 |
  thin@8r512+13r64 .810.  Refined oracle (one step at the routed point, d KL/d alpha): WORSE (or230@4r256 .657, or220@8r256 .752,
  or210@8r512 .867).  => even a 50%-full oracle routing at r=512 in layers 4-12 caps CF-probe fgh at ~.87.
- dev_E (91 REAL sd, 54 CF pairs, 46 CF-probe pairs, JB-hard): thin@13r64 1.000 / 1.000 / .969 / JB 2 lost 3 gained = floor;
  every config that thins state rows anywhere in layers 4-12 loses CF-probe and JB-hard, oracle-routed or not.
- lat check: mixed-width runtime (grouped Triton GEMMs, gather/scatter rows) vs tw.py thin reference: TV <= .0016 on 12 real
  questions (dense lean2 vs tw.py: <= .0055), same argmax 12/12.  (identity-perm check was garbage: tuning pass ran inside it.)
- Running: full_F (thin@13r64, 12r64, 13r128, 11r64 on all 2475 q, no oracle needed), heal (k=10, r=128, 1200 train items),
  then healE evals, then lat bench (queue4, exclusive GPU).
- Accidentally killed queue2.sh (pkill pattern matched my own shell); full_F kept running, heal started early, concurrently.

## 14:15
- full_F (depth cuts, all 2475 q, measured): @13r64 .60x: REAL .991 LONG .933 CF 1.0 CF-probe .952 JB 2/3 REAL-label .792;
  @13r128 .61x: .997/.945/1.0/.971; @12r64 .56x: .957/.933/.991/.952; @11r64 .52x: collapse (.853/.830/.743/.562).
  LONG sd never reaches .95 at <= .6x for any config (floor .988).
- Heal (k=10, r=128, 1200 train items, 300 updates): online train KL .027 -> .009, but eval unchanged
  (REAL sd .912 -> .923, CF-probe fgh .638 -> .638, TV .061 -> .063). Does not transfer.
- Latency (A10G, lean2 fused + mixed-width grouped GEMMs, T state + 125 q rows, 20 reps): dense 25.6 / 57.0 / 204.2 ms at
  256/1000/4000; depth cut @13r64 20.5 / 40.7 / 135.7 (1.25/1.40/1.50x); 30% @4r256 18.4/39.1/125.5; all-thin @8r512+13r64
  18.7/35.5/113.0 (1.37/1.61/1.81x).
- VERDICT: token-adaptive width dead by its oracle-first criterion (CF-probe fgh <= .867, LONG <= .945 at <= .6x).
  DRAFT_REPORT.md written. GPU work finished (~3.2 A10G-hours). Box left idle (auto-terminates).
