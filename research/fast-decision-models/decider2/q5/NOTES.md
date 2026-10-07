# Q5 notes (box q5): structure accuracy. B11 (decision-aware 2:4), B12 (per-deployment dead MLP neurons), B13 exactness, B7 (per-deployment calibration)

## Summary (kept at the top; updated as results land)
- (nothing measured yet)

## Log

### 2026-10-06 21:14 PDT start
- Read BRIEF10 in full (incl. addenda 1-3; addendum 3 gives Q5 = B11, B12, B13 checks, B7), IDEAS_EXPLORED.md, FAST_DECISION_MODEL.md sec 5.
- Read Q1 NOTES, q1lib/q1fmt/q1spec/q1cal/q1hd/q1b13/q1eval/q1score, H1 h1lib/h1score/NOTES, evalkit README.
- Box q5 not ready (boxes/q5.ready absent; retry_q45.log empty at 21:13). Writing code meanwhile; will poll every ~5 min.

### 21:35 PDT code written (laptop copies in ~/decider2/q5/code); box q5 launched 21:33 (g5.8xlarge, TTL ~07:33), setup pending
- q5lib.py: S5 = structure installed on Q1's model via g.qfn (Q1/H1 files imported, never edited):
  - 2:4 along K in the kernel basis (bf16 natural basis, bf16 rotated basis, int8 / int4 rotated with H1 FORMAT v1 arithmetic); row role 'all' or
    's' (state rows sparse, question rows dense). Selection: mag0 (magnitude), wanda, sgpt (SparseGPT with the layer Hessian), sgptd (SparseGPT with
    the decision-weighted Hessian H_d = sum_t w_t x_t x_t^T, w_t = Fisher-weighted ||dKL/dy_t||^2), fisher (decision Fisher diagonal of the weights,
    E[Gamma^2] with Gamma = per-request weight gradient of the decision, times OBS compensability 1/(H_kk d_k^2), H_d updates). int8/int4 codes by a
    second SparseGPT pass with the mask fixed and joint quantization.
  - neuron removal: dead (max|m| < tol), var (Var(m_j) ||Wd_j||^2), dsal (decision saliency of mean replacement, sum lam (sum_t g_tj (m_tj - mu_j))^2),
    obs/obsd (greedy structured OBS on Wd with plain / decision-weighted covariance and output metric G_out), mean-replacement bias folded exactly.
- q5cal.py: one decision-gradient pass per domain (bank / ret train-split, calibration tau tasks) accumulating per GEMM Hs,Hq,Hds,Hdq, Fisher diagonal
  (kernel basis Fr, natural Fn), G_out (Wo, Wd), and neuron statistics for all 24 layers (+ per question name). GEN = bank + ret.
- q5dev.py: screening on train-split DEV requests (160 banking + 96 retail, tau tasks disjoint from calibration): KL/TV/flips vs in-runtime bf16.
- q5kit.py: all 3227 questions (or banking subset), resumable preds. q5score.py (laptop): kit + banking-subset scores vs the bar.
- q5b13.py: B13 exactness (layer-23 state rows K/V only, with NaN for everything else they would compute; layer-0 Win per-token table) + vocab coverage.

### 22:07 PDT box q5 ready 21:56 (A10G 23 GB, 32 vCPU, 124 GB RAM, torch 2.14.1+cu130, nvcc 12.8); TTL ~07:33 PDT
- Put on box: ~/work/q1 (q1lib, q1spec, q1fmt, cfgs.json: copies), ~/work/h1 (H1 code), ~/work/g2 (g2lib, subsets), ~/work/q5 (mine).
- H1 GPTQ Hessians (h1calib --n 64, 145k tokens, train split): done in ~2.5 min.
- Smoke test (4-request calibration, layer 22 only; indicative, not results): all S5 paths run; 2:4 masks verified (zero share 0.500 in bf16,
  0.5000-0.5046 in int8/int4 codes, extra zeros = rounding to 0). Peak GPU 10.7 GB at T=1749 for a 1-layer calibration pass.
- Ops lesson: box.sh run with '&' hangs the ssh unless 'setsid nohup ... < /dev/null > log 2>&1 &'. pgrep/pkill -f match their own ssh command line.
- Main queue started 22:06: q5cal bank/ret for layers 12-15 (+ neuron stats all 24 layers), 16-19, 20-23; q5nstat; dev screen set 1; then 8-11, 4-7, 0-3.

### 22:22 PDT B12 neuron statistics (MEASURED, q5cal --neur: 160 banking + 160 retail train-split cal requests; res/res_nstat.json)
- Dead neurons: 0 of 147,456 SwiGLU neurons (24 x 6144) have max |m| < 0.01 on either domain; every neuron reaches |m| >= 0.074 somewhere;
  only 3 (bank) / 1 (ret) in layer 23 stay below 0.1. => exact dead-neuron removal saves nothing on this model.
- Sparse firing exists only at layer 0-1: 5156 (bank) / 5367 (ret) of layer 0's neurons exceed |m| = 0.1 on < 1% of rows (median |m| 0.013).
- Decision saliency (Fisher-weighted first-order effect of replacing a neuron by its mean): layer shares (gen) layers 0-11: 92%, 12-22: 7%, 23: 1.3%.
  Within a layer, neurons holding 90% of the layer's saliency: layers 0-9: 3500-4300 of 6144; 10-12: 1400-3300; 13-21: 14-1000; 22: 48-59;
  23: 35-49. 99%: 0-9 ~5700; 13-21 1600-4300; 22 700-1000; 23 200-270.
- Banking vs retail: Spearman of per-neuron saliency 0.52-0.62 (layers 0-11), 0.64-0.91 (12-23). Banking's least-salient half carries
  1-2.5% of banking saliency and 0.7-2.7% of retail saliency at layers 13-19; at layers 20-21 retail's least-salient half carries 10-20% of
  banking saliency (domain-specific there).
- Per question (banking): one question's 99%-saliency set is 5000-5700 neurons at layers 2-10 (no per-question savings early), 400-1400
  at layer 22.
- Published placeholder best_structure.json (format + these facts).

### 22:50 PDT B11 dev screen 1 (MEASURED, 256 train-split DEV requests: 160 banking + 96 retail, tau tasks disjoint from calibration;
### metrics vs the in-runtime bf16; 2:4 on all 4 GEMMs of layers 12-22 = 46.2% of GEMM MACs; calibration 'gen' = 160 bank + 160 ret requests)
| selection (bf16 2:4) | rows | basis | bank KL | bank flips | ret KL | ret flips | TV (all) |
| magnitude | all | natural | 4.5e-2 | 11/160 | 2.1e-2 | 1/96 | .090 |
| Wanda | all | natural | 2.5e-2 | 10/160 | 1.3e-2 | 5/96 | .065 |
| SparseGPT, layer Hessian | all | natural | 2.3e-3 | 2/160 | 2.9e-3 | 3/96 | .023 |
| SparseGPT, decision-weighted Hessian | all | natural | 9.6e-4 | 3/160 | 7.7e-4 | 0/96 | .013 |
| decision Fisher diag x OBS compensability | all | natural | 9.8e-4 | 1/160 | 7.0e-4 | 0/96 | .013 |
| SparseGPT, layer Hessian | all | rotated | 5.0e-3 | 10/160 | 2.2e-3 | 2/96 | .030 |
| SparseGPT, decision-weighted | all | rotated | 1.2e-3 | 3/160 | 6.4e-4 | 1/96 | .014 |
| decision Fisher | all | rotated | 1.7e-3 | 5/160 | 1.0e-3 | 0/96 | .016 |
| SparseGPT, decision-weighted (state-row Hessian) | STATE rows only | natural | 1.0e-4 | 1/160 | 1.6e-4 | 0/96 | .0049 |
- Decision weighting cuts KL 2.4x (bank) / 3.7x (ret) vs per-layer SparseGPT; 20-50x vs magnitude. The Hadamard-rotated basis (needed by
  int8/int4 activations) costs 1.3-2x with decision weighting, 2x without.
- Row role is the big lever: state rows sparse + question rows dense = 10x lower KL than all rows sparse.

### 23:12 PDT reference points on the same dev set (MEASURED) and the full-kit dense run
- b8 (W8A8-GPTQ, 8 GEMMs bf16; passes the bar on the kit per H2/J15): bank KL 9.5e-5, 1/160 flips; ret 9.2e-5, 0/96; TV .0040.
- w4q8 (state rows W4A4, question rows W8A8; fails per Q1): bank KL 3.7e-3, 6/160; ret 6.0e-3, 5/96; TV .027.
  => my dev KL scale: ~1e-4 is W8A8-level, ~4e-3 fails clearly.
- More dev rows: SparseGPT layer Hessian, state rows only: bank 1.5e-4 / ret 2.3e-4 / TV .0058 (decision-weighted: 1.0e-4 / 1.6e-4 / .0049).
  MLP only (Wgu+Wd, 30.2% of MACs), all rows, decision-weighted: 2.9e-4 / 2.3e-4 / .0070. Mixer only (Win+Wo, 15.9%): 6.3e-4 / 5.5e-4 / .011.
- b8 + int8 2:4 natural basis (all rows, L12-22, decision-weighted): smoothing 0: bank 1.26e-3 / ret 9.1e-4; smoothing 0.5: 1.22e-3 / 6.4e-4.
- Full kit, in-runtime bf16 ('dense', MEASURED, 3227 q, 23 min): REAL 5/1083 = 0.46% flips vs hobson, TV .0031, CF retention 1.000,
  CF-probe .971, JB-hard .531, REAL-label .785, LONG 2/471 (matches H1's dense exactly).
- Banking subset caveat (MEASURED from refs): hobson tracks 1 of 161 banking CF pairs (53/125 retail, 55/120 airline), so banking CF retention
  is not measurable; banking uses REAL (883 q), LONG (471 q), CF-probe (36 tracked pairs), REAL-label.
- Ops: at 23:01 two jobs (b8 kit 9.8 GB + w4q8 dev reference 12.5 GB) touched 22.3 GB total for a moment (one allocation retried). From now on
  every Q5 process caps itself at 0.44 of the GPU (q5lib reads ~/work/q5/memfrac; env Q5_MEMFRAC overrides, e.g. 0.88 for calibration alone).

### 23:45 PDT B11 int8 / int4 2:4 over b8 (MEASURED, dev; L12-22 = 45.9% of GEMM MACs, b8's bf16 GEMM 12.Wo left dense)
| config | rows sparse | bank KL | bank flips | ret KL | ret flips | TV | vs b8 alone: bank KL |
| b8 + int8 2:4 natural basis, smoothing 0.5, decision-weighted | state | 1.6e-4 | 1/160 | 3.2e-4 | 0/96 | .0061 | 9.4e-5 |
| b8 + int8 2:4 rotated basis, decision-weighted | state | 2.8e-4 | 4/160 | 2.3e-4 | 0/96 | .0071 | 2.0e-4 |
| b8 + int4 2:4 rotated basis, decision-weighted | state | 3.9e-4 | 2/160 | 2.8e-4 | 0/96 | .0081 | 3.0e-4 |
| b8 + int8 2:4 natural, smoothing 0 / 0.5, decision-weighted | all | 1.26e-3 / 1.22e-3 | 2 / 2 | 9.1e-4 / 6.4e-4 | 0 / 0 | .015 / .014 | |
| b8 + int8 2:4 rotated, decision-weighted | all | 1.13e-3 | 3/160 | 7.4e-4 | 0/96 | .014 | |
| b8 + int8 2:4 natural 0.5, decision Fisher | all | 1.16e-3 | 2/160 | 9.6e-4 | 0/96 | .014 | |
| b8 + int8 2:4 natural 0.5, layer-Hessian SparseGPT | all | 2.7e-3 | 3/160 | 1.7e-3 | 0/96 | .021 | |
| b8 + int8 2:4 natural 0.5, magnitude | all | 2.8e-2 | 5/160 | 1.5e-2 | 5/96 | .068 | |
- Decision weighting halves the KL in int8 too (2.7e-3 -> 1.2e-3). Fisher-diagonal and decision-weighted Hessian are equal within noise.
- Arithmetic (speed, not measured): state rows int4 2:4 on L12-22 at 4x the int8 dense rate, rest W8A8-b8, 1000-token state + 110 q rows
  (state 90% of rows) -> GEMM time ~0.70x of b8 if the sparse kernel reaches its nominal rate; int8 2:4 there -> ~0.79x.
- Ops: dev screening now reuses cached prefix residuals (layers 8-23, CPU RAM) so a config starting at layer L recomputes only L..23;
  kit runs for L12-22 configs use an on-disk layer-12 cache per base format (--start 12, checked against full forwards on 20 questions).

### 23:58 PDT full kit: b8 in my emulation (MEASURED, 3227 q, 50 min while sharing the GPU)
- b8 (H1 GPTQ calibration recomputed on q5): REAL 5/1083 = 0.46% vs hobson (paired vs in-runtime bf16: 2 lost / 2 gained, p 1.0), TV .0047
  (brief: .0048), CF retention .982 (107/109; bar .99), CF-probe .962 (bar .95), JB-hard .538 (McNemar p .5), REAL-label .788, LONG 3/471.
  => b8 itself misses the CF bar by one pair in this draw (H1's draw had 1.000). The 2 lost CF pairs and the 4 lost CF-probe pairs are all
  threshold cases: the in-runtime bf16 P(expected) of the failing item is 0.503-0.523; b8 moves it 0.01-0.03 across 0.5 (q5cfloss.py).
  CF retention at .99 with 109 tracked pairs allows 1 loss, so it is decided by a handful of items sitting within 0.02 of 0.5.
- Cached-prefix kit path verified: max |d logit| = 0.0 on 20 questions vs the full forward.
- b8 + int4 2:4 ALL rows (L12-22): dev bank KL 2.9e-3, 9/160 flips -> fails clearly.

### 00:10 PDT per-layer 2:4 scan (MEASURED, dev; bf16 natural basis, decision-weighted SparseGPT; one layer at a time; bank KL)
| layer | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| all rows | 1.5e-3 | 1.7e-3 | 1.7e-3 | 2.0e-3 | 7.3e-4 | 4.4e-4 | 7.7e-5 | 4.0e-5 | 1.9e-5 | 8.8e-6 | 9.1e-6 | 8.2e-6 | 5.9e-6 | 6.2e-6 | 7.3e-6 | 1.0e-5 |
| state rows only | 5.1e-4 | 5.7e-4 | 4.4e-4 | 4.3e-4 | 6.7e-5 | 3.1e-5 | 9.2e-6 | 7.6e-6 | 5.3e-6 | 5.1e-6 | 4.7e-6 | 4.7e-6 | 4.0e-6 | 3.6e-6 | 4.2e-6 | 3.0e-6 |
- Singles add up roughly to the joint (sum of all-row singles 12-22 = 1.35e-3 vs joint 9.6e-4). Layers 8-11 are out even for state rows.
- Hybrid configs built from this: all rows sparse on 16/17-23 (each ~1e-5), state rows only on 12/13-15/16.
- Q4 (their NOTES, MEASURED kernels): int8 2:4 state rows L12-22 GEMM time 0.924x of b8 at 1000+125 rows, int4 2:4 state rows L12-22
  0.828x; int4 sparse on Ampere is PAIR-wise (keep 2 of 4 adjacent nibble pairs, 4:8 with pairs) -> my element-wise int4 2:4 is not
  hardware-valid. Added prec 'int4p' (pair-wise mask; pair saliency = sum of its two OBS saliencies). The queued element-wise int4 kit run
  was replaced.
### 00:10 PDT B12 dev (MEASURED; removal of 25% of neurons in each of layers 12-23 = 8.2% of GEMM MACs)
| ranking | compensation | bank KL | bank flips | ret KL | ret flips | TV |
| variance x ||Wd_j||^2 | mean bias only | 1.8e-3 | 5/160 | 3.7e-4 | 0/96 | .015 |
| decision saliency (mean replacement) | mean bias only | 3.7e-4 | 3/160 | 1.1e-4 | 0/96 | .0062 |
| decision saliency | decision-weighted least-squares update of the kept Wd columns + bias | 4.2e-5 | 1/160 | 3.4e-5 | 0/96 | .0027 |
| dead (max|m| < 0.1): removes 1-3 neurons | - | 0 | 0 | 0 | 0 | 0 |

### 00:21 PDT FIRST FULL-KIT PASS: bf16 2:4 on state rows of layers 12-22 (MEASURED, 3227 q; best_structure.json updated)
- s24.L12-22.sgptd.bf16n.s (46.2% of every state row's GEMM MACs sparse; decision-weighted SparseGPT, gen calibration):
  REAL 7/1083 = 0.65% (bar 0.70%), paired vs bf16 6 lost / 4 gained p .754, TV .0062, CF 1.000, CF-probe .952 (bar .95), JB-hard .523
  (p 1.0), REAL-label .790, LONG 4/471. Banking: REAL 7/883, LONG 4/471, CF-probe .972 (36), REAL-label .749. PASSES, thin margins.
  Lost CF-probe pairs are again threshold items (in-runtime bf16 P(expected) 0.491-0.520).
- B12 dev, 50% of neurons in each of layers 12-23 (16.5% of GEMM MACs): decision saliency + decision-weighted LS compensation bank 1.3e-4 /
  ret 1.0e-4 / TV .0045; decision-weighted greedy OBS 1.7e-4 / 8.7e-5 / .0048; plain greedy OBS (layer covariance) 1.0e-3 / 2.4e-4 / .011;
  decision saliency without compensation 1.7e-3; variance ranking 5.5e-3. Global 5% of all neurons by decision saliency (lands in 15-23):
  9.8e-6.
- ARITHMETIC (q5speed.py, nominal Ampere rates; Q4's measured kernels run 0.1-0.12 slower than nominal for 2:4): b8 + int4 2:4 state rows
  L12-22 0.70x of b8 GEMM time (Q4 measured 0.83x); + B13 0.65x; b8 + B12 50% neurons L12-23 + B13 0.80x; B12 75% 0.72x.
  => with layers 0-11 too sensitive for structure, the GEMM-time bar (0.65x) is not reachable by structure on layers 12-23 alone.

### 00:55 PDT SECOND FULL-KIT PASS, deployable: b8 + int8 2:4 state rows L12-22 (MEASURED, 3227 q)
- b8+s24.L12-22.sgptd.int8n0.5.s: REAL 6/1083 = 0.55%, paired vs bf16 5/4 p 1.0, TV .0069, CF .991 (108/109), CF-probe .952 (100/105),
  JB-hard .531 p 1.0, REAL-label .7925, LONG 4/471. Paired vs b8 itself: REAL 5/4, LONG 2/1. Banking: REAL 6/883, CF-probe .944 (36).
  Lost CF/CF-probe pairs: all threshold items (bf16 P(expected) 0.491-0.523). best_structure.json updated (arrays on box, keys listed).
- Dev, more (MEASURED): int4 2:4 pair-wise (Ampere) = element-wise within noise (state L12-22: 3.9e-4 both); state rows L12-23 int4p
  3.9e-4; L13-23 int4p 1.8e-4 / ret 1.5e-4 / TV .0054 (layer 12 = half the int4 error); int8n0.5 state L13-23 1.3e-4 / 1.8e-4 / .0050.
  Putting QUESTION rows through int4 2:4 fails (hybrids h1/h3/h4 with int4 all rows on L16/17-23: bank KL 1.0-1.5e-3, 4/160 flips).
  int8 2:4 all rows on L16/17-23 + state rows L12-15/16: 2.0-2.3e-4 (vs 1.6e-4 state-only L12-22). Fisher criterion for int4p state rows
  4.6e-4 vs decision-weighted Hessian 3.9e-4.
- B12 global allocation by decision saliency, no compensation: G12-23 20% of their neurons 2.8e-5, 30% 1.05e-4; G0-23 20% 2.4e-4 (bank),
  retail KL 10-20x lower than banking for the same removal.

### 01:10 PDT B12 with GLOBAL allocation + compensation (MEASURED, dev) -- the strongest structural result so far
| config (decision saliency + decision-weighted LS compensation) | neurons removed | GEMM MACs removed | bank KL | ret KL | TV | flips |
| uniform 50% per layer, L12-23 | 36,864 | 16.5% | 1.3e-4 | 1.0e-4 | .0045 | 2/256 |
| uniform 75% per layer, L12-23 | 55,296 | 24.7% | 3.7e-4 | 2.5e-4 | .0078 | 1/256 |
| GLOBAL ranking across L12-23, 50% of their neurons | 36,864 | 16.5% | 1.4e-5 | 8.2e-6 | .0015 | 1/256 |
| GLOBAL, 60% | 44,237 | 19.8% | 2.5e-5 | 1.4e-5 | .0019 | 0/256 |
| GLOBAL, 75% | 55,296 | 24.7% | 5.4e-5 | 2.3e-5 | .0027 | 0/256 |
| uniform 25% per layer, L8-23 | 24,576 | 11.0% | 7.3e-4 | 5.0e-4 | .011 | 4/256 |
| GLOBAL across L8-23, 50% / 60% | 49,152 / 58,982 | 22.0% / 26.4% | 4.6e-5 / 1.3e-4 | 1.6e-5 / 3.4e-5 | .0024 / .0038 | 0 / 0 |
- Global 75% of L12-23 keeps per layer: L12 6136, L13 5125, L14 2889, L15 1613, L16 644, L17-22 182-344, L23 587 neurons: the late-layer
  MLPs are almost entirely replaced by a bias + a least-squares-updated remainder (cf. R1: hobson's last 8 layers barely change decisions).
  Global across L8-23 removes nothing from layers 8-12.
- ARITHMETIC (nominal rates): b8 + global 75% L12-23 = 0.75x of b8 GEMM time (dense GEMMs, just narrower); + B13 0.73x; + int4 2:4 on the
  state rows of the mixers (Win, Wo) of L13-23 0.63x (meets the 0.65x bar nominally; Q4's measured 2:4 kernels run ~0.1 slower).
- B7 first look (dev): banking-only calibration (160 req) of uniform 50% L12-23 is WORSE than gen (320 req = same 160 banking + 160 retail)
  even on banking dev (1.9e-4 vs 1.3e-4; retail 3.9e-4 vs 1.0e-4) -> size-limited calibration, not deployment mismatch. Queued equal-size
  retail-only (160 req) versions to separate the two effects.

### 01:22 PDT full kit: b8 + int4 2:4 pair-wise state rows L13-23 = NEAR MISS (MEASURED)
- REAL 8/1083 = 0.74% (bar .70%), paired vs bf16 4/1 p .375, TV .0066, CF .972 (106/109), CF-probe .962, JB-hard .538 (p .5), REAL-label
  .7875, LONG 4/471; banking REAL 8/883, CF-probe .917 (33/36). Lost CF pairs: threshold items (bf16 P 0.510-0.517).
- B7 precision (dev, MEASURED): b8 with GPTQ calibrated on 64 BANKING states: bank KL 1.2e-4, ret 7.8e-5, TV .0044 vs b8 with H1's pool
  sample (64 states, ~77% banking): 9.5e-5 / 9.2e-5 / .0040 -> no gain on banking; difference within GPTQ draw noise.
- b8 + uniform B12 50% L12-23: dev 2.0e-4 / 2.0e-4 (b8 alone 9.5e-5); 75% uniform: 3.9e-4.

### 01:40 PDT more kit + dev (MEASURED)
- KIT uniform B12 50% L12-23 (bf16, 16.5% of MACs removed): REAL 8/1083 = 0.74%, paired vs bf16 7/4 p .55, TV .0076, CF .991, CF-probe .943
  (bar .95), JB-hard .546 (p .25), REAL-label .790 -> narrow miss (dev KL 1.3e-4). The global allocation (dev 10x lower) is queued.
- DEV stack: b8 + global 75% L12-23 neurons + int4 2:4 pair-wise on STATE rows of the mixers (Win, Wo) of L13-23: bank 2.1e-4, ret 1.5e-4,
  TV .0058, 1/256 (b8 alone 9.5e-5; the int8 config that passes the kit had 1.6e-4 / 3.2e-4 / .0061). Same stack with int8n mixers 1.65e-4 /
  1.7e-4 / .0054; with bf16 2:4 mixers (no quantization) 9.2e-5 / 6.1e-5 / .0038. With 85% neurons + int4p mixers: 4.1e-4 / 1.9e-4.
  ARITHMETIC 0.63x of b8 GEMM time (nominal). Kit run queued (qb).
- DEV global B12 sweep, L12-23: 50% 1.4e-5, 60% 2.5e-5, 75% 5.4e-5, 85% 2.5e-4, 90% 9.5e-4 (bank KL); global over L8-23 at 75% of their
  neurons 2.2e-3 (it starts removing from 11-12). => ~75% of the late MLP neurons (~25% of all GEMM MACs) is the knee.
- B7 (dev): global 75% calibrated on banking only: bank 6.1e-5 / ret 6.8e-5 vs gen 5.4e-5 / 2.3e-5; global 50% bank-only 1.6e-5 / 1.1e-5 vs
  gen 1.4e-5 / 8.2e-6. b8's GPTQ calibrated on 64 retail states: bank 1.4e-4 / ret 6.9e-5 (bank-cal 1.2e-4 / 7.8e-5; pool 9.5e-5 / 9.2e-5).
  => per-deployment calibration buys nothing measurable on its own deployment here; mismatched calibration costs up to 3x on the other one.
- Ops: w4q8.fbank/fret dev references hit the 0.42 cap (two GPTQ code copies); rerun at the end with a higher cap if time permits.

### 01:50 PDT B7 for B11 on the full kit (MEASURED)
- s24.L12-22.sgptd.bf16n.s calibrated on banking only (160 req) vs generic (160 banking + 160 retail):
  kit REAL 7/1083 vs 7/1083; paired vs bf16 4/2 (p .69) vs 6/4; TV .0059 vs .0062; CF 1.000 vs 1.000; CF-probe .971 vs .952;
  JB-hard .523 vs .523; REAL-label .7875 vs .790; LONG 5/471 vs 4/471.
  banking subset: REAL 7/883 vs 7/883; LONG 5 vs 4; CF-probe 36/36 vs 35/36; REAL-label .746 vs .749.
  => equal within 1-2 items; the banking-only masks cost nothing measurable on retail/airline/JevBench either.
- Kit for the stack (b8 + global 75% neurons + int4p state-row mixers L13-23) running; bf16 global 75% B12 next in qa.

### 02:02 PDT KIT: B12 global 75% of L12-23 neurons, bf16 (MEASURED; 55,296 neurons, 24.7% of GEMM MACs removed)
- REAL 8/1083 = 0.74% (bar: <= 7), paired vs bf16 3 lost / 0 gained p .25, TV .0052, CF 1.000, CF-probe .981, JB-hard .546 (p .25),
  REAL-label .785, LONG 5/471; banking REAL 8/883, CF-probe 36/36. Misses only REAL flips, by one question.
- The 3 lost REAL questions (vs bf16) are threshold items: hobson P(its answer) 0.502 / 0.496 (binary) or a top-2 near-tie (0.328);
  the bf16 runtime is at 0.500 / 0.485 already; the config moves them 0.003-0.004. Same pattern for the int4p near miss (0.503-0.508).
- Queued kit: global 60% (dev KL 2.5e-5, 19.8% of MACs), bf16 and over b8.

### 02:14 PDT KIT: full stack fails (MEASURED)
- b8 + global 75% L12-23 neurons + int4 2:4 pair-wise on state rows of Win/Wo L13-23: REAL 10/1083 = 0.92%, paired vs bf16 8/3 p .23,
  TV .0075, CF .982, CF-probe .962, JB-hard .538, REAL-label .7875. Errors add: dev KL 2.1e-4 vs 1.6e-4 for the passing int8 config.
- Near the bar the outcome is decided by threshold items (TV .0052 failed by 1 REAL flip, TV .0069 passed): every lost REAL question and
  CF pair in every config has bf16 P(answer) within ~0.02 of the decision boundary.
- best_structure.json updated (accurate settings, near misses, failed, pending).

### 02:20 PDT B7 at equal calibration size (MEASURED, dev; 160 train-split requests per calibration set; bank KL / ret KL)
| structure | banking-calibrated | retail-calibrated | generic (both, 320) |
| B12 global 75% L12-23 | 6.1e-5 / 6.8e-5 | 1.7e-3 / 7.5e-5 | 5.4e-5 / 2.3e-5 |
| B12 uniform 50% L12-23 | 1.9e-4 / 3.9e-4 | 2.2e-3 / 7.7e-5 | 1.3e-4 / 1.0e-4 |
| B11 bf16 2:4 state rows L12-22 | (kit: equal to generic) | 1.6e-4 / 1.6e-4 | 1.0e-4 / 1.6e-4 |
- Neuron removal calibrated on the WRONG deployment (retail) damages banking decisions 27x (global 75%) to 12x (uniform 50%); banking-
  calibrated removal transfers to retail at no cost (banking traffic covers retail's patterns, not the reverse). Restricting calibration
  to the deployment buys nothing over generic traffic that contains it. 2:4 masks are much less calibration-sensitive (1.6x).

### 02:29 PDT KIT: b8 + B12 global 75% (MEASURED)
- REAL 10/1083 = 0.92%, paired vs bf16 8/3 p .23, TV .0064, CF .991, CF-probe .943, JB-hard .538 p .5, REAL-label .7875 -> fails (b8 and
  B12 errors add; bf16 B12 alone had 8 flips, TV .0052). b8 + global 60% (19.8% of MACs) queued.
- B7 dev (equal size): b8 + int8 2:4 state L12-22 retail-calibrated: bank 2.6e-4 / ret 3.0e-4 (generic 1.6e-4 / 3.2e-4); int4p L13-23
  retail-calibrated 2.4e-4 / 1.6e-4 (generic 1.8e-4 / 1.5e-4).
- B13 dense on all 3227 started 09:27 UTC (base + layer-23 rerun + full table0 forward per question).

### 02:40 PDT KIT PASS with margin: B12 global 60% of L12-23 neurons, bf16 (MEASURED; 44,237 neurons, 19.8% of GEMM MACs)
- REAL 6/1083 = 0.55%, paired vs bf16 1 lost / 0 gained (p 1.0), TV .0039 (bf16 floor .0031; b8 .0047), CF 1.000, CF-probe .971 (= bf16),
  JB-hard .546 (p .25), REAL-label .785, LONG 3/471; banking REAL 6/883, LONG 3/471, CF-probe 35/36, REAL-label .743.
  Kept neurons per layer: 12: 6144, 13: 6124, 14: 5324, 15: 3981, 16: 2280, 17: 1229, 18: 1066, 19: 737, 20: 501, 21: 488, 22: 698, 23: 919.
- B13 vocab (MEASURED): banking train traffic 31.9M tokens, 13,376 distinct ids; they cover 99.75% of banking eval tokens (6.79M) and
  93.4% of distinct eval ids -> per-deployment layer-0 table 13,376 x 8224 = 220 MB bf16 (110 MB int8); misses fall back to the GEMM.

### 02:53 PDT timestamp correction
- Headers from "00:52" to "04:55" were written 0.5-2.3 h ahead of the clock (I mis-converted the box's UTC clock). Corrected above from the
  box queue logs (UTC - 7 h): e.g. the bf16 2:4 kit finished 07:19 UTC = 00:19 PDT; the B12 60% kit 09:38 UTC = 02:38 PDT. Box q5 booted
  04:33 UTC, TTL ends 14:33 UTC = 07:33 PDT. best_structure.json "updated" fields corrected the same way.

### 03:17 PDT KIT PASS, deployable B12: b8 + global 60% of L12-23 neurons (MEASURED; 19.8% of GEMM MACs removed)
- REAL 3/1083 = 0.28%, paired vs bf16 1 lost / 3 gained (p .625), TV .0054, CF 1.000, CF-probe .952, JB-hard .538 (p .5), REAL-label .7875,
  LONG 3/471; banking REAL 3/883, LONG 3/471, CF-probe 33/36, REAL-label .746. best_structure.json: listed first, with kept neurons per layer.
- Late queue (box): second disjoint calibration set (160 banking + 160 retail) for layers 12-23, then dev of 75% with 2x calibration.

### 03:25 PDT B13 exactness, bf16 runtime, all 3227 questions (MEASURED / VERIFIED)
- skip23 (layer 23: state rows compute only K/V by a separate GEMM over Win rows 4096:5120; their q/gate columns and Wo/Wgu/Wd outputs set
  to NaN): 0 of 3227 logit vectors contain NaN -> those values are never read [V]. Numerically vs the unmodified run: max |d logit| 0.031,
  1 flip (REAL-agree), bit-identical logits on 3/3227: computing layer-23 GEMMs on row subsets changes cuBLAS's bf16 rounding.
- table0 (layer 0 Win as a per-token table: the GEMM run once per distinct id, then gathered): bit-identical on 3027/3227 (93.8%; all 471
  LONG), max |d logit| 0.071, 1 flip (CF). Non-identical cases: cuBLAS picks a different kernel for the smaller M.
- => both are exact as functions (the table is a function of the token id; skipped values are never read); in this bf16 emulation they
  re-roll bf16 rounding like any GEMM-shape change (1 flip each in 3227 = 0.03%, vs the bf16 runtime's own 0.46% vs hobson).
- Dev (MEASURED): b8 + int4 2:4 pair-wise state rows L14-23 (40.2% of state-row MACs): bank 1.4e-4, ret 9.2e-5, TV .0046 (b8 alone
  9.5e-5 / 9.2e-5 / .0040); int8 L12-13 + int4 L14-23: 1.8e-4 / 3.3e-4. Kit for L14-23 running.

### 03:45 PDT KIT PASS: b8 + int4 2:4 pair-wise state rows L14-23 (MEASURED; 40.2% of state-row GEMM MACs on int4 2:4)
- REAL 3/1083 = 0.28%, paired vs bf16 1/3 p .625, TV .0054, CF .991, CF-probe .962, JB-hard .531 (p 1.0), REAL-label .790, LONG 2/471;
  banking REAL 3/883, LONG 2/471, CF-probe 34/36, REAL-label .749. (L13-23 missed by one REAL flip; layer 13 costs ~4e-5 dev KL.)
- B13 in b8 (both changes, 600 questions = every 5th): 0 flips, bit-identical logits on 583/600 (every REAL, LONG, CF, CF-probe item);
  the 17 differing items are short JevBench requests where the emulator's int8 GEMM falls back to a bf16 matmul for <= 16 rows
  (h1lib.imm), which is not exact for K = 2048 int8 products; the deployed int32 kernel would be exact.

### 03:52 PDT GEMM timing of B12 shapes (MEASURED, box q5 with the GPU otherwise idle: 0 MiB / 0% before; q5gemmtime.py)
- The 96-GEMM sequence of one forward (b8 map: int8 GEMMs via torch._int_mm = cuBLASLt s8, 8 bf16 GEMMs via torch.mm), fresh inputs,
  30 reps, median (p95 within 0.1%). Ratios vs b8 (b8 = 31.33 ms at M = 1125; Q4's CUTLASS b8 GEMM time is 27.0 ms):
| M (state + 125 q rows) | B13 | B12 50% global | 60% global | 75% global | 60% + B13 | 75% + B13 | uniform 50% |
| 381 | 0.966 | 0.843 | 0.810 | 0.768 | 0.791 | 0.752 | 0.833 |
| 1125 | 0.949 | 0.849 | 0.817 | 0.771 | 0.790 | 0.745 | 0.845 |
| 4125 | 0.944 | 0.848 | 0.818 | 0.770 | 0.789 | 0.742 | 0.848 |
- Nominal arithmetic said 0.80 / 0.75 for 60% / 75%: measured within 0.02. Narrower dense GEMMs keep their rate down to ~180 kept neurons.
- B7 for Q1's 4-bit row-role format w4q8 (dev): GPTQ on 64 banking states: bank KL 4.6e-3 / ret 5.9e-3 vs pool sample 3.7e-3 / 6.0e-3.

### 03:58 PDT (clock-checked) B7 for Q1's w4q8, all three GPTQ calibrations (MEASURED, dev; bank KL / ret KL, flips)
- pool sample (H1, 64 states, ~77% banking): 3.7e-3 / 6.0e-3 (6/160, 5/96); banking-only 64: 4.6e-3 / 5.9e-3 (6, 6); retail-only 64:
  4.7e-3 / 4.2e-3 (7, 2). At 4 bits, a matched calibration helps the domain the pool under-represents (retail: KL -30%, flips 5 -> 2 of
  96) and does nothing for banking, which dominates the pool. Late queue: 2nd calibration set (bank2, ret2) for L12-23 running.

### 04:10 PDT late dev (MEASURED; 2nd disjoint calibration set bank2 / ret2 = 160 + 160 more train-split requests, layers 12-23)
| B12 global 75% L12-23 calibrated on | requests | bank KL | ret KL | TV |
| banking (bank) | 160 | 6.1e-5 | 6.8e-5 | .0035 |
| banking (bank + bank2) | 320 | 4.8e-5 | 5.2e-5 | .0030 |
| retail (ret + ret2) | 320 | 2.1e-3 | 1.6e-5 | .0079 |
| generic (bank + ret) | 320 | 5.4e-5 | 2.3e-5 | .0027 |
| generic x2 (all four) | 640 | 4.5e-5 | 2.4e-5 | .0026 |
- At equal size (320), matched calibration is 12% better on banking and 30% better on retail than generic; mismatched (retail -> banking)
  is 44x worse. Doubling generic calibration to 640 helps 17% on banking. global 60% with 640: 1.9e-5 / 8.0e-6 (320: 2.5e-5 / 1.4e-5).
- b8 + B12 60% + int4 2:4 pair-wise on state rows of Win/Wo L14-23: bank 1.3e-4, ret 1.1e-4, TV .0047 (b8 + int4p all four GEMMs L14-23,
  which passed the kit: 1.4e-4 / 9.2e-5 / .0046); int8n mixers instead: 1.5e-4 / 1.1e-4 / .0047. ARITHMETIC 0.69x of b8 GEMM time (+B13).
- Kit runs launched 04:06: the stack above (fin), and B12 75% gen2 bf16 (fin2).

### 04:37 PDT final kit runs (MEASURED)
- B12 global 75% with 2x calibration (640 requests), bf16: REAL 5/1083 = 0.46%, paired vs bf16 1/1 p 1.0, TV .0050, CF 1.000, CF-probe .962,
  JB-hard .546 (p .25), REAL-label .7825, LONG 5/471 -> PASSES (24.7% of GEMM MACs removed). With 320 requests it missed by one REAL flip.
- Stack b8 + B12 60% + int4 2:4 pair-wise state rows of Win/Wo L14-23: REAL 5/1083, paired 2/2 p 1.0, TV .0059, CF .982, CF-probe .952,
  JB-hard .538, REAL-label .785 -> misses CF by one pair: the same two airline asked_for_human pairs b8 alone loses (bf16 P .514/.510).
  Nominal 0.69x of b8 GEMM time with B13.
- b8 + G75 gen2 dev: 1.5e-4 / 9.3e-5 / .0048 (= b8 + G75 gen, which failed the kit with 10 flips): dev KL does not separate them.

### 04:56 PDT final kit run + wrap-up
- b8 + B12 75% (640-request calibration) + int4 2:4 pair-wise state rows of Win/Wo L14-23 (nominal 0.642x of b8 GEMM time): REAL 10/1083 =
  0.92%, paired 8/3 p .23, TV .0067, CF .982, CF-probe .952, REAL-label .7825 -> fails. Every b8 + 75% setting fails (10 flips).
- best_structure.json FINAL. DRAFT_REPORT.md written. All results copied to ~/decider2/q5/res (preds/, dev_res.json, dev_table.txt,
  res_gemmtime_*.json, res_b13.json, res_nstat.json, scores.json, cfgs_q5.json, logs/logs_small.tgz).
- Box q5 left idle (no jobs); it auto-terminates at its 10 h TTL (07:33 PDT). I did not reset the timer.
