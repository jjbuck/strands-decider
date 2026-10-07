# J12 NOTES

## 2026-10-06 09:14 start
- Read BRIEF8, FAST_DECISION_MODEL, F7_REPORT, evalkit README. Box j12 not ready yet.

## 09:35 design (while box boots)
- Direction chosen: XR = "exact relation readout" module. Deterministic typed-literal extractor (regex: numbers/amounts, dates (ISO, US, month-name), phones, emails, ids, quoted strings, key:value values) gives slots with exact eq-keys, order values (num/date groups), record/block ids, and exact question-join bits (slot eqkey in question literals; slot's block contains a question literal).
- In-network: at layers 11 and 17, question rows only (state rows untouched -> state prefix cache stays exact, causal), H=16 heads, each with two learned pointers p_a,p_b over slots (+null). Exact readout E[f(a,b)] = p_a^T M_f p_b for f in {eq, a<b, a>b, same_block, date diff > t (8 thresholds x2 directions), null_a, null_b}: 22 feats/head computed in O(Q*N) by scatter/cumsum/gather (no N^2). W_o zero-init -> starts exactly as hobson.
- Training: 2 arms on same data: XR+LoRA r16 vs LoRA-only control (attribution: H7 showed data+LoRA alone lifts CF-probe when templates are in-distribution). Synthetic exact tasks on train-split states with templates/field names/tools DISJOINT from CF-probe's (so CF-probe is a template-OOD transfer test). Pointer aux loss on synthetic items (gold slots known by construction).
- Lit fetched: 1912.04971 (NMN over text, differentiable compare/count over numbers+dates), 1808.00508 NALU, 2001.05016 NAU, 1910.06701 NumNet, 2406.09308 TransNAR, 1511.04834 Neural Programmer, 1907.05242 product keys, 2412.09764 memory layers at scale, 2203.08913 Memorizing Transformers, 1605.06640 diff. Forth, 1909.00109 BERT calculator, 2105.02761 NAR, 2405.17399 Abacus, 1506.03134 Ptr-Net, 2302.04761 Toolformer.
## 10:25 both arms training concurrently (xr, ctl), ~11.5 s/update each -> ETA ~12:25
- Base hobson on synthetic dev (300 pairs, my templates): pair flip .323 (its CF-probe flip is .328: the dev set reproduces the weakness).
  per kind: date_window 0.00, num_cmp .16, user_amt .13, date_cmp .21, ident2 .25, num_const .32, ident1 .52, id_eq .55, status_eq .59, exists .63.
- XR arm at upd 50: ptr loss 20 -> 6.3, pointer top-1 hit .43; train agree_real fluctuates .86-1.0 (LoRA drift to watch).
- Prepared: eval_xr.py (all 3227 evalkit q + syn dev), diag_xr.py (pointer hit on CF-probe gold literals), lat_xr.py (lean2 fused + XR, CUDA graph), arm 'xrf' (frozen torso+head, XR only) for a follow-up.
## 09:58 box j12 ready (09:40); smoke tests pass
- (1) my Prep+fwd vs evalkit merged_full refs: max|dp| .007, argmax 30/30. (2) XR at init: max|dlogit| 0.0 (function-preserving).
- (3) exact readout with one-hot gold pointers equals ground truth on 261/261 synthetic dev items (after fixing a strict-threshold bug in date windows).
- (5) fwd+bwd (ckpt) 0.92 s at 2730 tokens, 7.5 GB.
- Synthetic data: 3200 train pairs (11 kinds), 300 dev pairs (other seed). Templates/fields/tools disjoint from CF-probe.
- Launched arm xr: 700 updates x 8 seq, maxtok 3072, LoRA lr 1e-4, head 2e-4, XR 1e-3, mix real .45 / v5 .15 / syn .40, ptr aux .3.
## 11:10 restarted xr/ctl at --updates 450 (from their upd-50 checkpoints; 2 concurrent procs ran 16 s/update each, no faster than sequential). Launched arm xrf (frozen torso+head, XR only, maxtok 2560) as 3rd process. GPU 17.1 GB.
- Train-time signals at ~upd 110-130: acc_syn xr .85 / ctl .83 (both learn the in-distribution synthetic tasks; teacher .60); agree_real drifts to .83-.89 in both LoRA arms (KL .05-.08) -> expect agree_sd losses like H7; xrf is the arm that cannot drift.
- XR pointer: ptr loss 20 -> 5.0, head-0 top-1 on gold .56.
## 11:45 mid-run CF-probe eval OOMed (4th process; training procs unaffected: all 3 alive). No more concurrent GPU jobs until training ends (~13:15).
- ctl acc_syn .92-1.0 by upd 150-170: LoRA alone fits the in-distribution synthetic tasks. Decisive test = OOD transfer (CF-probe, CF identity) + xr vs ctl.
- Literal stats (laptop regex): REAL-agree median 84 literals/state (p90 166), LONG 304 (p90 447), CF-probe 221; ~46-69 literals per 1k tokens. Python extraction 2-7 ms on laptop CPU (must overlap layers 0-10 or move to Rust).
## 12:45 killed arm xrf at upd ~275 (kept s250.pt): frozen torso + XR only does NOT learn: train acc_syn .59 = teacher's .59, ptr loss 9.4 (xr arm 3.8). Negative result: the injected exact features are not usable by frozen downstream layers within this budget. Freed GPU for xr/ctl.
- xr upd 290: acc_syn .94, ptr top-1 .58, ptr loss 3.8; ctl upd 280: acc_syn .94. In-distribution synthetic accuracy is equal: the module is not needed in-distribution.
## 13:00 xrf s250 (frozen torso+head, XR only) on synthetic dev: pair flip .567 vs hobson .323 [M]. Correction to 12:45 note: it does learn, mainly the equality/identity kinds:
  ident1 .52->.97, user_amt .13->.97, status_eq .59->.90, ident2 .25->.56, id_eq .55->.64; but date_cmp .21->.35, num_cmp .16->.16, date_window 0->.14.
## 13:30 xrf s250 on CF-probe [M]: flip .394 (hobson .328), acc .691 (.653). id_match .54->.84, status_equal_distract .25->.55; date_order 0 (hobson .06), amount_vs_limit .30/.26 (hobson .33/.24).
- Diagnosis [S]: equality is symmetric, order is not. With the symmetric pointer aux loss a head's (a, b) orientation is arbitrary, so lt/gt bits are unreadable downstream. Launched v2 = arm xr with --ptr_order 1 (a = operand mentioned first in the question; ordered loss); otherwise identical. ck_xr2. 3 procs again (17.8 GB).
## 14:10 xr and ctl finished 450 updates (s450.pt). Evals launched (CF-probe first, then all 3227, then syn dev), concurrently with xr2 training.
## 14:35 KEY RESULT [M]: CF-probe (template-OOD for my synthetic data): xr acc .995 flip .991; ctl (LoRA only, same data) acc .995 flip .991; hobson .653/.328.
  per kind xr/ctl: amount .98-1.0, date_order 1.0, date_order_distract .97/.94, id_match_distract .95/1.0, status_distract 1.0/1.0.
  => the module adds nothing on CF-probe; ~700 synthetic pairs + LoRA r16 alone close hobson's CF-probe gap. Killed xr2 (ordered-pointer ablation is moot here).
- Next: HARD synthetic stress set (4-7k-token states, 3 distractor records, near-tie values) to test where exactness could matter (length/distractors), eval-only for base/xr/ctl/xrf.
## 15:10 HARD stress set (160 pairs; 3.5-7k-token train-split states; 3 distractor records of the same schema; near-tie values: $0.01-0.50, 1-2 days, window +-1 day) [M]
- hobson: pair flip .150 (record comparisons ~0: date_cmp 0, num_cmp .05, status_eq .06, id_eq 0).
- xrf (frozen torso, XR only): flip .325 (user_amt 1.00, ident2 .75, exists .78; order kinds ~0).
- xr / ctl running. Full-suite evals of xr / ctl ~60% done (LONG is slow).
## 16:20 DECISIVE ABLATION [M]: xr model with XR switched off at inference (same LoRA+head): CF pairs 235 -> 237, identity 22 -> 24, human_insert 98 -> 98, CF-probe 317 -> 317 (discordant 6 vs 8, p .79).
  XR output is not zero (median max|dp| .026; 25% of items move > .05) but changes no pair decision. The xr-vs-ctl CF gap (235 vs 189, p<1e-4; identity 22 vs 9; human_insert 98 vs 67) is therefore a training-trajectory effect (incl. pointer-aux gradients into LoRA / run variance), not the module's computation.
- Full suites [M] (hobson / ctl / xr): JB-hard .523/.515/.538 (McNemar p 1.0/.82); JB-all .723/.714/.719; REAL-label .785/.752/.755 (both FAIL .78 bar); CF acc .594/.696/.754, flip .268/.466/.579; CF-probe .653/.995/.995 flip .328/.991/.991; REAL agree_sd 1/.893/.876; LONG agree_sd 1/.915/.903; SHUF both_right .245/.452/.568; CF>=4k 1/76, 3/76, 6/76.
- syn dev flip: base .323, xrf .567, xr .937, ctl .973. HARD: base .150, xrf .325, xr .931, ctl .975.
- Latency (exclusive A10G, CUDA graph, 14 inputs): T=64 base 15.51 / +XR 16.08 (+0.55 ms); T=128 15.88 / 16.45 (+0.57); host prep ~1 ms.
## 16:55 done
- Pointer diag (xr, CF-probe, items where gold located 60-70%): best head p_a*p_b on gold pair >= .98 for amount/date/status (incl. distractors), .35-.44 id_match. Module points right out-of-template but is redundant.
- REAL-label vs hobson McNemar: ctl 7 vs 20 (p .019), xr 10 vs 22 (p .050).
- Latency full table in results/lat_xr.json. DRAFT_REPORT.md written. Checkpoints remain on box (ck_xr, ck_ctl, ck_xrf); preds + scores on laptop.
## 13:52 final: DRAFT_REPORT.md complete (1476 words excl. table separators). Box idle; checkpoints on box only.
