# J12: exact-relation primitives inside hobson-v19

Labels: [M] measured, [V] verified, [S] speculation. Box j12 (A10G). Code: `~/decider2/j12/code/`. Timeline: `NOTES.md`. Results: `results/`, `preds/`, `scores_*.json`.

## Hypothesis
- **First principles [S].** Many decisions over agent states reduce to a typed relation: an amount against a limit, one date before another, an id equal to the one on file, a field belonging to the named record. A transformer has no exact comparator: digits are tokenized one at a time, so date order becomes soft multi-token matching. A comparator over parsed values is exact and almost parameter-free; the network only has to learn where to point.
- **Claim.** A few such operations would buy what parameter scale buys, and fix hobson's date_order .06, amount .33 and identity 0.
- **Prior art (abstracts fetched).** XR follows the argument-pointer-plus-exact-executor pattern of NMN over text (1912.04971: differentiable compare and count over numbers and dates), NumNet (1910.06701), Neural Programmer (1511.04834) and BERT-with-a-calculator (1909.00109), not learned arithmetic units (NALU 1808.00508, NAU 2001.05016). It is placed inside the network as TransNAR does (2406.09308), and it covers exact content-addressable memory (hash-join bits, as opposed to product keys 1907.05242 or kNN memory 2203.08913) and a typed field graph; question-row queries act as the compiled operator program.
- **Why it is not marginal.** It changes the class of the computation: relations are exact whatever the scale, pointing works over a slot set and does not weaken with distance, and it runs on question rows only, so the state prefix and its caches stay exactly hobson's.

## What was built
- **`xlit.py` [V], a typed-literal extractor.** Regexes find amounts, dates (ISO, US, month-name), phones, emails, ids, quoted strings and `key: value` values. Each literal gets an exact equality key, an order value, its record, its field name, and exact question-join bits (it equals a question literal; its record contains one; its field is named in the question).
- **The XR module (`xrlib.py`).** It is applied after layers 11 and 17, on question rows only, with 16 heads.
  - Each head has two learned pointers p_a and p_b over the slots plus a null slot. The readout is exact: E[f(a,b)] = p_aᵀ M_f p_b for f in {eq, a<b, a>b, same block, date gaps over 1–365 days in both directions, null}, 22 features per head in O(Q·N), with no N×N matrix.
  - W_o starts at zero; 18.6M parameters, 0.8% of hobson.
  - Checks [V]: the base forward matches the `merged_full` references (max |dp| .007, 30/30 argmax); XR at init changes the logits by exactly 0; with one-hot gold pointers the readout is exact on 261/261 synthetic items.
- **Synthetic exact tasks (`gen_xr.py`).** 3,200 pairs on train-split states in 11 kinds (number, constant, date and date-window comparisons; status and id equality; stated identity against the record; amount against a cap; existence); half carry a distractor record. Tools, fields and phrasings are disjoint from CF-probe's, so CF-probe tests template transfer.
- **Arms.** All start from hobson-v19 with F7-style losses (KL to frozen hobson on train_pool states, CE+KL on train_v5 gold, CE+0.2·KL on synthetic pairs, option shuffling) and the same 3,600 sequences (450 updates of 8).
  - **ctl:** LoRA r16 on every projection of all 24 layers, plus the head.
  - **xr:** ctl plus XR plus a pointer auxiliary loss on synthetic items.
  - **xrf:** hobson frozen, XR only (s250 evaluated).

## Results: every suite [M]
All 3,227 evalkit questions, with token ids identical to the references. p is a paired McNemar test against hobson.

| | hobson | ctl (LoRA + data) | xr (LoRA + data + XR) |
|---|---|---|---|
| JB-all acc | .723 | .714 (p .80) | .719 (p 1.0) |
| JB-hard acc | .523 | .515 (p 1.0) | .538 (p .82) |
| temporal_numeric / multi_hop | 3/15, 8/18 | 3/15, 7/18 | 2/15, 6/18 |
| REAL-label | .785 | **.752 (p .019)** | **.755 (p .050)** |
| REAL agree_sd / LONG agree_sd | 1 / 1 | .893 / .915 | .876 / .903 |
| LONG agree | 1 | .955 | .949 |
| CF acc / flip | .594 / .268 | .696 / .466 | .754 / .579 |
| CF identity / amount_insert / human_insert (flip) | 0 / .10 / .35 | .18 / .71 / .43 | .43 / .74 / .63 |
| CF ≥4k-token states | 1/76 | 3/76 | 6/76 |
| CF-probe acc / flip | .653 / .328 | **.995 / .991** | **.995 / .991** |
| date_order (+distractor) | .06 / .03 | 1.0 / .94 | 1.0 / .97 |
| amount_vs_limit (+distractor) | .33 / .24 | .98 / 1.0 | 1.0 / 1.0 |
| SHUF both_right | .245 | .452 | .568 |
| Brier JB-all / REAL-label / CF | .348 / .347 / .526 | .353 / .365 / .417 | .361 / .371 / .378 |
| synthetic dev pair flip (300 pairs) | .323 | .973 | .937 |
| HARD pair flip (160 pairs) | .150 | **.975** | .931 |

HARD uses 3.5–7k-token states with 3 same-schema distractor records and near-ties ($0.01–0.50, 1–2 days, windows ±1 day).

**Attribution: is the module doing the work? [M]**
- **XR switched off at inference** in the xr model: CF pairs 235 → 237, identity 22 → 24, human_insert 98 → 98, CF-probe 317 → 317 (discordant 6 against 8, p .79). The XR output is not zero (median max |dp| .026) but changes no pair decision.
- **xr against ctl** on CF: 235 against 189 pairs (p < 1e-4). The edge includes human_insert (+31), which has no literal the module could use, and it survives XR-off. It is a training-trajectory effect (run variance, or pointer-loss gradients into the LoRA), not the module's computation.
- **The pointers generalize anyway.** On CF-probe items where my locator found the gold slots (60–70%), some head puts p_a·p_b ≥ .98 on the exact gold pair for amount, date and status, distractors included (id_match .35–.44). The module points correctly out of template; the fine-tuned torso already computes the answer.
- **xrf (frozen torso) gains partially:** synthetic .567, HARD .325, CF-probe flip .394 (id_match .54 → .84, status_distract .25 → .55; date_order 0, amount .26–.30). [S] The symmetric pointer loss leaves a/b orientation arbitrary, so frozen layers can use equality bits but not a<b bits.

## Latency [M]
A10G, d1 lean2 fused bf16 runtime (all fusions), full 24 layers, one CUDA graph per input, 14 distinct real inputs, exact state lengths. Values are median / p95 in ms.
- 1q: one question of 93 tokens.
- 4q: one packed-equivalent sequence (state plus 4 real questions), with XR on all question rows. This is not hobson's shared-prefix engine.

| T | 1q base | 1q +XR | Δ | 4q base | 4q +XR | Δ |
|---|---|---|---|---|---|---|
| 64 | 15.51 / 15.58 | 16.08 / 16.17 | +0.55 | 32.75 | 33.70 | +0.95 |
| 128 | 15.88 / 15.99 | 16.45 / 16.58 | +0.57 | 36.97 | 37.90 | +0.97 |
| 256 | 25.40 / 25.50 | 25.99 / 26.12 | +0.59 | 43.72 | 44.68 | +1.03 |
| 400 | 32.92 / 33.05 | 33.54 / 33.69 | +0.62 | 48.60 | 49.62 | +1.08 |
| 1000 | 57.47 / 57.55 | 58.33 / 58.70 | +0.85 | 80.32 / 83.80 | 81.77 / 85.16 | +1.28 |
| 4000 | 202.4 / 206.2 | 204.8 / 208.7 | +2.41 | – (no inputs with 4 short questions) | | |

- **GPU:** +1–3.5% (+0.55–2.4 ms), growing with slot count (5–299 slots).
- **Host:** Python extraction and indexing take 1.0–5.9 ms up to 1000 tokens and 42.7 ms at 4000, serialized here. [S] It is needed only after layer 11, so it can overlap the first 11 GPU layers (about 26 ms at 1000 tokens); compiled, it should take under 1 ms.
- **Projections [S]:** Δ is about 40 small kernels, so about 0.4–0.9 ms on the 3090, 4090 and 5090 at ≤1000 tokens: +2–8% over the 27.4 / 13.7 / 10.0 ms fp16-accumulation bases.

## Verdict
**The hypothesis fails, and the reason is clean.**
- Hobson's exact-reading failures are gaps in its training distribution, not capacity failures of a 2B model. The same data with a 0.66% LoRA and no module lifts CF-probe from .328 to .991 (disjoint templates), HARD from .150 to .975, and CF identity from 0 to .18 (.43–.47 in the xr run, with XR on or off).
- The exact comparator learned correct out-of-template pointers but is redundant: switching it off changes no decision, and the control matches or beats it on every exact suite.
- **So exact primitives do not buy what scale buys: the gap scale closes is not exact comparison.** [M] temporal_numeric (2–3/15) and multi_hop (6–8/18) did not move in any arm; they need multi-step composition (unit conversion, time zones, aggregation). [S] That and robustness on real traffic are plausibly where Open-Jev 9B's 0.775 against 0.645 comes from; the 9B was not measured.
- **Neither arm passes the new-architecture bar.** JB-all and JB-hard show no significant drop and CF / CF-probe beat hobson widely, but REAL-label is .752 / .755 (p .019 / .050) against ≥ .78 and agree_sd is .88–.89: the same augmentation tilt H7 saw.
- **Caveats:** one seed per arm and 450 updates (run variance as large as human_insert +31 means CF differences between runs need several seeds); the synthetic tasks share CF-probe's format (inserted records, true/false questions); HARD has no multi-operand arithmetic.

## The single decisive next step
Exact comparison and lookup are closed: data suffices. The only open case for in-network primitives is multi-operand arithmetic (N line items against a limit, date plus N days across time zones, unit conversion), where a 2B model lacks an algorithm. Run the same matched xr/ctl pair there with 3 seeds and the XR-off ablation; if ctl learns it from data too, close the direction. Independently, the measured lever for detail reading is the ctl recipe rebalanced toward real-state KL (all 46k train-pool pairs, synthetic share ≤ 15%), gated on REAL-label ≥ .78 at CF-probe ≥ .95.
