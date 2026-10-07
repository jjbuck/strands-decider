# J7: a reader-native input layer for hobson (box j7, A10G)

Labels: [M] measured, [V] verified, [A] arithmetic from measured anchors, [S] speculation. Code is in `~/decider2/j7/code/`, the timeline in `NOTES.md`, scores and latency in `results/`.

## 1. Hypothesis, from first principles
- **A reader's vocabulary is free.** A decision model has no LM head, so an embedding row is a gather that costs zero FLOPs. A generator's vocabulary has to be emittable and pays V × d per output token.
- **Cost is linear in tokens.** GEMMs are 87% of the time at 1,000 tokens.
- **So the reader can carry its deployment's own vocabulary.** Real gate states are mostly repeated knowledge-base pages, tool schemas and record layouts.
  - (a) That vocabulary should cut tokens 1.5–3x, and latency with them, at the same 2B and the same 24 layers.
- **Side channels are free too.**
  - (b) A canonical value-identity code (`04/17/1979` = `April 17, 1979`; one changed digit gives an unrelated code) should make identity matching a one-head primitive. hobson gets 0% on identity edits.
  - (c) 18 of 24 layers are GDN, whose only notion of position is the order of the recurrence. Structure can enter only as additive coordinates (field, record, role, turn) or through RoPE in 6 layers.
- **Why this is not marginal.** Tokenizer, embeddings and features become per deployment. If it held, every request on every card would be about 2x faster with the same model, stacking with W8A8 and the compiled schema.

## 2. What I built
- **`superbpe.py` (a).** BPE over Qwen ids: every new token concatenates tokens hobson already reads. It is SuperBPE-style (arXiv 2503.13423): merges cross spaces and punctuation, never digits or newlines. Trained on 34M train-split tokens in 36 s; the 4k, 16k and 64k vocabularies are nested prefixes.
- **`chan.py` (b, c).** Zero-initialised per-token channels, so the untrained model is exactly hobson [V: 160/160]:
  - canonical value identity (hashed codes and a learned projection);
  - Abacus digit place (arXiv 2405.17399);
  - field key, record instance (TAPAS-like, arXiv 2004.02349);
  - role × section;
  - turn recency.
- **`j7lib.py`.**
  - Each super-token embedding = mean of its constituents + shared per-position gains + a per-token delta. The shared part is ZeTT-like (arXiv 2405.07883); the mean start follows AdaptiVocab's weighted combination (arXiv 2503.19693).
  - RoPE sits at each token's original position.
  - Trained by token distillation (arXiv 2505.20133): relative MSE to hobson's rows at aligned positions at 6 layers, plus KL. Early-layer detokenization (arXiv 2410.05864) motivates it.
- **Training.** F7-style: LoRA r16 on every projection plus the head, in-process KL to frozen hobson plus gold CE, train split only, 400 updates of 8 sequences (about 7M tokens) per arm.
  - My own detail augmentation: 4,000 pairs of stated value vs record, lookup, amount above a threshold, date before a date (formats differ), half with distractors. No evalkit template, field, tool or wording, and no CF identity kind.
  - The arms:
    - **C:** control, Qwen tokens;
    - **V:** C plus channels;
    - **T:** super-token transplant, vocabulary level drawn per example;
    - **T2:** T, but user and assistant lines stay at Qwen granularity;
    - **TV:** T plus channels plus augmentation;
    - **TL / TL2:** T plus 700 / 1,400 more updates at 64k (about 20M / 33M teacher tokens).

## 3. Tokens [M]
| vocabulary added | REAL states | LONG | CF | CF-probe | JevBench | questions |
|---|---|---|---|---|---|---|
| 4k | 1.60x | 1.59x | 1.52x | 1.50x | 1.02x | 1.23x |
| 16k | 1.97x | 2.06x | 1.85x | 1.71x | 1.03x | 1.31x |
| 64k | **2.41x** | **2.67x** | 2.14x | 1.91x | 1.04x | 1.59x |
| 64k, messages protected | 1.90x | 2.26x | 1.73x | 1.55x | – | 1.59x |

## 4. Accuracy, every suite [M]
All 3,227 questions are scored. Each p-value is a paired McNemar test against hobson. CF and CF-probe show item accuracy and pair accuracy (both items right). Brier is on REAL-label.

| model | REAL tokens | JB-all (p) | JB-hard (p) | REAL-label (p) | REAL agree_sd | LONG agree_sd | CF acc / pair | CF-probe acc / pair | Brier |
|---|---|---|---|---|---|---|---|---|---|
| hobson | 1x | .723 | .523 | .785 | 1 | 1 | .594 / .268 | .653 / .328 | .347 |
| C | 1x | .719 (1.0) | .523 (1.0) | .782 (1.0) | .907 | .939 | .607 / .310 | **.952 / .903** | .358 |
| V | 1x | .723 (1.0) | .531 (1.0) | **.795** (.50) | .907 | .915 | **.622 / .335** | .933 / .866 | .357 |
| T, 4k | 1.60x | .710 (.45) | .508 (.69) | .777 (.70) | .844 | .897 | .583 / .264 | .642 / .306 | .354 |
| T, 16k | 1.97x | .693 (.09) | .492 (.34) | .772 (.50) | .809 | .861 | .548 / .192 | .639 / .287 | .360 |
| T, 64k | 2.41x | .710 (.65) | .538 (.79) | .730 (**.0007**) | .801 | .836 | .550 / .200 | .620 / .244 | .387 |
| T2, 64k | 1.90x | .688 (.12) | .515 (1.0) | .733 (**.001**) | .777 | .861 | .544 / .187 | .619 / .250 | .385 |
| TV, 64k | 2.41x | .697 (.31) | .500 (.66) | .750 (**.02**) | .806 | .842 | .575 / .249 | .817 / .637 | .389 |
| TL, 64k (1,100 updates) | 2.41x | .706 (.45) | .515 (1.0) | .767 (.25) | .841 | .855 | .568 / .234 | .619 / .263 | .372 |
| TL2, 64k (1,800 updates) | 2.41x | .688 (.10) | .492 (.42) | .765 (.20) | .832 | .885 | .573 / .244 | .628 / .266 | .359 |
| zero-shot transplant, 16k | 1.97x | .727 | .561 | .705 (**.0001**) | .679 | .430 | .555 / .202 | .605 / .222 | .432 |

- **Held-out detail pairs** (400, unseen values), pair accuracy: hobson .323, C **.960**, V .965. Plain digit tokens read amounts, cross-format dates, ids and distractor bindings after 400 updates.
- **The value channel is unused.** Zeroing it in the trained V gives CF .622 → .619, identity 4/51 → 4/51, CF-probe .933 → .938.
- **V against C, paired.** CF: 14 pairs only V gets right, 4 only C (p .03). CF-probe: 6 against 18 (p .02, all date order).
- **Identity CF pairs:** hobson 0/51, C 1/51, V 4/51. All three move P(true) the right way but keep a strong "true" prior.
- **Transplant budget curve** (64k, REAL agree_sd): .668, .772, .795, .801, .841 and .832 at 100, 200, 300, 400, 1,100 and 1,800 updates.
  - TL beats T on REAL-label by 23 pairs to 8 (p .01) and on CF by 26 to 12 (p .03).
  - TL2 against TL is flat on REAL (13/16, p .71). LONG agree_sd still rises, from .855 to .885, and so does CF fgh, from .68 to .75.
  - REAL agreement plateaus near .84 with this recipe.
- **Why human requests fade.** P(true) on the request item is .425 for hobson and .374 / .318 / .287 for T at 4k / 64k / T2. The signal dims even when the message lines are not merged, so the damage is in reading the merged context.
- **[V] Runtime fidelity.** The fused runtime (LoRA merged, folded super and channel tables) against j7lib: 24/24 for C, T and TV, median |dp| at most .0009.

## 5. Latency [M]
A10G with exclusive use of the GPU, fused bf16 runtime (lean2 + TTL), CUDA graph, fresh real states, 20 timed repetitions after 3 warm-ups. Medians in ms; p95 is within 0.1 ms everywhere.
- Super-token rows are the Qwen rows divided by the REAL median compression.
- The timed span includes the extended gather, RoPE at original positions and the channels.

| Qwen state tokens | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| hobson, 1 q | 15.24 | 15.56 | 22.84 | 28.14 | 57.06 | 201.17 |
| hobson + channels | 15.29 | 15.60 | 22.90 | 28.22 | 57.35 | 202.47 |
| super 16k | 9.33 | 14.70 | 15.51 | 22.33 | 37.36 | 108.46 |
| super 64k | 9.53 | 9.74 | 15.24 | 15.53 | **27.82** | **92.50** |
| super 64k + channels | 9.58 | 9.79 | 15.28 | 15.58 | 27.89 | 93.06 |
| hobson, 4 q (state once, branches) | 48.60 | 48.94 | 55.64 | 67.31 | 93.25 | 243.30 |
| super 64k, 4 q | 44.80 | 45.22 | 45.55 | 51.63 | 64.24 | 126.37 |

- **Real traffic** [A: measured curve × each request's own token counts]: the REAL median goes from 128.6 to 61.4 ms (2.00x), the LONG median from 253 to 110 ms (2.39x).
- **Overheads.** Channels cost at most 0.7%. On the host, in Python, per 1,000 tokens: merge 1.2 ms, channel annotation 3.0 ms (tokenizing is 2.2 ms).

## 6. Projections [A]
These apply FAST_DECISION_MODEL's per-card ratios at about 1,000 tokens (3090 0.52, 4090 0.26, 5090 0.19, with fp16 accumulation). They are optimistic for the shorter rows, whose floor scales only with bandwidth.

| 1 q, Qwen 1000 / 4000 tokens | A10G | 3090 | 4090 | 5090 |
|---|---|---|---|---|
| hobson | 57.1 / 201 | 29.7 / 105 | 14.8 / 52 | 10.8 / 38 |
| super 64k | 27.8 / 92.5 | 14.5 / 48 | 7.2 / 24 | 5.3 / 18 |

[S] With W8A8 (0.64x at 1,000 tokens), 64k would give about 18 ms on the A10G and 9 ms on a 3090, if it were accurate.

## 7. Verdict
- **(a) The token cut is real; the transplant is not accurate at this budget.**
  - [M] 1.6–2.7x fewer tokens, giving 2.0x (REAL) to 2.4x (LONG) lower latency on real traffic.
  - [M] With 7M to 33M tokens of token distillation, no level keeps hobson's function: agree_sd .78–.84, CF fgh .47–.75.
  - [M] REAL-label drops significantly at 64k after 400 updates. It recovers to .765–.767 (p .20–.25) after 1,100–1,800, but that is still below .78.
  - Keeping the natural-language turns at Qwen granularity does not help.
  - [M] REAL agreement plateaus at about .84 between 20M and 33M tokens with LoRA r16 plus per-token embeddings. That points at the recipe's capacity, not only at its token count.
  - [S] ZeTT closes its gap with full continued training on under 1B tokens. The negative is for a short LoRA transplant, not for the idea.
- **(b) Refuted as the bottleneck.**
  - [M] Exact reading is a data problem. Plain digit tokens with 400 updates of out-of-template augmentation take CF-probe from .653 to .952 and held-out value pairs from .32 to .96.
  - [M] Canonical identity codes go unused.
  - [S] Identity CF stays near 0 in every arm while P(true) moves the right way. That fits a question-prior problem, not an encoding one.
- **(c)** [M] V against C (all channels together) is +14/−4 on CF pairs (p .03) and −12 on date-order pairs (p .008). The value-identity part is inert, so the CF gain is the structure channels or noise. No capability change.
- **The best model that passes the bar is V**, at hobson's speed (channels +0.3 ms at 1,000 tokens). C, the same recipe without channels, is close: V wins on CF (p .03) and C on CF-probe (p .02). The gain is a data fix, not an input fix:
  - JB unchanged;
  - REAL-label .795;
  - CF .622;
  - CF-probe .933;
  - REAL agree_sd .91.

  The fast combination, TV (2.0x), fails REAL-label (.750, p .02).

## 8. Decisive next step
Transplant 16k (1.97x) with a **full-rank** fine-tune, not LoRA r16: the dense token-distillation loss on 0.3–1B tokens over the whole train pool and the F7 data, multi-GPU. Gate it on REAL agree_sd ≥ .95 and CF fgh ≥ .95.
- If it passes, it is a 2x multiplier on every card that stacks with W8A8 and the compiled schema.
- If agree_sd stays near .80, retire input-side token reduction for GDN hybrids.
