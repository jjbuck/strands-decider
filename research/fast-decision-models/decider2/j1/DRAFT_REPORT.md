# J1: a 2B bidirectional encoder decider at equal size (T5Gemma-2B encoder, pointer head, masked state cache)

Labels: **[M]** measured on box j1 (A10G), **[V]** verified by a check, **[A]** arithmetic, **[S]** speculation.

## 1. Hypothesis, from first principles
Hypothesis: a decision model needs no causal mask, so an encoder could change both cost and capability.
- **Cost.** Brooker's e1b served at about 30–35 ms against 60–65 for v19 on a 3090.
- **Capability.** Bidirectional reading, and under e1a also a state that reads the question, could fix hobson's weak detail reading (CF flip .268).

The arithmetic predicts the cost half fails:
- The T5Gemma-2B encoder has 2.03B non-embedding weights: 4.05 GFLOP per token, 1.46x hobson's 2.77 [A].
- 26 global bidirectional layers add 5% of FLOPs at 1000 tokens and 17% at 4000 [A].
- Decisions are prefill and compute-bound above about 150 rows. So in an equally fused runtime the encoder must be 1.2–1.7x slower at every length, and can win only on accuracy [A].

This is not marginal. It replaces the causal decoder with a different architecture family at the same size, and it tests two of the brief's targets: the causal mask, and reading the state before the question.

## 2. What I built (`~/decider2/j1/code/`)
- **Checkpoint.** google/t5gemma-* is gated: the laptop token's request is "awaiting review" (HTTP 403). I used the ungated encoder-only extraction `Minthy/t5gemma-2b-2b-ul2-encoder-only` (Gemma terms; 2.614B parameters). **Caveat: I could not hash-check it against Google's files.**
- **Torso.** Plain torch. Per-layer outputs match HF's T5GemmaEncoderLayer at delta cos ≥ 0.9998 [V].
  - The model is chaotic in bf16: HF's own bf16 run puts 70 of 1501 rows below 0.99 cos against fp32 [M].
  - Dropping the attention softcap adds about that same noise [M], so I dropped it, which allows flash attention.
- **Training.** Packed rows with varlen FA2 forward and backward inside an autograd Function.
  - Masked mode: the state attends to the state; the question attends to state plus question.
  - Verified against a dense-mask reference: per-layer cos ≥ 0.9999, gradient cos 0.9995 [V].
  - Throughput 4,170 tok/s.
- **Recipe: F7 exactly.**
  - Data: g3's F7 rows regenerated with the same seeds. The Qwen twin matches g3 id for id on 110,633 rows, so hobson's teacher logits were reused [V].
  - Size: 108,547 rows (6,000 KL-only real states), 40.5M tokens.
  - Head and loss: pointer head of dim 256; LoRA r16 on all 7 projections; CE + KL(hobson).
  - Schedule: 3,392 steps in 2.7 h.
- **e1a fork.** The same run's step-1700 optimizer state, continued to step 3392 with full attention, so the state reads the question.
- **Fused runtime, `encrt.py`.**
  - Fusions: merged LoRA; folded pre-norms; GeGLU epilogue; one fused kernel for post-norm, residual and sum-of-squares; one RoPE kernel; per-shape tile autotuning.
  - Attention: FA2. The state is cached and shared by all M questions in one packed pass, with log-sum-exp merging.
  - One CUDA graph per exact shape.
  - Matches the training path on 47 real questions: 47/47 argmax, median |Δp| .0005 [V].
- **hobson comparator.** d1 lean2 / H4 TTL in the same harness. For fairness I gave its Triton GEMMs the same tile autotuning, which made it 13% faster at 64 tokens [M].

## 3. Accuracy (evalkit, absolute; per-kind temperatures fit on held-out train-distribution rows) [M]

| suite | e1b (masked) | e1a fork | hobson | e1b vs hobson |
|---|---|---|---|---|
| JB-all | .688 | .688 | .723 | McNemar 21/29, p .32 |
| JB-hard | .492 | .477 | .523 | 20/24, p .65 |
| REAL-label (Opus) | **.770** | .762 | .785 | 12/18, p .36; below the .78 bar |
| CF pair flip | **.170** | .180 | .268 | items 16/63, p < .001 |
| CF-probe flip | **.222** | **.316** | .328 | items 69/104, p .010 |
| REAL / LONG agree_sd | .809 / .842 | .812 / .842 | 1 | F7 hob12: .884 / .885 |
| SHUF: decision changes | .222 | .227 | .339 | the encoders read the state less |
| Brier: JB-all / REAL-label / CF / CF-probe | .397 / .359 / .591 / .451 | .394 / .366 / .589 / .417 | .348 / .347 / .521 / .410 | |
| ECE: same suites | .054 / .066 / .224 / .059 | .043 / .079 / .221 / .031 | .048 / .116 / .111 / .083 | |
| option rotations: unchanged / mean \|Δp\| | **.949 / .036** | .951 / .039 | .922 / .058 | 1,997 renderings |

- **e1b, masked.** It learned the training distribution: held-out val_v5 .866 against hobson's .873, adequacy .818 against .808. It does not transfer to real tau3 states. CF human_insert is .05 against .35; CF-probe amount_vs_limit .09 against .33. **Bidirectional reading does not read details better at equal size and budget; it reads them worse.** Its one advantage is lower option-order sensitivity.
- **e1a, state reads the question.** CF-probe rises from .222 to .316 (paired pairs 14 vs 44, p 1e-4), matching hobson.
  - The gains are in binding under distractors: date_order .14 / .16 against hobson's .06 / .03, and status_distract .48 against .25.
  - It loses amount_vs_limit, .05 against .33.
  - CF, REAL-label and JB are unchanged.

## 4. Latency (A10G, exclusive GPU, n=20, fresh real states, median ms, p95 ≤ +1.5%) [M]

**1 question** (state T plus a 92-token real question):

| T | 64 | 128 | 256 | 400 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|
| encoder, fused | 15.5 | 19.0 | 27.8 | 36.7 | 85.8 | 159.8 | 330.2 |
| encoder, 21/26 layers local-512 | | | 27.9 | | 85.6 | 151.8 | 294.4 |
| hobson, fused | 13.2 | 14.7 | 20.8 | 28.1 | 56.7 | 105.2 | 199.2 |
| encoder / hobson | 1.17 | 1.30 | 1.34 | 1.30 | 1.51 | 1.52 | 1.66 |

**4 and 15 questions** (T = 64 / 1000 / 4000):

| case | 64 | 1000 | 4000 |
|---|---|---|---|
| Q4, encoder (one packed pass) | 34.1 | 99.4 | 353.4 |
| Q4, hobson branch per question | 45.5 | 89.4 | 238.8 |
| Q4, hobson one-pass lower bound | 26.8 | 69.9 | 215.9 |
| Q15, encoder | 130.9 | 202.1 | 467.8 |
| Q15, hobson branch per question | 163.1 | 210.1 | 365.9 |
| Q15, hobson lower bound | 90.0 | 138.6 | 279.5 |

The curves are in `results/lat_curves.png`.

- **Where the time goes.** At 156 rows, GEMMs alone take 14.4 of 15.5 ms, at 44 TFLOPS; the weight-streaming floor is about 8 ms. The whole request runs at 54 TFLOPS at 1000 tokens and 61 at 4000. The runtime is at the A10G's GEMM rate, so the gap is FLOPs, not overhead.
- **Crossover: none** against fused hobson, from 64 to 4000 tokens. The encoder beats only hobson's branch-per-question path: up to about 700 state tokens at 4 questions and about 1,300 at 15, because that path streams the weights M+1 times. A packed hobson (lower bound) wins everywhere.
- **Brooker's ~2x advantage was serving overhead.**
- **e1a latency.** The same as e1b for 1 question [A]. For M questions it needs M state passes [A].
- **Long-context fix (question 3): local-512 on 21 of 26 layers, question rows global.**
  - Saving: 0% at 1000 tokens, 5% at 2000, 11% at 4000. The cap is the attention share of FLOPs [A].
  - Zero-shot accuracy cost: LONG agree_sd .842 → .758, CF flip .170 → .126, REAL-label .770 → .765, CF-probe .222 → .244 (noise), JB flat.
  - It is a small saving with a real cost on long states.

## 5. Projections [A]
I applied FAST_DECISION_MODEL's factors for GEMM-bound fp16 accumulation (3090 ×0.52, 4090 ×0.26, 5090 ×0.19) to both models. **fp16 accumulation is untested on the encoder [S].**

| state tokens, 1 question | 3090 encoder / hobson | 4090 encoder / hobson | 5090 encoder / hobson |
|---|---|---|---|
| 256 | 14.5 / 10.8 | 7.2 / 5.4 | 5.3 / 4.0 |
| 1000 | 44.6 / 29.5 | 22.3 / 14.7 | 16.3 / 10.8 |
| 4000 | 172 / 104 | 86 / 52 | 63 / 38 |

## 6. Frontier and verdict
- **Dominated.** At 1000 tokens the encoder takes 85.8 ms against hobson's 56.7. REAL-label is lower (.770 vs .785), CF flip lower (.17–.18 vs .27), and Brier worse on every suite [M].
- **Below the half-latency control.** F7's hob12 (26.6 ms) beats it on agree_sd (.884), CF fgh (.78) and JB-hard (.546) [M].
- **Verdict.** The encoder's speed was a serving artifact. Its cost is set by width (d 2304 and FFN 9216, against 2048 and 6144 plus a cheap GDN), which local attention cannot fix. At this budget, bidirectional reading from a UL2 base reads real states worse than hobson's decoder.

## 7. The single decisive next step
The encoder does not survive. One finding transfers.
- **The finding.** Letting the state read the question (e1a) was the only change here that improved detail binding: +9.4 CF-probe points over the masked model [M].
- **Its cost.** Nothing extra for 1-question requests.
- **Test it at hobson's cost.**
  - Train hobson in question-then-state order (causal, so the state reads the question).
  - Use the F7 recipe and budget, with no CF augmentation.
  - Run a state-then-question control from the same init.
  - Score CF-probe and CF with paired tests, and REAL-label at the .78 bar.
- **What the outcome means [S].** If the gain holds without losing agreement, question-aware reading is worth shipping for the 1-question majority. If it does not, the gain was specific to bidirectionality, and the encoder line is closed.
