# J10: a natively ternary 2B decider (BitNet b1.58 2B4T)

Files: `~/decider2/j10/` (`code/`, `results/`, `scores_j10.json`, `res_lat_dedup.jsonl`, `proj_j10.md`, `NOTES.md`). **[M]** measured, **[V]** verified source, **[A]** arithmetic, **[S]** speculation.

## 1. Hypothesis and first principles
**The bet.** Ternary weights are exact in s4 and FP4. If activations born low-bit were also 4-bit-safe, a 2B decider would run at the int4 or FP4 tensor rate (4x bf16) without the rounding that broke post-hoc 4-bit on hobson. Short prompts would also gain from about 8x fewer weight bytes.

**The arithmetic before any run** [A]:
- **Model size** [V, config]: BitNet 2B4T has 30 layers, d 2560, a ReLU² GLU of 6912, and full attention in every layer.
  - That is 2.085B GEMM weights, or **4.17 GFLOP per token against hobson's 2.77 (1.5x)**.
  - It has 30 attention layers to hobson's 6.
- **No ternary tensor core exists.** W1.58A8 runs on int8 MMA, the same rate as hobson's W8A8, which already passes fidelity. So BitNet-A8 should be slower than hobson-W8A8 wherever prefill is compute-bound.
- **The weight-byte saving should be worthless.** The A10G's int8 ridge is about 190 FLOP/byte, so 8-bit weights stop binding above about 95 rows. Every decision request carries a question of at least about 85 tokens.
- **What would change the cost class** is W1.58A4, and only if natively ternary activations tolerate 4 bits.

## 2. Why it is not marginal
**The upside.** It removes the doc §5 error budget by construction, for 4x GEMM throughput on Ampere and Ada and FP4 on Blackwell at 2B.

**The downside.** A clean failure closes the low-bit-from-scratch branch.

## 3. What I built
1. **Verified the published quality** [V, arXiv 2504.12285 Table 2].
   - BitNet 2B4T averages 55.01. Qwen2.5-1.5B scores 55.72 in bf16, 52.15 with GPTQ-int4 and 51.17 with AWQ-int4 (weight-only).
   - MMLU is 53.2 against 60.3.
   - Born-low-bit beats post-hoc int4 here, but its knowledge sits below a bf16 1.5B.
2. **My own torso.** It matches HF BitNetForCausalLM at hidden cosine 0.999 and equal LM loss [M].
3. **The decider, by F7's recipe.**
   - Data: F7's 102,547 labelled rows plus 3,600 KL-only real states, 30.2M tokens, 2.4 h.
   - Teacher logits are F7's own hobson logits. I re-rendered the rows with F7's seeds, and the Qwen ids are identical to F7's on every row.
   - **LoRA r16 lives in the latent weights.** The deployed weights are ternary(W0 + 2BA) per tensor, so the model stays exactly ternary. A low-rank straight-through backward avoids dense weight gradients. 0.33% of the codes moved [M].
4. **Kernels.** bitnet.cpp's GPU kernel is a decode-only GEMV [V], so I wrote my own prefill runtime (`rt_j10.py`), under a CUDA graph:
   - fused Triton glue;
   - a Triton int8 GEMM, and a **W2A8 GEMM** that unpacks 2-bit weights to int8 MMA in registers;
   - CUTLASS s8 and s4 GEMMs.

   Fidelity on 40 real questions: hidden cosine ≥ 0.99994 against the trained model, argmax 38 of 40, both flips at TV ≤ 0.024 [M].
5. **The 4-bit study.** Crest factors, A4 emulation, and a 500-step A4 QAT.
6. **BitDistill-lite.** hobson ternarised post hoc, then the same latent-LoRA QAT with KL from hobson (31 min).

## 4. Accuracy against hobson [M]

| suite | ternary decider | hobson | paired |
|---|---|---|---|
| JB-hard | .385 | .523 | McNemar 8/26, **p .003** |
| JB-all | .619 | .723 | 9/33, **p .0003** |
| REAL-label | .733 | .785 | 13/34, **p .003** |
| REAL agree / agree_sd | .870 / .803 | 1 / 1 | F7 0.8B: – / .824 |
| LONG agree / agree_sd | .879 / .818 | 1 / 1 | no long-state labels exist |
| CF acc / flip / fgh | .576 / .254 / .596 | .594 / .268 / 1 | 48/62, p .21 |
| CF-probe acc / flip / fgh | .589 / .203 / .40 | .653 / .328 / 1 | 83/124, **p .005** |
| SHUF change | .323 | .339 | |
| Brier JB-all / REAL-label | .464 / .392 | .348 / .347 | ECE .112 vs .116 |

- **The bar fails on JB-hard, JB-all, REAL-label and CF-probe.**
- **The losses are spread across every reading family:** multi_hop 5/8, long_policy 4/7, adequacy 7/10, ordinal 8/12. JB-hard sits below hobson's no-state baseline (.462).
- **In-distribution held-out sets show the same gap:** synthetic documents score .79 / .68 against hobson's .88 / .78. The one gain is CF amount edits, .32 against .10.
- **Overall it lands below F7's Qwen3.5-0.8B decider** (JB-hard .469, REAL-label .767, agree_sd .824).

**hobson made ternary (BitDistill-style)** [M]:

| | JB-all | REAL-label | REAL agree_sd | CF acc |
|---|---|---|---|---|
| post hoc, untrained (W1.58A8) | .299 | .400 | .202 | .438 |
| + latent-LoRA QAT, 8.1M tokens | .429 | .655 | .451 | .477 |

The QAT recovers about a third of the gap, at roughly 0.1% of BitDistill's ~10B-token continued-pretraining budget. It fails at this budget.

## 5. Latency [M]
A10G, exclusive; fresh ids, H2D + graph replay + D2H; median of 30, p95 within 0.3%. 1 question = 85 tokens; 4 = 408, packed with a masked state. hobson: same box, the doc's fused lean2 runtime.

| ms, 1 question | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| hobson bf16 (same box) | 15.5 | 15.9 | 25.3 | 32.8 | 57.1 | 201.6 |
| hobson, doc (bf16 / W8A8) | 9.4 | – | 15.7 | – | 52.7 / 36.3 | 200.5 / 142 |
| ternary, bf16 MMA | 21.2 | 21.1 | 32.7 | 39.5 | 89.6 | 330.9 |
| W1.58A8, int8 codes (Triton) | 13.1 | 15.7 | 23.4 | 27.2 | 65.1 | 242.7 |
| W1.58A8, 2-bit packed (Triton) | 12.7 | 16.7 | 23.0 | 26.6 | 60.7 | 231.5 |
| **W1.58A8, CUTLASS int8** | **10.6** | **12.0** | **18.1** | **21.1** | **48.3** | **206.1** |
| A4 qkv + gate/up, CUTLASS | 8.0 | 8.9 | 13.7 | 16.0 | 37.2 | 160.1 |
| A4 all GEMMs (inaccurate) | 6.6 | 7.5 | 11.0 | 13.1 | 30.1 | 137.4 |

| ms, 4 questions | 64 | 256 | 1000 | 4000 |
|---|---|---|---|---|
| hobson bf16 (one causal pass, cost proxy) | 32.9 | 43.7 | 83.6 | 215.8 |
| W1.58A8, CUTLASS int8 | 22.8 | 33.4 | 67.5 | 246.5 |
| A4 all GEMMs | 15.0 | 21.9 | 44.5 | 172.9 |

- **Weight bytes buy nothing.** 4x fewer bytes (2-bit against int8 codes, same kernel family) gains 3% at 149 rows and 7% at 1,085 rows.
- **Long prompts lose to attention.** At 4000 tokens, attention is 45 ms (30 layers) against hobson's whole 201.6 ms bf16 pass.
- **At equal precision the ternary model is slower:**
  - accurate W1.58A8 is 48.3 ms at 1000 tokens against hobson-W8A8's 36.3;
  - A4 on all GEMMs is 30.1 ms against hobson-W4A4's 23.5.
- **The only wins are against bf16 hobson on the same box:** 1.46x at 64 tokens and 1.18x at 1000. The doc's 9.4 ms anchor at 64 tokens, with a shorter question, beats even that.

**Projections** use H2's method and device table [A]. ms at 1000 / 64 tokens, 1 question:

| | 3090 | 4090 | 5090 |
|---|---|---|---|
| W1.58A8, CUTLASS int8 | 24.9 / 5.4 | 13.0 / 2.7 | 9.3 / 2.0 |
| A4 qkv + gate/up | 19.4 / 4.2 | 10.6 / 2.2 | 7.5 / 1.6 |
| A4 all (5090: FP4, needs block scales) | 15.9 / 3.5 | 9.0 / 2.0 | 6.2 / 1.4 |
| hobson W8A8 (doc, schema layout) | 18.3 | 10.7 | 7.3 |

## 6. Does born-ternary make 4 bits safe? [M]
- **Crest factor (max/rms per token) at the GEMM inputs of base BitNet:**

  | GEMM input | median crest | p99 crest |
  |---|---|---|
  | qkv | 5.2 | 10.3 |
  | o | 6.1 | 11.2 |
  | gate/up | 4.1 | 9.2 |
  | **down** (after ReLU²) | **30.7** | **71** |

  hobson's is 15–40. Per-token int4 rounding error is 18–25%, and 39% on down; 16-element block scales bring it to 6–9%.
- **Flips against the A8 model:**

  | A4 configuration | REAL | CF-probe |
  |---|---|---|
  | per token, all GEMMs | 16.2% | 32% |
  | qkv + gate/up only | 7.1% | 19% |
  | block-16 on all GEMMs | 3.5% | 7.2% |

  hobson post-hoc: 9–13.5% with all GEMMs at 4 bits; 2.2–2.6% with 53% at 4 bits.
- **The rotation trick is unavailable.** Ternary weights cannot absorb a Hadamard rotation (WH is not ternary), and that trick is what rescued hobson.
- **A4 QAT for 500 steps** on qkv + gate/up:

  | | JB-hard | REAL-label | REAL agree_sd | CF-probe acc |
  |---|---|---|---|---|
  | after A4 QAT | .454 | .728 | .731 | .583 |
  | A8 model | .385 | .733 | .803 | .589 |

  REAL agree_sd stays where A4 without QAT left it (.734).

## 7. Verdict: negative on both axes
- **Accuracy** [M]. The natively ternary 2B is significantly worse than hobson, and below the F7 0.8B decider. Its 4T-token ternary pretraining buys roughly 1–1.5B-class knowledge [V].
- **Cost class** [M, A]. Ternary does not change the cost class of prefill on tensor-core GPUs.
  - The MMA rate follows activation precision, and the question tokens alone make prefill compute-bound.
  - BitNet's 1.5x FLOPs and 30 attention layers make it slower than hobson at equal precision. That is measured at 1000 and 4000 tokens, and arithmetic at shorter lengths.
- **The premise.** Native A8 training gives low crest factors on three of four inputs, but not int4-safe ones. The ReLU² down-projection is as peaked as hobson's worst.
- **What would be transformative:** W1.58 + block-scaled FP4 activations, trained at A4 from the start (BitNet v2's online Hadamard [V, arXiv 2504.18415]), on a base as strong as Qwen3.5-2B. Neither exists [S].

## 8. Single decisive next step
Nothing further on Ampere or Ada.

On Blackwell, run one test: full-weight A4 QAT of hobson to W1.58 + NVFP4 (BitDistill with billions of continued-pretraining tokens). Gate it on ≤1% REAL flips and CF-probe fgh ≥ 0.95. If it fails, close the branch.
