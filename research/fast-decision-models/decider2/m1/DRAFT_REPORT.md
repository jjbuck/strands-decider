# M1: all-attention hobson (draft report)

Agent M1, box m1 (A10G), 2026-10-06 22:07 to 2026-10-07 01:40 PDT.

Claims are marked measured, verified (checked against an independent reference) or arithmetic.

## What I built

1. **The conversion** (ARCH.md v1, the shapes N1 is timing). The 18 GDN mixers become causal softmax attention with 16 heads of 128 (MHA). Each is built from its own GDN layer's parameters: the in- and out-projections, the short conv and the gated-RMSNorm output gate.
   - **q/k norm and RoPE:** per-head RMSNorm on q and k with a learned gain, which carries the softmax temperature. RoPE matches hobson's attention layers (64 of 128 dims, theta 1e7).
   - **Gates as biases:** the GDN's write gate, log sigmoid(b_j), and its decay, G_i − G_j with G the cumulative log decay, become attention biases. They ride in 6 of each head's 128 dims, split into 3 bf16 parts, so plain FlashAttention runs the layer.
   - **Bias check (verified):** relative error 1.85e-3 against an explicit fp32 softmax, against 1.99e-3 for plain bf16 SDPA.
2. **Training** (m1train.py).
   - **Stage 1, transfer:** each converted mixer is fit full-rank to its GDN layer's residual contribution, on the teacher's own inputs. All 18 are fit in one teacher pass.
   - **Stage 2, end to end:** KL to hobson's decisions, plus relative MSE at layers 5/11/17/23 on the answer and option rows, plus train_v5 CE + KL, plus at most 10% cf_aug. LoRA r32 elsewhere.
3. **Fused runtime** (m1lat.py): lean2/TTL GEMMs, one Triton prep kernel (conv, SiLU, norm, RoPE, bias dims), FlashAttention and a gated-norm kernel.
   - **Agreement with eager M1 (verified):** argmax 56/56, median |dp| .0011, max .0079.
   - **hobson in this harness (verified):** within 0.5 ms of J3's numbers.

## Permission denials

- **Timer reset denied:** resetting the box's shutdown timer (`sudo shutdown -c; sudo shutdown -h +480`) was denied as "Modify Shared Resources".
- **Training relaunch denied:** relaunching the final end-to-end run without the timer change was also denied.
- I did not retry either; evaluating existing checkpoints was allowed.

**Consequences:** neither variant (a)'s planned 2.75-hour distillation nor the bottom-up staged variant (b) ran. The distilled checkpoints below saw 1.6–2.3M tokens; read them as early points on the learning curve.

## Results against both bars (all 3,227 evalkit questions; paired McNemar against hobson-v19; measured)

Checkpoints: **untrained**; **transfer** (stage 1 only, 20 min, 1.64M tokens); **d3** (transfer plus 52 end-to-end updates, 0.63M tokens, of the small mixer parameters only: gains, decay, conv, norm gain; a diagnostic run).

| metric | hobson | untrained | transfer | d3 |
|---|---|---|---|---|
| JB-all | .723 | .290 | .645 (p .001) | .645 (p .001) |
| JB-hard | .523 | .308 | .408 (p .004) | .415 (p .009) |
| REAL-label | .785 | .440 | .728 | **.770** |
| REAL state-dependent agreement | 1 | .361 | .812 | **.861** |
| LONG state-dependent agreement | 1 | .333 | .824 | .903 |
| CF pair accuracy | .268 | .012 | .219 (p .002) | .192 (p < .001) |
| CF-probe pair accuracy | .328 | .053 | .244 (p .002) | .263 (p .02) |
| CF / CF-probe retention | 1 / 1 | .00 / .04 | .73 / .53 | .69 / .54 |
| REAL-label Brier | .347 | .585 | .388 | .374 |
| RuleTaker depth 3 (300 held-out rows) | .867 | – | .833 (p .076) | .827 (p .023) |
| RuleTaker depth 5 (300) | .750 | – | .687 (p .003) | .700 (p .017) |

**Strict bar:** no checkpoint passes. All have significant JB-all and JB-hard drops, REAL-label below .78, and CF and CF-probe below hobson.

**Relaxed bar:** d3 passes REAL-label (.770 ≥ .765) and REAL state-dependent agreement (.861 ≥ .85). It fails JB-all (.645 < .693), CF (.192) and CF-probe (.263).
- **For scale:** hobson keeping a random 50% of state rows after layer 7 scores .870 (evalkit baseline).
- **Where the CF losses are:** human_insert (.21 against .35) and amount_insert (.01 against .10).
- **Deduction:** RuleTaker drops 3–6 points (p .003–.08).

**Other points:**
- **Merging is not additive.** d3 plus a LoRA-only run's LoRA (checkpoint arithmetic, no training) scores JB-all .662 (p .016), REAL state-dependent agreement .801 and CF .121.
- **The train-split dev set overstated fidelity.** After transfer it gave state-dependent agreement .926 (n = 95), against .812 on REAL, about 4 standard errors apart.

## How far the untrained conversion is from hobson (measured)

The conversion is a warm start, not function-preserving.
- **Layer error:** the converted layers' residual contribution has mean relative MSE 3.9 against the GDN's. Above 1 is worse than adding nothing.
  - The direction is partly right: cosine .26–.62 in layers 0–16 and .75–.93 in layers 17–22.
  - The magnitudes are too large: softmax averages raw values; the delta rule outputs residuals.
  - Dropping the short conv raises the error to 70.
- **One layer converted at a time** (158 dev questions): agreement is .35–.73 for 7 of the 9 layers in 0–10, and ≥ .98 for layers 14–22.

## Learning curves (measured)

| stage | tokens | mean local relative MSE | dev agreement / state-dependent agreement | REAL state-dependent agreement |
|---|---|---|---|---|
| untrained | 0 | 3.93 | .372 / .284 | .361 |
| transfer, step 30 | 0.28M | 0.32 | – | – |
| transfer, step 90 | 0.84M | 0.090 | – | – |
| transfer, end | 1.64M | 0.067 (still falling) | .918 / .926 | .812 |
| d3 | +0.63M | – | KL to hobson .0165 → .0081 (20-request subset) | .861 |

**Why end-to-end distillation drifted** (measured, from the transfer checkpoint).
- **Full recipe:** full-rank updates of the 378M mixer weights at learning rate ≥ 5e-6 pushed the hidden-state error from .033 to .058–.12 within 40–80 updates. A no-update probe confirmed the start was as good as the dev set.
- **Single-factor runs, ~52 updates each:**
  - mixers only (lr 1e-5): KL .0165 → .0151, hidden error worse;
  - LoRA only (5e-5): KL .0132, hidden error better;
  - small parameters only (1e-4): KL .0081, hidden error better.

## Latency against hobson (measured)

Conditions: A10G, bf16, CUDA graph, exact lengths, fresh banking states, 20 warm reps. p95 is within 0.2 ms of the median.

| T | hobson, 1 question (one sequence) | M1 | hobson, 4 questions (state pass + packed questions) | M1 |
|---|---|---|---|---|
| 64 | 15.56 ms | 15.08 (0.97x) | 52.69 | 48.34 (0.92x) |
| 256 | 25.36 | 25.26 (1.00x) | 57.87 | 54.27 (0.94x) |
| 1000 | 57.10 | 56.54 (0.99x) | 95.20 | 93.87 (0.99x) |
| 4000 | 201.34 | 210.99 (1.05x) | 246.08 | 265.38 (1.08x) |

**The long-state cost at 4,000 tokens (measured, profiler):**
- The GEMMs take 169.9 ms in both models.
- M1's attention takes 30.8 ms against 8.2 for hobson. That is 1.26 ms per converted layer, against 1.37 per hobson attention layer.
- M1 removes the 13.1 ms GDN scan.
- **Net:** +9.6 ms with one question, +19.3 ms with four.

These are below the brief's +4% / +14% arithmetic, which counted the added attention but not the removed scan.

**Two fixed implementation costs:** appending the bias dims (136-wide heads) cost 15 ms at 4,000 tokens, and an outer-dimension torch.cumsum cost 10.8 ms. After both fixes the biases cost nothing measurable (210.58 ms without them).

## Projections (arithmetic)

**Method:** GEMMs use the fp16-accumulation rate (J14's roofline split). FlashAttention accumulates in fp32, which runs at half rate on GeForce. Other kernels scale with bandwidth.

M1 / hobson, one question:

| card | T = 64 | 256 | 1000 | 4000 | ms at T = 1000, hobson → M1 |
|---|---|---|---|---|---|
| 3090 | 0.98 | 1.02 | 1.02 | 1.13 | 30.9 → 31.4 |
| 4090 | 0.94 | 0.97 | 0.95 | 1.03 | 16.9 → 16.1 |
| 5090 | 0.97 | 1.00 | 0.98 | 1.08 | 11.8 → 11.6 |

With four questions at 4,000 tokens: 1.18 (3090), 1.08 (4090), 1.13 (5090).

## What this means

- **GPUs:** M1 is latency-neutral to 1,000 tokens and 5–8% slower at 4,000 (measured).
- **Inferentia:** any speed gain has to come from there, where hobson's GDN took 72 of 152 ms (J8). N1 is timing these shapes.
- **Accuracy:** at 2.3M tokens the best checkpoint passes 2 of the relaxed bar's 5 clauses (7.8 points below hobson on JB-all, 7.6 on CF pairs).
- **Learning:** 52 small-parameter updates moved REAL state-dependent agreement from .812 to .861 and REAL-label from .728 to .770, so the curve was still steep where training stopped.

## Single most valuable next step

If N1 shows M1 is faster on Inferentia, run the blocked ~6-hour distillation with this recipe:
- start from the transfer checkpoint;
- train LoRA r32 and the small mixer parameters;
- freeze the full-rank mixer weights, or hold them at ≤ 2e-6 with the stage-1 local loss as an anchor;
- include train_v5 CE + KL for JevBench.

This needs permission to launch the run and a box lifetime that covers it plus about 1 hour of evaluation.

## Files

~/decider2/m1/: ARCH.md, NOTES.md, code/, results/, preds/. Checkpoints are on box m1 only (~/work/m1/ck_transfer, diag/), until its 08:07 PDT poweroff.
