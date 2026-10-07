# Fast decision models

This package is the record of a research program on decision latency, run in October 2026. It asked how fast Strands
Decider 2B (hobson-v19) can answer a request whose state is about 1,000 tokens long, on an RTX 3090-class GPU, without
losing its accuracy, and what it would take to get to about 10 ms per request.

hobson-v19 is a Qwen3.5-2B model with a pointer head: 24 layers of width 2,048, of which 18 are Gated DeltaNet (GDN, a
linear-attention recurrence) and 6 are full attention. It answers yes/no, multiple-choice and score questions about an
agent's state in one forward pass, and returns calibrated probabilities without generating text. Every idea in the program
keeps the model at hobson's size; smaller models appear only as controls. Several dozen research agents ran the
experiments, in rounds D to H and then J, Q, M and N. The full write-up is [FAST_DECISION_MODEL.md](FAST_DECISION_MODEL.md).

## What was found

These are the main results from the report's [Summary](FAST_DECISION_MODEL.md#summary). Latencies were measured on an
NVIDIA A10G unless they are marked as projected. "8-bit" means the W8A8 runtime of idea P1. The fidelity bar is defined
under [Hardware and measurement conventions](#hardware-and-measurement-conventions).

- **Engineering alone made hobson 2.8x faster with the same decisions (S1).** A fused runtime with one CUDA graph per
  request length took a request with a 1,000-token state from 148.7 ms to 52.7 ms, and a 4,000-token request from about
  520 ms to 200 ms. The stock engine launched 3,054 GPU kernels per request, and 74% of them came from the LoRA adapter
  being applied unmerged.
- **The time that remains is matrix-multiply arithmetic.** At 1,000 tokens, matrix multiplies take 87% of the 52.7 ms,
  attention takes 1.7%, and all token mixing together about 10%. A faster attention kernel or a new sequence mixer can
  therefore save a few percent at most. Latency falls substantially only with fewer rows (tokens), fewer layers per row,
  or cheaper arithmetic per row.
- **Two speedups keep hobson's decisions as reliable as its own runtime.** 8-bit weights and activations with a Hadamard
  rotation (P1) run 1.57x faster than bf16 at 1,000 tokens (36.3 ms against 57.0 ms). An exit at layer 16 (R1) changed 0 of
  6,216 real decisions, and its mean latency on 120 recorded requests was 0.688x the 8-bit runtime's.
- **The best configuration that keeps hobson's decisions is C1 (P11 + R1).** The 32 least sensitive of hobson's 96
  matrix multiplies run in 4-bit on the state's tokens, the rest stay in 8-bit, and the model exits at layer 16 when the
  exit head is confident. On 120 recorded requests its mean latency was 50.6 ms against 79.7 ms for the 8-bit runtime
  (0.635x), and 0.65% of real decisions differed from hobson's, inside the 0.7% bar. At 1,000 tokens it projects to
  13.1 ms on an RTX 3090, 7.8 ms on a 4090 and 5.3 ms on a 5090.
- **Accurate 4-bit arithmetic everywhere is blocked by one error source:** rounding the activations of the state's tokens
  in layers 0–11 (P10, B1–B6). It resisted every format and training method tried. The error is concentrated by token:
  the 5% most sensitive state tokens carry 63% of the decision's sensitivity to it, so the open lead is precision chosen
  per token by the question (B1).
- **Pruning chosen by the decision's own sensitivity works in layers 12–23 (B11–B13).** Decision-aware 2:4 sparsity and
  per-deployment neuron removal have 2–40x lower decision error than the standard per-layer criteria, keep decisions inside
  the fidelity bar, and bring matrix-multiply time to 0.82–0.92x of the 8-bit runtime's. Two exact eliminations (B13)
  save another 4% of latency.
- **Removing rows gives 2–5x, but not yet at hobson's accuracy.** Precompiling a deployment's fixed documents (A6) or
  questions (A3, A4), or a domain vocabulary (A7), cut latency 2–5x. After LoRA retraining on 3–40M tokens, each fell
  short of the accuracy bar, mostly on two 23-way procedure questions.
- **A decision-native layout met an accuracy bar for the first time (M2).** State tokens attend only within their own
  segment (a message, a tool output, a document) for 12 layers, while the question's tokens read every segment through
  all 24. It is 1.7–1.8x faster than hobson on recorded requests in bf16. Documents become exactly precomputable: with
  every document computed once and spliced in, 228 of 228 decisions were unchanged.
- **hobson reads details poorly, and training data fixes it (E1).** hobson gets both items of a counterfactual pair right
  on 27% of pairs. Adding counterfactual examples to training, with no change to the architecture, raised the
  synthetic-probe (CF-probe) pair accuracy from .328 to .95–.99.
- **Other hardware.** Inferentia2 runs hobson in 132 ms at 1,000 tokens on one core, at 0.83x the A10G's cost per decision
  (HW3). AMX CPUs cost 12–16x the A10G's cost per decision, and an Apple M4 Pro takes 469 ms (HW1, HW2).
- **Where 10 ms at 1,000 tokens stands.** On an RTX 4090, C1 projects to 7.8 ms and plain 8-bit to 11.6 ms. On an RTX
  3090 no measured configuration gets there: C1 projects to 13.1 ms. Getting below 10 ms on a 3090 needs C1 combined with
  fewer rows (M2 or A6 compiled documents, A7 vocabulary), or exact context parallelism over 4 GPUs (S5, predicted at
  8–10 ms). Neither has been run.

## How the package is laid out

| Path | Contents |
| --- | --- |
| [`FAST_DECISION_MODEL.md`](FAST_DECISION_MODEL.md) | The report: every idea, what was measured, the evaluation kit and bars, the wall-time floor, recommended next steps. |
| [`figures/`](figures/) | `fig1`–`fig7`, the result figures; `ideas/d01`–`d22`, the concept diagrams; `data/*.csv`, the numbers plotted in `fig1`–`fig7`. |
| [`docs/`](docs/) | [`CPU_DECISION_MODEL_MEMO.md`](docs/CPU_DECISION_MODEL_MEMO.md), the CPU analysis and a proposed CPU experiment; [`STEERING_BENCH_LITERATURE.md`](docs/STEERING_BENCH_LITERATURE.md), the companion literature review. |
| `decider2/` | The research tree: code, results, training recipes and agent notes. `decider2/MANIFEST.md` lists what is included and `decider2/EXCLUDED.md` what was left out. |

The report refers to this tree as `~/decider2/`, the location it had on the laptop where the program ran. In this package
it is the `decider2/` folder next to the report. Agent names in the report (J15, Q2, ...) are the lower-case folder names
(`j15/`, `q2/`). Inside `decider2/`:

| Folder | Contents |
| --- | --- |
| `evalkit/` | The evaluation kit: suites, hobson's reference outputs, the train/eval split and the scorer (`evalkit.py`). |
| `analysis/` | The post-processing pipeline that scores every agent's predictions and draws the figures, and the diagram scripts. |
| `d1/`, `d2/`, `systems/` | D round: the fused bf16 runtime (`d1/lean2.py`), an int8 matrix multiply, and the packed multi-question forward. |
| `h1/`–`h7/` | H round: 8- and 4-bit formats, the low-bit runtime, error propagation, this-that-model, learned rotations, mixed-row precision, the schema-first fine-tune. |
| `F7_REPORT.md` | F round: the equal-latency smaller-model controls and the fine-tuning recipe most later agents reused. The other F agents' folders are not in the tree. |
| `j1/`–`j15/` | J round: encoder, bidirectional GDN, depth split, pretraining, short-input kernels, compiled questions, vocabulary, Inferentia, compiled documents, BitNet, slot model, comparator, adaptive width, compiled question rows, early exit. |
| `q1/`–`q5/` | Q round: the 4-bit program (BRIEF10, ideas B1–B14, P10, P11, C1). |
| `m1/`, `m2/`, `n1/` | M and N rounds: decision-native mixers and layouts, and the Inferentia GDN kernel (BRIEF11). |
| `k1/`, `k2/` | The combined stack with ablations (BRIEF9): its library, trainer and runtime, paused and kept for resuming. |
| `recovered/` | Copies of the working directories of five GPU boxes (`g1/`–`g5/`), recovered after the laptop reboot that lost the original tree. They hold G1's structured-matrix code (R8), G2's per-token precision emulation that the H round built on, G3's mixture-of-experts code (R7), G4's fixed-budget pretraining code, and copies of D1's profiling and fusion scripts. |
| `shrink/`, `tokens/`, `training/` | Early-round shared code: depth truncation, token selection and eviction, the training mixes and JevBench evaluation. |
| `bundle/` | `setup_box.sh`, the script that set up the Python environment on each GPU box. |
| `BRIEF7.md`–`BRIEF11.md` | Each round's rules, hypotheses and pass bars. |

Most agent folders have a `NOTES.md` log and a `DRAFT_REPORT.md` (`REPORT.md` in most of the H round), which hold the
full results behind each entry of the report.

## Where each idea's code is

The table maps every idea ID in the report to the agent folders that did the work (inside `decider2/`) and the key files,
with paths relative to this package. "Training" includes calibration, data generation and sensitivity ranking. "Runtime,
kernels or analysis" includes the latency harness. "Results" names score and latency files; the agent's `NOTES.md` and
`DRAFT_REPORT.md` explain them. A dash means there is no file of that kind, and "not located" means the work was done but
its code is not in the tree.

| ID | Idea | Agent folders | Training | Runtime, kernels or analysis | Results |
| --- | --- | --- | --- | --- | --- |
| S1 | Lean fused runtime | `d1/`, its box copy `recovered/g1/d1/` | – | `decider2/d1/lean2.py`, `decider2/recovered/g1/d1/bench_d1.py` | `decider2/recovered/g1/d1/NOTES.md`, `decider2/recovered/g1/d1/results/` |
| S2 | Faster or replaced attention | `d1/`, `j1/` | – | `decider2/recovered/g1/d1/prof_d1.py`, `decider2/j1/code/encrt.py` | `decider2/recovered/g1/d1/results/prof_base_256_1000_4000.json`, `decider2/j1/results/lat_enc_loc512.json`, `decider2/j1/results/scores_A_loc512.json` |
| S3 | Megakernel | `d1/`, `j5/`; B14 for `q4/`'s kernel | – | `decider2/recovered/g1/d1/mega.py`, `decider2/j5/code/ramp.py` | `decider2/recovered/g1/d1/results/mega_L4.json`, `decider2/j5/res_ramp.json` |
| S4 | Layer streaming, co-scheduling | D round; not located | – | not located | – |
| S5 | Exact context parallelism on 4 GPUs | F6; its code was lost with the original tree | – | not located | – (predictions only, in the report) |
| S6 | One packed pass for several questions | `systems/`, `j8/` | – | `decider2/systems/g/packed.py`, `decider2/systems/g/test_packed.py`, `decider2/j8/code/hob.py` | `decider2/systems/g/bench3_2b_single_packed4_host.json` |
| S7 | fp16 accumulation | `d1/` | – | `decider2/recovered/g1/d1/fp16acc_08b.py` | `decider2/recovered/g1/d1/NOTES.md` |
| S8 | Rust tokenizer | `d1/` | – | `decider2/recovered/g1/d1/host_d1.py` | `decider2/recovered/g1/d1/results/host_tok.json`, `decider2/recovered/g1/d1/results/host_e2e.json` |
| S9 | Activation sparsity, library 2:4 | early round; not located | – | not located (`q4/`'s custom 2:4 kernels are under B11) | – |
| S10 | Strassen–Winograd | `q4/` | – | `decider2/q4/code/q4st.cu`, `decider2/q4/code/q4st.py`, `decider2/q4/code/q4bit.py` | `decider2/q4/res/res_strassen.jsonl`, `decider2/q4/res/res_bit.json` |
| S11 | Decisions computed as the conversation happens | proposed, not built; the text-reuse measurement script was not located | – | – | – |
| P1 | W8A8 with rotation and GPTQ | `h1/`, `h2/`, `j5/`, `j15/` | `decider2/h1/FORMAT.md`, `decider2/h1/code/h1calib.py`, `decider2/h1/code/h1sens.py`, `decider2/h1/code/h1select.py`, `decider2/h2/code/h2gptq.py`, `decider2/j15/code/j15gptq.py` | `decider2/h2/code/qrt.py`, `decider2/h2/code/qk.py`, `decider2/h2/code/qgemm.py`, `decider2/j5/code/j5rt.py`, `decider2/j5/code/sk.py` | `decider2/h2/scores.json`, `decider2/h2/res_bench_v4.jsonl`, `decider2/h1/res/precmap_w8a8_b8.json` |
| P2 | W4A4 | `h1/`, `h2/`, `h3/`, `h6/` | `decider2/h1/code/h1sens.py`, `decider2/h1/code/h1select.py` | `decider2/h2/code/qrt.py`, `decider2/h2/code/g2s4.cu`, `decider2/h6/code/qrt6.py`, `decider2/h3/code/errprop.py` | `decider2/h2/scores.json`, `decider2/h6/scores.json`, `decider2/h3/res/errA_int4_rot.json` |
| P3 | 4-bit weights only (W4A16) | `j5/`, `j10/` | `decider2/j5/code/wq.py` | `decider2/j5/code/j5rt.py`, `decider2/j5/code/sk.py` | `decider2/j5/scores_wq.json`, `decider2/j5/scores_wq_table.md` |
| P4 | NVFP4 block scales (emulated) | `h5/`, `h3/` | – | `decider2/h5/code/h5fp4.py`, `decider2/h3/code/errprop.py` | `decider2/h5/scores.json`, `decider2/h5/box/res/fp4.json`, `decider2/h3/res/errA_nvfp4.json` |
| P5 | Learned rotations (SpinQuant) | `h5/` | `decider2/h5/code/h5rot.py`, `decider2/h5/code/h5rotl.py`, `decider2/h5/code/h5kron.py` | `decider2/h5/code/h5build.py`, `decider2/h5/code/h5suite.py` | `decider2/h5/scores.json` |
| P6 | LoRA quantization-aware training | `h1/`, `h5/`, `h6/`; earlier, G2 in `recovered/g2/` | `decider2/h1/code/h1qat.py`, `decider2/h5/code/h5qat.py`, `decider2/h6/code/h6qat.py`, `decider2/recovered/g2/g2/g2qat.py` | – | `decider2/h1/res/curve_eval.json`, `decider2/h6/scores.json` |
| P7 | Hyperspherical layers | `h3/` | `decider2/h3/code/train_h3.py`, `decider2/h3/code/pretrain_tiny.py` | `decider2/h3/code/errprop.py` | `decider2/h3/scores.json`, `decider2/h3/res/errA_int4.json`, `decider2/h3/res/tiny_evalq.json` |
| P8 | Speculative precision | `j15/` | `decider2/j15/code/j15learn.py` | `decider2/j15/code/j15run.py`, `decider2/j15/code/j15bench.py`, `decider2/j15/code/casc2.py` | `decider2/j15/res/analysis.json`, `decider2/j15/res/costs.json` |
| P9 | Native ternary 2B (BitNet b1.58) | `j10/` | `decider2/j10/code/train_j10.py`, `decider2/j10/code/train_j10_lib.py`, `decider2/j10/code/tern_hob.py` | `decider2/j10/code/bitnet_j10.py`, `decider2/j10/code/rt_j10.py` | `decider2/j10/scores_j10.json`, `decider2/j10/crest_j10.json` |
| P10 | Row-role precision | `q1/`, `q2/` | – | `decider2/q1/code/q1fmt.py` (emulation), `decider2/q2/code/q2rt.py`, `decider2/q2/code/q2gemm.cu` | `decider2/q2/res/scores.json`, `decider2/q1/res/scores.json` |
| P11 | k64rr mixed 8/4-bit | `q2/`, `h1/`, `h6/` | `decider2/h1/code/h1sens.py` (GEMM ranking), `decider2/q2/code/q2map_k64rr.json` | `decider2/q2/code/q2rt.py`, `decider2/q2/code/q2bench.py` | `decider2/q2/res/scores.json`, `decider2/q2/res/gemm_tables.json` |
| R1 | Exit at layer 16 | `j15/` | `decider2/j15/code/j15learn.py`, `decider2/j15/code/mkdev.py` | `decider2/j15/code/j15run.py`, `decider2/j15/code/j15bench.py`, `decider2/j15/code/exit_an.py`, `decider2/j15/code/sweep_exit.py` | `decider2/j15/res/exit_sweep.json`, `decider2/j15/res/depth_scores.json` |
| R2 | Depth split | `j3/` | `decider2/j3/code/train_dt.py`, `decider2/j3/code/dtlib.py`, `decider2/j3/code/dtset.py` | `decider2/j3/code/dt_lat.py`, `decider2/j3/code/costmodel.py` | `decider2/j3/results.json`, `decider2/j3/scores.json`, `decider2/j3/lat_dt.json` |
| R3 | Token selection and routing | F1, F2: not located; earlier selection and eviction scripts are in `tokens/` | `decider2/tokens/heal.py` | `decider2/tokens/select_exp.py`, `decider2/tokens/qr_sweep.py` | `decider2/tokens/select_res.json` |
| R4 | Learned pooling | F3: not located | – | – | – |
| R5 | Small reader feeding the 2B | F4: not located | – | – | – |
| R6 | Token-adaptive width | `j13/`; earlier, G2's per-token width in `recovered/g2/` | `decider2/j13/code/fit.py`, `decider2/j13/code/heal.py`, `decider2/recovered/g2/g2/g2train.py` | `decider2/j13/code/sens.py`, `decider2/j13/code/route.py`, `decider2/j13/code/tw.py`, `decider2/j13/code/mixed.py`, `decider2/j13/code/lat.py` | `decider2/j13/results.json`, `decider2/j13/lat.json`, `decider2/j13/spectra.json` |
| R7 | MLP pruning, mixture-of-experts | G3 in `recovered/g3/`; the early-round pruning is not located | `decider2/recovered/g3/g3/train_g3.py` | `decider2/recovered/g3/g3/moe.py`, `decider2/recovered/g3/g3/moe_lean.py` | `decider2/recovered/g3/g3/NOTES.md` (G3 produced no measurements) |
| R8 | Structured matrices | G1 in `recovered/g1/` | `decider2/recovered/g1/g1/train_g1.py` | `decider2/recovered/g1/g1/btt.py`, `decider2/recovered/g1/g1/kbench.py`, `decider2/recovered/g1/g1/g1lib.py` | `decider2/recovered/g1/g1/res/`, `decider2/recovered/g1/g1/kb1.jsonl` |
| R9 | Looped layers | `j11/` (its belief arm); the early toy tests are not located | `decider2/j11/code/train.py` | `decider2/j11/code/models.py` | `decider2/j11/runs/S_belief/score.json` |
| R10 | Smaller models (controls) | F7, reconstructed in `F7_REPORT.md`; G3 replicated its recipe | `decider2/F7_REPORT.md`, `decider2/recovered/g3/g3/train_g3.py` | `decider2/recovered/g3/g3/f7code/lat.py`, `decider2/recovered/g3/g3/f7code/models.py` | `decider2/F7_REPORT.md` |
| B1 | Precision where the decision looks | `q1/`, `q2/` | – | `decider2/q1/code/q1spec.py`, `decider2/q1/code/q1router.py`, `decider2/q2/code/q2gemm.cu` | `decider2/q1/sensitivity.json`, `decider2/q1/res/spec/`, `decider2/q1/res/router_w4q8.json` |
| B2 | Dithered rounding | `q1/` | – | `decider2/q1/code/q1fo.py`, `decider2/q1/code/q1fmt.py` | `decider2/q1/res/fo_summary.json`, `decider2/q1/res/dm/srq8.json` |
| B3 | Prototype subtraction | `q1/`, `q2/` | `decider2/q1/code/mkproto.py` | `decider2/q1/code/q1b3.py`, `decider2/q2/code/q2pro.py`, `decider2/q2/code/bench_pro.py` | `decider2/q1/res/res_b3.json`, `decider2/q1/res/dm/proto256fq8.json` |
| B4 | Decision-level rounding and QAT | `q3/` | `decider2/q3/code/q3train.py`, `decider2/q3/code/q3lib.py`, `decider2/q3/code/q3gptq.py` | `decider2/q3/code/q3eval.py` | `decider2/q3/scores.json`, `decider2/q3/scores_table.md`, `decider2/q3/curve_scores.json` |
| B5 | Decision-weighted transform coding | `q1/`, `q2/` | `decider2/q1/code/q1b5.py` | `decider2/q2/code/q2gemm.cu`, `decider2/q2/code/basis_cost.py` | `decider2/q1/res/b5/`, `decider2/q2/res/res_basis_cost.json` |
| B6 | Self-certifying 4-bit decisions | `q1/` | – | `decider2/q1/code/q1fo.py` | `decider2/q1/res/fo_summary.json` |
| B7 | Per-deployment calibration | `q5/` | `decider2/q5/code/q5cal.py`, `decider2/q5/code/q5hcal.py` | `decider2/q5/code/q5dev.py` | `decider2/q5/res/dev_res.json`, `decider2/q5/res/scores.json` |
| B8 | More bits for value tokens | `q1/`, `q2/` | – | `decider2/q1/code/q1tok.py`, `decider2/q1/code/q1fo.py`, `decider2/q2/code/q2rt.py` | `decider2/q1/res/fo_summary.json`, `decider2/q2/res/scores.json` |
| B9 | Sampled matrix multiplies | `q1/`, `q2/` | – | `decider2/q1/code/q1fmt.py`, `decider2/q2/code/q2gemm.cu` | `decider2/q1/res/dm/b9f75q8.json` |
| B10 | Precision on option differences | `q1/` | – | `decider2/q1/code/q1fo.py` | `decider2/q1/res/fo_summary.json` |
| B11 | Decision-aware 2:4 sparsity | `q5/`, `q4/` | `decider2/q5/code/q5cal.py`, `decider2/q5/code/q5lib.py` | `decider2/q4/code/q4sp.cu`, `decider2/q4/code/q4sp.py`, `decider2/q4/code/bench_sp.py`, `decider2/q4/code/tune_sp.py` | `decider2/q5/res/scores.json`, `decider2/q4/res/gemm_time_b11.json`, `decider2/q4/res/res_b11_e2e.jsonl` |
| B12 | Per-deployment neuron removal | `q5/` | `decider2/q5/code/q5cal.py`, `decider2/q5/code/q5lib.py` | `decider2/q5/code/q5nstat.py`, `decider2/q5/code/q5gemmtime.py` | `decider2/q5/res/res_nstat.json`, `decider2/q5/res/res_gemmtime_1125.json`, `decider2/q5/res/scores.json` |
| B13 | Exact dead-computation elimination | `q4/`, `q5/`, `q1/` | – | `decider2/q4/code/q4rt.py`, `decider2/q5/code/q5b13.py`, `decider2/q1/code/q1b13.py` | `decider2/q4/res/res_b13_exact.json`, `decider2/q4/res/res_b13_e2e.jsonl`, `decider2/q5/res/res_b13.json` |
| B14 | Persistent short-request kernel | `q4/`, `j5/` | – | `decider2/q4/code/q4pk.cu`, `decider2/q4/code/q4pk2.cu`, `decider2/q4/code/bench_pk.py`, `decider2/q4/code/pk_barrier.py` | `decider2/q4/res/res_pk.jsonl`, `decider2/q4/res/res_pk2.jsonl` |
| A1 | 2B bidirectional encoder | `j1/` | `decider2/j1/code/train_j1.py`, `decider2/j1/code/prep_j1.py` | `decider2/j1/code/encj1.py`, `decider2/j1/code/encrt.py`, `decider2/j1/code/lat_enc.py` | `decider2/j1/results/scores_A.json`, `decider2/j1/results/scores_B.json`, `decider2/j1/results/lat_enc_v2.json` |
| A2 | Bidirectional GDN | `j2/` | `decider2/j2/code/j2train.py`, `decider2/j2/code/j2lib.py`, `decider2/j2/code/j2data.py` | `decider2/j2/code/j2lat.py` | `decider2/j2/score_final.json`, `decider2/j2/lat.json` |
| A3 | Question-first compiled schema | `h7/`; F5's variant is not located | `decider2/h7/code/train_h7.py`, `decider2/h7/code/h7lib.py`, `decider2/h7/code/gen_cf.py` | `decider2/h7/code/h7lat.py`, `decider2/h2/code/qrt.py` (compiled schema in the low-bit runtime) | `decider2/h7/scores.json`, `decider2/h7/lat_h7.json` |
| A4 | Questions compiled into weights | `j6/` | `decider2/j6/code/j6lib.py`, `decider2/j6/code/train_a.py`, `decider2/j6/code/train_e.py`, `decider2/j6/code/train_h.py`, `decider2/j6/code/train_p.py` | `decider2/j6/code/lat_j6.py` | `decider2/j6/scores.json`, `decider2/j6/scores_hyper_heldout.json`, `decider2/j6/lat_final.json` |
| A5 | Compiled question rows | `j14/` | `decider2/j14/code/j14train.py`, `decider2/j14/code/j14lib.py` | `decider2/j14/code/j14rt.py`, `decider2/j14/code/j14lat.py` | `decider2/j14/score_ladder.txt`, `decider2/j14/lat_table.txt` |
| A6 | Precompiled deployment documents | `j9/` | `decider2/j9/train_j9.py`, `decider2/j9/j9lib.py` | `decider2/j9/j9lat.py`, `decider2/j9/const_analysis.py` | `decider2/j9/final_scores.json`, `decider2/j9/lat_real_affine.json` |
| A7 | Domain vocabulary | `j7/` | `decider2/j7/code/superbpe.py`, `decider2/j7/code/train_j7.py` | `decider2/j7/code/lat_j7.py`, `decider2/j7/code/traffic_lat.py` | `decider2/j7/results/score_T64k.json`, `decider2/j7/results/score_TL2.json`, `decider2/j7/results/traffic_lat_64k.json` |
| A8 | Value-identity and structure features | `j7/` | `decider2/j7/code/chan.py`, `decider2/j7/code/gen_aug.py`, `decider2/j7/code/train_j7.py` | – | `decider2/j7/results/score_CV.json` (with channels), `decider2/j7/results/score_C.json` (without) |
| A9 | Built-in exact comparator | `j12/` | `decider2/j12/code/train_xr.py`, `decider2/j12/code/xrlib.py`, `decider2/j12/code/gen_xr.py` | `decider2/j12/code/lat_xr.py` | `decider2/j12/scores_xr.json`, `decider2/j12/scores_ctl.json`, `decider2/j12/results/diag_xr.json` |
| A10 | Decision pretraining | `j4/` | `decider2/j4/code/gen_dp.py`, `decider2/j4/code/mk_ft.py`, `decider2/j4/code/train.py` | – | `decider2/j4/results.json`, `decider2/j4/curves.json` |
| A11 | Slot model from scratch | `j11/` | `decider2/j11/code/synth.py`, `decider2/j11/code/models.py`, `decider2/j11/code/train.py` | `decider2/j11/code/lat.py` | `decider2/j11/results.json`, `decider2/j11/runs/` |
| A12 | Dual encoders, late interaction | early round; not located | – | – | – |
| A13 | Pointer programs, JSON reader | early round; not located | – | – | – |
| M1 | All-attention hobson | `m1/`, `n1/` | `decider2/m1/code/m1lib.py`, `decider2/m1/code/m1conv.py`, `decider2/m1/code/m1train.py` | `decider2/m1/code/m1lat.py`, `decider2/n1/code/m1.py` (Inferentia timing) | `decider2/m1/ARCH.md`, `decider2/m1/results/scores.json`, `decider2/m1/results/lat_hobson_and_m1_head136.json` |
| M2 | Segment-isolated state | `m2/`, `n1/` | `decider2/m2/code/m2seg.py`, `decider2/m2/code/m2lib.py`, `decider2/m2/code/m2train.py` | `decider2/m2/code/m2lat.py`, `decider2/m2/code/m2pre.py`, `decider2/n1/code/m2.py` (Inferentia timing) | `decider2/m2/ARCH.md`, `decider2/m2/res/full_scores.json`, `decider2/m2/res/lat_real.json` |
| M3 | Shallow state, deep reading | `m2/`; see also `j3/` (R2) | `decider2/m2/code/m2train.py` | `decider2/m2/code/m2tf.py`, `decider2/m2/code/m2hold.py` | `decider2/m2/res/tf1/f12.json`, `decider2/m2/res/tf1/f8.json`, `decider2/m2/res/full/N_k8.json`, `decider2/m2/res/hold_C.json` |
| M4 | Global summary slots | not built | – | – | – |
| M5 | Exact top-k reads for question rows | `m2/` | – | `decider2/m2/code/m2tf.py` | `decider2/m2/res/tf1/topk16.json`, `decider2/m2/res/tf1/topk64.json`, `decider2/m2/res/tf1/topk256.json` |
| M6 | Linear attention without the erase term | not planned | – | – | – |
| M7 | Local attention as the only mixer | `j1/` | – | `decider2/j1/code/encrt.py` | `decider2/j1/results/scores_A_loc512.json`, `decider2/j1/results/lat_enc_loc512.json` |
| C1 | k64rr + layer-16 exit | `q2/`, with `j15/`'s exit recipe | `decider2/q2/code/q2exit.py` | `decider2/q2/code/q2rt.py`, `decider2/q2/code/q2bench.py`, `decider2/q2/code/q2exit_an.py` | `decider2/q2/res/exit_q2b_k64x16.json`, `decider2/q2/res/proj_cascade.json` |
| HW1 | Inferentia2, AMX CPU | `j8/` | – | `decider2/j8/code/hob.py`, `decider2/j8/code/gdn_nki.py`, `decider2/j8/code/nbuild.py`, `decider2/j8/code/nrun.py`, `decider2/j8/code/cpu_lat.py`, `decider2/j8/code/q8_cpu.py` | `decider2/j8/scores.json`, `decider2/j8/results/cpu_lat_compile_bd_all.json`, `decider2/j8/results/scores_cpu_fullkit.json` |
| HW2 | Apple M4 Pro | early round; not located | – | – | – |
| HW3 | Pipelined NKI GDN kernel | `n1/` | – | `decider2/n1/code/gdn11.py`, `decider2/n1/code/hob5.py`, `decider2/n1/code/kbench.py`, `decider2/n1/code/costs.py` | `decider2/n1/results/scores_n11.json`, `decider2/n1/results/kprof_gdn11_gdn11_H16_T1152_n3.json`, `decider2/n1/results/lat_L1152_S32_n11_C128.json` |
| E1 | Counterfactual training data | `h7/`, `j3/`, `j7/`, `j12/`, `m2/` | `decider2/h7/code/gen_cf.py`, `decider2/j7/code/gen_aug.py`, `decider2/j12/code/gen_xr.py`, `decider2/j12/code/train_xr.py` | – | `decider2/j12/scores_ctl.json`, `decider2/j7/results/score_aug.json`, `decider2/h7/scores.json` |
| E2 | this-that-model | `h4/` | – | `decider2/h4/tt_lean.py`, `decider2/h4/tt_eval.py`, `decider2/h4/tt_lat.py`, `decider2/h4/tt_quant.py` | `decider2/h4/scores.json`, `decider2/h4/lat_tt.json`, `decider2/h4/quant_scores.json` |
| E3 | Evaluation noise floors | `evalkit/` | – | `decider2/evalkit/evalkit.py`, `decider2/evalkit/kitrun.py` | `decider2/evalkit/README.md`, `decider2/evalkit/refs/` |

## How to reproduce

The code was written to run in two places: GPU boxes (A10G, Inferentia2 or a CPU instance), where it lived under
`~/work/<agent>/`, and a laptop, where the tree lived at `~/decider2/` and the scoring and plotting ran. Many scripts
therefore contain those absolute paths. The simplest way to use them unchanged is to link the tree into place from the
repository root:

```bash
ln -s "$PWD/research/fast-decision-models/decider2" ~/decider2
```

`decider2/bundle/setup_box.sh` records the environment every GPU box got: the Python packages (among them
`transformers>=5.1`, `peft` and `flash-linear-attention`), a strands-decider checkout installed in editable mode
from `~/work/sd`, and the `StrandsAgents/strands-decider-2B-hobson-v19` and `Qwen/Qwen3.5-2B-Base`
checkpoints downloaded from the Hugging Face Hub. The archive it unpacked held copies of folders that are in `decider2/`
(`d1/`, `evalkit/`, `h2/code/`, `h4/tt_lean.py`, `h7/code/`, `shrink/`, `systems/`, `tokens/`, `training/`). Agents ran their own code
from `~/work/<agent>/` on their boxes; `decider2/q2/code/README.md` is an example of the build and run steps.

### The figures

`decider2/analysis/run_all.sh` rebuilds the seven result figures and their data in under a minute. It runs four steps
from `decider2/analysis/`:

1. `catalog.py` finds every agent's prediction files under `~/decider2/<agent>/` and normalises them into `norm/`
   (about 258 MB, left out of the package because this step regenerates it);
2. `metrics.py` scores them with the eval kit at `~/decider2/evalkit` and writes `results/metrics.csv` and `.json`;
3. `latency_grid.py` reads the measured latency curves from the agent folders and writes `results/latency_at_1000.txt`;
4. `figures.py` draws the figures as SVG and CSV in `analysis/figures/`, and as PNG in `analysis/figures/png/` if
   `rsvg-convert` (librsvg) is installed. Without it the PNG step is skipped silently.

All four scripts locate the tree with `os.path.expanduser('~/decider2')`, so they need the symlink above or edited paths.
The last lines of `run_all.sh` then copy the PNGs to `~/code/jit-eval/docs/figures/` and the CSVs to
`~/code/jit-eval/docs/figures/data/`, creating that folder if it is missing. That was the working copy of the report on
the original laptop, not this package. To update this package's [`figures/`](figures/), change those `cp` lines to point
at `research/fast-decision-models/figures/` and its `data/` folder, or copy the outputs by hand.

The concept diagrams in `figures/ideas/` are not made by `run_all.sh`. They come from `ideas_diagrams.py`,
`ideas_diagrams_s.py` and `ideas_diagrams_m.py` in the same folder, which need `fontTools` and `rsvg-convert` and write to
`~/code/jit-eval/docs/figures/ideas/` unless the default folder in `sketch.py` is changed.

Running in place rewrites `catalog.csv`, `norm/`, `results/` and `figures/` inside `decider2/analysis/`. To leave both the
package and your home folder untouched, run on a copy, with `HOME` pointed at a scratch folder that holds the copy as
`decider2/`. The outputs then land in `<scratch>/code/jit-eval/docs/figures/`:

```bash
mkdir -p /tmp/fdm && cp -R research/fast-decision-models/decider2 /tmp/fdm/
cd /tmp/fdm/decider2/analysis && HOME=/tmp/fdm bash run_all.sh
HOME=/tmp/fdm python3 ideas_diagrams.py && HOME=/tmp/fdm python3 ideas_diagrams_s.py && HOME=/tmp/fdm python3 ideas_diagrams_m.py
```

Run this way on 2026-10-07 on a copy of this package's `decider2/`, the pipeline catalogued 443 prediction sets and
rebuilt all seven result figures and their CSV files byte for byte identical to the ones in [`figures/`](figures/), in
about 45 s on a laptop. The three diagram scripts rebuilt all 22 diagrams identically.

### Scoring predictions with the eval kit

`decider2/evalkit/README.md` describes the suites, the prediction format, the metrics and the baselines. In short,
`evalkit.py` is plain Python plus numpy and finds its suites and hobson's references relative to its own file, so it works
from any location. `EK.all_question_items()` yields the 3,227 deduplicated questions to answer. A model's answers go in a
dict `preds[item_id][question_name] = {label: probability}`. `EK.report(preds)` prints every suite against hobson and the
baselines that read less of the state, and `EK.score(suite, preds)` returns one suite's metrics.

Inputs must be rendered exactly as the references were. `kitrun.py` gives the exact token ids every reference used, but it
runs only in the GPU-box layout: it needs this repository's `strands_decider` package, the hobson-v19 checkpoint in the
local Hugging Face cache, and `decider2/tokens/` at `~/work/tokens`.

### Training recipes

- **The base fine-tuning recipe** is F7's, reconstructed in section 2 of `decider2/F7_REPORT.md`: a pointer head, LoRA
  rank 16 on every projection, cross-entropy on gold labels plus KL divergence to hobson, and 39.8M tokens in one epoch.
  Most later trainers reuse it. Its code survives in G3's replica, `decider2/recovered/g3/g3/train_g3.py`, and in
  trainers that copy it, such as `decider2/j1/code/train_j1.py` and `decider2/j10/code/train_j10.py`.
- **Each idea's trainer** is in its agent's `code/` folder; the Training column of the table above names it. The
  "Training recipes" section of `decider2/MANIFEST.md` lists every trainer in the tree with its driver scripts, and the
  agents' `NOTES.md` files record the arguments, budgets and learning curves of each run.
- **Shared data tools:** `decider2/training/` builds the fixed pilot training mixes (`mkmix.py`), labels them with the
  v19 teacher (`teacher_v19.py`) and scores checkpoints on JevBench (`jbeval.py`). The counterfactual and detail-reading
  generators are `decider2/h7/code/gen_cf.py`, `decider2/j7/code/gen_aug.py` and `decider2/j12/code/gen_xr.py`.
- **Training data:** every run trained only on train-split requests; the split is `decider2/evalkit/split.json`. The two
  inputs most trainers read are not in the package because of their size: the train pool `evalkit/train_pool.jsonl`
  (185 MB, 12,747 requests) and `training/data/train_v5.jsonl` (82 MB). The generated training sets built from them are
  left out too, but the scripts that wrote them are included.

## What is not included

`decider2/EXCLUDED.md` lists every file and folder left out, with the reason. In summary:

- **Model weights and checkpoints.** No trained adapter, exit head, quantized weight file or checkpoint is included (43
  files, about 2.1 GB, were left out). Most of them lived only on the GPU boxes and were never copied back. Reproducing a
  result that depends on a trained model means retraining it with the recipe in its agent folder. The hobson-v19
  checkpoint itself is not in the package either.
- **Large data files:** every file over 25 MB, including the train pool, `train_v5.jsonl`, the generated training sets,
  and `analysis/norm/`.
- **Third-party material:** two checkouts of strands-decider (`ext/`), paper PDFs, and copied Neuron documentation.
- **Orchestration:** the scripts that launched and reached the GPU boxes, their SSH keys and the per-box records.
- **Logs, data copies and binaries in `recovered/`,** images and archives, report drafts, and the `archive/` folder of
  earlier report versions.
- **Code that is not in the tree at all.** The original tree was wiped in a laptop reboot, and only part of it was
  recovered from the GPU boxes. The F-round folders (only F7's reconstructed report remains), F6's context-parallel
  runtime (S5), and the code behind S4, S9, S11's text-reuse measurement, R3's F1 and F2 runs, R4, R5, A12, A13 and HW2
  are not here.

## Hardware and measurement conventions

This is a short version of the report's [How to read this report](FAST_DECISION_MODEL.md#how-to-read-this-report).

- **Scope.** Requests are cold and independent: nothing is cached across requests. The 1–15 questions of one request may
  share work over the same state, and per-deployment constants (fixed question specs and documents) may be precomputed.
- **Hardware.** Every GPU number was measured on NVIDIA A10G GPUs (AWS g5.2xlarge, 24 GB). The A10G is the RTX 3090's
  chip family (GA102), with the same bf16 tensor-core rate per SM per clock, so arithmetic-bound times carry over roughly
  1:1; the 3090 has 1.56x the memory bandwidth. RTX 3090, 4090 and 5090 numbers are projections from A10G measurements;
  none of those cards was measured. GeForce cards run fp16 with fp16 accumulation at twice the bf16 rate.
- **Measured** means wall-clock time end to end for one request, at batch 1, in a CUDA graph, with exclusive use of the
  GPU and fresh inputs on each call: the median of at least 12 warm repetitions. **Arithmetic** means a model built from
  measured parts. **Projected** means arithmetic carried over to hardware that wasn't measured. Speculation is marked.
- **Lengths** are exact, with no bucket padding. "At 1,000 tokens" means a state of exactly 1,000 tokens plus one real
  question of 85–125 tokens. "Real requests" are recorded gate requests from the tau3 agent runs.
- **Accuracy** is measured on the evaluation kit (3,227 questions from held-out real traffic, counterfactual pairs and
  JevBench), not on JevBench alone. A speedup meant to keep hobson's function must meet the fidelity bar: at most 0.7% of
  real decisions differ from hobson's, with no significant difference from the bf16 runtime's own decisions (McNemar test),
  counterfactual retention of at least .99 (CF) and .95 (CF-probe), JB-hard accuracy not significantly below hobson's, and
  REAL-label accuracy of at least .78. A change to the model itself is judged by the accuracy bar instead. Both bars are
  defined in the report's [How we measure accuracy](FAST_DECISION_MODEL.md#how-we-measure-accuracy).
- **IDs.** Each idea has a category letter and a number: S systems, P numeric precision, R reducing work per token, B
  decision-specific numerics and structure, A architecture, M mixers and layout, C combinations, HW hardware, E data and
  evaluation. The figures use the same labels.
