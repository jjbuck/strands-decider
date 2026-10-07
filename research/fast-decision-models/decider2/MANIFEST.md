# decider2: manifest

A filtered, redacted mirror of `~/decider2`, the working tree of the fast decision model program for hobson-v19
(Strands Decider, a 2B Qwen3.5 decision model). Each research agent has its own folder with its notes, report, code,
results and predictions; the shared folders hold the evaluation kit, training tools and the figure pipeline. What was
left out, and why, is in `EXCLUDED.md`. Sizes are decimal (1 MB = 10^6 bytes).

Most agents ran on a remote GPU box; paths such as `~/work/<agent>` in the code refer to the box, and
`~/decider2/<agent>` to this tree. The orchestration scripts that launched and reached the boxes are not included.

| folder | what it is | files | size |
|---|---|---|---|
| `analysis/` | The post-processing pipeline: `catalog.py` normalises every agent's predictions, `metrics.py` scores them on every evalkit suite, `latency_grid.py` holds the measured A10G latency curves, `figures.py` and `ideas_diagrams.py` draw the report figures, and `registry.py` maps each plotted configuration to its idea ID. `results/` and `figures/*.csv` hold the scored numbers; `fonts/` holds the Virgil font (SIL OFL 1.1, see `fonts/README.md`). | 25 | 1.3 MB |
| `evalkit/` | The capacity-sensitive evaluation kit every agent scored against: suites (JevBench subsets, REAL-agree, LONG, CF, CF-probe, SHUF, REAL-label), hobson-v19 and base-model references, the train/eval task split and the scoring code. | 33 | 46.2 MB |
| `training/` | Shared training-data tools and data: the pilot-mix builder, the hobson-v19 teacher pass, the JevBench evaluator, JevBench public, and the pilot and holdout training files. | 10 | 30.6 MB |
| `bundle/` | The box setup script that installed the Python environment and downloaded hobson-v19 on each GPU box (the code tarball it unpacked is not included). | 1 | 1 KB |
| `d1/` | An earlier-round lean 2B runtime with hand-written Triton fusions for the long, compute-bound prefill path. | 1 | 14 KB |
| `d2/` | An earlier-round W8A8 int8 Triton GEMM with a fused dequantisation epilogue, benchmarked against cuBLAS bf16. | 1 | 5 KB |
| `systems/` | The earlier-round lean Qwen3.5 inference runtime (fused weights, fla kernels, CUDA graphs) with its benchmarks and profiles. | 57 | 0.3 MB |
| `tokens/` | Earlier-round experiments on training-free token eviction and input-level token selection in hobson-v19, plus a LoRA test of healing eviction. | 26 | 5.1 MB |
| `shrink/` | Earlier-round depth truncation of hobson-v19: latency by number of layers and a recovery fine-tune of the truncated model. | 10 | 0.8 MB |
| `h1/` | H1 (Brief 7): W8A8 and W4A4 recipes for hobson-v19 in exact emulation, covering GPTQ, per-GEMM precision maps, the W8A8 error budget and LoRA QAT. | 81 | 13.2 MB |
| `h2/` | H2 (Brief 7): an end-to-end low-bit runtime for hobson-v19 on the A10G, with CUTLASS int4/int8 GEMMs, a W4A8 mixed-input GEMM and fused Triton glue kernels. | 62 | 13.9 MB |
| `h3/` | H3 (Brief 7): how 4-bit error propagates through hobson-v19, and a test of hyperspherical (nGPT-style) layers as a 4-bit co-design. | 119 | 13.2 MB |
| `h4/` | H4 (Brief 7): this-that-model-1.0 compared with hobson-v19 on accuracy, latency and quantisation; the two share the same torso. | 64 | 11.6 MB |
| `h5/` | H5 (Brief 7): learned-rotation W4A4 (a learned residual rotation, per-GEMM Kronecker rotations and a learned clip) on full-depth hobson-v19, plus LoRA QAT. | 66 | 14.1 MB |
| `h6/` | H6 (Brief 7): mixed W4A4/W8A8 precision through the deployed kernels, with question rows kept in bf16, and QAT for the schema-first layout. | 96 | 23.8 MB |
| `h7/` | H7 (Brief 7): training the compiled question schema (questions compiled once per deployment) into the full 24-layer hobson-v19 (idea A3). | 43 | 6.6 MB |
| `j1/` | J1 (Brief 8): a 2B bidirectional encoder decider built on the T5Gemma-2B encoder at hobson's size (A1). | 48 | 7.9 MB |
| `j2/` | J2 (Brief 8): a bidirectional GDN-hybrid encoder made from hobson's own torso (A2). | 59 | 4.1 MB |
| `j3/` | J3 (Brief 8): the depth-split decision transformer, in which state rows stop at layer 8 and question rows continue to layer 24 (R2). | 62 | 10.5 MB |
| `j4/` | J4 (Brief 8): decision pretraining at equal size, using a self-supervised verification corpus before the decider fine-tune (A10). | 79 | 11.8 MB |
| `j5/` | J5 (Brief 8): millisecond decisions on short inputs with the full 2B model, testing weight-only 4-bit and int8 short-M kernels (P3). | 56 | 10.9 MB |
| `j6/` | J6 (Brief 8): deployed questions compiled into the weights as per-question LoRA deltas and slot rows, including a hypernetwork variant (A4). | 51 | 5.2 MB |
| `j7/` | J7 (Brief 8): a reader-native input layer for hobson, with a domain super-token vocabulary and value-identity side channels (A7, A8). | 93 | 17.2 MB |
| `j8/` | J8 (Brief 8): unchanged hobson-v19 on AWS Inferentia2 and on Sapphire Rapids AMX CPUs (HW1). | 46 | 3.2 MB |
| `j9/` | J9 (Brief 8): compiling the deployment's constant blocks (knowledge-base documents, hook notes, tool schemas) once per deployment (A6). | 83 | 6.5 MB |
| `j10/` | J10 (Brief 8): a natively ternary 2B decider built from BitNet b1.58 2B4T, plus a ternary distillation of hobson (P9). | 71 | 5.6 MB |
| `j11/` | J11 (Brief 8): decision-native architectures (slot, belief and value-channel models) trained from scratch at matched compute (A11). | 101 | 33.5 MB |
| `j12/` | J12 (Brief 8): exact-relation (comparator) modules inside hobson-v19, trained on synthetic exact-reasoning pairs (A9). | 43 | 16.3 MB |
| `j13/` | J13 (Brief 8): token-adaptive width on full 24-layer hobson-v19, tested first with an oracle router (R6). | 64 | 63.4 MB |
| `j14/` | J14 (Brief 8): compiling the question bundle while keeping hobson's state-first layout (A5). | 74 | 25.4 MB |
| `j15/` | J15 (Brief 8): speculative precision (a W4A4 draft re-run in W8A8 when unsure), which did not pay, and the layer-16 depth exit that did (P8, R1). | 106 | 63.0 MB |
| `k1/` | K1 (Brief 9): the combined-stack library, training recipe and accuracy harness; paused by the coordinator before any training run. | 12 | 10.1 MB |
| `k2/` | K2 (Brief 9): the runtime and latency harness for the combined stack; paused with K1. | 7 | 0.1 MB |
| `q1/` | Q1 (Brief 10): decision sensitivity and 4-bit formats for hobson-v19 in exact emulation (B1 to B3, B5, B6, B8 to B10). | 134 | 12.4 MB |
| `q2/` | Q2 (Brief 10): 4-bit decision GEMM kernels on the A10G, row-role precision, the k64rr mixed 8/4-bit map and k64rr with the layer-16 exit (P10, P11, C1). | 113 | 22.5 MB |
| `q3/` | Q3 (Brief 10): decision-level rounding, decision-level weight scales and quantisation-aware distillation for 4-bit hobson-v19 (B4). | 76 | 17.4 MB |
| `q4/` | Q4 (Brief 10): fewer and better-used multiplies: 2:4 sparse GEMMs, exact dead-computation elimination, Strassen-Winograd and a persistent kernel (B11, B13, S10, B14). | 67 | 11.0 MB |
| `q5/` | Q5 (Brief 10): decision-aware structure: 2:4 sparsity masks, per-deployment MLP neuron removal and per-deployment calibration (B7, B11, B12, B13 checks). | 58 | 14.6 MB |
| `m1/` | M1 (Brief 11): all-attention hobson, converting the 18 GDN mixers to softmax attention and training the converted model (M1). | 63 | 7.8 MB |
| `m2/` | M2 (Brief 11): a decision-native layout with segment-isolated state, shallow state rows and global question readers (M2, M3, M5). | 150 | 20.1 MB |
| `n1/` | N1 (Brief 11): hobson-v19 on Inferentia2 with a pipelined NKI GDN kernel, plus timing of the M1 and M2 shapes (HW3). | 108 | 0.8 MB |
| `recovered/` | Copies of the five earlier-round GPU box directories (g1 to g5) recovered after a laptop reboot; only notes, code and small result JSON are kept. | 593 | 16.1 MB |
| `BRIEF7.md` | Brief 7, the H agents' assignment: accurate low-bit decision models co-designed with architecture and inference. | 1 | 5 KB |
| `BRIEF8.md` | Brief 8, the J agents' assignment: transformative ideas for fast, accurate decision models. | 1 | 8 KB |
| `BRIEF9.md` | Brief 9, the K agents' assignment: the combined stack and its pre-registered ablations. | 1 | 14 KB |
| `BRIEF10.md` | Brief 10, the Q agents' assignment: faster matrix multiplies and 4-bit activations, with addenda B5 to B14. | 1 | 24 KB |
| `BRIEF11.md` | Brief 11, the M and N agents' assignment: mixers and layout designed for decisions. | 1 | 12 KB |
| `F7_REPORT.md` | The recipe and results of F7's dense equal-latency controls (R10), reconstructed from F7's final report; most trainers here reuse this recipe. | 1 | 3 KB |

Total before this file and `EXCLUDED.md`: 3120 files, 598.3 MB.

## Training recipes

Most trainers reuse F7's recipe (`F7_REPORT.md`): a pointer head of dim 256, LoRA r16/α32 on every projection, lr 1e-4 (head 1e-3),
AdamW(0.9, 0.95), 3% warmup and cosine decay, 32 rows per step, one epoch, and a loss of CE on gold labels plus KL to hobson-v19.
IDs in brackets are the idea IDs in `../FAST_DECISION_MODEL.md`. Every script listed here is included; the data each one reads is
included only when it is under 25 MB (see `EXCLUDED.md`), and no trained weights are included.

**Shared**
- `training/mkmix.py`: builds the fixed pilot training mixes from `train_v5` (the same mix for every torso).
- `training/teacher_v19.py`: hobson-v19 teacher logits on `training/data/pilot_train.jsonl`, the distillation targets.
- `training/jbeval.py`: scores a checkpoint on JevBench public and the pilot held-out sets (used by the recipes; not a trainer).
- `shrink/train_trunc2.py`: recovery fine-tune of a depth-truncated hobson-v19 (first L layers, fresh LoRA, CE plus KL to the 24-layer model); earlier round.
- `tokens/heal.py`: LoRA self-distillation meant to heal state-token eviction [R3]; earlier round.

**H agents (Brief 7)**
- `h1/code/h1qat.py`: QAT by KL distillation to bf16 hobson, with LoRA in the rotated weight space and straight-through rounding, for W4A8 and W8A8 [P6]. Drivers: `h1/code/jobs/qatA.sh`, `qatB.sh`. GPTQ calibration: `h1/code/h1calib.py` [P1].
- `h3/code/train_h3.py`: LoRA fine-tune arms of hobson with hyperspherical head and norm variants and optional int4 or NVFP4 fake-quant QAT [P7, P4]. Drivers: `h3/code/run_main*.sh`, `run_tail*.sh`.
- `h3/code/pretrain_tiny.py`: from-scratch pretraining of a standard and an nGPT-style small transformer on identical data, to compare W4A4 robustness [P7].
- `h5/code/h5rot.py`, `h5/code/h5rotl.py`: learn the residual rotation R1 (end-to-end and local objectives) [P5]; `h5/code/h5qat.py`: short LoRA QAT on the rotated, quantised model, KL to bf16 hobson [P6].
- `h6/code/h6teach.py`: builds the training set and bf16 hobson teacher distributions; `h6/code/h6qat.py`: QAT and a bf16 control for the schema-first layout under a mixed W8A8/W4A4 precision map [P2b, P6]. Drivers: `h6/code/run_queue*.sh`, `run_chain*.sh`.
- `h7/code/gen_cf.py`: CF-style reading augmentation on train-split states [E1]; `h7/code/train_h7.py`: LoRA fine-tune for the compiled-schema layout, distilled from state-first hobson [A3].

**J agents (Brief 8)**
- `j1/code/prep_j1.py`: F7's rows tokenised for both Qwen and T5Gemma; `j1/code/train_j1.py`: the F7 recipe on the T5Gemma-2B encoder [A1]. Drivers: `j1/code/run_main.sh`, `chain_prep.sh`.
- `j2/code/j2data.py`: the F7-recipe mix and hobson teacher pass; `j2/code/j2train.py`: masked next-token adaptation, then the F7 fine-tune of the bidirectional GDN model [A2]. Drivers: `j2/code/queue*.sh`.
- `j3/code/train_dt.py`: the depth-split decision transformer distilled from hobson (F7 recipe plus CF augmentation and a dense loss on question rows) [R2]; `j3/code/dtset.py`: the option-order-invariant variant. Drivers: `j3/code/pipeline.sh`, `pipeline4.sh`; arguments in `a4_args.txt`, `set_args.txt`.
- `j4/code/gen_dp.py`: the self-supervised decision-pretraining corpus; `j4/code/mk_ft.py`: the fine-tune row list; `j4/code/teach.py`: hobson teacher; `j4/code/train.py`: pretraining and fine-tune arms [A10]. Drivers: `j4/code/drive*.sh`.
- `j6/code/train_a.py`: per-question adapters with question-blind state rows [A4]; `j6/code/train_h.py`: hypernetwork from question spec to weight delta [A4]; `j6/code/train_e.py`: early versus late reading test; `j6/code/train_p.py`: capacity probe for many-way questions. Driver: `j6/code/pipeline.sh`.
- `j7/code/superbpe.py`: the domain super-token vocabulary [A7]; `j7/code/gen_aug.py`: detail-reading augmentation pairs [E1]; `j7/code/train_j7.py`: arms C, V, T and TL (F7 recipe, plain layout) [A7, A8]. Drivers: `j7/code/pipe*.sh`.
- `j9/train_j9.py`: hobson trained for the compiled-deployment layout, with the student compiling constant blocks itself [A6].
- `j10/code/train_j10.py`: the F7 recipe on the BitNet b1.58 2B4T torso with LoRA applied to the latent weights before ternary rounding [P9]; `j10/code/tern_hob.py`: ternary (W1.58A8) distillation of hobson [P9]; `j10/code/bitnet_j10.py`: the torso. Drivers: `j10/code/chain*.sh`.
- `j11/code/synth.py`: synthetic decision corpus with exact labels; `j11/code/prep.py`, `j11/code/teacher.py`: rows and hobson teacher logits; `j11/code/train.py`: FLOP-budgeted from-scratch training of each arm in `j11/code/models.py` [A11]. Run settings: `j11/runs/*/meta.json`.
- `j12/code/gen_xr.py`: synthetic exact-reasoning pairs on train-split states; `j12/code/train_xr.py`: fine-tune of hobson with and without the XR modules [A9]. Arguments: `j12/results/*_args.json`.
- `j13/code/fit.py`: thin-path bases from GEMM-input second moments; `j13/code/heal.py`: trains the thin-path factors by KL to the dense path [R6].
- `j14/code/j14train.py`, `j14/code/j14train_util.py`: distil hobson's state-first decisions into the compiled-question layout [A5].
- `j15/code/mkdev.py`: train-split DEV and EXIT sets; `j15/code/j15learn.py`: depth-exit heads (hobson's head plus a residual adapter) trained by KL to the full model [R1]; `j15/code/j15gptq.py`: GPTQ calibration-draw variance.

**K agents (Brief 9)**
- `k1/code/stacktrain.py`: the Brief 9 combined-stack recipe (LoRA r32 plus the enabled components); `k1/code/mkassets.py`: the deployment assets it trains with. Paused before any run.

**Q agents (Brief 10)**
- `q2/code/q2exit.py`: the layer-L exit head on k64rr, using J15's recipe [C1].
- `q3/code/q3gptq.py`: GPTQ initialisation; `q3/code/q3train.py`: decision-level weight scales, AdaRound rounding and full-weight QAD [B4]. Drivers: `q3/code/chain*.sh`.
- `q5/code/q5cal.py`: a per-deployment decision-gradient calibration pass (no weights are trained) [B7, B12].

**M agents (Brief 11)**
- `m1/code/m1train.py`: teacher-forced transfer of each converted mixer, then end-to-end distillation from hobson [M1]; `m1/code/m1conv.py`: conversion checks. Drivers: `m1/code/run_a*.sh`, `run_b.sh`.
- `m2/code/m2train.py`, `m2/code/m2train_util.py`: distil the decision-native layout from frozen hobson (LoRA r32 plus any new mixer modules) [M2, M3]. Drivers: `m2/code/phase3*.sh`, `pipeline*.sh`, `run_sweep1.sh`.

**Earlier rounds (`recovered/`)**
- `recovered/g1/g1/train_g1.py`: structured (low-rank, BTT) MLP layers healed by self-distillation, with a dense-LoRA control [R8]; `recovered/g1/g1/prep_train.py`: rows and teacher logits.
- `recovered/g2/g2/g2train.py`: per-token nested width (MoNE/MatFormer style) self-distilled from hobson [R6]; `recovered/g2/g2/g2qat.py`: rotated W4A4 quantisation-aware distillation [P2, P6]. Drivers: `recovered/g2/g2/jobs/*.sh`.
- `recovered/g3/g3/train_g3.py`: the F7 recipe on an MoE torso [R7]; `recovered/g3/g3/prep_g3.py`, `teacher.py`: rows and teacher logits. Drivers: `chain_train.sh`, `chain_teacher.sh`.
- `recovered/g4/g4/pretrain.py`: from-scratch LM pretraining at a fixed training-FLOP budget (dense, slot and Monarch-MLP arms); `finetune.py`, `lmft.py`: the identical decision and LM-loss fine-tunes; `prep_lm.py`, `prep_ft.py`: their data. `recovered/g5/g4/` is a later copy that adds `prep_ft_short.py`. Drivers: `q_g4*.sh`, `q*_g5.sh`.

**No training:** h2 (GPTQ codes only, `h2/code/h2gptq.py`), h4, j5, j8, k2, q1, q4 and n1.

