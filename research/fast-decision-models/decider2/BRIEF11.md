# Brief 11: mixers and layout designed for decisions (M1, M2, N1)

You are one of three new agents working for a principal engineer who wants a decision model that is radically faster. The engineer explicitly rejects marginal ideas and literature reproductions; wants ideas that lean into what makes a decision model unique; and doesn't care whether a technique would hurt a general-purpose LLM.

**Read first:**
- this brief;
- `~/code/jit-eval/docs/IDEAS_EXPLORED.md`: the whole table, especially the new "M · Mixers and decision-native layout" rows, HW1, HW3, R2, A6 and "The floor on wall time";
- `~/code/jit-eval/docs/FAST_DECISION_MODEL.md`.

## The model and the premise

**hobson-v19:**
- **Architecture:** Qwen3.5-2B with 24 layers of width 2048. 18 are Gated DeltaNet (GDN; 16 heads, key/value dimension 128) and 6 are full attention (layers 3, 7, 11, 15, 19 and 23; 8 query heads × 256, 2 key/value heads). Each has a SwiGLU MLP of width 6144, and a pointer head reads K+1 rows per question.
- **Layout:** the state comes first and the questions after it, with a causal mask, so state rows never see the question.
- **Rows:** a *state row* is a token of the state; a *question row* is a token of a question (instructions, options, `<answer>`). Only question rows are read out.

**What is measured (agent Q1, this week, train-split data):** at layers 18–23, state rows carry only 0–1.7% of the decision's gradient sensitivity. Layer 23's state rows need only their K/V. The question rows do the reading. In earlier work:
- state rows through only 8 layers met the accuracy bar (R2);
- an exit at layer 16 changed 0 of 6,216 decisions (R1);
- question tokens' content rows are where hobson reads the state (A5/J14).

**The premise:** a decision model needs the state *read*, through a narrow channel. It does not need every state token processed to full depth or mixed with every other state token. A layout built on that runs as dense, fixed-shape matmuls, which suits GPUs and suits Inferentia's systolic arrays even more.

## The ideas (IDs from the table)

| ID | idea | agent |
|---|---|---|
| M1 | All-attention hobson: the 18 GDN layers replaced by softmax attention, initialized from hobson's projections, distilled from hobson | **M1** |
| M2 | Segment-isolated state encoding: state rows attend only within their segment (message, tool output, document); question rows read every segment | **M2** |
| M3 | Reader attention: cheap local mixing for state rows at every depth; question rows cross-attend to all state rows at every layer | **M2** |
| M4 | 32–64 global summary slots that state rows write to and read from, for reasoning that chains facts across segments | **M2** |
| M5 | Exact top-k retrieval for question rows | **M2** (inside M3) |
| HW3 | A pipelined NKI GDN kernel and a layout cleanup on Inferentia2, plus timing M1's and M2's shapes there | **N1** |

## The bars

**Accuracy**, on all 3,227 evalkit questions (`~/decider2/evalkit`), with paired McNemar tests against hobson-v19. Report both bars:
- **Strict:** no significant drop on JB-all or JB-hard; REAL-label ≥ .78; CF and CF-probe pair accuracy ≥ hobson's (.268 / .328).
- **Relaxed** (the engineer accepts a marginal loss for an outsized gain): JB-all ≥ .693; REAL-label ≥ .765; REAL state-dependent agreement ≥ .85; CF and CF-probe pair accuracy ≥ hobson's.

Also report:
- the RuleTaker depth-3 and depth-5 held-out probe (`training/data/train_v5.holdout.jsonl`; J3's `dt_holdout.py`), because layouts that cut state-to-state mixing are expected to lose there;
- Brier on REAL-label.

**Latency:**
- A10G, exclusive GPU, exact lengths T = 64, 256, 1000 and 4000, with 1 and 4 questions;
- ≥ 12 warm reps; median and p95;
- fused runtimes comparable to hobson's (`d1/lean2.py`, H2's QRT), not eager PyTorch;
- projections to 3090, 4090 and 5090 as in the doc.

## Training recipe (M1, M2)

- **Data:** distillation from frozen hobson-v19 (in process), on the train split only (never evalkit eval items):
  - real states from `evalkit/train_pool.jsonl`, loss KL to the teacher's decisions;
  - `training/data/train_v5.jsonl` gold rows, loss CE + KL;
  - optionally ≤ 15% detail augmentation (`h7/data/cf_aug.jsonl`).
- **Dense row loss:** relative MSE of hidden states at layers 5, 11, 17 and 23 on the answer and option rows.
- **Parameters:** full-rank on the new mixer weights; LoRA r32 elsewhere (full fine-tunes of the whole model forgot: J14).
- **Budget:** as many tokens as fit about 6 hours, with checkpoints every ~30 minutes scored on a fixed train-split dev set (J14's `is_dev` hash split). Report learning curves.
- **Reference code:**
  - `~/decider2/j3/code/train_dt.py` and `dtlib.py` (depth split; teacher in process; dense rows);
  - `~/decider2/j2/code/` (function-preserving mixer surgery on hobson, packed varlen);
  - `~/decider2/j14/code/j14train.py`;
  - `~/decider2/q3/code/` (Q3's current QAT trainer and lean student).

  Copy what you need; never edit other agents' files.

## The three agents

### M1: all-attention hobson (box `m1`, A10G)
1. **Convert.** Replace each GDN layer's mixer with causal softmax attention of the same width, initialized from the GDN layer's own q, k, v and output projections and gates (16 heads × 128). Keep RoPE consistent with hobson's attention layers, keep q/k norms, and keep the output gate. Everything else stays hobson's.
   - **Untrained, measure:** agreement, and how far from hobson.
   - **Variants:** (a) all 18 layers converted at once; (b) a schedule converting a few layers at a time (bottom-up or top-down), distilling each stage.
2. **Distil and evaluate** against both bars.
3. **Measure latency** in a fused runtime: SDPA/FlashAttention for all 24 layers; hobson's GEMMs unchanged.
4. **Report:** accuracy against tokens of training, latency at every length against hobson, and the long-state cost (4,000 tokens).

### M2: decision-native layout, M2–M5 (box `m2`, A10G)
1. **Training-free measurements first. They decide the program.**
   1. **Segment the state.** Define segments from the rendered state's structure: messages, tool outputs, documents, hook notes. Use J9's segmenter (`~/decider2/j9/j9lib.py`, `pieces`) as a start.
   2. **Measure dependence on cross-segment mixing.** In hobson's 6 attention layers, mask attention from state rows to other segments' state rows; question rows keep full attention. In the 18 GDN layers, run each segment's recurrence separately for state rows. Question rows read a composition of the segment states: J9's exact affine composition, S ← A_i(S − S_U) + E_i, or a sum.
      - **Report:** agreement with hobson, flips, CF and CF-probe retention, and REAL-label for each.
      - **Variants:** isolation in all layers; only in layers ≥ k; only in attention or only in GDN.
   3. **Measure dependence on state depth with full-depth question reading:** state rows frozen after layer k (they keep providing their layer-k K/V to later layers) for k = 4, 8 and 12. This is R2's question, measured untrained, together with segment isolation.
2. **Then build and distil the best layout:**
   - state rows with segment-local attention (or windowed attention as the fallback);
   - question rows with global attention to all state rows at every layer (M3);
   - optionally summary slots (M4), and top-k reads for question rows (M5).

   Report against both bars, with the RuleTaker probe.
3. **The precompilation benefit.** Show that constant documents become exactly precomputable under segment isolation: identical decisions with compiled and live document blocks. Measure the latency on J9's 84 real banking requests (`~/decider2/j9/lat_real_affine.json` lists them) against hobson and A6.
4. **Latency** of the layout in a fused runtime at exact lengths, against hobson.

### N1: Inferentia2 (box `n1`, inf2)
J8's work is in `~/decider2/j8/`:
- `DRAFT_REPORT.md` and `NOTES.md`;
- `code/gdn_nki.py` (the exact NKI GDN kernel);
- `code/hob.py` (pure-torch hobson with a matmul-only GDN);
- `HobNL` (the Neuron layout), and the nbuild / nrun scripts.

Its kernel was correct, exact to 3e-7, but latency-bound. Each head was one dependent chain: per 128-token chunk, about 35 Tensor Engine operations including 14 transposes, each followed by a PSUM→SBUF copy. The (I+L)⁻¹ used 6 levels of block doubling. The engines were 19–24% busy. GDN took 72 of 152 ms at T = 1000, and the rest of the model reached about 40 of about 95 TFLOPS per core because of 472 GFLOP of compiler-inserted transposes and 3.4 GB of spill.

1. **Build a pipelined multi-head NKI GDN kernel.** Interleave ≥ 4 heads per program, or stack heads along the free dimension. Use a shorter triangular solve (64-token chunks, or 32-blocks with forward substitution). Drop most transposes using operand order and (AB)ᵀ = BᵀAᵀ. Keep intermediates in PSUM. Double-buffer chunks. Stay exact (≤ 1e-5 against J8's fp64 reference).
2. **Clean up the layout:** remove the compiler-inserted transposes and spills around the GEMMs, and fuse GDN with its projections where possible.
3. **Measure hobson end to end** on one NeuronCore at T = 64, 256, 1000 and 4000, with 1 and 4 questions. Check decisions at the noise floor on J8's eval subset. Compute the cost per million decisions on inf2.xlarge with both cores, using J8's method.
4. **Time M1's and M2's shapes on inf2.** This is latency only, so it needs no trained weights: random weights of the right shapes.
   - M1: all-attention hobson.
   - M2: segment-local state attention (fixed 128- or 256-token blocks) with global question-row attention.
5. **Report** against J8's numbers (152 ms; $16.0 per million against the A10G's $17.7). J8's threshold was ≤ 90 ms at T = 1000 on one core, which is half the A10G's cost per decision.

**Your box:**
- **Hardware:** an inf2 instance (one Inferentia2 chip, 2 NeuronCores) with the AWS Deep Learning AMI Neuron (Ubuntu 22.04).
- **What's there:** the coordinator copied `~/work/bundle` (evalkit, h2/code, d1, …) and strands-decider (`~/work/sd`). Activate the AMI's Neuron PyTorch venv (`ls /opt/aws_neuronx_venv_*`).
- **You must download:** hobson-v19 from Hugging Face (`StrandsAgents/strands-decider-2B-hobson-v19`) and Qwen3.5-2B-Base.
- **Before compiling anything:** check the Neuron SDK versions against J8's (torch-neuronx 2.8, neuronx-cc 2.23).

## Infrastructure and rules (all agents)

- **Boxes.** Use only `~/decider2/box.sh <BOX> run "<cmd>" | put <local> <remote dir under ~/work> | get <remote path under ~/work> <local> | status`. Wait for `~/decider2/boxes/<BOX>.ready`, checking every ~5 minutes with `sleep 290`.
- **Box lifetime.** Each box auto-terminates 10 hours after launch. You may reset your own box's timer once (`sudo shutdown -c; sudo shutdown -h +N`); note it in NOTES.md.
- **Commands.** One command must finish in under 30 minutes. Run long jobs with nohup and make them resumable. GPU memory must stay under 22 GB on A10G boxes.
- **No models on the laptop.** Don't run torch or any model locally, and keep weights and checkpoints on the boxes. Small JSON outputs come back to the laptop.
- **No other AWS resources.** Don't touch any and don't launch instances.
- **Writing.** Write only under `~/decider2/<you>/`:
  - `NOTES.md` with timestamps every ~15 minutes;
  - `DRAFT_REPORT.md` (the harness blocks files named REPORT.md);
  - results JSON.

  Copy results off the box at least every 30 minutes. You may read other agents' folders but never write to them.
- **Permissions.** If a permission is denied, don't work around it; report it.
- **Language.** Plain and specific. Every number gets its metric, baseline and unit. No verdict labels ("survives", "dominated", "cost-class"); say what was measured and what it means. Mark each claim as measured, verified or arithmetic.

## Report

At most 1,500 words. Return it as your final message and also write it to `DRAFT_REPORT.md`. Cover:
- what you built;
- the measured results against both bars;
- latency against hobson;
- projections;
- the single most valuable next step.
