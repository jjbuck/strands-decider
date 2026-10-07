# J9: compile the deployment, not just the questions (box j9, A10G)

Labels: [M] measured, [V] verified, [A] arithmetic, [S] speculation. Everything is in `~/decider2/j9/`.

## Hypothesis
Prefill is GEMM-bound (86-91% of hobson's time), so cost scales with the tokens that pass through the 2B. Most of a gate state is the deployment, not the request: the hook frame, KB documents injected per procedure, hook notes and tool schemas, interleaved with the conversation. Compile every constant once per deployment, as the question specs already are, and a request costs O(|dynamic|) instead of O(|deployment| + |dynamic|).

It is not marginal. It removes 41-52% of real banking tokens from the request path without changing size, precision or depth, it multiplies every other lever, and it grows with the KB and policy an agent carries. It is not cross-request caching: the cache depends only on the deployment.

## 1. How much of a state is a deployment constant [M]
Hobson tokenizer. Blocks are KB document bodies, hook notes and tool schemas of at least 32 tokens, plus the frame.
- **Strict** compiles only those kinds: a bounded, canonical library.
- **Loose** also compiles runs of lines that recur in 3 or more train tasks.

Piece tokenizations concatenate to the exact state tokens (0 mismatches).

| traffic | mean tokens | loose | strict |
|---|---|---|---|
| raw gate logs (31,013 records → 24,399 unique states) | 2,143 | **46.5%** | **34.0%** |
| banking | 2,127 | 52.1% | 40.6% |
| banking hooks: after_tool / after_model / before_invocation / before_model / before_tool | | 67 / 52 / 41 / 34 / 8% | 60 / 44 / 13 / 30 / 6% |
| retail | 2,225 | 20.2% | 3.2% |
| REAL-agree / LONG / CF / CF-probe / JevBench | | 37 / 55 / 34 / 25 / 0% | 25 / 48 / 27 / 15 / 0% |

- **What the constants are.** KB document bodies are 72% of the compiled banking text; hook notes and tool schemas are about 6%. The frame is about 45 tokens, so an exact frame-prefix cache, the only strictly function-preserving piece, buys about 1%.
- **Library.** Strict banking has 2,781 blocks (1.16M tokens; 604 distinct documents), and 389 of them cover 90% of uses. K/V costs 12 KB per token: 14.2 GB in all, about 2 GB for the hot set [A].

## 2. What I built
- **Segmenter** (`j9lib.pieces`): frame, constant blocks keyed by text, and dynamic pieces.
- **Per-block compile in a universal context U = `<state>\n`**, independent of request and order. It stores:
  - attention K/V, re-rotated exactly to the runtime position;
  - the GDN affine transfer S_out = A·S_in + B. The delta rule is linear in S, so blocks compose in any order as S ← A_i(S − S_U) + E_i. Verified against direct recurrence at relative error 1.3e-4 [V];
  - the conv tail.
- **Layout R:** [frame][blocks in order of first appearance][dynamic pieces, document headers left in place][question]. A request is one live segment continuing the compiled cache. The fused runtime (`j9lat.py`) matches the reference: argmax 10/10, max |dp| median .004 [V].
- **Compile adapter** (`train_j9.py --adapter compile`): a LoRA (r16, 24 layers) that acts only on the compile rows.
  - The live path (frame, dynamic state, question, head) is exactly hobson's weights, so states without constants (JevBench, airline) are hobson by construction.
  - Loss: KL to native-layout hobson, plus relative MSE of hidden rows (layers 5/11/17/23, question rows plus 128 live rows), the F7/H7 recipe.
  - Data: train_pool only; 400 updates of 8, 40% of them focused on the procedure questions from update 200.
- **Negative result [M].** Three all-weights LoRA fine-tunes (lr 1e-4 to 3e-5) each worsened a held-out probe within 100 updates (KL .026 → .029): optimizer noise on an already-close model. The compile adapter took the same probe from .0258 to .0095.

## 3. Accuracy, every suite [M]
All 3,227 questions, untruncated. p is an exact McNemar test against hobson.

| | hobson | compiled, untrained | + compile adapter (s400) |
|---|---|---|---|
| JB-all | .723 | .727 (no constants: live path = hobson) | .727 (p 1.0) |
| JB-hard | .523 | .531 (same function as merged hobson) | .531 (p 1.0) |
| REAL-label (bar ≥ .78) | .785 | .775 (p .50) | .767 (p .17) |
| CF pair acc | .268 | .256 (p .36) | .256 (p .30) |
| CF-probe pair acc | .328 | .344 (p .51) | .344 (p .46) |
| CF / CF-probe fgh | 1 / 1 | .890 / .848 | .908 / .886 |
| REAL agree / agree_sd / tv | 1 / 1 / 0 | .936 / .899 / .050 | .952 / .908 / .035 |
| LONG agree / agree_sd / tv | 1 / 1 / 0 | .911 / .800 / .081 | .936 / .861 / .054 |
| REAL-label Brier / ECE | .347 / .116 | .359 / .116 | .359 / .094 |
| SHUF both_right | .245 | .240 | .240 |

- **Where the flips are [M].** The two 23-way procedure questions (`needed_procedure`, `procedure`) are the gap. They are the questions about the documents themselves, where H7 also failed.

  | agree_sd | untrained | s200 | s400 |
  |---|---|---|---|
  | procedure questions, REAL | .750 | .771 | .792 |
  | procedure questions, LONG | .408 | .531 | .612 |
  | every other question, REAL / LONG | .956 / .966 | .940 / .966 | .952 / .966 |

  The procedure rows are still rising at s400.
- **Decomposition** (untrained, every 3rd item; agree_sd REAL / LONG) [M]:
  - exact recompute in the new order: .947 / .830, so most of the loss is the reorder;
  - adding the context-free compile: .921 / .774;
  - in-place splicing: .868 / .792;
  - GDN composition variants are within noise (GDN-invisible blocks .912 / .811).
- **K/V-only library [M].** With GDN-invisible blocks at s400 (trained affine), REAL / LONG agree_sd is .887 / .861, CF pair accuracy .264 (p .77) and CF-probe .344. That is at most 2 points lower, and it removes the 9.4 MB per-block GDN states.

## 4. Latency [M]
A10G, bf16 fused runtime, CUDA graph per exact shape, 15 warm reps, fresh inputs, exclusive GPU, K/V-only composition. Medians in ms (p95 within 0.1 ms) for T state tokens plus 125 per question; c is the compiled share.

| T | 64 | 128 | 256 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|
| hobson, 1 q | 15.2 | 15.5 | 22.8 | 28.1 | 57.1 | 201.4 |
| compiled c=.37, 1 q | 15.0 | 16.3 | 16.4 | 23.7 | 45.0 | 140.0 |
| compiled c=.55, 1 q | 9.7 | 15.8 | 17.3 | 24.2 | 39.2 | 102.8 |
| hobson, 4 q | 48.5 | 48.9 | 55.5 | 67.2 | 93.3 | 243.5 |
| compiled c=.55, 4 q | 48.3 | 49.1 | 50.9 | 57.8 | 70.9 | 144.5 |

- **Real length distribution** (84 eval-split requests, exact shapes, first question):

  | | hobson mean (ms) | compiled mean (ms) | speedup |
  |---|---|---|---|
  | banking | 180.5 | 94.9 | **1.90x** (median per request 1.80x) |
  | LONG | 263 | 133 | 1.98x |
  | REAL-agree | 121 | 82 | 1.47x |
  | retail | 117 | 102 | 1.14x |

  Affine GDN composition adds 1-4 ms.
- **Below the token ratio at 1000 tokens or fewer [M].** 543 live rows cost 33.3 ms of GEMM against 46.0 ms for 1,000 rows (small-M efficiency), plus about 5 ms of cache assembly. There is no gain below 256 tokens.
- **Projections** [A] (GEMM time scaled by fp16-accumulation peak, 6.3 ms of weight streaming by bandwidth, calibrated to the doc's §9), 1 q, c=.55, hobson in parentheses:

  | T | 3090 | 4090 | 5090 |
  |---|---|---|---|
  | 1000 | 20.6 (29.6) | 10.8 (14.6) | 7.7 (10.7) |
  | 4000 | 52.6 (102.2) | 24.4 (45.5) | 18.5 (35.3) |

  W8A8 would multiply the GEMM part by a further 1.6x [S].

## Verdict
- **The axis is real [M].** Deployment constants are 46.5% of gate-state tokens (34% with a strict library) and about half of banking. Compiling them (loose library) gives 1.9x on banking latency, with no change to size or precision.
- **The order problem is solved exactly for GDN and RoPE [V].** The GDN part barely matters, because documents are read through attention, so a K/V-only library suffices.
- **It is not function-preserving.** The compile adapter keeps JevBench identical by construction and leaves CF and CF-probe statistically unchanged. It cuts TV by 30-33%, but REAL agree_sd is .908 (LONG .861).
- **It narrowly misses the architecture bar:** REAL-label .767 against ≥ .78 (n.s. against hobson, p .17), and CF pair accuracy .256 against .268 (n.s.).
- **The residual is the procedure questions,** caused mostly by moving the documents rather than by compiling them. It was still shrinking when training stopped.

## Decisive next step
Train the compile adapter 10x longer (about 16k examples, procedure-focused, K/V-only library), with a document-identity auxiliary loss (header ↔ compiled block). Gate on REAL-label ≥ .78, CF ≥ .268 and procedure agree_sd ≥ .95, then run it in the 8-bit runtime.
