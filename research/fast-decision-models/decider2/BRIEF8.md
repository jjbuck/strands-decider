# Brief 8: transformative ideas for fast, accurate decision models

You are one of 10 research agents working for a principal engineer at AWS AgentCore. Each agent has a different direction. The engineer has rejected several rounds of work as incremental:
- pruning, eviction, token selection and routing, LoRA merging;
- smaller or shallower models;
- quantization tweaks.

**Assume that obvious, marginal or low-hanging-fruit ideas will not be fruitful.** Your job is to explore truly transformative ideas, ones that change the cost class or the capability of the model, and to back them with first-principles arithmetic and a decisive measured experiment. Be bold in the hypothesis and rigorous in the test. A clean negative result on a bold idea is worth more than a positive result on a timid one.

## Reimagine the transformer for this job
We explicitly do NOT need a general-purpose autoregressive generative model. We need a focused System-One decision model: f(state, questions, options) → a few calibrated probabilities. Question every piece of the decoder-LLM machinery that exists only to serve generation:
- the causal mask;
- next-token pretraining;
- the vocabulary-sized LM head and the embedding table;
- KV caches;
- per-token outputs at every position;
- uniform depth and width for every token;
- a generic tokenizer;
- reading the state before the question.

Ask what you would design if the only output were a ranking over K declared options, the input were often structured (JSON, tool outputs, conversations), most requests were short, and questions were fixed per deployment. Bold redesigns can still start from pretrained weights, so you do not need trillions of tokens to test them.

## Read first
- `~/code/jit-eval/docs/FAST_DECISION_MODEL.md`, in full. It covers everything measured so far, what failed and why; do not repeat it.
- Marc Brooker's two posts:
  - https://brooker.co.za/blog/2026/09/28/engineering-system-one.html (how hobson/strands-decider was built);
  - https://brooker.co.za/blog/2026/10/04/encoders.html ("Why not an encoder?").

  Fetch them with WebFetch.

**Key facts from the encoder post:**
- A T5Gemma 2.6B encoder with a pointer head and a masked state cache (e1b) ties hobson-v19 on JevBench accuracy (171 against 168 of 231), with slightly worse Brier.
  - The masked cache: state tokens attend to each other bidirectionally; each question attends to the state and to itself.
- On an RTX 3090 it serves with a **median of about 30–35 ms against about 60–65 ms for v19**. That compares Brooker's own serving stacks, not our fused runtime.
- The encoder is faster at short prompts and slower at long ones: 26 full-attention layers against v19's GDN hybrid, which has a higher fixed cost and better scaling.
- Real decision traffic is short. In JF100, 9 of 10 requests are under 400 tokens; JevBench's median input is about 105 tokens. Our gate logs have a median of 964 tokens with a long tail.
- e1a, the encoder without masking, beats v19 on 7 of 9 held-out sets but loses on JevBench and JF100.
- Brooker's open question: "Is being able to attend to the question when reading the state actually a weakness?"

**Our own anchors** (A10G, full 24-layer hobson, fused runtime `~/decider2/d1/lean2.py` plus `~/decider2/systems/g/lean.py`), in ms:

| tokens | 64 | 256 | 512 | 1000 | 4000 |
|---|---|---|---|---|---|
| bf16 | 9.4 | 15.7 | 28.3 | 52.7 | 200.5 |
| W8A8-GPTQ (passes fidelity) | | | | 36.3 | 142 |

- Int4 GEMMs run 2.75–4.3x faster than bf16, but 4-bit decisions are not yet accurate.
- The compiled question schema makes 15 questions cost about the same as 1, but it needs a trained schema-first model; the best so far reaches 0.69 agreement.
- hobson gets only 27% of counterfactual pairs right; it reads details poorly.
- The stock engine is 67.5 ms at 64 tokens, so Brooker's serving floor is largely framework overhead that our runtime already removed.

## The hard constraint
- **Smaller or shallower models do not count as answers.** A model must be about 2B parameters (or more), or must match hobson's accuracy at equal or greater capability.
- **A different architecture at equal size is fine**: an encoder, MoE, hybrid, low-bit-native model and so on.
- **Function-preserving systems ideas are fine** at any size.

## Evaluation (mandatory for any model you produce)
The kit is at `~/decider2/evalkit/`; read `README.md`. Put it on your box with `~/decider2/box.sh <BOX> put ~/decider2/evalkit/ evalkit`.

**Report absolute accuracy, not just agreement with hobson.** A different architecture may legitimately differ from hobson:
- JB-all and JB-hard, with a paired McNemar test against hobson;
- REAL-label (Opus labels; hobson .785);
- CF and CF-probe ground-truth pair accuracy (hobson .268 and .328);
- LONG: agreement plus REAL-label on long states;
- Brier and calibration.

**The bar for a new architecture:** no significant drop on JB-all or JB-hard, REAL-label ≥ .78, and CF / CF-probe accuracy ≥ hobson's.

**Function-preserving methods** use the fidelity bar instead: flips against hobson at the bf16 runtime floor (about 0.4%), with paired tests.

**Latency.** Measure on the A10G at exact 64, 128, 256, 400, 1000 and 4000 tokens, with 1 and 4 questions, using exclusive use of the GPU, at least 12 warm reps, fresh inputs, and median plus p95. Project to the 3090 (same per-SM bf16 rate as the A10G, 1.56x bandwidth, fp16 accumulation at 2x) and to the 4090 and 5090.

**Training data**:
- `~/decider2/training/data/train_v5.jsonl` (gold, the v19 mix);
- `~/decider2/evalkit/train_pool.jsonl` (real train-split states for KL distillation from hobson);
- the distillation recipe in `~/decider2/F7_REPORT.md`.

Never train on evalkit eval items.

## Infrastructure and rules
- **Box.** You own one box, named in your direction. A launcher is creating them; wait for `~/decider2/boxes/<BOX>.ready` by checking every ~5 minutes with `sleep 290`.
- **Box commands.** Use only `~/decider2/box.sh <BOX> run "<cmd>" | put <local> <remote dir under ~/work> | get <remote path under ~/work> <local> | status`.
  - The box has the strands-decider repo at `~/work/sd`, the venv at `~/venv` (`source ~/venv/bin/activate`), hobson-v19 and Qwen3.5-2B-Base in the HF cache.
  - The bundle contains systems/, tokens/, shrink/, training/, d1/, evalkit/, h2/code (the low-bit runtime), h4/tt_lean.py (the schema-cache runtime) and h7/code.
  - pip install anything else on the box.
- **Box lifetime.** It auto-terminates 10 hours after launch. Finish within about 8 hours of it being ready.
- **Commands.** One command must finish in under 30 minutes. Run longer work with nohup on the box and make it resumable. GPU memory must stay under 22 GB.
- **No models on the laptop.** Do not run any model or torch code locally, and keep weights and checkpoints on the box.
- **No other AWS resources.** Do not touch any, and do not launch instances.
- **Writing.** Write ONLY under `~/decider2/<you>/`:
  - `NOTES.md` with timestamps every ~15 minutes;
  - `DRAFT_REPORT.md`, because the harness blocks files named REPORT.md;
  - results JSON.

  Copy results off the box at least every 30 minutes. The laptop may sleep, which stalls you; jobs on the box keep running.
- **arXiv.** Fetch `https://arxiv.org/abs/<id>` with WebFetch, and cite only IDs you fetched.
- **Permissions.** If a permission is denied, do not work around it; report it.

## Report
At most 1500 words. Return it as your final message and also write it to `DRAFT_REPORT.md`. Cover:
- the transformative hypothesis, from first principles;
- why it is not marginal;
- what you built;
- measured results against hobson on every suite;
- latency curves across lengths;
- projections;
- the verdict;
- the single decisive next step if it survives.

Mark each claim as measured, verified or speculation.
