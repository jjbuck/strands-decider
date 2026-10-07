# Brief 7: accurate low-bit decision models, co-designed with architecture and inference

For a principal engineer at AWS AgentCore. They want next-generation ideas, not incremental ones. Their pointers:
- rotated low-bit compute (QuaRot INT8/INT4);
- hyperspherical architecture co-design that makes 4-bit accurate (`~/Downloads/611_Introducing_Accurate_4_Bit.pdf`, ICLR 2026 submission);
- this-that-model-1.0 (arXiv 2609.23886): a 2B typed-decision model fine-tuned from decider-2b, with schema-first prefix caching and proper-scoring-rule training.

They are pointers, not limits. The goal is the decision model of the next hardware generation: compute at 4 bits on the tensor cores, accuracy kept by co-design, and only the work that varies per request computed at runtime.

## What is already measured (A10G, which has the same chip family as the RTX 3090)

- **hobson-v19, the reference decider.** A Qwen3.5-2B torso:
  - 24 layers: 18 Gated DeltaNet, plus attention at layers 3, 7, 11, 15, 19 and 23;
  - d = 2048, FFN 6144;
  - a pointer head over option tokens.

  The fused runtime runs it in **52.7 ms at 1000 tokens** (46 ms GEMM, about 6.7 ms other kernels) and 200 ms at 4000. The 3090 projection with fp16 accumulation is about 27 ms.
- **Int4/int8 tensor-core GEMMs at hobson's shapes.** These are G2's CUTLASS kernels. They are on box g2 at `~/work/g2`: `g2s4.cu`, `libg2s4.so`, `kbench.py`; results are in `res_prec.json`. A local copy is in `~/decider2/recovered/g2/g2/`.

  | precision | speedup over bf16, M = 1000 to 4000 |
  |---|---|
  | int8 | 1.64–2.36x |
  | int4 | **2.75–4.29x** |

  The GEMM shapes measured: gdn_in, attn_in, out, gate_up and down.
- **Rotated W8A8 with no training:** 1.20% decision flips on real requests, CF fgh 0.97, GEMM speedup 1.98x.
- **W4A4 QAT checkpoints** from G2 (LoRA r32 + STE, KL + residual MSE): `~/work/g2/lora_qat44_a_s{500,1000,1500}.pt` on g2. They have not been scored.
- **Compiled schema / question bundle:** question tokens first, with their GDN states, conv tails and K/V precomputed once per deployment. For 15 questions over a 1000-token state this took 214 → 85 ms in one runtime (F5). this-that-model uses the same mechanism.
- **Smaller or shallower models DO NOT COUNT.** hob12, the 0.8B and other depth or width reductions trivially speed inference by shrinking the model, and the engineer has rejected them as breakthroughs. Every speed claim must be for the full 24-layer hobson, or for a model of equal size and capability.
- **The arithmetic target for int4.**
  - On a 3090: about 22.8 ms of fp16-accumulation GEMM shrinks to about 6–7 ms at int4. Adding about 4.3 ms of non-GEMM work and 1.2 ms of host time gives about 11.5–12 ms for the 2B.
  - Shrinking the non-GEMM floor (fusing quantization into norms, the GDN kernels) and compiling the schema gives about 9–10 ms.
  - An RTX 5090 with NVFP4 has about 4x the 3090's int4 rate.
- **Supporting evidence:** NVFP4 W4A4 on all linear layers of a Gated DeltaNet hybrid matched BF16 within noise (arXiv 2609.04098).

## Evaluation (mandatory)

The kit is at `~/decider2/evalkit/`; read `README.md`. Put it on your box with `~/decider2/box.sh <BOX> put ~/decider2/evalkit/ evalkit`.

Suites:
- JB-hard (130 tasks), JB-long (77);
- REAL-agree, LONG;
- CF (counterfactual pairs), CF-probe (JSON-field probes);
- SHUF, REAL-label.

hobson's references and baselines are in `evalkit/refs`.

**Kill criterion, pre-registered.**
- **Low-bit numerics:** decision flips under 0.5% on REAL-agree, CF and CF-probe fgh of at least 0.97, and a paired McNemar test on JB-hard showing no significant drop. Low-bit methods compute the same function approximately, so they must be nearly lossless.
- **Architecture co-design** (same model size): state-dependent agreement of at least 0.95 on REAL-agree and LONG, CF and CF-probe fgh of at least 0.90, and JB-hard flagged if more than 3 points below hobson.
- **Every speed claim:** a measured A10G end-to-end time at exact 1000 and 4000 tokens, plus a 3090/4090/5090 projection with stated assumptions.

## Rules

- **Boxes.** Use your own box only, via `~/decider2/box.sh <BOX> run|put|get|status`. The old `/tmp/decider2` tree was lost in a laptop reboot; everything now lives in `~/decider2`. The boxes are g1–g5, all A10G, and they shut down at **05:43 PDT**.
- **Protect your results.** Copy results off the box to `~/decider2/<you>/` at least every 30 minutes. The laptop may sleep, which stalls you: run long jobs on the box with nohup, make them resumable, and log to files.
- **Commands.** One command under 30 minutes. Latency measurements need exclusive use of the GPU (no other jobs running), at least 12 warm reps, fresh inputs, and median plus p95. GPU memory under 22 GB.
- **No models on the laptop.** Do not touch other AWS resources. sudo on your box is for apt only.
- **arXiv.** Fetch `https://arxiv.org/abs/<id>` with WebFetch, and cite only IDs you fetched.
- **Notes** go in `~/decider2/<you>/NOTES.md` every ~15 minutes, with timestamps.
- **If a permission is denied,** do not work around it; report it.

## Report (≤ 1500 words; return it as your final message, and save it to `~/decider2/<you>/REPORT.md`)

Cover:
- the idea;
- what you built;
- measured latency and projections;
- every suite next to the bf16 hobson references;
- the verdict against the kill criterion;
- the decisive next step.

Mark each claim as verified, measured or speculation.
