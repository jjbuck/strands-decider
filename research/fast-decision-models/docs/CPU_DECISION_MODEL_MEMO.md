# Memo: running the decision model on CPU, co-located with the agent

October 2026. Status: notes for later. Nothing here has been built yet.

## 1. The question

A Strands agent runs on AgentCore. Its LLM (Sonnet-class) is served by Bedrock over the network. Can the decision model (hobson-v19: a 2B Qwen3.5 GDN hybrid with a pointer head) run on the same host as the agent, on CPU?

The motivation comes from a Strands community advocate. He wants a pay-per-request option. Lambda is "not super fast", SageMaker Serverless won't run the model, Bedrock bring-your-own-model is expensive, and he doesn't want to run an EC2 for it. His attempt is at <https://github.com/ryancormack/strands-decider-lambda> (the repo returned 404 when fetched) with a write-up on AWS Builder Center (not fetched).

## 2. What the hosting options provide

From the AgentCore docs and quotas page, October 2026:

| | AgentCore microVMs | AgentCore Instances | Lambda |
|---|---|---|---|
| CPU / memory | **max 2 vCPU / 8 GB per session**, not adjustable | any EC2 type in your account | up to about 6 vCPU / 10 GB (general knowledge, not re-verified here) |
| Architecture | Linux containers, **arm64 only** | x86_64 or arm64 | x86_64 or arm64 |
| Accelerators | none | NVIDIA g4dn, g5, g6, g6e, gr6, g6f, gr6f, g7e; Inferentia2 (inf2) | none |
| Artifact size | Docker image ≤ 2 GB; direct code ≤ 250 MB zipped / 750 MB unzipped; session storage 1 GB | your EBS volumes | container image up to 10 GB (general knowledge) |
| Session | up to 8 h, idle timeout 15 min | **each session is its own EC2 instance**, up to 14 days, up to 20 agents per session | per-invocation |
| Billing | consumption-based; CPU billed only while active | EC2 pricing in your account | per request plus GB-seconds |

**Consequences:**
- In a microVM, the 2B weights (4 GB in bf16, 2 GB in int8, about 1 GB in int4) don't fit in the image. They have to be fetched at session start, from S3 or a persistent filesystem.
- A co-located GPU on Instances means one GPU per user session, because a session is an instance.

## 3. The physics on CPU

**Measured, agent J8** (c7i.8xlarge Sapphire Rapids, 16 pinned cores, AMX-BF16, torch inductor with prepacked weights, exact hobson function):

| state tokens | 64 | 256 | 1,000 | 4,000 |
|---|---|---|---|---|
| 1 question | 95.7 ms | 214.5 ms | 577 ms | 2,199 ms |
| 4 questions | 208.9 ms | 317.2 ms | 809 ms | 2,273 ms |

- **Fidelity:** 99.50% agreement with hobson on all 3,227 eval questions, at the bf16 noise floor.
- **AMX throughput:** oneDNN AMX-BF16 reaches 0.5–1.3 TFLOPS per core at hobson's shapes. int8 gave no consistent GEMM gain over bf16 in J8's microbenchmarks.
- **Cost per million decisions at 1,000 tokens:**

  | hardware | cost |
  |---|---|
  | c7i.8xlarge | $229 |
  | A10G | $17.7 |
  | Inferentia2 | $16.0 |

**Arithmetic for a 2-vCPU arm64 microVM** (Graviton-class; no AMX; bf16/i8mm via SVE):
- **Throughput:** about 0.05–0.15 TFLOPS sustained.
- **Work per decision:** hobson costs about 2.75 GFLOP per token, so:

  | input | estimated latency |
  |---|---|
  | median short decision (140 tokens) | about 3–6 s |
  | 1,000-token state | about 25–45 s |

- **It's compute-bound almost everywhere.** Streaming 2 GB of int8 weights costs about 100 ms at roughly 20 GB/s, while one row of compute costs about 20 ms, so compute dominates above about 5 rows.
- **On CPU, latency is FLOPs per decision.** That matches the advocate's "not super fast" on Lambda.

## 4. Levers, re-ranked for CPU

From the GPU program, in `docs/FAST_DECISION_MODEL.md` and `~/decider2/*/DRAFT_REPORT.md`, ordered by expected value on CPU:

| lever | effect on CPU | evidence | accuracy |
|---|---|---|---|
| **Incremental state across hooks** (keep the GDN recurrent state and attention K/V between hook calls in the same session; process only new tokens) | 5–20x fewer tokens per decision (arithmetic) | not measured. The GDN state is a fixed ~19 MB plus about 12 KB of attention K/V per token. This was out of scope for the GPU study, but it is legitimate when co-located. | exact, if the renderer keeps the state prefix stable and puts hook-specific text after it |
| **Questions compiled into weights** (J6 "late": per-question delta on the K+1 answer rows only) | question cost goes from 100–1,000 tokens to about 3–10 rows | measured on GPU: 2–5x for multi-question requests | REAL-label unchanged; agreement .86 (procedure questions are the residual) |
| **Precompiled deployment documents** (J9 compile adapter) | 34–46% of tokens become compile-time | measured on GPU: banking 1.9x | REAL-label .767 (n.s.) |
| **Depth-split** (J3 DT-A8: state rows through 8 of 24 layers) | 0.35x state FLOPs | measured on GPU: 1.7–2.4x | passes the new-architecture bar; weaker on deep deduction (RuleTaker d3) |
| **Layer-16 early exit** (J15) | about 0.69x | measured on GPU: 0 of 6,216 real decisions changed | near-lossless |
| **Domain vocabulary** (J7 super-tokens) | 1.6–2.4x fewer tokens | measured on GPU: 2.0x on real traffic | REAL-label .767 with a short LoRA (n.s.); needs full fine-tuning |
| **int8 W8A8** (AMX-INT8 on x86, VNNI, i8mm on Graviton3/4) | about 2x over bf16 in principle | measured on GPU: rotated W8A8 + GPTQ near the noise floor (0.65–1.1% flips depending on calibration draw). J8 saw no consistent int8 gain on AMX. | near-lossless |
| **Compile and fuse** (OpenVINO, ONNX Runtime, inductor freezing) | about 1.2–2x over eager | J8 used inductor. GDN needs custom CPU kernels: the chunked scan under inductor was pathological (2.6 s at 1,000 tokens). | exact |
| **Unstructured sparsity with CPU sparse kernels** (DeepSparse-style) | about 2–4x | CPU-only lever (GPUs can't exploit it); not measured here | needs a sparse fine-tune; pruning-class loss |
| **Ternary / 2-bit weights with lookup-table kernels** (T-MAC, bitnet.cpp) | potentially 2–4x over int8 | CPU-only lever (GPU found no gain) | J10: native BitNet-2B is significantly worse; ternary hobson collapses without billions of retraining tokens |
| **4-bit weights** (GGUF Q4_K, GPTQ) | about 1x for prefill (memory and bandwidth only) | unpacks to int8 before the dot | 2–6% flips at 2B (StartLux: Q4_K_M at 94% agreement at 2B) |
| **Smaller model** (0.1–0.3B) | 7–25x | F7 controls: 0.8B and 12-layer variants lose 4–12 agreement points | the blunt instrument; acceptable only under the latency-first rule |

**Stacked estimate (arithmetic, unverified)** for a hook decision with incremental state, compiled questions and documents, depth-split, early exit, int8 and compile:
- 2-vCPU microVM: about 0.1–0.5 s.
- 16-core AMX instance: about 10–40 ms.

Compressing the model alone (compile + int8 + sparsity or LUT) is about 4–10x. That leaves a 2-vCPU microVM at roughly 3–10 s at 1,000 tokens.

## 5. Economics
- **Shared accelerators win on cost.** A multi-tenant decision endpoint on GPU or Inferentia2 batches across tenants and costs about $16–18 per million decisions at 1,000 tokens, even before batching. J8 measured the 16-core CPU at $229.
- **On-device wins only on network latency, data locality and operational simplicity.**
- **The advocate's real ask is a pay-per-request, serverless, hosted decider:** a managed "decision" primitive in Bedrock or AgentCore, as Jev offers. Batching makes that 10x+ cheaper than per-session CPU.

## 6. Proposed experiment (about 3 h, one agent, two boxes)
1. **Boxes:**
   - c8g.large (2 Graviton4 vCPU), as a proxy for the microVM;
   - c7i.4xlarge or c7i.8xlarge (AMX).
2. **Engines,** each at exact 64, 256, 1,000 and 4,000 tokens, 1 and 4 questions, on both boxes:
   - llama.cpp GGUF (BF16, Q8_0);
   - ONNX Runtime and OpenVINO int8;
   - torch inductor bf16/int8.

   GDN support has to be checked per engine; llama.cpp has Qwen3.5 hybrid support.
3. **Incremental state:** replay real tau3 agent traces hook by hook, carrying GDN state and attention K/V. Measure new tokens per hook and per-hook latency.
4. **Compiled questions:** reuse J6's "late" adapters (on box j6's checkpoints, if still available; otherwise retrain).
5. **Fidelity on every engine:** evalkit flips against hobson, CF and CF-probe fgh, JB-hard McNemar.
6. **Report:**
   - latency against length;
   - per-hook latency on real traces;
   - cost per million decisions against an A10G and an inf2 endpoint;
   - a go/no-go for the microVM.

**Kill criterion:** if the best stacked configuration can't reach ≤ 300 ms per hook decision on the 2-vCPU box at hobson-level fidelity, recommend Instances (CPU or small GPU) or a shared endpoint instead.

## 7. Open questions
- What CPU does an AgentCore microVM actually get (Graviton generation, SVE width, i8mm/bf16 support)? Is "2 vCPU" two physical cores?
- Is the hook state append-only in practice? The steering hooks inject and transform messages, so the renderer may need changes.
- How large is the session-start cost of fetching weights (1–4 GB) into a microVM? Does Runtime V2 snapshot start help?
- Could AgentCore host a shared, batched decision endpoint as a managed feature (the advocate's ask)?
