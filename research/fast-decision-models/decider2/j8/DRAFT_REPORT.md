# j8: hobson-v19, unchanged, on AWS silicon (Inferentia2 and Sapphire Rapids AMX)

Labels: **[M]** measured on my boxes (j8a inf2.8xlarge, j8b c7i.8xlarge); **[V]** verified from a primary source; **[D]** from FAST_DECISION_MODEL.md (A10G anchors, 3090 projections); **[S]** arithmetic or speculation.

## 1. Hypothesis, from first principles

Prefill of a 2B decider is compute-bound, so the cost class is set by peak FLOPs per dollar at batch 1.
- **Inferentia2.** 190 BF16 TFLOPS per chip, 2 NeuronCore-v2 per chip (TensorEngine ">90 TFLOPS" each), 32 GiB HBM **[V, Neuron docs]**. inf2.xlarge carries the same single chip for $0.76/h; inf2.8xlarge costs $1.97/h **[V, aws.amazon.com]**. That is 250 peak TF·h per dollar, against about 58 for the A10G (≈70 TF at $1.212/h): a **4.3x lever [S]**.
- **AMX.** On c7i.8xlarge (16 cores, $1.428/h), oneDNN AMX-BF16 GEMMs at hobson's shapes reach 8.4–20.5 TFLOPS at M=1000, i.e. **0.5–1.3 TF/core**. The brief's 1–2 TF/core is not reached. At M=4000 they reach 4.3–6.2 TFLOPS **[M]**. int8 (`torch._int_mm` and oneDNN dynamic quant) reaches 5–21 TOPS, no consistent gain over bf16 **[M]**. That is 6–14 TF·h per dollar, **4–10x worse than the A10G before any overhead [S]**.

So the bold bet is that the same function on AWS's own accelerator moves cost per decision by up to 4x. The CPU cannot win on cost, whatever the software.

## 2. Why this is not marginal

It changes the silicon and the price per FLOP, not the model: hobson's weights and function, no accuracy trade, hardware AgentCore can buy.

## 3. What I built

- **`hob.py`: a pure-torch hobson forward.** The LoRA is merged in fp32 from the raw safetensors. No fla/Triton; traceable.
  - **GDN with matmuls only.** (I+L)⁻¹ is computed by recursive 2×2 block inversion, which is exact block forward substitution.
  - **The obvious Neumann/doubling inverse is unstable.** Binomial growth took the state to 3e16 by layer 8 **[M]**.
  - **Validation.** It matches an fp64 recurrent reference to 4e-8 **[M]**, and the stock strands-decider CPU engine to max|dp| ≤ 0.0015 **[M]**.
  - **Packed path.** A state-once + M-question path with masked state padding is identical to separate passes **[M]**.
- **Inferentia2** (torch-neuronx 2.8 trace, neuronx-cc 2.23; NxDI has no GDN model).
  - **What compiles.** The whole model compiles as one graph. Sequences are right-padded to multiples of 128, which is exact for the real rows because everything is causal. Unpadded lengths hit a pathological lowering: 100–180 ms per GDN layer **[M]**.
  - **Custom op: an NKI GDN kernel** (`gdn_nki.py`). One program per head, C=D=128, fp32. It has the block inverse, initial and final state, and a two-phase schedule (state-independent tiles first, then a sequential state pass).
    - **Exactness.** Hardware output matches the fp64 recurrence to 3e-7 **[M]**.
    - **Speed.** 4.1 ms device time for 16 heads at T=1152 **[M]**.
  - **`HobNL`, a layout for Neuron.** Head-major projections via batched matmul, pre-transposed weights, per-head einsum output projections, and NKI GDN. Single-sequence and packed versions are exact against `hob.py` on CPU **[M]**.
- **SPR.** `hob.py` in bf16 under torch 2.14 inductor with freezing, which uses prepacked oneDNN AMX linears. It runs on 16 pinned cores with jemalloc.

## 4. Decisions against the GPU reference

Evalkit refs. Subset, seed 0: all 231 JevBench, 300 random REAL-agree questions, 120 CF pairs, 80 CF-probe pairs.

| | n | agree | flips | TV | JB-all / JB-hard acc (hobson .7229 / .5231) | McNemar vs hobson | CF fgh | CF-probe fgh | REAL-label (hobson-same) |
|---|---|---|---|---|---|---|---|---|---|
| SPR bf16, exact lengths **[M]** | 931 | .9957 | 4 | .0031 | .7229 / .5231 | 0/0 | 1.000 | .967 | .7434 (.7345) |
| inf2 HobNL, ≤4096 tokens **[M]** | 837 | .9928 | 6 | .0027 | .7359 / .5462 | 3/0, p .25 | 1.000 | .966 | .7455 (.7364) |

- **Counterfactuals.** flip_rel against hobson is 0.97–1.03 on both ports **[M]**. Absolute accuracies on the full kit are below.
- **Where the inf2 flips are.** All 6 lie where hobson's top probability is 0.36–0.51.
- **Against the floor.** The bf16 runtime floor is 0.37–0.46% flips. inf2 has 6 flips where about 3.5 are expected at the floor (Poisson p ≈ 0.14), so both ports pass the function-preserving bar.
- **Coverage.** 94 subset questions above 4096 tokens are not covered on inf2.
- **Full kit on SPR (all 3,227 questions) [M].** Agreement .9950 (16 flips), TV .0031. REAL-agree .9963 (agree_sd .9971) and LONG .9958 (agree_sd .9879), both identical to the GPU merged_full runtime. CF fgh 1.000, acc .596 (hobson .594). CF-probe fgh .962 (merged_full .971, one pair). REAL-label .7875 (hobson .785). SHUF both_right .251 (hobson .246).

## 5. Latency

Median ms. Warm, fresh inputs each call, n=20–30, p95 within 1 ms unless noted. inf2 is one NeuronCore, with padding to a multiple of 128 included.

| state tokens | A10G fused bf16 **[D]** | inf2 HobNL, 1 q **[M]** | inf2, 4 q packed **[M]** | SPR 16 cores, 1 q / 4 q **[M]** |
|---|---|---|---|---|
| 64 | 9.4 | **30.9** | 94.8 | 95.7 / 208.9 |
| 256 | 15.7 | **43.1** | 110.0 | 214.5 / 317.2 |
| 1000 | 52.7 | **152.0** | 181.1 | 577 / 809 |
| 4000 | 200.5 | **537.5** | – | 2199 / 2273 |

- **What the inf2 port went through at T=1000** **[M]**:
  - **Naive XLA graph:** 457 ms, with 48.5 GB of SBUF spill traffic and DMA 88% busy.
  - **With the NKI GDN:** 416 ms.
  - **With HobNL layout:** 169 ms.
  - **With the two-phase kernel:** 152 ms.
- **Decomposition at T=1000.** The same graph with the GDN core skipped runs in 80.2 ms, so **GDN takes 72 ms (47%)**, or 4.0 ms per layer, against 3.4 ms total for GDN on the A10G with fla **[M/D]**. The rest of the model reaches about 40 effective TFLOPS.
- **The NKI kernel is latency-bound, not FLOP-bound.** A bf16-operand variant runs in 3.71 ms with engines only 19–24% busy **[M]**.
- **Two NeuronCores in parallel** (synchronized 18.7 s windows): 169.7 ms each, against 168.6 ms alone. There is no interference **[M]**.
- **SPR profile at T=1000.** AMX linears about 39%, fused elementwise about 36%, and the GDN chunk loop (1530 small fp32 bmm calls) about 23% **[M]**.

## 6. Cost per million decisions

Latency-bound serving, one request per core, on-demand prices **[S on M/D/V]**.

| T | A10G | A10G W8A8 | inf2.xlarge, both cores | inf2.xlarge, 1 core | c7i.8xlarge |
|---|---|---|---|---|---|
| 64 | $3.16 | – | **$3.26** | $6.52 | $38.0 |
| 256 | $5.29 | – | **$4.55** | $9.10 | $85.1 |
| 1000 | $17.74 | $12.22 | **$16.04** | $32.09 | $229 |
| 4000 | $67.5 | $47.8 | **$56.7** | $113.5 | $872 |

- **4-question requests on inf2.xlarge** cost $10.0, $11.6 and $19.1 per million requests at T = 64, 256 and 1000.
- **inf2.8xlarge** is 2.6x the inf2.xlarge figures. inf2.xlarge assumes its 4 vCPUs cover tokenization and ~0.7 ms host work per call **[S]**.
- **Against the 3090.** Its projected 27.4 ms (bf16 with fp16 accumulation) and 18.3 ms (W8A8) at T=1000 **[D]** are 5.5–8.3x faster per request than one NeuronCore.

## 7. Verdict

- **AMX CPU: clean negative [M].**
  - **Latency.** 96 ms at 64 tokens and 577 ms at 1000 tokens on 16 cores: 10–14x the A10G's latency.
  - **Cost.** 12–16x the A10G's cost per decision.
  - **The ceiling.** The measured AMX rate (≤1.3 TF/core) bounds even a perfect runtime at about 200 ms for T=1000 **[S]**.
  - **Colocation.** It only makes sense if 16 otherwise idle cores can be burned for half a second per decision.
- **Inferentia2: an exact port and cost parity today, with a latency penalty [M].**
  - **Fidelity.** The full GDN-hybrid forward runs at the noise floor.
  - **Cost.** inf2.xlarge with both cores matches or beats the A10G's bf16 cost per decision at every length (0.84–1.03x).
  - **Latency.** Single-request latency is 2.7–3.3x the A10G's.
  - **The gap.** The 4.3x FLOPs-per-dollar lever is not yet realized. The gap is almost entirely the GDN kernel (72 of 152 ms) plus GEMM efficiency.
  - **Projection [S].** With a GDN as cheap as fla's, T=1000 projects to about 84 ms per core, or $8.9 per million (0.5x the A10G). Adding A10G-level GEMM efficiency gives about 60 ms and $6.3 per million (0.36x).
  - **Is it a cost-class change?** Not demonstrated. Parity is.

## 8. The single decisive next step

**Write a pipelined multi-head NKI GDN kernel** that interleaves 2–4 heads per program, keeps all state-independent tiles in bf16, and double-buffers chunks. Then re-measure T=1000 on one NeuronCore.
- **Threshold.** ≤90 ms makes inf2.xlarge half the A10G's cost per decision at identical decisions, which is a cost-class result.
- **Kill test.** If GDN cannot drop below about 1 ms per layer, inf2 stays at parity and the GPU remains the default.
