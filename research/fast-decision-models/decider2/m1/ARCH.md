# M1 architecture: all-attention hobson (for N1's Inferentia timing)

Status: **v1, 2026-10-06 23:40 PDT. Final shapes; training uses exactly this.** (v0 had 136-wide heads; replaced, see the end.)

## What changes and what does not

hobson-v19 = Qwen3.5-2B torso, 24 layers, width d = 2048, SwiGLU MLP 6144, pointer head. Embedding vocab 248,320 (lookup only).

- **Unchanged:** the 6 full-attention layers (3, 7, 11, 15, 19, 23), every MLP, every RMSNorm, the embedding, the final norm, the pointer head.
- **Changed:** the mixer of the 18 former GDN layers (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22).

## The new mixer (one per former GDN layer)

Shapes per request of T rows (batch 1).

| step | op | shape | notes |
|---|---|---|---|
| 1 | pre-norm | zero-centred RMSNorm, [T, 2048] | as hobson; folded into the next GEMM in the fused runtime |
| 2 | in-projection GEMM | [T, 2048] x [2048, 8224] -> [T, 8224] | same shape as hobson's GDN in_proj: q 2048, k 2048, v 2048, z 2048, b 16, a 16 |
| 3 | short causal depthwise conv, kernel 4, + SiLU on q, k, v | 6144 channels | kept: dropping it raised the untrained layer error from 3.9 to 70 (relative MSE) |
| 4 | split heads | q, k, v each [T, 16, 128] | **16 heads, head dim 128, MHA (16 KV heads, no GQA)** |
| 5 | q/k norm | per-head RMSNorm over dims 0..121 with a learned gain [16, 128] | the q gain carries the softmax temperature |
| 6 | RoPE | dims 0..63 of each head, rotate-half pairing (d, d+32), theta 1e7, 32 frequencies | identical to hobson's attention layers (they rotate 64 of 256 dims) |
| 7 | gate biases, in-head | dims 122..127 of q and k: q = [G1, G2, G3, 1, 1, 1], k = [1, 1, 1, B1, B2, B3] | G = cumsum_t g_t per head, g_t = -exp(A_log) softplus(a_t + dt_bias) (GDN's decay gate); B = log sigmoid(b) - G (GDN's write gate and the decay's key side). Each value is split into 3 bf16 parts so the logit gets q.k/sqrt(128) + (G_i - G_j) + log sigmoid(b_j) to fp32 precision. q content dims are pre-scaled by 1/sqrt(128), attention scale 1.0 |
| 8 | causal softmax attention | 16 heads x 128, T x T causal | plain FlashAttention / SDPA. Question rows of a multi-question request attend to the state + their own question |
| 9 | output gate | per-head gated RMSNorm: gn_w[128] * rmsnorm(o) * SiLU(z) | identical to hobson's GDN output gate |
| 10 | out-projection GEMM | [T, 2048] x [2048, 2048] | as hobson |

Then residual add, post-norm and the unchanged SwiGLU MLP: [T, 2048] x [2048, 12288] (gate|up), SiLU*mul, [T, 6144] x [6144, 2048].

Small per-layer extras: the decay glue ([T, 32] elementwise) and one cumulative sum over T per head ([16, T], innermost dimension).
On the A10G these cost nothing measurable once the scan runs over the innermost dimension (an outer-dimension torch.cumsum cost 0.6 ms per layer at T = 4000).

## Arithmetic (per request of N rows)

- GEMMs unchanged: 2 x 57.0M multiply-adds per row per layer, as hobson.
- Attention core per converted layer: about 2 N^2 x 2048 FLOPs causal (QK^T and PV at 16 x 128). hobson has this in 6 layers (8 x 256 query width, GQA 2 KV heads); M1 in 24.
- Removed: the GDN chunked delta rule (triangular solve) in 18 layers.

## Measured on the A10G (fused bf16 runtime, CUDA graph, 20 reps, untrained-conversion weights)

| T (one sequence, 1 question) | hobson | M1 | M1 / hobson |
|---|---|---|---|
| 64 | 15.56 ms | 15.08 ms | 0.97 |
| 256 | 25.36 | 25.26 | 1.00 |
| 1000 | 57.10 | 56.54 | 0.99 |
| 4000 | 201.34 | 210.99 | 1.05 |

At T = 4000 the attention kernels take 30.9 ms in M1 (1.26 ms per converted layer) against 8.2 ms for hobson's 6 attention layers, and the GDN scan's 13.1 ms is gone.

## For Inferentia timing (N1)

Random weights of these shapes are enough:
- 24 layers; every layer = Win GEMM, mixer, Wo GEMM, MLP.
- Former-GDN layers: Win [8224, 2048]; depthwise conv k=4 on 6144 channels + SiLU; per-head RMSNorm on q and k; RoPE on 64 of 128 dims;
  **16-head MHA, head dim 128, causal**; gated RMSNorm with SiLU(z); Wo [2048, 2048]. The 6 bias dims are inside the 128 (no extra width);
  for timing you may fill dims 122..127 with any values.
- Attention layers: as hobson (Win [5120, 2048] = 8 q heads x 256 + a 256-wide sigmoid gate per head, 2 KV heads x 256; q/k RMSNorm; RoPE 64 of 256).
- Exact lengths T = 64, 256, 1000, 4000 (+ one question of about 100 tokens, or 4 questions as branches).

## Change log

- v0 (22:30): gate biases in 8 extra dims (head dim 136). Measured on the A10G: 136 is not a native FlashAttention size, M1 took 236.5 ms at T = 4000
  against 210.6 ms with 128-dim heads. v1 moves the 6 bias dims inside the head (122 content dims); untrained layer fidelity unchanged (3.927 vs 3.926 relative MSE).
