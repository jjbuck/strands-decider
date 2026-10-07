# F7 recipe and results: dense equal-latency controls (reconstructed from F7's final report; the original files were lost)

## 1. Models
- **q08:** Qwen3.5-0.8B-Base. Hidden 1024, FFN 3584, 24 layers (18 GDN + 6 attention). 0.996 GFLOP per state token, 0.36x of hobson.
- **hob12:** hobson-v19's first 12 layers, LoRA merged, with a fresh pointer head and fresh LoRA. 0.50x the FLOPs of hobson.
- **q08L12:** the first 12 layers of the 0.8B, with attention at layers 3, 7 and 11. About 0.18x the FLOPs.

## 2. Recipe (v19 as closely as feasible)
- **Model setup.**
  - Pointer head of dim 256.
  - LoRA r16/α32 on every projection, including the GDN in_proj_* and out_proj.
  - LoRA dropout 0; bf16 adapters with fp32 master weights.
- **Optimizer.**
  - lr 1e-4, head lr 1e-3.
  - AdamW(0.9, 0.95), 3% warmup, cosine schedule, gradient clip 1.0.
  - 32 rows per step, 1 epoch.
  - Length-grouped batches, option shuffling, ordinal smoothing 0.1.
- **Data: 108,547 rows, 39.8M tokens.**
  - train_v5 minus v19's 3% stratified val split: 97,432 rows.
  - v19 synthetic documents: generated_v16 2,148, generated_v18 1,667, adequacy_gen 1,300.
  - 6,000 KL-only real states from `evalkit/train_pool.jsonl` (train split; zero eval tasks), one question each, up to 9.2k tokens.
- **Loss.**
  - CE(gold) + 1.0·KL(hobson ‖ student) on every labelled row, plus 1.0·KL on the real rows.
  - The teacher target is hobson's calibrated distribution (logits / T_kind).
  - Teacher and student see identical token ids; the tokenizers are verified identical.
  - hobson's labelling took 47 min, and its argmax matched gold on 0.887 of rows.
- **Throughput.** About 8k tokens/s on an A10G using:
  - the fla conv;
  - bf16 adapters;
  - compiled RMSNorms;
  - checkpointing only for long rows.

  The q08 run took 94 minutes; hob12 runs at about 1.15 s per step.
- **Plateau.** All arms plateaued over the last quarter of training. More data (all 46k train-pool pairs, plus multistep_v14, 2 epochs) is the lever.

## 3. Latency (A10G, fused lean2 runtime, LoRA merged, CUDA graph, exact T, n=20, p95 within 0.05 ms)

| model | 256 | 1000 | 4000 | GEMM / other at 1000 |
|---|---|---|---|---|
| hobson | 16.19 | 52.88 | 200.95 | 46.03 / 6.94 |
| q08 | 9.16 | 24.13 | 95.85 | 17.37 / 6.84 |
| hob12 | 8.14 | 26.59 | 101.04 | 23.01 / 3.61 |
| q08L12 | 4.61 | 12.14 | 48.20 | 8.69 / 3.48 |

- **3090 projections with fp16 accumulation:** hobson about 32, q08 about 15.7, hob12 about 16.3, q08L12 about 7.9 ms.
- **Packed multi-question passes (A10G, 1000-token state):**
  - q08: 54.3 ms for 1 question with a 1,014-token procedure question, 39.1 / 51.2 / 79.9 ms for 4 / 8 / 15 questions;
  - hobson: 111.3 / 83.4 / 103.5 / 154.4 ms.

## 4. Suites

| | hobson | q08 | hob12 | q08L12 |
|---|---|---|---|---|
| JB-hard | .523 | .469 | .546 | .415 |
| REAL agree_sd | 1 | .824 | .884 | .766 |
| LONG agree_sd | 1 | .867 | .885 | .873 |
| REAL-label | .785 | .767 | .735 | .748 |
| CF fgh | 1 | .661 | .780 | .532 |
| CF-probe fgh | 1 | .543 | .571 | .429 |
| SHUF | .339 | .336 | .323 | .289 |
