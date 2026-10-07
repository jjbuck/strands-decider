# G3 NOTES: sparse capacity (MoE) as the architecture lever, batch-1 prefill

## [15:40 PDT] start (box g3 not launched yet)
- Read BRIEF6, BRIEF5, FAST_DECISION_MODEL.md, f8/NOTES.md, evalkit README, f7/NOTES.md + f7 code (prep/teacher/train/ev/score/lat), d1 lean2.py + NOTES.
- F7 recipe to replicate: corpus = train_v5 (97,432 after 3% val split) + v19 synthetic docs (5,115) = 102,547 labelled rows + 6,000 real-state KL-only rows
  (train_pool, seed 5, 1 random question each); 39.8M tokens; 1 epoch = 3,392 steps x 32 rows; pointer head (dim 256) + LoRA r16 on all projections,
  lr 1e-4 (head 1e-3), 3% warmup + cosine, AdamW(0.9,0.95), clip 1; loss CE(gold, ordinal-smoothed) + 1.0*KL(hobson||student) on every row.
- F7 q08 final (dense 0.8B, 23.9 ms fused): JB-hard 0.469, REAL agree_sd 0.824, LONG agree_sd 0.867, CF fgh 0.661, CF-probe fgh 0.543, REAL-label 0.767.

### Hub scan (metadata only, laptop; no models run) [verified from HF API]
| model | total | active (body) | layers x d | experts | licence | transformers |
|---|---|---|---|---|---|---|
| ibm-granite/granite-3.1-1b-a400m-base | 1.33B | ~0.38B | 24 x 1024 | 32, top-8, ffn 512 | Apache-2.0 | native (granitemoe) |
| ibm-granite/granite-3.1-3b-a800m-base | 3.30B | ~0.80B | 32 x 1536 | 40, top-8, ffn 512 | Apache-2.0 | native |
| Tiiny(PowerInfer)/SmallThinker-4BA0.6B-Instruct | 4.27B | ~0.63B | 32 x 1536 | 32, top-4, ffn 768, ReLU-GLU, pre-attn router | Apache-2.0 | remote code (written for 4.53; imports HybridCache/LossKwargs) |
| utter-project/EuroMoE-2.6B-A0.6B-2512 | 2.61B | ~0.37B | 24 x 1024 | 64, top-8, ffn 512 | Apache-2.0 | native (mixtral); multilingual; 4k ctx |
| microsoft/Phi-tiny-MoE-instruct | 3.76B | 1.1B | 32 x 4096 | 16, top-2 | MIT | remote; 4k ctx; d=4096 makes attention 0.67B |
| allenai/OLMoE-1B-7B-0125 | 6.92B | 1.3B | 16 x 2048 | 64, top-8 | Apache-2.0 | native; 14 GB read/req |
| LiquidAI/LFM2.5-8B-A1B(-Base) | 8.47B | ~1.5B | hybrid | | LFM licence | native; 17 GB read/req |
| arcee-ai/Trinity-Nano-Base | 6.12B | ~1B | | | other | afmoe |
| ibm-granite/granite-4.0-h-tiny-base | 6.94B | ~1B | hybrid mamba | | Apache-2.0 | native |
- Published quality: granite-3.1-3b-a800m-base MMLU 48.3, granite-3.1-1b-a400m-base MMLU 26.5 (near chance), granite-3.1-2b dense 52.9 (granite model card).
  SmallThinker-4BA0.6B-Instruct MMLU 66.1 (0-shot CoT) vs Qwen3-1.7B 64.2 / Qwen3-0.6B 43.3 in its own card. => SmallThinker is the strongest per active FLOP.
- Weight-read floor per request on A10G at ~510 GB/s achieved (bf16): granite-1b 2.6 GB -> 5 ms; granite-3b 6.4 GB -> 12.5 ms; SmallThinker 8.2 GB body -> 16 ms;
  OLMoE 13.8 GB -> 27 ms (already > dense 0.8B's 23.9 ms total); LFM2.5-8B 17 GB -> 33 ms. Only totals <= ~4B are in the latency game.
- Literature [verified via arxiv_q]: Ling-mini-beta (2507.17702) 0.85B active matched a 6.1B dense at 1T tokens (EL ~7x) -- but at an activation ratio
  of ~1/32 the total is ~17B params => ~34 GB per request read at batch 1. Joint MoE scaling laws (2502.05172): MoE can be memory-efficient too.
  Fine-grained MoE scaling (2402.07871). OLMoE 2409.02060. SmallThinker 2507.20984. MegaBlocks 2211.15841.

### Plan
1. Box: install; download SmallThinker, granite 1b/3b, EuroMoE (OLMoE shapes only, synthetic weights, for the latency model).
2. Own implementation of the MoE decoder (no remote code): per-model config (granite multipliers; SmallThinker pre-attention router, ReLU-GLU,
   sigmoid-normalised top-k). Validate against HF (granite native; SmallThinker against its remote code if it imports, else against a reference loop).
3. Inference runtime: CUDA graph, my Triton grouped GEMM (sorted tokens, per-tile expert id, fused SwiGLU/ReLU-GLU epilogue, gate-weighted
   scatter-add in the down GEMM) + torch._grouped_mm and per-expert cuBLAS loop as references; real text routing (latency is data dependent for MoE).
   Split: weight-read floor (bare streaming of the same bytes) vs compute (GEMM FLOPs at measured rate).
4. Train SmallThinker (primary) into a decider with F7's recipe: same corpus/rows (F7 prep code, same seeds, re-tokenised twin), hobson teacher logits
   computed on the Qwen-tokenised twin (identical to F7's), LoRA r16 on attention + per-expert LoRA on all expert projections, pointer head.
5. evalkit all suites; plot quality vs latency with hobson, q08, (F7 hob12 / q08L12 if published).

## [16:00 PDT] box g3 launched 15:58 (pending setup). Coordinator request (scout g0 row 3) added:
1. synthetic stack ~2.2B total / 0.33B active: L24 x d1024 (H16 hd64 KV4), E=64 top-8, ffn_e=448 (8 x 448 = 3584 = Qwen3.5-0.8B's FFN width,
   so its active-equivalent dense twin is the 0.8B's shape), 125 rows/expert at 1000 tokens; grouped GEMMs bf16, W8A16, W4A16 experts.
   Dense twin (same attention, FFN 3584) timed with the same runtime.
2. best pretrained small MoE fine-tuned vs dense control at equal active params (SmallThinker 0.63B active vs F7 q08 0.5B body).
3. optional: per-question-spec routing (expert subset per spec).
Kill: latency > 0.4x fused 2B (21 ms A10G) or JB-hard > 3 pts below control.
Code written while waiting: moe.py (torso, LoRA, grouped_mm), moe_lean.py (runtime + Triton grouped GEMMs), bench_lat.py, prep_g3.py, train_g3.py, ev_g3.py, vllm_bench.py.
