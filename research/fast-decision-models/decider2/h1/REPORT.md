# H1 report (coordinator's condensed copy of H1's final message)

## Main results
- **W8A8-GPTQ8-qb16, all suites:**
  - 7 of 1083 REAL flips against hobson (0.65%), against the bf16 runtime's own 5 of 1083 (0.46%); paired p .69;
  - CF fgh 1.000, CF-probe fgh .990;
  - JB-hard .538 (McNemar against hobson: 0 lost, 2 gained);
  - REAL-label .790.

  This is at the noise floor. H2's runtime with the same recipe plus b8 gave 0.18%.
- **Noise floor.** Running a single GEMM in fp32, with no quantization, moves decisions by about as much as one W8A8 GEMM. The fixed 0.5% bar therefore sits at the floor; use TV and paired tests.
- **W8A8 error budget.** Weight and activation rounding contribute equally. GPTQ8 is the biggest lever (flips 14 → 7). Layer 23's down-projection carries 23% of the decision KL; layers 14–21 carry about 2%.
- **qb16** (question rows in bf16) cuts W8A8 TV to .0036, against a .0031 floor. It is free in the schema layout.
- **4-bit calibration KL against bf16 dense:**

  | config | KL |
  |---|---|
  | W8A8-GPTQ8 | 1.8e-4 |
  | W4A8-GPTQ | 4.7e-3 |
  | W4A4-GPTQ | 0.052 |
  | W4A4 RTN | 0.084 |
  | k48 mixed map (48 GEMMs at W8A8) | 0.0077 |

  b/a gates in bf16 and clip 0.85 each help about 13%. The rotation seed moves KL by about 15%.
- **Best 4-bit:** W4A4-k48 + GPTQ + b/a gates bf16 + qb16, emulated on a subset. REAL-SD flips 0.58%, CF fgh .972, CF-probe fgh .933. It misses the CF-probe bar narrowly. It has not been run on all suites in the deployed kernels.
- **LoRA-QAT did not move the eval metrics.** The W4A8 run cut train KL 3.5x without improving eval; the W8A8 run at lr 1e-4 got worse.

## Latency
Measured by H2 on the A10G, 1 question, plain layout, T = 1000 / 4000:

| precision | T = 1000 | T = 4000 |
|---|---|---|
| bf16 | 57.0 ms | 203.6 ms |
| W8A8-b8 | 36.3 ms | 142.1 ms |
| W4A4 | 23.5 ms | 84.7 ms |
| k48 | 29.2 ms | 110.0 ms |

## Next step
1. Run W4A4-k48 + qb16 on all suites in H2's runtime.
2. Re-register the low-bit bar against the bf16 runtime, using a paired test and TV rather than a raw 0.5%.
3. Get 4-bit accuracy from co-design (noise-aware training, schema-first with bf16 question rows) rather than post-hoc LoRA-QAT.

## Conduct
- Permission denied: stopping G2's leftover job queue on g2. H1 did not work around it.
- Files: `NOTES.md`, `FORMAT.md`, `code/` and `res/` are in this folder. The checkpoints and Hessians are on g2 at `~/work/h1`.
