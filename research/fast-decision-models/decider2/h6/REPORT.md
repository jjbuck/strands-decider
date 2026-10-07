# H6 report (coordinator's condensed copy of H6's final message)

## Mixed precision through the deployed kernels: all 3227 questions, hobson layout
Every 4-bit config uses GPTQ codes and bf16 GDN b/a gates. qb = question rows in bf16. kN = the N most sensitive GEMMs at W8A8, the rest at W4A4.

| config | % of GEMM work at 4-bit | REAL flips | TV | paired vs bf16 (lost/gained) | CF fgh | CF-probe fgh | JB-hard |
|---|---|---|---|---|---|---|---|
| H2 bf16 | 0 | 0.37% | .0033 | – | 1.000 | .971 | .538 |
| k48, all rows | 53 | 2.59% | .0246 | 28/4 | .972 | .848 | .538 |
| k48+qb | 53 | 1.20% | .0096 | 10/1 | 1.000 | .924 | .538 |
| k56+qb | 46 | 1.02% | .0056 | 8/1 | .982 | .943 | .531 |
| **k64+qb** | 38 | **0.46%** | **.0040** | **2/1** | **1.000** | **.962** | **.523 (0/0)** |
| W8A8-b8 (H2) | 0 | 0.18% | .0047 | 2/4 | 1.000 | .971 | .523 |

- **k64+qb sits one CF-probe pair short of the .97 bar** against hobson's references. Against the in-runtime bf16 it passes every test (0.28% flips, CF .973, CF-probe .971).
- **The answer rows must stay high-precision.**
- **A more aggressive map fails:** W4A4 everywhere except layer 0 and the attention projections gives 2.03% flips.

## QAT
No evaluation gain, in either recipe.
- LoRA on all rows at lr 1e-4: worse, 5.6% flips at step 300.
- LoRA on state rows only: decisions reshuffle within noise; TV is unchanged at .010–.011.

## Schema-first
- **Untrained hobson does not work in a schema-first layout.**
  - single slot: behaves like the no-state baseline;
  - H7's slot sets: 57% flips.
- **The bf16 schema control did not learn the layout in budget.** After 75 updates: 28% flips, agree_sd .36.

## Latency (A10G, ms)

| config | T=1000, 1q | T=4000, 1q | T=1000, 15q | T=4000, 15q |
|---|---|---|---|---|
| bf16, hobson layout | 56.9 | 203.6 | 237.4 | 398.2 |
| bf16, schema sets | 52.8 | 199.4 | 61.3 | 217.1 |
| W8A8-b8, hobson layout | 36.2 | 142.3 | – | – |
| W8A8-b8, schema sets | 33.6 | 139.0 | 40.6 | 155.9 |
| k64+qb, hobson layout | 40.4 | 128.9 | – | – |
| k64, schema sets, bundle and slots bf16 | 34.2 | 123.9 | 46.6 | 144.4 |
| W4A4 throughout, schema mixed rows | 27.9 | 89.5 | 39.7 | 111.8 |

- **The bf16 slot rows cost about 5 ms:** they read a bf16 copy of every weight plus about 1,100 small kernels.
- **3090 projections:** k64 mixed 22.2 ms at T=1000 and 73.6 at T=4000; W8A8-b8 schema 18.4 / 77.4; bf16 29.5 at T=1000.

## Verdict
- No configuration with W4A4 GEMMs passes against hobson's references. k64+qb misses CF-probe by one pair.
- The 4-bit pass/fail boundary is between 38% and 46% of GEMM work.
- Speed gain over W8A8-b8: none at 1000 tokens, 11% at 4000.

## Next step
1. One grouped GEMM per projection for the state and slot rows, removing the bf16 weight read (estimate −13–16% against W8A8-b8).
2. 16-element block scales (NVFP4) to raise the 4-bit share.
3. A trained schema-first model.
