# Q2 code: build and use

Box layout: copy this folder to `~/work/q2/` (`box.sh q2 put ~/decider2/q2/code/ q2/`). H2's code must be at `~/work/h2/` (bundle `h2/code/*` copied up one level) with its CUTLASS libs built (`setup_q2.sh` does all of this, plus H1-recipe GPTQ codes).

## Build

```
cd ~/work/q2
# q2gemm: all variants (~3 min); add -DQ2_FAST to build only plain s4 (10 s) for main-loop experiments
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr --expt-extended-lambda \
    -DNDEBUG -shared -Xcompiler -fPIC -Xptxas -v -o libq2gemm.so q2gemm.cu > build.log 2>&1
source ~/venv/bin/activate
python test_q2.py 0,1,2,3,4,5,6,7,8,9      # every variant x config against the FORMATS.md torch reference; prints ALL OK
python bench_pro.py test                     # Triton prologues (RTN, SR, dX, g64, split, row-role, B6 stat, B3 search) vs torch
```
No CUTLASS headers are needed for q2gemm (raw PTX). `q2sp.cu` (2:4 sparse via CUTLASS GemmSparseUniversal) is an UNTESTED starting point handed to Q4: it builds with `-I ../cutlass/include -I ../cutlass/tools/util/include`, but only config 8 runs for s8 (identity-metadata check passes, rel err 2e-4) and its s4 path gives wrong results; the other configs fail `initialize` (rc 47).

## Files

| file | what |
|---|---|
| `q2gemm.cu` | the GEMM family (FORMATS.md sections 2-10): s4 / s8 / bf16 main loop, bf16 or int8 tail segment (B1, ResQ, B5), group scales (g64/g128), B9 sampled K tiles, two-problem row-role launch with rowmap (B8), batched (gridDim.z), epilogues bf16 / SwiGLU / fp16-alpha / int32 / fp32, bias, B3 table gather, B6 stat |
| `q2k.py` | ctypes wrapper (`Prob`, `Params`, `prob(...)`, `run(variant, cfg, p0, p1=None, batch=1)`), int4 pack/unpack, torch references (`quant_rows`, `ref_int`, `hash_u`, `skip_mask`) |
| `q2pro.py` | Triton quantize prologues (modes 0-8) and the B3 prototype search; torch reference `proto_ref` |
| `test_q2.py`, `bench_pro.py test` | bit-exactness tests |
| `bench_gemm.py` | GEMM timing at hobson's shapes (`base`: cuBLAS bf16, H2 CUTLASS, J5, Q2; `var`: B1/B5/B9/group/split/row-role variants) |
| `cfgscan.py` | per-config timing of one shape |
| `setup_q2.sh` | box setup (CUTLASS clones, H2 libs, GPTQ codes) |
| `qk2.py` | H2's glue kernels (qk.py) with the row-role quantizer (rows >= q0, or a POS map, to int8; question-row scale x32 for QRT2C) |
| `q2rt.py` | runtime: `QRT2` (q2gemm only, B8 partitions via `set_rowmask`) and `QRT2C` (CUTLASS W4A4 state rows + q2gemm int8 question rows, `Q2SIDE=1` side stream, CUTLASS for all-int8 GEMMs); per-GEMM maps `map:<json>` with 'rr'/'w8'/'w4' |
| `q2bench.py` | end-to-end latency (graph replay, fresh ids, 20 reps) with kernel split; names b8, w4a4, w8a8, w4q8, w4q8c, w8q2, w4q2, rrm<frac>, map:/cmap:<json> |
| `q2eval.py`, `score_q2.py` | all 3,227 evalkit questions through a runtime; laptop scorer with the BRIEF10 bar |
| `q2map_k{24,40,48,56,64}rr.json` | prefixes of H6's sensitivity ranking at all-int8, rest row-role |
| `bench_rr.py`, `bench_rr2.py`, `rr2.py`, `test_sk.py`, `bench_qz.py`, `basis_cost.py`, `gemm_tables.py`, `proj_q2.py` | row-role GEMM options, W4A8 question rows, split-K, fused B1 Z, dense-basis cost, GEMM sums, projections |

## Variants and configs

`q2k.V`: s4, s8, bf16, s4_tbf16 (B1 bf16 tail), s4_ts8 (int8 tail: ResQ / B5), s4g64, s4g128, s4skip (B9), s4skip_tbf16, s8_ts8, s8_tbf16. Row-role: pass a second `Prob` (always s8) to `run`; supported for s4, s4_tbf16, s4_ts8, s4g64, s4skip, s4skip_tbf16. Configs 0-9 are (BM, BN, warps, stages, K-tile bytes) in `q2gemm.cu` (`C0`..`C9`); pick the fastest per shape with `bench_gemm.best`.
