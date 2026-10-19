# Strands Decider 2B on AWS Lambda, CPU only: KleidiAI int8 kernels + SnapStart

This directory builds on the torch-free Lambda recipe in `examples/lambda/` (pull request #77). It replaces that recipe's llama.cpp build with one that uses the int8 matrix instructions of Graviton3 and Graviton4, plus four changes to llama.cpp itself. PR #77's handler, memfd weight loader and numpy head ship unchanged in the SnapStart package.

Measured for `strands-decider-2B-qwen3.5-v1-2610` on arm64 Lambda at 10,240 MB (six vCPUs):

| | Released engine (torch fp32) | PR #77 | This recipe |
|---|---|---|---|
| Forward at 256 / 1,024 / 4,096 tokens, six Graviton3 cores | 3.50 / 12.3 / 54.1 s | 1.33 / 5.33 / 24.1 s | 0.61 / 2.74 / 14.4 s |
| Warm round trip, 266-token question, Graviton3 sandbox | | 2.06 s | 0.80 s |
| Cold start with SnapStart | | 5.8 s | 4.85 s |
| Official Decision Index 0.3 (released engine: 28.48) | 28.48 | | 28.43 |
| JevBench public set, harness at 1bcc55eb (released: 180/231) | 180 | | 180 |

The accuracy rows come from the full Decision Index, JEVal, BEV, MJB and JevBench suites run through the deployed function: about 400,000 decisions. Macro accuracy is within 0.002 of the released engine on every suite (largest gap: MJB, +0.0016).

## What is different from PR #77's build

1. **Kernels.** llama.cpp is built with `GGML_CPU_ALL_VARIANTS` and KleidiAI (`-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CPU_KLEIDIAI=ON`). Each sandbox loads the best variant for its CPU: i8mm (`SMMLA`) on Graviton3 and 4, dot product on Graviton2. This accounts for almost all of the 2× at short prompts.
2. **Hadamard-folded weights.** KleidiAI quantises each token's activations with one scale per row, which costs accuracy on this model. `weights/fold_hadamard.py --block 512` stores every matmul weight as W·D·H. llama.cpp's `prism.hadamard` path applies the same rotation to each matmul input at runtime, which restores the accuracy at about 6% of the latency.
3. **llama.cpp patches**, applied by `llama/patch_llama.py` at commit f74e2fa6a:
   - **prefill**: compute only the rows the batch marks as outputs, and skip the LM head;
   - **gdn**: a single-pass Gated DeltaNet kernel;
   - **kskip**: `GGML_KLEIDIAI_SKIP` to keep chosen weights off KleidiAI;
   - **sme**: drop the SME variants GCC 11 cannot build.
4. **Runtime settings** (`shim/decshim.c`):
   - fp32 KV cache;
   - flash attention off;
   - a persistent context;
   - `DEC_UBATCH=512` (n_ubatch), which is 28% faster at 4,096 tokens with the same answers;
   - a shared-state fork that encodes a request's state once for all of its questions.

## Layout

| Directory | Contents |
|---|---|
| `llama/` | `patch_llama.py`, the llama.cpp patches |
| `shim/` | `decshim.c`, the C shim: `dec_load`, `dec_eval`, `dec_eval_fork` |
| `weights/` | `merge_export.py` (merge the LoRA in fp32) and `fold_hadamard.py` (the H512 fold). Convert with llama.cpp's `convert_hf_to_gguf.py --outtype f32`, then `llama-quantize … Q8_0` |
| `snapstart/` | The deployed package: PR #77's zip with `engine.py` (our shim behind PR #77's `ForwardShared` interface), `app.py` (adds an S3-shard eval mode and timing), `eval_np.py`, build scripts (`build_ours.sh`, `build_shim.sh`, `build_snap.sh`, `box_build.sh`), `deploy.py` and `snap_bench.py` |
| `container/` | The first deployment: a container image with the released torch head (`lcpp_engine.py`), background init and weights from S3. It has no SnapStart; its cold start is 42 s |
| `microvm/` | A Lambda MicroVM image (HTTP server with ready and validate hooks) and its in-region client |
| `bench/` | Forward ladder (`shim_bench.py`), parity against the released engine (`parity_bench.py`, `compare.py`), shared-state fork check (`fork_test.py`), drivers and cold-start tests, and `lambda_shim.py` (serves `/v1/systemone` over a Lambda invoke so the JevBench harness can call the function) |
| `rotation/` | The rotation-placement study: a QuaRot-style residual fold (`fold_quarot.py`), selective runtime rotation and Python un-rotation. On this model it matches H512 at half the runtime cost but was slightly less accurate at scale, so it is not deployed |

## Notes and caveats

- **Answers depend slightly on the Graviton generation.** Kernels differ by CPU, so the same request differs by up to about 0.2 in log-probability between Graviton generations. The top answer agreed on every row we checked. Within one generation, outputs repeat exactly. PR #77's single-ISA build gives the same answers everywhere.
- **SnapStart snapshots per generation.** We observed separate snapshots on Graviton3 and Graviton2 and no illegal instructions. No SnapStart restore landed on Graviton4 in our runs.
- **SnapStart ephemeral storage.** SnapStart limits ephemeral storage to 512 MB. The memfd loader keeps the weights out of `/tmp`.
- **Cold start without SnapStart** is bound by Lambda's per-environment network quota (625 Mbit/s by default, quota L-14320ECE). The 2 GB download takes about 25 s whatever the client or S3 storage class.
- **Two overheads, both fixed in `snapstart/`.**
  - `app.py` sets `OPENBLAS_NUM_THREADS=1`. Otherwise numpy's OpenBLAS workers busy-wait on the cores llama.cpp needs, which adds about 35 ms per forward.
  - `engine.py` asks the torso only for the rows the head reads, instead of every question token as PR #77's interface does.
  - Answers are bit-identical either way.
- **Placeholders to fill in.** Scripts are as run in our measurements. Bucket names, roles and image URIs are placeholders: set `DECIDER_BUCKET`, `DECIDER_LAMBDA_ROLE_ARN`, `ECR_REPO` and `BASE_IMAGE`.
