# lambda-kleidiai P1 baseline: pinned build, phase timers, benchmark harness

The reference every lambda-kleidiai experiment compares against. It deploys the recipe in `snapstart/` exactly as
measured (same libraries, weights, flags and handler code), adds phase timers that do not change the request path,
and benchmarks cold, SnapStart-restore and warm starts with enough samples to report run-to-run variance.

```bash
AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh <run-id>
```

`run.sh` needs an arm64 Linux host with Docker (a Graviton instance or an arm64 DevSpace) and about 60 GB of disk.
It runs `build.sh`, `deploy.py up`, `bench.py` and `summarize.py`, then deletes the published versions so the
SnapStart cache stops billing. The results are in `results/<run-id>/results.json`. Knobs: `REGION` (default
`us-east-2`), `ROUNDS` (5), `CONCURRENCY` (10), `WARM` (20), `PREFIX` (`lkai-p1`), `TEARDOWN`
(`versions`, `all` or `none`).

## What the example is (read from `7147a30`)

| | |
|---|---|
| Runtime | `python3.12` managed runtime, zip package. SnapStart supports Python 3.12 and later, so this runtime qualifies. The runtime version is pinned per function (`RuntimeManagementConfig` Manual, set by `deploy.py` from the first invocation's `Runtime Version ARN`) |
| Architecture | `arm64`. Lambda places it on Graviton2, 3 or 4. The portable llama.cpp build loads the best `libggml-cpu-*` variant per host |
| Memory | 10,240 MB (6 vCPUs), 512 MB ephemeral storage (SnapStart's maximum), 900 s timeout |
| SnapStart | `ApplyOn: PublishedVersions`. Loading and warm-up run at import (`examples/lambda/app/handler.py`), so the snapshot holds a loaded model |
| Weight fetch | `examples/lambda/app/loader.py`: one `HeadObject`, then 64 MiB ranged `GetObject`s on 32 threads (a ~2 GiB object is 32 ranges), each resumed from the byte where it broke off. The bytes go into a `memfd`, not `/tmp`, and llama.cpp mmaps `/proc/self/fd/<n>` |
| Model | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` at Hub revision `216d1ccb`. The base is `Qwen/Qwen3.5-2B-Base` at the revision the checkpoint's `provenance.json` names. The LoRA is merged in fp32, every matmul weight is H512 Hadamard-folded, and the result is converted to GGUF f32 and quantized to Q8_0 (about 2 GB). The pointer head, tokenizer and temperatures ship in the zip (`assets/`) |
| Loader | llama.cpp `f74e2fa6a` with `llama/patch_llama.py`, built with `GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS` and KleidiAI. It is reached through `shim/decshim.c` by ctypes. It needs a readable, mmappable path to the GGUF, the `libggml-cpu-*` variants in `lib/`, and `libgomp`, `libstdc++` and `libgcc_s` bundled |
| Deploy tooling | The source used boto3 scripts (`snapstart/deploy.py`) against hand-staged S3 objects. `examples/lambda/` has a CDK stack for the PR #77 build. This baseline uses boto3 (`deploy.py`) so that versions and configuration can be driven per sample |

## What this directory changes

Nothing in the request path. `snapstart/app.py`, `engine.py` and PR #77's `handler.py` and `loader.py` ship
unchanged. The function's handler is `lkai_entry.handler`, which wraps them:

- `shim/decshim.c` gains `dec_load_phases_ns`. It reads `CLOCK_MONOTONIC` four times inside `dec_load_seq`
  (backend load, `llama_model_load_from_file`, `llama_init_from_model`) and changes nothing else.
- `phases.py` holds the standard-library timers and the parsers for the `LKAI {json}` stdout records and Lambda's
  `REPORT` line.
- `lkai_entry.py` times each wrapped call: the imports, `loader.download_to_memfd`, the shim's load phases, the
  asset load, the warm-up inference (the first inference) and PR #77's `_load`. It registers the SnapStart
  `before_snapshot` and `after_restore` hooks, and for every invocation it records the start class, the handler
  time and PR #77's `latency_ms` (`engine.evaluate`). The `REPORT` line is left alone.
- With `LKAI_PHASES=0` the module imports `app` and calls `app.handler` directly, with no timers. This is the
  REPORT-only arm.

`deploy.py` creates two functions from the same zip with the same configuration. `<prefix>-inst` has
`LKAI_PHASES=1` and `<prefix>-rep` has `LKAI_PHASES=0`. The instrumentation overhead is the difference in Lambda's
own numbers between the two (`results.json` → `instrumentation_overhead`).

## Phases

| Phase | Where it is measured | Start classes |
|---|---|---|
| `runtime_to_import` | process start (`/proc/self/stat`) to `lkai_entry` import | cold, snapshot |
| `imports` | numpy, pydantic, tokenizers, `strands_decider`, engine, loader | cold, snapshot |
| `weight_download` (+ `weight_mib_s`) | `loader.download_to_memfd` | cold, snapshot |
| `ggml_backend_load` | decshim: `ggml_backend_load_all_from_path` + `llama_backend_init` | cold, snapshot |
| `model_load_mmap` | decshim: `llama_model_load_from_file` (mmap, tensor setup, KleidiAI repack) | cold, snapshot |
| `context_init` | decshim: `llama_init_from_model` | cold, snapshot |
| `assets_load` | tokenizer, head and config (`engine_load` − `shim_init`) | cold, snapshot |
| `warmup_inference` | the handler's warm-up `evaluate` (the first inference) | cold, snapshot |
| `model_ready` | PR #77's `_load`, end to end | cold, snapshot |
| `report_init_duration` | REPORT `Init Duration` | cold |
| `report_restore_duration` | REPORT `Restore Duration` (and the billed value) | restore |
| `restore_to_invoke` | `after_restore` hook to the first invocation | restore |
| `first_request_inference` | `engine.evaluate` on the probe request, in a new environment | cold, restore |
| `first_request_total` | Init or Restore Duration + Duration | cold, restore |
| `steady_state_inference` | `engine.evaluate`, overall and per request | warm |
| `report_duration`, `report_billed_duration`, `report_max_memory_mb`, `client_round_trip` | REPORT and client | all |

"snapshot" means the init that runs inside `publish_version`. Its phases come from the `before_snapshot` record in
CloudWatch Logs (`results.json` → `phases.<arm>.restore.snapshot_init`), because no invocation sees that init.

Durations are never computed across a snapshot boundary. A restored environment's monotonic clock need not
continue from the snapshot's, so `restore_to_invoke` uses two monotonic reads taken after the restore.

## How the harness forces each start class

Each round runs these steps for both arms. The arm order alternates between rounds.

1. Set a new `COLD_TAG` on `$LATEST`. A configuration change retires every `$LATEST` environment.
2. **cold**: `CONCURRENCY` simultaneous invokes of `$LATEST`. Each needs a new on-demand environment.
3. Publish a version from that configuration and wait until its snapshot is ready.
4. **restore**: `CONCURRENCY` simultaneous invokes of that version. Each restores a new environment.
5. **warm**: `WARM` sequential invokes per arm on that version, interleaved request by request.
6. Delete both versions. A version's SnapStart cache is billed as long as it exists.

The harness checks each invocation's expected class against its REPORT line: `Init Duration` means cold,
`Restore Duration` means restore, and neither means warm. Mismatches are counted and excluded from the phase
distributions; they are never relabelled. Cold and restore waves send `requests.json`'s probe request. Warm
invocations cycle through all five requests.

## Correctness reference

The first run writes `reference/outputs.json`: one answer set per request and CPU part. Kernels differ by Graviton
generation (README, "Notes and caveats"), and within one generation the outputs repeat exactly. Later runs, and
every experiment, are compared against it on the same request and CPU part (`results.json` → `correctness`:
`bit_identical`, `choice_flips`, `max_abs_delta`).

## Results format

`results/<run-id>/results.json` is `lkai-result/1` (`results/schema.md`). It covers every field the initiative brief
asks each experiment to publish. P2's dashboard ingests it. Raw evidence sits beside it: `invocations.jsonl` (one
row per invocation, with the log tail), `logs/` (CloudWatch events for the run's window) and `conditions.json`.

## Security and cost

- The bucket blocks public access and denies non-TLS requests, and objects use SSE-S3.
- The execution role may `GetObject` the one GGUF key and write its own two log groups. Nothing else.
- Both functions are reachable only by direct `Invoke` with IAM credentials. They have no Function URL.
- Committed results never include account IDs or bucket names: `deploy.py` keeps those in `build/deployment.json`,
  which is git-ignored with `build/`.
- At us-east-2 prices, one default run (5 rounds × 2 arms × (10 cold + 10 restore + 20 warm)) costs about $6.
  Most of that is the 3-hour SnapStart cache minimum on 10 versions and about 100 cold starts that each download
  2 GB. `results.json` → `cost` gives the computed figure for the run. `deploy.py teardown --all` removes
  everything.
