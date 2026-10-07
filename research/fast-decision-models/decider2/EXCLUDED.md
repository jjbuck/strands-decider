# decider2: what was left out

This mirror is public, so it leaves out model weights, credentials and orchestration, third-party material and very
large files. Every excluded path stays in the private source tree. Counts and sizes are per category; 1 MB = 10^6 bytes.

| category | left out |
|---|---|
| Model weights, checkpoints and tensors | 43 files, 2101.1 MB |
| Files over 25 MB, logs over 5 MB, recovered JSON over 5 MB, `analysis/norm/` | 480 files, 2108.3 MB |
| Dropped to keep the package under about 600 MB | 2 files, 25.1 MB |
| Third-party code, papers and documentation | 593 files, 364.3 MB |
| AWS orchestration, SSH keys and box records | 121 files, 0.4 MB |
| Non-result files in `recovered/` | 381 files, 367.0 MB |
| Images, PDFs, archives, pickles and marker files | 50 files, 731.1 MB |
| Report drafts and old idea notes | 18 files, 0.6 MB |
| `archive/` | 4 files, 0.4 MB |
| `__pycache__` and `.pyc` | 38 files, 0.5 MB |

## Weights and checkpoints

- No `.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt`, `.npz`, `.neff` or `.ntff` file is included, nor any `.npy` over 1 MB, nor anything under a `ckpt/` or `checkpoints/` directory.
- These include H7's four LoRA checkpoints (`h7/ckpt/`, 59.7 MB each), the earlier rounds' trained models and pre-tokenised row tensors in `recovered/` (up to 201 MB each), and small tensors such as `h5/box/res/devref_240.pt` and `j15/smoke/b8.jsonl.taps.000.pt`.
- The same rules also removed four tokenised LM validation files (`recovered/g*/g4/data*/val.bin`, 4 MB each) and nine vocabulary-map and token-count arrays (`.npy`, 2 MB each) from `recovered/`; they are data, not weights.
- Most agents kept their checkpoints on the GPU boxes and never copied them back; their reports say so. The training recipes that produced them are included (see `MANIFEST.md`).

## Size cap

- Every file over 25 MB is left out; the full list is at the end of this file.
- The 5 MB caps on logs and on `recovered/` result JSON removed nothing extra: every log outside `recovered/` is under 5 MB, logs in `recovered/` are left out as non-results, and no `recovered/` result JSON exceeds 5 MB.
- `analysis/norm/` (258 MB, 459 files) is left out: it holds normalised copies of the agents' prediction files, and `analysis/catalog.py` regenerates it from the included predictions.
- Large generated training sets are left out: `j4/data/dp.jsonl`, `j4/data/ft.jsonl`, `h7/data/cf_aug.jsonl`, `j12/data/xr_train.jsonl`, `k1/data/j7aug.jsonl`, `k1/data/EXIT.jsonl` and `j15/suites/EXIT.jsonl`, plus `j8/ids.jsonl` (pre-tokenised ids). The scripts that wrote them are included (`j4/code/gen_dp.py`, `j4/code/mk_ft.py`, `h7/code/gen_cf.py`, `j12/code/gen_xr.py`, `j7/code/gen_aug.py`, `j15/code/mkdev.py`), so each set can be rebuilt from its source data (mainly `train_pool.jsonl` and `train_v5.jsonl`).
- `evalkit/train_pool.jsonl` (185 MB, the 12,747 train-split requests) is left out. Rescoring needs only `evalkit/suites/`, `evalkit/refs/` and `evalkit/split.json`, which are included; retraining needs `train_pool.jsonl` and `training/data/train_v5.jsonl` (82 MB), which are not.

## Package budget

With everything else in, the mirror came to 645 MB. First, files in `recovered/` that are data rather than results were left out
(five copies of `tokens/real_reqs.json`, 19.3 MB, and a 2.6 MB metadata dump; see below). Then the largest prediction files were
dropped. Both are byte-identical to copies that are kept, so no result is lost:

- `j13/box/dev_E.jsonl` (6.8 MB): byte-identical to j13/dev_E.jsonl (kept).
- `j13/box/full_D.jsonl` (18.3 MB): byte-identical to j13/full_D.jsonl (kept).

## Third-party material

- `ext/` (345 MB): two checkouts of strands-decider with their git history: `sd-main` (strands-labs/strands-decider, this repository's upstream) and `sd-bidi` (a fork's `bidi-b1` branch).
- `papers/` (1.4 MB): three third-party paper PDFs.
- `n1/ref/nki_doc1.txt`, `n1/ref/nki_doc2.txt`: copied AWS Neuron NKI API documentation.
- `recovered/g3/g3/hub_meta.json` (2.6 MB): a dump of Hugging Face Hub model metadata.

## Credentials and orchestration

- Left out whole: `box.sh`, `launch.sh`, `launch_n1.sh`, `retry.sh`, `retry_big.sh`, `setup_named.sh`, `userdata.sh`, `watch_matrix.sh`, the SSH key pair `gpu_key2` and `gpu_key2.pub`, and the `boxes/` directory (111 files: per-box SSH configs, instance records, setup logs and readiness markers).
- No other file contained a private key, an AWS access key, a GitHub or Hugging Face token, an AWS secret or an SSH key, so no code file had to be left out for that reason.
- The agents' own helper scripts that call `box.sh` (for example `h6/code/sync.sh`, `j11/code/fetch.sh`) are included; they contain no host, key or account, and do not run without `box.sh`.

## Other left-out files

- **Images and documents:** `analysis/figures/*.svg` and `analysis/figures/png/` (the curated figures are in `../figures/`; `analysis/run_all.sh` regenerates these), a few agent PNG plots, and `h4/this-that-model.pdf` (a third-party model paper).
- **Archives:** `bundle/work_bundle.tgz` (49.5 MB, a code tarball for the boxes), `j3/box_logs/j3_logs.tgz` and `q5/res/logs/logs_small.tgz`. Their contents could not be checked line by line.
- **Pickles and marker files:** `.pkl` data in `recovered/`, `.done` / `.latest` / `.state` markers, `READY_v*`, and the backup `m1/untrained_local.json.bak` (`m1/untrained_local.json` is kept).
- **Non-result files in `recovered/`:** logs and `.out` files, CSVs, `.raw` sweeps, a compiled `.so`, markers, and data copies (`evalkit/suites`, `evalkit/refs`, `training/data`, `training/jevbench_public`, `tokens/data`, `tokens/real_reqs.json`, `shrink/jb_public`, `g4/data*`, `g4/evalkit_suites`). The top-level copies of the shared data are included.
- **Per instructions:** `analysis/report_parts/` (report drafts), `analysis/_old_ideas.txt` and `archive/`.

## Redactions in included files

Every included text file was scanned and redacted in the copy (never in the source).
- EC2 instance ids in agent notes were replaced with `i-REDACTED` (10 occurrences in 9 NOTES files).
- One line in `j8/NOTES.md` about where box credentials came from was replaced with `[internal reference removed]`.
- A private IPv4 address in a synthetic bash task in `training/data/pilot_train.jsonl` was replaced with `IP_REDACTED` (2 occurrences).
- No AMI, security-group, subnet, VPC, volume or ENI id, AWS account id, ARN, `--profile` value or internal hostname remained in any included file.
- Kept on purpose: the documentation-range addresses 203.0.113.24 and 198.51.100.77 (RFC 5737, reserved for examples) inside a synthetic JevBench scenario (`evalkit/suites/JB-*.jsonl`, `training/jevbench_public/hard.jsonl` and its copies in `shrink/` and `tokens/`), so the evaluation inputs stay exactly as scored; the profiler version string `2.28.23.0` in `n1/results/base1152_profile_summary.json`; and the public pricing host `aws.amazon.com` cited in `j8/NOTES.md` and `j8/DRAFT_REPORT.md`.
- Every JSON and JSONL file that parsed before redaction still parses.

## Every file over 25 MB that was left out

| size | path | why |
|---|---|---|
| 201.0 MB | `recovered/g5/g4/runs/S60_slot/final.pt` | weights/checkpoints |
| 184.7 MB | `evalkit/train_pool.jsonl` | size cap |
| 184.7 MB | `recovered/g1/evalkit/train_pool.jsonl` | size cap |
| 184.7 MB | `recovered/g2/evalkit/train_pool.jsonl` | size cap |
| 184.7 MB | `recovered/g3/evalkit/train_pool.jsonl` | size cap |
| 171.1 MB | `recovered/g5/g4/runs/S60_mudd/final.pt` | weights/checkpoints |
| 170.7 MB | `recovered/g5/g4/runs/S60_dense/final.pt` | weights/checkpoints |
| 145.5 MB | `recovered/g4/g4/data/lmeval.pkl` | other file types |
| 145.5 MB | `recovered/g5/g4/data/lmeval.pkl` | other file types |
| 144.4 MB | `recovered/g5/g4/data/ft.pkl` | other file types |
| 137.9 MB | `recovered/g4/g4/runs/S125_dense3/final.pt` | weights/checkpoints |
| 129.2 MB | `recovered/g3/g3/rows_corpus_q.pt` | weights/checkpoints |
| 125.7 MB | `recovered/g3/g3/rows_corpus_s.pt` | weights/checkpoints |
| 121.2 MB | `recovered/g1/g1/runs/denseft/step100.pt` | weights/checkpoints |
| 121.2 MB | `recovered/g1/g1/runs/denseft/step200.pt` | weights/checkpoints |
| 111.0 MB | `recovered/g2/g2/lora_qat44_a_s1000.pt` | weights/checkpoints |
| 111.0 MB | `recovered/g2/g2/lora_qat44_a_s1500.pt` | weights/checkpoints |
| 111.0 MB | `recovered/g2/g2/lora_qat44_a_s500.pt` | weights/checkpoints |
| 96.4 MB | `recovered/g4/g4/data/ft.pkl` | other file types |
| 96.4 MB | `recovered/g5/g4/data/ft_long_only.pkl` | other file types |
| 96.4 MB | `recovered/g5/g4/data_old/ft.pkl` | other file types |
| 90.5 MB | `recovered/g5/g4/runs/S60_dense3/final.pt` | weights/checkpoints |
| 82.4 MB | `recovered/g1/training/data/train_v5.jsonl` | size cap |
| 82.4 MB | `recovered/g2/training/data/train_v5.jsonl` | size cap |
| 82.4 MB | `recovered/g3/training/data/train_v5.jsonl` | size cap |
| 82.4 MB | `recovered/g4/training/data/train_v5.jsonl` | size cap |
| 82.4 MB | `recovered/g5/training/data/train_v5.jsonl` | size cap |
| 82.4 MB | `training/data/train_v5.jsonl` | size cap |
| 76.3 MB | `recovered/g3/g3/rows_real_q.pt` | weights/checkpoints |
| 74.6 MB | `recovered/g3/g3/rows_real_s.pt` | weights/checkpoints |
| 71.0 MB | `j4/data/dp.jsonl` | size cap |
| 68.4 MB | `recovered/g4/g4/data/train_states.jsonl` | size cap |
| 68.4 MB | `recovered/g5/g4/data/train_states.jsonl` | size cap |
| 68.3 MB | `j4/data/ft.jsonl` | size cap |
| 59.7 MB | `h7/ckpt/h7_sets_s1300.pt` | weights/checkpoints |
| 59.7 MB | `h7/ckpt/h7_sets_s300.pt` | weights/checkpoints |
| 59.7 MB | `h7/ckpt/h7_sets_s600.pt` | weights/checkpoints |
| 59.7 MB | `h7/ckpt/h7_sets_s900.pt` | weights/checkpoints |
| 55.5 MB | `k1/data/j7aug.jsonl` | size cap |
| 54.2 MB | `j12/data/xr_train.jsonl` | size cap |
| 50.2 MB | `j8/ids.jsonl` | size cap |
| 49.5 MB | `bundle/work_bundle.tgz` | size cap |
| 47.9 MB | `h7/data/cf_aug.jsonl` | size cap |
| 42.3 MB | `recovered/g1/g1/rows_train.pt` | weights/checkpoints |
| 36.1 MB | `j15/suites/EXIT.jsonl` | size cap |
| 36.1 MB | `k1/data/EXIT.jsonl` | size cap |
| 32.2 MB | `ext/sd-bidi/.git/objects/pack/pack-8365c2d6e1db689e478b75424af50c5884816e4f.pack` | third-party |
| 31.6 MB | `ext/sd-main/.git/objects/pack/pack-785da667ee21b294d4de853267bcb173d5ec61e9.pack` | third-party |
