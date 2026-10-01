#!/usr/bin/env bash
# recipe.sh on one multi-GPU host, timed per stage. Each stage is `bash training/recipe.sh STAGE`
# with NGPU set; this script adds timing, logs, row-count checks and the S3 copy
# (training/aws/README.md, section 4).
#
# Run from the repo root. Environment:
#   PY          python of the venv (required)
#   S3_PREFIX   s3://bucket/prefix for outputs (required; "none" disables S3)
#   RUN_DIR     timing, logs, config copies (default: $PWD/runs/recipe)
#   NGPU        GPUs to use (default: all that nvidia-smi lists)
#   GIT_REV     code revision to record (default: git rev-parse HEAD, else the .hobson-rev
#               that training/aws/scripts/sync-code.sh ships, else "unknown")
#   PARENT_CONFIG, TRAIN_CONFIG  the two training configs, as in recipe.sh (default
#               configs/train-parent.yaml, configs/train.yaml); each file must exist
#   CKPT        the checkpoint that train writes and calibrate and eval read, as in recipe.sh
#               (default checkpoints/hobson-2b-recipe); a path under checkpoints/
#   SEED        if set, parent and train use config copies with `seed: $SEED`
#   FAST        1 = gradient_checkpointing off + precompute_frozen_kl (speed only; NGPU=8, 80 GB)
#   ALLOW_COUNT_MISMATCH=1  warn instead of fail on a row-count difference
#
# Usage: training/run_recipe.sh STAGE [STAGE ...]   (every name is checked before a stage runs)
#   cpu = build, fetch -> multistep -> adequacy, generated (concurrently, no GPU)
#   gpu = teacher parent replay train calibrate eval
#   all = cpu then gpu (the v19 route)
#   or any single recipe.sh stage; catchall and distill run only when named
set -euo pipefail

: "${PY:?set PY to the venv python}"
: "${S3_PREFIX:?set S3_PREFIX (s3://bucket/prefix, or none)}"
[ -f training/recipe.sh ] && [ -d src/strands_decider ] || { echo "run from the repo root" >&2; exit 2; }
export PY PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
RUN_DIR="${RUN_DIR:-$PWD/runs/recipe}"
mkdir -p "$RUN_DIR/logs" data checkpoints reports
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="${NGPU:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"  # no `| head`: SIGPIPE + pipefail
else
  NGPU="${NGPU:-1}"; GPU_NAME="none"
fi
export NGPU
# gpus_of STAGE: the GPUs that `recipe.sh STAGE` uses; fails for a name that has no
# check_STAGE here. No group (cpu, gpu, all) includes catchall or distill.
gpus_of() {
  case "$1" in
    build|fetch|multistep|generated|adequacy|catchall|distill) echo 0 ;;
    teacher|parent|replay|train) echo "$NGPU" ;;
    calibrate|eval) echo 1 ;;
    *) return 1 ;;
  esac
}
for s in "$@"; do
  case "$s" in cpu|gpu|all) ;; *) gpus_of "$s" >/dev/null || { echo "unknown stage: $s" >&2; exit 2; } ;; esac
done
# The EC2 instance type, else the hostname. -f: another cloud's metadata endpoint answers
# with an HTML error page; stages.jsonl quotes host_shape without escaping, so accept a token only.
HOST_SHAPE="$(t=$(curl -sf -m 2 -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null) && \
  curl -sf -m 2 -H "X-aws-ec2-metadata-token: $t" \
    http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)"
[[ "$HOST_SHAPE" =~ ^[A-Za-z0-9._-]+$ ]] || HOST_SHAPE="$(hostname)"
GIT_REV="${GIT_REV:-$(git rev-parse HEAD 2>/dev/null || cat .hobson-rev 2>/dev/null || echo unknown)}"
SEED="${SEED:-}" FAST="${FAST:-0}"
case "$FAST" in 0|1) ;; *) echo "FAST=$FAST: must be 0 or 1" >&2; exit 2 ;; esac
# Checkpointing off fits 80 GB only with the forwards an 8-rank plan makes (PERF: 48 GiB).
[ "$FAST" = 0 ] || [ "$NGPU" = 8 ] || { echo "FAST=1 is measured for NGPU=8 only (got $NGPU)" >&2; exit 2; }
case "$SEED" in ''|*[!0-9]*) [ -z "$SEED" ] || { echo "SEED=$SEED: must be a non-negative integer" >&2; exit 2; } ;; esac
CKPT="${CKPT:-checkpoints/hobson-2b-recipe}"
[[ "$CKPT" =~ ^checkpoints(/[A-Za-z0-9._-]+)+$ && "/$CKPT/" != */./* && "/$CKPT/" != */../* ]] \
  || { echo "CKPT=$CKPT: must be a path under checkpoints/ of [A-Za-z0-9._-] names, none '.' or '..'" >&2; exit 2; }
export CKPT  # recipe.sh train, calibrate and eval read it; check_train and check_calibrate upload it

log() { echo "[run_recipe $(date -u +%H:%M:%SZ)] $*" >&2; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------ configs
# config_for CONFIG: CONFIG itself, or with SEED / FAST an edited copy in $RUN_DIR/configs/.
# Either way the config as run is kept in $RUN_DIR/configs/: hf_export ships
# training/configs/ from there, and the checkpoint's train_config.json is not the YAML.
# A copy is never overwritten: a RUN_DIR holds one run, so a stage whose config would
# differ from the copy an earlier stage left (a plain run after SEED or FAST) is refused.
config_for() {
  local src="$1" dst new
  [ -f "$src" ] || { echo "$src: no such config" >&2; return 1; }
  dst="$RUN_DIR/configs/$(basename "$src")"; new="$dst.new"
  mkdir -p "$RUN_DIR/configs"; cp "$src" "$new"
  if [ -n "$SEED" ]; then
    [ "$(grep -c '^seed:' "$src")" = 1 ] || { echo "$src: expected exactly one 'seed:' line" >&2; return 1; }
    sed -i.bak "s/^seed:.*/seed: $SEED  # SEED override from training\/run_recipe.sh/" "$new"
  fi
  if [ "$FAST" = 1 ]; then
    [ "$(grep -c '^gradient_checkpointing:' "$src")" = 1 ] || { echo "$src: expected one 'gradient_checkpointing:' line" >&2; return 1; }
    sed -i.bak "s/^gradient_checkpointing:.*/gradient_checkpointing: false  # FAST override from training\/run_recipe.sh/" "$new"
    printf '%s\n' "# FAST override from training/run_recipe.sh (speed only)" "precompute_frozen_kl: true" >>"$new"
  fi
  rm -f "$new.bak"
  if [ -e "$dst" ] && ! cmp -s "$new" "$dst"; then
    rm -f "$new"
    echo "$dst: the copy an earlier run left differs from this run's config (SEED or FAST changed); use another RUN_DIR" >&2
    return 1
  fi
  mv "$new" "$dst"
  if [ -z "$SEED" ] && [ "$FAST" = 0 ]; then echo "$src"; return 0; fi
  diff "$src" "$dst" >&2 || true
  echo "$dst"
}
PARENT_CONFIG="${PARENT_CONFIG:-configs/train-parent.yaml}" TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.yaml}"
for c in "$PARENT_CONFIG" "$TRAIN_CONFIG"; do  # stages.jsonl quotes the names without escaping
  [[ "$c" =~ ^[A-Za-z0-9._/-]+$ ]] || { echo "config path $c: only [A-Za-z0-9._/-] is allowed" >&2; exit 2; }
done
[ "$PARENT_CONFIG" = "$TRAIN_CONFIG" ] || [ "$(basename "$PARENT_CONFIG")" != "$(basename "$TRAIN_CONFIG")" ] \
  || { echo "PARENT_CONFIG and TRAIN_CONFIG need different file names (both are copied into $RUN_DIR/configs/)" >&2; exit 2; }
PARENT_CONFIG_SRC="$PARENT_CONFIG" TRAIN_CONFIG_SRC="$TRAIN_CONFIG"  # the names the user gave, for stages.jsonl
PARENT_CONFIG="$(config_for "$PARENT_CONFIG")"
TRAIN_CONFIG="$(config_for "$TRAIN_CONFIG")"
export PARENT_CONFIG TRAIN_CONFIG

# ------------------------------------------------------------------ S3
s3_on() { [ "$S3_PREFIX" != "none" ]; }
s3_put() {  # s3_put PATH...  (files or dirs, relative to the repo root)
  s3_on || return 0
  local p
  for p in "$@"; do
    if [ -d "$p" ]; then aws s3 sync --only-show-errors "$p/" "$S3_PREFIX/$p/"
    elif [ -f "$p" ]; then aws s3 cp --only-show-errors "$p" "$S3_PREFIX/$p"
    else log "s3_put: $p missing"; return 1; fi
  done
}
s3_run() { s3_on && aws s3 sync --only-show-errors "$RUN_DIR/" "$S3_PREFIX/run/" || true; }
if s3_on; then
  ( while sleep 300; do s3_run; done ) &
  SYNC_PID=$!
  trap 'kill $SYNC_PID 2>/dev/null || true; s3_run' EXIT
fi

# ------------------------------------------------------------------ checks and outputs
expect_rows() {  # expect_rows FILE N [FILE N ...]
  local n
  while [ "$#" -gt 0 ]; do
    n="$(wc -l <"$1" | tr -d ' ')"
    if [ "$n" = "$2" ]; then echo "OK   $1: $n rows"
    elif [ "${ALLOW_COUNT_MISMATCH:-0}" = 1 ]; then echo "WARN $1: $n rows, expected $2"
    else echo "FAIL $1: $n rows, expected $2"; return 1; fi
    shift 2
  done
}
checksum() { sha256sum "$@" | tee -a "$RUN_DIR/sha256.txt"; }

# check_STAGE: after `recipe.sh STAGE` succeeds; counts from recipe.sh and training/steps.md.
check_build() {
  local n  # not grep's own output file: check output is appended to build.log
  if n="$(grep -c skipped "$RUN_DIR/logs/build.log")"; then echo "$n recipe(s) skipped" >&2; return 1; fi
  expect_rows data/train_v5.jsonl 100449
  checksum data/train_v5.jsonl data/train_v5.holdout.jsonl data/holdout_v5_norule.jsonl
  s3_put data/train_v5.jsonl data/train_v5.holdout.jsonl data/holdout_v5_norule.jsonl
}
check_fetch() {
  checksum data/raw/contract-nli.zip data/raw/musique_data_v1.0.zip data/raw/helpsteer2/*.jsonl.gz
  s3_put data/raw/contract-nli.zip data/raw/musique_data_v1.0.zip data/raw/helpsteer2
}
check_multistep() {
  expect_rows data/multistep_v14.jsonl 12909 data/multistep_v14_eval.jsonl 4084
  checksum data/multistep_v14.jsonl data/multistep_v14_eval.jsonl
  s3_put data/multistep_v14.jsonl data/multistep_v14_eval.jsonl
}
check_generated() {
  expect_rows data/generated_v16.jsonl 2148 data/generated_v16_eval.jsonl 350 \
    data/generated_v18.jsonl 1667 data/generated_v18_eval.jsonl 247 \
    data/generated_v16p.jsonl 2148 data/generated_v18p.jsonl 1667 \
    data/para_pairs_v16_eval.jsonl 678 data/para_pairs_v18_eval.jsonl 484 \
    data/flips_v20.jsonl 718 data/flips_v20_eval.jsonl 120
  set -- data/generated_v16.jsonl data/generated_v16_eval.jsonl data/generated_v18.jsonl data/generated_v18_eval.jsonl \
    data/generated_v16p.jsonl data/generated_v18p.jsonl data/para_pairs_v16_eval.jsonl data/para_pairs_v18_eval.jsonl \
    data/flips_v20.jsonl data/flips_v20_eval.jsonl
  checksum "$@"; s3_put "$@"
}
check_catchall() {
  expect_rows data/catchall_v20.jsonl 5000 data/catchall_v20_eval.jsonl 800
  set -- data/catchall_v20.jsonl data/catchall_v20_eval.jsonl
  checksum "$@"; s3_put "$@"
}
check_distill() {
  expect_rows data/teacher_v20.jsonl 72434
  checksum data/teacher_v20.jsonl data/teacher_v5_qwen35-4b.jsonl data/replay_v14_multistep.jsonl
  s3_put data/teacher_v20.jsonl
}
check_adequacy() {
  expect_rows data/adequacy_hs2.jsonl 4866 data/adequacy_hs2_eval.jsonl 234 \
    data/adequacy_gen.jsonl 1300 data/adequacy_gen_eval.jsonl 302
  set -- data/adequacy_hs2.jsonl data/adequacy_hs2_eval.jsonl data/adequacy_gen.jsonl data/adequacy_gen_eval.jsonl
  checksum "$@"; s3_put "$@"
}
check_teacher() {
  expect_rows data/teacher_multistep_v14.jsonl 12909
  checksum data/teacher_multistep_v14.jsonl data/teacher_multistep_v14_train.jsonl
  s3_put data/teacher_multistep_v14.jsonl data/teacher_multistep_v14_train.jsonl
}
check_parent() { s3_put checkpoints/hobson-2b-recipe-parent; }
check_replay() {
  expect_rows data/replay_parent_multistep.jsonl 12909
  checksum data/replay_parent_multistep.jsonl
  s3_put data/replay_parent_multistep.jsonl
}
check_train() { s3_put "$CKPT"; }
check_calibrate() { s3_put "$CKPT"; }
check_eval() { s3_put reports; }

# ------------------------------------------------------------------ stages
# stage NAME: `bash training/recipe.sh NAME` into logs/NAME.log, one JSON line in stages.jsonl
# (the config names as given; the copies as run are in $RUN_DIR/configs/), then
# check_NAME. Exits on failure.
stage() {
  local name="$1" gpus logf="$RUN_DIR/logs/$1.log" start t0 t1 rc
  gpus="$(gpus_of "$name")" || { echo "unknown stage: $name" >&2; exit 2; }
  start="$(utc)"; t0="$(date +%s)"
  log "start $name (gpus=$gpus) -> $logf"
  set +e
  if [ "$gpus" = 0 ]; then bash training/recipe.sh "$name"; else
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; bash training/recipe.sh "$name"; fi >"$logf" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then ( set -euo pipefail; "check_$name" ) >>"$logf" 2>&1; rc=$?; fi
  set -e
  t1="$(date +%s)"
  printf '{"stage":"%s","start_utc":"%s","end_utc":"%s","wall_s":%d,"gpus_used":%s,"host_shape":"%s","gpu_name":"%s","exit_code":%d,"git_rev":"%s","parent_config":"%s","train_config":"%s","ckpt":"%s"}\n' \
    "$name" "$start" "$(utc)" "$((t1 - t0))" "$gpus" "$HOST_SHAPE" "$GPU_NAME" "$rc" "$GIT_REV" \
    "$PARENT_CONFIG_SRC" "$TRAIN_CONFIG_SRC" "$CKPT" >>"$RUN_DIR/stages.jsonl"
  if [ "$rc" -ne 0 ]; then
    log "FAIL $name (exit $rc, $((t1 - t0)) s); last lines of $logf:"
    tail -n 30 "$logf" >&2 || true
    s3_run; exit "$rc"
  fi
  log "done $name in $((t1 - t0)) s"
  s3_run
}

cpu() {
  local pids=() pid bad=0
  ( stage build ) & pids+=("$!")
  ( stage fetch && stage multistep && stage adequacy ) & pids+=("$!")
  ( stage generated ) & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  [ "$bad" = 0 ] || { log "a CPU stage failed (see stages.jsonl)"; exit 1; }
}
gpu() { for s in teacher parent replay train calibrate eval; do stage "$s"; done; }

# ------------------------------------------------------------------ main
[ "$#" -gt 0 ] || { sed -n 2,25p "$0"; exit 2; }
log "host=$HOST_SHAPE gpus=$NGPU ($GPU_NAME) rev=$GIT_REV seed=${SEED:-config} fast=$FAST parent=$PARENT_CONFIG train=$TRAIN_CONFIG ckpt=$CKPT run_dir=$RUN_DIR s3=$S3_PREFIX"
for s in "$@"; do
  case "$s" in cpu) cpu ;; gpu) gpu ;; all) cpu; gpu ;; *) stage "$s" ;; esac
done
log "finished: $*"
