#!/usr/bin/env bash
# Idempotent host setup for the hobson v17 host (Ubuntu 24.04 DLAMI, run as root).
#   setup-host.sh [code-name]      (default code-name: aws-infra)
# From your machine: training/aws/scripts/ssm-run.sh -t 1800 -f training/aws/image/setup-host.sh <code-name>
# Expects s3://$HOBSON_BUCKET/code/<code-name>.tgz (training/aws/scripts/sync-code.sh --upload-only).
# HOBSON_BUCKET must be set (ssm-run.sh exports it); HOBSON_BUCKET_REGION defaults to us-west-2.
# Replicates training/README.md#setup (Qwen3.5 torso, Linux): Python 3.12 venv, torch 2.7.1
# cu126, transformers 5.17.0, peft 0.21.0, flash-linear-attention,
# `pip install -e ".[dev,train]"` (the train extra has the corpus build's datasets).
# causal_conv1d is NOT installed, same as that reference environment.
set -euo pipefail
CODE_NAME="${1:-aws-infra}"
BUCKET="${HOBSON_BUCKET:?set HOBSON_BUCKET (training/aws/scripts/ssm-run.sh exports it)}"
BUCKET_REGION="${HOBSON_BUCKET_REGION:-us-west-2}"
H=/opt/hobson
T0=$(date +%s)
step() { echo "[setup $(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s] $*"; }

step "packages"
export DEBIAN_FRONTEND=noninteractive
need=()
for p in tmux git jq build-essential python3.12-venv python3.12-dev pigz; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if (( ${#need[@]} )); then apt-get update -qq && apt-get install -y -qq "${need[@]}"; fi
command -v aws >/dev/null || { apt-get install -y -qq unzip; curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip && unzip -qo /tmp/awscli.zip -d /tmp && /tmp/aws/install; }

step "scratch on NVMe instance store"
mkdir -p $H/{code,logs,scratch}
# The DLAMI already assembles the instance-store disks into one volume at /opt/dlami/nvme.
if ! mountpoint -q $H/scratch; then
  if mountpoint -q /opt/dlami/nvme; then
    mkdir -p /opt/dlami/nvme/hobson
    mount --bind /opt/dlami/nvme/hobson $H/scratch
  else
    echo "WARN: no /opt/dlami/nvme; scratch stays on the root EBS volume"
  fi
fi
mkdir -p $H/scratch/{hf,triton,work,tmp}
chmod 1777 $H/scratch/tmp

step "env.sh + helpers"
cat > $H/env.sh <<EOF
# sourced by ssm-run.sh and hobson-bg
export HOME=\${HOME:-/root}
export HOBSON_BUCKET=$BUCKET
export AWS_DEFAULT_REGION=$BUCKET_REGION
export HF_HOME=$H/scratch/hf
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRITON_CACHE_DIR=$H/scratch/triton
export TMPDIR=$H/scratch/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH=$H/venv/bin:/root/.local/bin:\$PATH
export VIRTUAL_ENV=$H/venv
EOF

cat > /usr/local/bin/hobson-pull <<'EOF'
#!/bin/bash
# hobson-pull <name>: extract s3://$HOBSON_BUCKET/code/<name>.tgz over /opt/hobson/code/<name>
set -euo pipefail
. /opt/hobson/env.sh
name="$1"; dst=/opt/hobson/code/$name; work=/opt/hobson/scratch/work/$name
mkdir -p "$dst" "$work/data" "$work/checkpoints"
tmp=$(mktemp -d)
aws s3 cp "s3://$HOBSON_BUCKET/code/$name.tgz" "$tmp/c.tgz" --only-show-errors
[[ -f "$dst/.hobson-manifest" ]] && cp "$dst/.hobson-manifest" "$tmp/old" || : > "$tmp/old"
# Links first, and kept: the archive has a data/ directory (data/synthetic/), which GNU tar
# would otherwise extract over the link, onto the root volume.
for d in data checkpoints; do [[ -e "$dst/$d" ]] || ln -s "$work/$d" "$dst/$d"; done
# The archive's data/ (data/synthetic/) goes straight into the scratch work directory, not
# through the link: GNU tar fails with EXDEV when it extracts through a link onto the NVMe mount.
tar --warning=no-unknown-keyword --no-same-owner --anchored --exclude=data --exclude=checkpoints \
  -xzf "$tmp/c.tgz" -C "$dst"
members=$(tar -tzf "$tmp/c.tgz")  # once, not `tar -t | grep -q`: under pipefail tar's SIGPIPE fails the test
for d in data checkpoints; do
  if grep -q "^$d/" <<<"$members"; then
    tar --warning=no-unknown-keyword --no-same-owner --anchored -xzf "$tmp/c.tgz" -C "$work" "$d"
  fi
done
for d in data checkpoints; do [[ -L "$dst/$d" ]] || { echo "hobson-pull: $dst/$d is not a link to $work/$d" >&2; exit 1; }; done
comm -23 <(sort "$tmp/old") <(sort "$dst/.hobson-manifest") | while IFS= read -r f; do rm -f "$dst/$f"; done
rm -rf "$tmp"
echo "hobson-pull: $name -> $dst ($(wc -l < "$dst/.hobson-manifest") files); data/ checkpoints/ -> $work"
EOF

cat > /usr/local/bin/hobson-bg <<'EOF'
#!/bin/bash
# hobson-bg <name> "<command>"   run detached as systemd unit hobson-<name>
# log: /opt/hobson/logs/<name>.log   exit code: /opt/hobson/logs/<name>.rc (written at the end)
# state: systemctl status hobson-<name>; stop: systemctl stop hobson-<name>
set -euo pipefail
name="$1"; shift; cmd="$*"
log=/opt/hobson/logs/$name.log; rcf=/opt/hobson/logs/$name.rc
if systemctl is-active --quiet "hobson-$name"; then echo "hobson-$name is already running"; exit 1; fi
systemctl reset-failed "hobson-$name" 2>/dev/null || true
rm -f "$rcf"
echo "[hobson-bg] start $(date -u +%FT%TZ): $cmd" >> "$log"
systemd-run --unit "hobson-$name" --quiet -p LimitNOFILE=1048576 -p LimitMEMLOCK=infinity \
  -p StandardOutput=append:"$log" -p StandardError=append:"$log" \
  /bin/bash -c "set -o pipefail; . /opt/hobson/env.sh; cd /opt/hobson; ( $cmd ); rc=\$?; echo \$rc > $rcf; echo \"[hobson-bg] exit=\$rc \$(date -u +%FT%TZ)\"; exit \$rc"
echo "started hobson-$name; log $log"
EOF

cat > /usr/local/bin/hobson-extend <<'EOF'
#!/bin/bash
# hobson-extend <minutes-from-now>: reschedule the safety poweroff (poweroff == terminate).
set -euo pipefail
m="${1:?minutes}"
shutdown -c 2>/dev/null || true
shutdown -h +"$m" "hobson safety timer"
date -u -d "+$m min" +%FT%TZ | tee /opt/hobson/shutdown-at
EOF
chmod +x /usr/local/bin/hobson-{pull,bg,extend}
. $H/env.sh

step "uv + venv"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null; }
[[ -x $H/venv/bin/python ]] || uv venv -q --python /usr/bin/python3.12 $H/venv
export UV_CACHE_DIR=$H/scratch/uv-cache UV_LINK_MODE=copy
PY=$H/venv/bin/python
echo "torch==2.7.1" > $H/constraints.txt

step "torch 2.7.1 cu126"
$PY -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.7.1") else 1)' 2>/dev/null || \
  uv pip install -q --python $PY torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

step "transformers peft fla"
uv pip install -q --python $PY -c $H/constraints.txt \
  transformers==5.17.0 peft==0.21.0 flash-linear-attention "huggingface_hub[cli]"

step "code ($CODE_NAME) + pip install -e .[dev,train]"
/usr/local/bin/hobson-pull "$CODE_NAME"
uv pip install -q --python $PY -c $H/constraints.txt -e "$H/code/${CODE_NAME}[dev,train]"

step "pre-download models into HF_HOME"
# torso (configs/train*.yaml base_model) at main; teacher at the revision pinned in
# src/strands_decider/data/teacher.py (--revision default), read from the synced code.
TEACHER_REV=$(grep -oE '"--revision", default="[0-9a-f]{40}"' "$H/code/$CODE_NAME/src/strands_decider/data/teacher.py" | grep -oE '[0-9a-f]{40}') \
  || { echo "ERROR: no pinned --revision default in src/strands_decider/data/teacher.py" >&2; exit 1; }
# Not `download && echo`: set -e does not stop at a failed command before `&&`.
$H/venv/bin/hf download Qwen/Qwen3.5-2B-Base --quiet >/dev/null
echo "downloaded Qwen/Qwen3.5-2B-Base@main"
$H/venv/bin/hf download Qwen/Qwen3.5-4B --revision "$TEACHER_REV" --quiet >/dev/null
echo "downloaded Qwen/Qwen3.5-4B@$TEACHER_REV"
ls $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/ $HF_HOME/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/
du -sh $HF_HOME/hub/* 2>/dev/null || true
step "DONE setup in $(( $(date +%s) - T0 ))s"
