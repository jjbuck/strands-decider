#!/usr/bin/env bash
# Run a shell command (or a local script file) on the hobson host via SSM, wait, print
# stdout/stderr, and exit with the remote exit code.
#
#   training/aws/scripts/ssm-run.sh nvidia-smi
#   training/aws/scripts/ssm-run.sh 'cd /opt/hobson/code/aws-infra && pytest -q tests -x'
#   training/aws/scripts/ssm-run.sh -f ./probe.sh [args]     # ship a local script, run it with args
#   training/aws/scripts/ssm-run.sh -t 7200 'long thing'     # remote execution timeout (s), default 3600
#
# The command runs as root under `bash -l`, cwd /opt/hobson, with HOBSON_BUCKET exported and
# /opt/hobson/env.sh sourced (venv on PATH, HF_HOME, etc.). Quoting is safe: the command is
# sent base64-encoded, so heredocs, pipes and quotes survive. Full output goes to
# s3://$HOBSON_BUCKET/ssm/<command-id>/ (no 24k truncation); this script prints it.
#
# LONG JOBS (> a few minutes): do not hold SSM open. Detach and log:
#   ssm-run.sh 'hobson-bg myjob "cd /opt/hobson/code/aws-infra && training/recipe.sh teacher"'
#   ssm-run.sh 'tail -n 50 /opt/hobson/logs/myjob.log'   # poll
#   ssm-run.sh 'systemctl status hobson-myjob --no-pager' # state / exit code
# hobson-bg (installed by setup-host.sh) wraps `systemd-run --unit hobson-<name>`,
# logs to /opt/hobson/logs/<name>.log, and survives SSM session end.
# Env: HOBSON_INSTANCE / HOBSON_REGION override the host (else training/aws/scripts/.state/host.env).
set -euo pipefail
source "$(dirname "$0")/common.sh"

TIMEOUT=3600; FILE=""; QUIET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--timeout) TIMEOUT="$2"; shift 2;;
    -f|--file) FILE="$2"; shift 2;;
    -q|--quiet) QUIET=1; shift;;
    --) shift; break;;
    -h|--help) sed -n 2,21p "$0"; exit 0;;
    *) break;;
  esac
done
if [[ -n "$FILE" ]]; then
  BODY=$(cat "$FILE")
  if [[ $# -gt 0 ]]; then BODY="set -- $(printf '%q ' "$@")"$'\n'"$BODY"; fi
else
  [[ $# -gt 0 ]] || die "usage: ssm-run.sh [-t sec] [-f file] [-q] <cmd...>"
  BODY="$*"
fi
resolve_host

SCRIPT=$(printf '%s\n' "export HOBSON_BUCKET=$(printf %q "$HOBSON_BUCKET") HOBSON_BUCKET_REGION=$(printf %q "$HOBSON_HOME_REGION")" \
  '[ -f /opt/hobson/env.sh ] && . /opt/hobson/env.sh' 'cd /opt/hobson 2>/dev/null || cd /' "$BODY")
B64=$(printf '%s' "$SCRIPT" | base64 | tr -d '\n')
REMOTE="f=\$(mktemp /tmp/ssm-XXXXXX.sh); echo $B64 | base64 -d > \$f; bash -l \$f; rc=\$?; rm -f \$f; exit \$rc"
PARAMS=$(python3 -c 'import json,sys; print(json.dumps({"commands":[sys.argv[1]],"executionTimeout":[sys.argv[2]]}))' "$REMOTE" "$TIMEOUT")

CID=$(aws ssm send-command --region "$HOBSON_REGION" --instance-ids "$HOBSON_INSTANCE" \
  --document-name AWS-RunShellScript --parameters "$PARAMS" \
  --output-s3-bucket-name "$HOBSON_BUCKET" --output-s3-key-prefix ssm --output-s3-region "$HOBSON_HOME_REGION" \
  --timeout-seconds 120 --comment "ssm-run" --query Command.CommandId --output text)
[[ $QUIET -eq 1 ]] || log "command $CID on $HOBSON_INSTANCE ($HOBSON_REGION)"

sleep 1; delay=1
while :; do
  st=$(aws ssm get-command-invocation --region "$HOBSON_REGION" --command-id "$CID" \
        --instance-id "$HOBSON_INSTANCE" --query Status --output text 2>/dev/null || echo Pending)
  case "$st" in Pending|InProgress|Delayed) sleep "$delay"; (( delay < 10 )) && delay=$((delay + 1));; *) break;; esac
done
inv=$(aws ssm get-command-invocation --region "$HOBSON_REGION" --command-id "$CID" --instance-id "$HOBSON_INSTANCE" --output json)
rc=$(echo "$inv" | python3 -c 'import json,sys; print(json.load(sys.stdin)["ResponseCode"])')

prefix="s3://$HOBSON_BUCKET/ssm/$CID/$HOBSON_INSTANCE/awsrunShellScript/0.awsrunShellScript"
out=$(aws s3 cp "$prefix/stdout" - --region "$HOBSON_HOME_REGION" 2>/dev/null) || \
  out=$(echo "$inv" | python3 -c 'import json,sys; print(json.load(sys.stdin)["StandardOutputContent"], end="")')
err=$(aws s3 cp "$prefix/stderr" - --region "$HOBSON_HOME_REGION" 2>/dev/null) || \
  err=$(echo "$inv" | python3 -c 'import json,sys; print(json.load(sys.stdin)["StandardErrorContent"], end="")')
[[ -n "$out" ]] && printf '%s\n' "$out"
[[ -n "$err" ]] && printf '%s\n' "$err" >&2
[[ $QUIET -eq 1 ]] || log "status=$st exit=$rc"
[[ "$rc" =~ ^[0-9]+$ ]] || rc=1
exit "$rc"
