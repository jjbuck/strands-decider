# shellcheck shell=bash
# Shared settings for the hobson v17 host scripts. Source it; do not run it.
# Override any value with an environment variable before sourcing, or put the overrides
# in training/aws/scripts/local.env (git-ignored, sourced first). AWS credentials come from the
# usual AWS CLI chain: set AWS_PROFILE if you do not use the default profile.
_hobson_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[[ -f "$_hobson_dir/local.env" ]] && source "$_hobson_dir/local.env"

log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

export HOBSON_HOME_REGION="${HOBSON_HOME_REGION:-us-west-2}"
# Bucket for code, data, checkpoints and SSM output. Default: hobson-v17-<account>-<region>,
# with the account of the current credentials.
if [[ -z "${HOBSON_BUCKET:-}" ]]; then
  HOBSON_ACCOUNT="${HOBSON_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
  [[ "$HOBSON_ACCOUNT" =~ ^[0-9]{12}$ ]] || die "cannot read the AWS account id (got '$HOBSON_ACCOUNT'); set HOBSON_BUCKET"
  HOBSON_BUCKET="hobson-v17-${HOBSON_ACCOUNT}-${HOBSON_HOME_REGION}"
fi
export HOBSON_BUCKET
export HOBSON_ROLE="${HOBSON_ROLE:-hobson-v17-host}"   # IAM role and instance profile name
export HOBSON_NAME_TAG="${HOBSON_NAME_TAG:-hobson-v17}"
export HOBSON_JOB_TAG="${HOBSON_JOB_TAG:-v17-replication}"
export HOBSON_REGIONS="${HOBSON_REGIONS:-us-west-2 us-east-1 us-east-2}"
# State written by the launchers (instance id + region) so other scripts find the host.
HOBSON_STATE_DIR="${HOBSON_STATE_DIR:-$_hobson_dir/.state}"
export HOBSON_STATE_DIR
AMI_PARAM="${AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"
ROOT_GB="${ROOT_GB:-1024}"

# Print the ids of $HOBSON_NAME_TAG hosts in region $1 (states $2, query field $3).
find_host() {
  aws ec2 describe-instances --region "$1" \
    --filters "Name=tag:Name,Values=$HOBSON_NAME_TAG" "Name=instance-state-name,Values=${2:-pending,running}" \
    --query "Reservations[].Instances[].${3:-InstanceId}" --output text
}

# Resolve the running host: $HOBSON_INSTANCE/$HOBSON_REGION if set, else the state
# file, else a tag lookup across both regions.
resolve_host() {
  if [[ -n "${HOBSON_INSTANCE:-}" && -n "${HOBSON_REGION:-}" ]]; then return 0; fi
  if [[ -f "$HOBSON_STATE_DIR/host.env" ]]; then
    # shellcheck disable=SC1091
    source "$HOBSON_STATE_DIR/host.env"
    [[ -n "${HOBSON_INSTANCE:-}" ]] && return 0
  fi
  local r id
  for r in $HOBSON_REGIONS; do
    id=$(find_host "$r")
    if [[ -n "$id" && "$id" != "None" ]]; then
      HOBSON_INSTANCE="${id%%[[:space:]]*}"; HOBSON_REGION="$r"
      export HOBSON_INSTANCE HOBSON_REGION; return 0
    fi
  done
  die "no running $HOBSON_NAME_TAG host found (set HOBSON_INSTANCE and HOBSON_REGION)"
}

ensure_sg() {  # $1 region -> prints sg id (no ingress; default all egress)
  local r="$1" vpc sg
  vpc=$(aws ec2 describe-vpcs --region "$r" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  sg=$(aws ec2 describe-security-groups --region "$r" --filters "Name=vpc-id,Values=$vpc" "Name=group-name,Values=hobson-v17-host" \
    --query 'SecurityGroups[0].GroupId' --output text)
  if [[ "$sg" == "None" || -z "$sg" ]]; then
    sg=$(aws ec2 create-security-group --region "$r" --vpc-id "$vpc" --group-name hobson-v17-host \
      --description "hobson v17 GPU host: no ingress, SSM only" \
      --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$HOBSON_NAME_TAG}]" \
      --query GroupId --output text)
  fi
  echo "$sg"
}

# Boot script: make the /opt/hobson layout; poweroff (== terminate, via
# InstanceInitiatedShutdownBehavior) after $1 minutes. $2, if given, is a line run first
# (e.g. one that sets m at boot, with $1 = '$m').
userdata() {
  printf '%s\n' '#!/bin/bash' 'mkdir -p /opt/hobson/{code,logs,scratch}' 'chown -R ubuntu:ubuntu /opt/hobson' ${2:+"$2"} \
    "shutdown -h +$1 \"hobson safety timer\" 2>&1 | tee /opt/hobson/logs/shutdown-timer.log" \
    "date -u -d \"+$1 min\" +%FT%TZ > /opt/hobson/shutdown-at"
}

# run_host region az shape ami subnet sg userdata log-label [extra run-instances args]
# Launches one tagged host, appends the attempt to launch-attempts.log, prints the instance
# id (or the error) and returns the run-instances exit code.
run_host() {
  local r=$1 az=$2 shape=$3 ami=$4 subnet=$5 sg=$6 ud=$7 label=$8 out rc=0
  local t="{Key=Name,Value=$HOBSON_NAME_TAG},{Key=hobson:job-type,Value=$HOBSON_JOB_TAG}"
  shift 8
  out=$(aws ec2 run-instances --region "$r" --image-id "$ami" --instance-type "$shape" "$@" \
    --iam-instance-profile "Name=$HOBSON_ROLE" \
    --subnet-id "$subnet" --security-group-ids "$sg" \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"Iops\":16000,\"Throughput\":1000,\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[$t]" "ResourceType=volume,Tags=[$t]" \
    --user-data "$ud" \
    --count 1 --query 'Instances[0].InstanceId' --output text 2>&1) || rc=$?
  mkdir -p "$HOBSON_STATE_DIR"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$r" "$az" "$label" "$(echo "$out" | tr '\n' ' ' | cut -c1-240)" \
    >> "$HOBSON_STATE_DIR/launch-attempts.log"
  echo "$out"
  return "$rc"
}

wait_ssm() {  # region instance-id tries: wait for running, then for SSM Online (tries x 10 s)
  local _ ping
  aws ec2 wait instance-running --region "$1" --instance-ids "$2"
  log "running; waiting for SSM agent"
  for _ in $(seq 1 "$3"); do
    ping=$(aws ssm describe-instance-information --region "$1" \
      --filters "Key=InstanceIds,Values=$2" --query 'InstanceInformationList[0].PingStatus' --output text)
    [[ "$ping" == "Online" ]] && { log "SSM Online"; return 0; }
    sleep 10
  done
  log "WARN: SSM not Online after $(( $3 / 6 )) min"
}
