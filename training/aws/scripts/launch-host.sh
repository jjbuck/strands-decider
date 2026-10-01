#!/usr/bin/env bash
# Launch ONE hobson-v17 GPU host (On-Demand, SSM only, no key pair).
# Tries shapes x AZs in order, us-west-2 then us-east-1, retrying on capacity errors.
# Idempotent: if a pending/running host tagged Name=hobson-v17 exists, prints it and exits.
#
# Usage: training/aws/scripts/launch-host.sh
#   SHAPES="p5.48xlarge p5e.48xlarge p5en.48xlarge g6e.48xlarge"  (default)
#   REGIONS="us-west-2 us-east-1"                                    (default)
#   ROOT_GB=1024  SHUTDOWN_MIN=720
set -euo pipefail
source "$(dirname "$0")/common.sh"

SHAPES="${SHAPES:-p5.48xlarge p5e.48xlarge p5en.48xlarge g6e.48xlarge}"
REGIONS="${REGIONS:-us-west-2 us-east-1}"
SHUTDOWN_MIN="${SHUTDOWN_MIN:-720}"

unset HOBSON_INSTANCE HOBSON_REGION
for r in $HOBSON_REGIONS; do  # every region we might have launched in
  id=$(find_host "$r" pending,running '[InstanceId,InstanceType,Placement.AvailabilityZone]')
  if [[ -n "$id" ]]; then log "host already exists in $r: $id"; echo "$id"; exit 0; fi
done

"$(dirname "$0")/ensure-infra.sh"
USERDATA=$(userdata "$SHUTDOWN_MIN")

for r in $REGIONS; do
  ami=$(aws ssm get-parameter --region "$r" --name "$AMI_PARAM" --query Parameter.Value --output text)
  sg=$(ensure_sg "$r")
  log "region $r ami=$ami sg=$sg"
  for shape in $SHAPES; do
    azs=$(aws ec2 describe-instance-type-offerings --region "$r" --location-type availability-zone \
      --filters "Name=instance-type,Values=$shape" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort)
    for az in $azs; do
      subnet=$(aws ec2 describe-subnets --region "$r" --filters "Name=availability-zone,Values=$az" Name=default-for-az,Values=true \
        --query 'Subnets[0].SubnetId' --output text)
      [[ "$subnet" == "None" ]] && continue
      log "try $shape $r/$az ($subnet)"
      if out=$(run_host "$r" "$az" "$shape" "$ami" "$subnet" "$sg" "$USERDATA" "$shape") && [[ "$out" == i-* ]]; then
        log "LAUNCHED $out $shape $r/$az"
        printf 'HOBSON_INSTANCE=%s\nHOBSON_REGION=%s\nHOBSON_AZ=%s\nHOBSON_SHAPE=%s\nHOBSON_LAUNCHED=%s\n' \
          "$out" "$r" "$az" "$shape" "$(date -u +%FT%TZ)" > "$HOBSON_STATE_DIR/host.env"
        wait_ssm "$r" "$out" 60; echo "$out $shape $r $az"; exit 0
      fi
      if echo "$out" | grep -qE 'InsufficientInstanceCapacity|Unsupported|InstanceLimitExceeded|VcpuLimitExceeded|capacity'; then
        log "no capacity: $(echo "$out" | grep -oE '\(([A-Za-z]+)\)' | head -1)"; continue
      fi
      die "run-instances failed with a non-capacity error: $out"
    done
  done
done
die "no capacity for [$SHAPES] in [$REGIONS]; try SHAPES='g6e.24xlarge g6e.12xlarge'. Attempts: $HOBSON_STATE_DIR/launch-attempts.log"
