#!/usr/bin/env bash
# Terminate every hobson host (tag Name=$HOBSON_NAME_TAG, default hobson-v17) in all hobson regions.
#   training/aws/scripts/teardown.sh          # dry run: list what would be terminated
#   training/aws/scripts/teardown.sh --now    # terminate, wait until every host is terminated
# Copy anything you need to S3 FIRST: instance store and root EBS are deleted.
# The bucket, IAM role and security groups are kept (cheap, reused by launch-host.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
NOW=0; [[ "${1:-}" == "--now" ]] && NOW=1
found=0
for r in $HOBSON_REGIONS; do
  ids=$(find_host "$r" pending,running,stopping,stopped)
  [[ -z "$ids" || "$ids" == "None" ]] && continue
  found=1
  aws ec2 describe-instances --region "$r" --instance-ids $ids \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime]' --output text
  if [[ $NOW -eq 1 ]]; then
    log "terminating in $r: $ids"
    aws ec2 terminate-instances --region "$r" --instance-ids $ids --output text >/dev/null
    aws ec2 wait instance-terminated --region "$r" --instance-ids $ids
  fi
done
[[ $found -eq 0 ]] && log "no $HOBSON_NAME_TAG hosts in [$HOBSON_REGIONS]"
if [[ $NOW -eq 1 ]]; then rm -f "$HOBSON_STATE_DIR/host.env"; else [[ $found -eq 1 ]] && log "dry run; pass --now to terminate"; fi
exit 0
