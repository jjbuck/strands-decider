#!/usr/bin/env bash
# Launch ONE hobson host into an EC2 Capacity Block (prepaid reservation), SSM only.
# Waits (polling every POLL_S s) until the reservation is `active`, then launches at once.
# Idempotent: if a pending/running host tagged Name=$HOBSON_NAME_TAG exists in REGION, prints it.
#
# Buy the block first (aws ec2 describe-capacity-block-offerings, then
# aws ec2 purchase-capacity-block); this script only launches into it.
#
# Usage: CR_ID=cr-0123456789abcdef0 REGION=us-east-2 training/aws/scripts/launch-capacity-block.sh
#   CR_ID, REGION  the Capacity Block reservation and its region (required)
#   SHUTDOWN_AT    absolute UTC safety poweroff (== terminate); default: 30 min before
#                  the reservation's end
#   ROOT_GB=1024  POLL_S=120
# The host gets the same Name tag (hobson-v17) and state file (.state/host.env) as
# launch-host.sh, so ssm-run.sh, sync-code.sh and teardown.sh find it unchanged. To run it
# next to another host, set HOBSON_NAME_TAG and HOBSON_STATE_DIR here and on every later call.
set -euo pipefail
source "$(dirname "$0")/common.sh"

CR_ID="${CR_ID:?set CR_ID to the Capacity Block reservation id (cr-...)}"
REGION="${REGION:?set REGION to the region of the reservation}"
POLL_S="${POLL_S:-120}"

id=$(find_host "$REGION")
if [[ -n "$id" && "$id" != "None" ]]; then log "host already exists: $id"; echo "$id"; exit 0; fi

read -r shape az end < <(aws ec2 describe-capacity-reservations --region "$REGION" --capacity-reservation-ids "$CR_ID" \
  --query 'CapacityReservations[0].[InstanceType,AvailabilityZone,EndDate]' --output text)
# EndDate looks like 2026-09-28T11:30:00+00:00. python3, not date -d: this runs on macOS too.
SHUTDOWN_AT="${SHUTDOWN_AT:-$(python3 -c 'import sys, datetime as d
e = d.datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")).astimezone(d.timezone.utc)
print((e - d.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))' "$end")}"
ami=$(aws ssm get-parameter --region "$REGION" --name "$AMI_PARAM" --query Parameter.Value --output text)
sg=$(ensure_sg "$REGION")
subnet=$(aws ec2 describe-subnets --region "$REGION" --filters "Name=availability-zone,Values=$az" Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text)
log "cr=$CR_ID $shape $REGION/$az ami=$ami subnet=$subnet sg=$sg shutdown-at=$SHUTDOWN_AT"

while :; do
  st=$(aws ec2 describe-capacity-reservations --region "$REGION" --capacity-reservation-ids "$CR_ID" \
    --query 'CapacityReservations[0].[State,AvailableInstanceCount]' --output text)
  log "reservation $CR_ID: $st"
  [[ "$st" == active* ]] && break
  [[ "$st" == scheduled* || "$st" == payment-pending* || "$st" == pending* ]] || die "reservation state $st"
  sleep "$POLL_S"
done

# The poweroff delay is computed on the host at boot, from the absolute SHUTDOWN_AT.
# shellcheck disable=SC2016  # $m and $(date ...) are for the host
USERDATA=$(userdata '$m' "m=\$(( ( \$(date -d $SHUTDOWN_AT +%s) - \$(date +%s) ) / 60 ))")
for attempt in 1 2 3 4 5; do
  out=$(run_host "$REGION" "$az" "$shape" "$ami" "$subnet" "$sg" "$USERDATA" "$shape/$CR_ID" \
    --instance-market-options MarketType=capacity-block \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}") \
    && [[ "$out" == i-* ]] && break
  log "attempt $attempt failed: $out"; sleep 30
done
[[ "$out" == i-* ]] || die "run-instances failed 5x: $out"
log "LAUNCHED $out $shape $REGION/$az"
printf 'HOBSON_INSTANCE=%s\nHOBSON_REGION=%s\nHOBSON_AZ=%s\nHOBSON_SHAPE=%s\nHOBSON_CR=%s\nHOBSON_LAUNCHED=%s\nHOBSON_SHUTDOWN_AT=%s\n' \
  "$out" "$REGION" "$az" "$shape" "$CR_ID" "$(date -u +%FT%TZ)" "$SHUTDOWN_AT" > "$HOBSON_STATE_DIR/host.env"
wait_ssm "$REGION" "$out" 90; echo "$out $shape $REGION $az"
