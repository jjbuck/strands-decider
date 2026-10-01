#!/usr/bin/env bash
# Idempotent: S3 bucket (private, SSE-S3) + IAM role/instance profile for the GPU host.
# Usage: training/aws/scripts/ensure-infra.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

B="$HOBSON_BUCKET"; R="$HOBSON_HOME_REGION"
if aws s3api head-bucket --bucket "$B" 2>/dev/null; then
  log "bucket $B exists"
else
  log "creating bucket $B in $R"
  aws s3api create-bucket --bucket "$B" --region "$R" \
    --create-bucket-configuration "LocationConstraint=$R" >/dev/null
fi
aws s3api put-public-access-block --bucket "$B" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$B" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
aws s3api put-bucket-tagging --bucket "$B" --tagging \
  "TagSet=[{Key=Name,Value=$HOBSON_NAME_TAG},{Key=hobson:job-type,Value=$HOBSON_JOB_TAG}]"

TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if aws iam get-role --role-name "$HOBSON_ROLE" >/dev/null 2>&1; then
  log "role $HOBSON_ROLE exists"
else
  log "creating role $HOBSON_ROLE"
  aws iam create-role --role-name "$HOBSON_ROLE" --assume-role-policy-document "$TRUST" \
    --tags "Key=Name,Value=$HOBSON_NAME_TAG" "Key=hobson:job-type,Value=$HOBSON_JOB_TAG" >/dev/null
fi
aws iam attach-role-policy --role-name "$HOBSON_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$HOBSON_ROLE" --policy-name hobson-bucket-rw --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
   {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetBucketLocation\"],\"Resource\":\"arn:aws:s3:::$B\"},
   {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:AbortMultipartUpload\",\"s3:ListMultipartUploadParts\"],\"Resource\":\"arn:aws:s3:::$B/*\"}]}"

if aws iam get-instance-profile --instance-profile-name "$HOBSON_ROLE" >/dev/null 2>&1; then
  log "instance profile $HOBSON_ROLE exists"
else
  aws iam create-instance-profile --instance-profile-name "$HOBSON_ROLE" >/dev/null
  sleep 5
fi
if ! aws iam get-instance-profile --instance-profile-name "$HOBSON_ROLE" \
     --query 'InstanceProfile.Roles[].RoleName' --output text | grep -qw "$HOBSON_ROLE"; then
  aws iam add-role-to-instance-profile --instance-profile-name "$HOBSON_ROLE" --role-name "$HOBSON_ROLE"
  log "added role to instance profile; waiting 15 s for IAM propagation"
  sleep 15
fi
log "ok: bucket=$B role=$HOBSON_ROLE profile=$HOBSON_ROLE"
