#!/bin/bash
# Upload the GGUF and the function zip, then deploy the stack.
#
#   BUCKET=my-bucket examples/lambda/deploy.sh
#
# Uses the AWS CLI's usual credentials and region (AWS_PROFILE, AWS_REGION). The
# bucket must exist in that region. Optional: STACK (default strands-decider-lambda),
# PREFIX (S3 key prefix, default strands-decider-lambda).
set -euo pipefail
cd "$(dirname "$0")"
: "${BUCKET:?set BUCKET to an S3 bucket in the deploy region}"
STACK=${STACK:-strands-decider-lambda}
PREFIX=${PREFIX:-strands-decider-lambda}

GGUF=build/model/decider-q8_0.gguf
ZIP=build/function.zip
for f in "$GGUF" "$ZIP"; do [ -f "$f" ] || { echo "missing $f -- run build.sh"; exit 1; }; done

MODEL_KEY=$PREFIX/model/decider-q8_0.gguf
# Key the zip by its hash, so a changed zip publishes a new version and an unchanged one does not.
CODE_KEY=$PREFIX/code/function-$(shasum -a 256 "$ZIP" | cut -c1-12).zip

LOCAL_SIZE=$(stat -f%z "$GGUF" 2>/dev/null || stat -c%s "$GGUF")
REMOTE_SIZE=$(aws s3api head-object --bucket "$BUCKET" --key "$MODEL_KEY" \
  --query ContentLength --output text 2>/dev/null || echo 0)
if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ]; then
  echo ">> s3://$BUCKET/$MODEL_KEY is already there, skipping the model upload"
else
  echo ">> uploading the model ($(du -h "$GGUF" | cut -f1)) -> s3://$BUCKET/$MODEL_KEY"
  aws s3 cp "$GGUF" "s3://$BUCKET/$MODEL_KEY" --only-show-errors
fi
echo ">> uploading the code -> s3://$BUCKET/$CODE_KEY"
aws s3 cp "$ZIP" "s3://$BUCKET/$CODE_KEY" --only-show-errors

echo ">> deploying $STACK (publishing a SnapStart version runs the full init once; a few minutes)"
aws cloudformation deploy --stack-name "$STACK" --template-file template.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides ModelBucket="$BUCKET" ModelKey="$MODEL_KEY" CodeKey="$CODE_KEY"
aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].Outputs' --output table
echo "Try it: python examples/lambda/client.py --stack $STACK ask"
