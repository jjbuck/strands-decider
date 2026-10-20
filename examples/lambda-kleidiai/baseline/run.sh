#!/bin/bash
# The baseline in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh [RUN_ID]
#
# build.sh (pinned weights + zip) -> deploy.py up -> bench.py -> summarize.py -> redact.py. Env knobs (defaults are the
# published baseline's): REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-p1 TEARDOWN=versions|all|none.
# Raw output stays in the git-ignored build/runs/<RUN_ID>/; results/<RUN_ID>/ is the same with the bucket name and
# account id replaced, which is what gets committed. The first run's answers become reference/outputs.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
RAW=$HERE/build/runs/$RUN_ID
PY=${PYTHON:-python3}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }

"$HERE/build.sh"
"$PY" "$HERE/deploy.py" up --region "$REGION" --prefix "${PREFIX:-lkai-p1}"
"$PY" "$HERE/bench.py" --region "$REGION" --rounds "${ROUNDS:-5}" --concurrency "${CONCURRENCY:-10}" \
  --warm "${WARM:-10}" --out "$RAW"
REF=$HERE/reference/outputs.json
if [ -f "$REF" ]; then
  "$PY" "$HERE/summarize.py" "$RAW" --reference "$REF" --meta "$HERE/meta.json"
else
  mkdir -p "$HERE/reference"
  "$PY" "$HERE/summarize.py" "$RAW" --write-reference "$HERE/build/outputs.json" --meta "$HERE/meta.json"
fi
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$HERE/build/deployment.json")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
"$PY" "$HERE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
[ -f "$REF" ] || "$PY" "$HERE/redact.py" "$HERE/build/outputs.json" "$REF" --bucket "$BUCKET" --account "$ACCOUNT"
cp "$HERE/build/manifest.json" "$HERE/results/$RUN_ID/build-manifest.json"
case "${TEARDOWN:-versions}" in
  versions) "$PY" "$HERE/deploy.py" teardown --region "$REGION" ;;
  all) "$PY" "$HERE/deploy.py" teardown --all --region "$REGION" ;;
esac
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
