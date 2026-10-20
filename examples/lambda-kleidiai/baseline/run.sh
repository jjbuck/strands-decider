#!/bin/bash
# The baseline in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh [RUN_ID]
#
# build.sh (pinned weights + zip) -> deploy.py up -> bench.py -> summarize.py. Env knobs (defaults are the
# published baseline's): REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-p1 TEARDOWN=versions|all|none.
# Results land in baseline/results/<RUN_ID>/; the first run's answers become reference/outputs.json if none exists.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
OUT=$HERE/results/$RUN_ID
PY=${PYTHON:-python3}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }

"$HERE/build.sh"
"$PY" "$HERE/deploy.py" up --region "$REGION" --prefix "${PREFIX:-lkai-p1}"
"$PY" "$HERE/bench.py" --region "$REGION" --rounds "${ROUNDS:-5}" --concurrency "${CONCURRENCY:-10}" \
  --warm "${WARM:-10}" --out "$OUT"
REF=$HERE/reference/outputs.json
if [ -f "$REF" ]; then
  "$PY" "$HERE/summarize.py" "$OUT" --reference "$REF" --meta "$HERE/meta.json"
else
  mkdir -p "$HERE/reference"
  "$PY" "$HERE/summarize.py" "$OUT" --write-reference "$REF" --meta "$HERE/meta.json"
fi
case "${TEARDOWN:-versions}" in
  versions) "$PY" "$HERE/deploy.py" teardown --region "$REGION" ;;
  all) "$PY" "$HERE/deploy.py" teardown --all --region "$REGION" ;;
esac
echo "RUN_DONE $OUT/results.json"
