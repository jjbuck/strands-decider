#!/bin/bash
# Run the packaged function zip in amazonlinux:2023 (linux/arm64) through its real handler,
# with the GGUF from build/model instead of S3: GET /health and POST /v1/systemone as
# Function URL events, then client.py's request as a direct invocation.
#
#   examples/lambda/local_test.sh
set -euo pipefail
cd "$(dirname "$0")"
CONTAINER=${CONTAINER:-finch}
"$CONTAINER" run --rm --platform linux/arm64 \
  -v "$PWD/build:/build:ro" -v "$PWD/client.py:/client.py:ro" \
  -e MODEL_PATH=/build/model/decider-q8_0.gguf \
  amazonlinux:2023 bash -c '
    dnf -q -y install python3.12 unzip >/dev/null
    mkdir /task && cd /task && unzip -q /build/function.zip
    PYTHONPATH=/task:/ LD_LIBRARY_PATH=/task/lib python3.12 -c "
import json
from client import REQUEST
from handler import handler

def url_event(method, path, body=None):
    return {\"rawPath\": path, \"requestContext\": {\"http\": {\"method\": method}},
            \"body\": json.dumps(body) if body is not None else None, \"isBase64Encoded\": False}

for method, path, body in [(\"GET\", \"/health\", None), (\"POST\", \"/v1/systemone\", REQUEST),
                           (\"GET\", \"/nope\", None)]:
    r = handler(url_event(method, path, body), None)
    print(method, path, r[\"statusCode\"], json.dumps(json.loads(r[\"body\"]), indent=1))
print(\"direct invocation\", json.dumps(handler(REQUEST, None)[\"answers\"][\"team\"]))
"'
