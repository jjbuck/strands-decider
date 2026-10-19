#!/bin/bash
# On the build box: MicroVM code artifact = Dockerfile + mvm_server.py + the H512 Q8_0 GGUF (stored, not deflated).
set -euxo pipefail
B=s3://${DECIDER_BUCKET:?set DECIDER_BUCKET}/lambda; Z=/opt/lam/mvmzip
rm -rf $Z && mkdir -p $Z && cd $Z
aws s3 cp --quiet $B/code/mvm/Dockerfile.h512 Dockerfile && aws s3 cp --quiet $B/code/image/mvm_server.py mvm_server.py
aws s3 cp --quiet $B/gguf/s2b-h512-Q8_0.gguf model.gguf
command -v zip || apt-get install -yq zip >/dev/null
zip -q -0 app.zip Dockerfile mvm_server.py model.gguf && aws s3 cp --quiet app.zip $B/mvm/app-h512.zip && ls -la app.zip
echo ZIP_DONE
