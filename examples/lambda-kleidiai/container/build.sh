#!/bin/bash
# build.sh <tag> <gguf-variant> [base-image] [cmake-extra]: stage the context (patched llama.cpp, see
# engine/patch_llama.py), build, push to ECR.
set -euxo pipefail
TAG=$1; Q=$2; BASE=${3:-public.ecr.aws/lambda/python:3.12}; EXTRA=${4:-}
B=s3://${DECIDER_BUCKET:?set DECIDER_BUCKET}/lambda; REPO=${ECR_REPO:?set ECR_REPO}
command -v docker || (apt-get install -yq docker.io >/dev/null && systemctl start docker)
C=/opt/lam/ctx-$TAG; rm -rf $C; mkdir -p $C/ckpt
cd /opt/lam && for f in Dockerfile handler.py mvm_server.py eval_native.py; do aws s3 cp --quiet $B/code/image/$f $C/$f; done
aws s3 cp --quiet $B/code/decshim.c $C/ && aws s3 cp --quiet $B/code/lcpp_engine.py $C/
tar -C /opt/lam -cf - --exclude='llama.cpp-patched/build' --exclude='llama.cpp-patched/build-*' --exclude='llama.cpp-patched/.git' --transform 's,^llama.cpp-patched,llama.cpp,' llama.cpp-patched | tar -x -C $C
grep -q 'strands-decider prefill patch' $C/llama.cpp/src/models/qwen35.cpp  # the prefill-only patch must be in
cp -r /opt/lam/strands-decider $C/ && rm -rf $C/strands-decider/{data,research,training,.git} 2>/dev/null || true
aws s3 cp --quiet --recursive $B/ckpt/ $C/ckpt/ && rm -f $C/ckpt/MANIFEST.sha256 && ls $C/ckpt  # no lora/: merged
# weights are not in the image: the function downloads GGUF_S3 at init (s3://.../lambda/gguf/s2b-<Q>.gguf)
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin ${REPO%%/*} >/dev/null
DOCKER_BUILDKIT=0 docker build -q --build-arg BASE=$BASE --build-arg CMAKE_EXTRA="$EXTRA" -t $REPO:$TAG $C
docker push -q $REPO:$TAG
docker images $REPO:$TAG --format '{{.Size}}'
echo BUILD_DONE $TAG
