#!/bin/bash
# v2 build + test box (c7g.8xlarge, Ubuntu 24.04 arm64): the v2/v2q zip from the h2h SnapStart zip, with libdecshim.so
# rebuilt from rotabl/decshim.c (DEC_UBATCH hook) and the eval mode added (app.py, eval_np.py, engine.py).
set -x
export HOME=/root DEBIAN_FRONTEND=noninteractive
B=s3://${DECIDER_BUCKET:?set DECIDER_BUCKET}/lambda; V=$B/results/v2; S=/opt/v2
mkdir -p $S/{code,base,out,gguf,logs,pydeps,res}
apt-get update -q >/dev/null && apt-get install -yq docker.io git unzip zip jq >/dev/null
systemctl start docker
command -v aws || snap install aws-cli --classic
export PATH=/snap/bin:$PATH
aws s3 cp --quiet --recursive $V/code/ $S/code/
aws s3 cp --quiet $B/results/h2h/snap/function.zip $S/base.zip
(cd $S/base && unzip -q ../base.zip)
sha256sum $S/base.zip
git clone -q https://github.com/ggml-org/llama.cpp $S/llama.cpp && git -C $S/llama.cpp checkout -q f74e2fa6a
git -C $S/llama.cpp log --oneline -1
python3 $S/code/patch_llama.py $S/llama.cpp
docker pull -q public.ecr.aws/lambda/python:3.12 && docker pull -q amazonlinux:2023
docker run --rm --platform linux/arm64 --entrypoint bash -v $S/llama.cpp:/src:ro -v $S/code:/code:ro \
  -v $S/base/lib:/lib0:ro -v $S/out:/out public.ecr.aws/lambda/python:3.12 /code/build_shim.sh
rm -rf $S/task && cp -a $S/base $S/task
cp $S/out/libdecshim.so $S/task/lib/libdecshim.so
cp $S/code/engine.py $S/code/app.py $S/code/eval_np.py $S/task/
echo "files added vs base:"; diff <(cd $S/base && find . -type f | sort) <(cd $S/task && find . -type f | sort)
echo "files changed vs base:"; (cd $S/base && find . -type f | sort | while read f; do cmp -s "$f" "$S/task/$f" || echo "  $f"; done)
docker run --rm --platform linux/arm64 -v $S/task:/task:ro -v $S/code:/code:ro amazonlinux:2023 bash /code/check_task.sh
(cd $S/task && rm -f ../function.zip && zip -q -r -X ../function.zip .)
echo "zip $(stat -c %s $S/function.zip) bytes, unzipped $(du -sb $S/task | cut -f1)"; sha256sum $S/function.zip
aws s3 cp --quiet $S/function.zip $V/function.zip
printf 'FROM amazonlinux:2023\nRUN dnf -q -y install python3.12 python3.12-pip >/dev/null && dnf clean all\n' > $S/Dockerfile.test
docker build -q -t v2test -f $S/Dockerfile.test $S >/dev/null
docker run --rm -v $S/pydeps:/pydeps v2test python3.12 -m pip install -q --target /pydeps boto3
aws s3 cp --quiet $B/gguf/s2b-h512-Q8_0.gguf $S/gguf/
aws s3 cp --quiet $B/results/rotabl/gguf/a6-Q8_0.gguf $S/gguf/
aws s3 cp --quiet $B/results/rotabl/q_and_scale.npz $S/gguf/
ls -la $S/gguf
echo BOX_BUILD_DONE
