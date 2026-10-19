#!/bin/bash
# Runs INSIDE public.ecr.aws/lambda/python:3.12 (arm64, AL2023, GCC 11), as image/Dockerfile's build stage and
# h2h/build_ours.sh did. Builds libdecshim.so from rotabl/decshim.c (the deployed shim + the DEC_UBATCH hook) against
# the h2h SnapStart zip's own libllama/libggml (the portable KleidiAI build of f74e2fa6a + patch_llama.py), with that
# source's headers.
#   /src   llama.cpp f74e2fa6a with patch_llama.py applied (ro)   /lib0  the base zip's lib/ (ro)
#   /code  v2 code (ro)                                            /out   libdecshim.so (rw)
set -euxo pipefail
dnf install -y gcc binutils >/dev/null
gcc -O2 -shared -fPIC -o /out/libdecshim.so /code/decshim.c -I/src/include -I/src/ggml/include -L/lib0 -lllama -lggml \
  -Wl,-rpath,'$ORIGIN'
gcc --version | head -1
objdump -p /out/libdecshim.so | grep -E "NEEDED|RUNPATH"
echo BUILD_SHIM_DONE
