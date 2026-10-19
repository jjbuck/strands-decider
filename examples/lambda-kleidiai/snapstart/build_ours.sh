#!/bin/bash
# Runs INSIDE public.ecr.aws/lambda/python:3.12 (arm64, AL2023, GCC 11), as image/Dockerfile's build stage.
#   /src/llama.cpp  f74e2fa6a with patch_llama.py applied (ro)    /code  h2h code (ro)    /out  results (rw)
# Writes /out/lib      the deployed build: portable variant set (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS) + KleidiAI,
#                      libgomp/libstdc++/libgcc_s, libdecshim.so
#        /out/lib-v82  the same source with PR #77's flags: armv8.2-a+fp16+dotprod, no KleidiAI, no DL, + libdecshim.so
set -euxo pipefail
dnf install -y gcc gcc-c++ cmake make git binutils >/dev/null
rm -rf /b && cp -a /src/llama.cpp /b && cd /b
COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF"
cmake -B build $COMMON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CPU_KLEIDIAI=ON >/dev/null
cmake --build build -j "$(nproc)" --target llama $(cmake --build build --target help | awk '{print $2}' | grep '^ggml-cpu' | sort -u) >/dev/null
mkdir -p /out/lib && rm -f /out/lib/* && cp -P build/bin/lib*.so* /out/lib/
cp -L /usr/lib64/libgomp.so.1 /usr/lib64/libstdc++.so.6 /usr/lib64/libgcc_s.so.1 /out/lib/
gcc -O2 -shared -fPIC -o /out/lib/libdecshim.so /code/decshim.c -I/b/include -I/b/ggml/include -L/out/lib -lllama -lggml \
  -Wl,-rpath,'$ORIGIN'
cmake -B build-v82 $COMMON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DGGML_BACKEND_DL=OFF \
  -DGGML_OPENMP=ON -DGGML_CPU_KLEIDIAI=OFF >/dev/null
cmake --build build-v82 -j "$(nproc)" --target llama >/dev/null
mkdir -p /out/lib-v82 && rm -f /out/lib-v82/* && cp -P build-v82/bin/lib*.so* /out/lib-v82/
cp -L /usr/lib64/libgomp.so.1 /usr/lib64/libstdc++.so.6 /usr/lib64/libgcc_s.so.1 /out/lib-v82/
gcc -O2 -shared -fPIC -o /out/lib-v82/libdecshim.so /code/decshim.c -I/b/include -I/b/ggml/include -L/out/lib-v82 -lllama \
  -lggml -Wl,-rpath,'$ORIGIN'
gcc --version | head -1
ls -la /out/lib /out/lib-v82
echo BUILD_OURS_DONE
