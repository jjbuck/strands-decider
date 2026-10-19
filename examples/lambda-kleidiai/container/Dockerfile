# Strands Decider 2B on llama.cpp for AWS Lambda (arm64 Graviton or x86_64), built inside the Lambda base image so
# glibc matches. Build context: this directory plus llama.cpp/, strands-decider/, ckpt/ (tokenizer, config, head) and
# model.gguf, staged by build.sh.
ARG BASE=public.ecr.aws/lambda/python:3.12
FROM ${BASE} AS build
RUN dnf install -y gcc gcc-c++ cmake make git >/dev/null && dnf clean all
COPY llama.cpp /src/llama.cpp
ARG CMAKE_EXTRA=""
RUN cd /src/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF ${CMAKE_EXTRA} >/dev/null \
 && cmake --build build -j $(nproc) --target llama \
      $(cmake --build build --target help | awk '{print $2}' | grep '^ggml-cpu' | sort -u) >/dev/null \
 && mkdir -p /opt/llama && cp -P build/bin/lib*.so* /opt/llama/ && cp -L /usr/lib64/libgomp.so.1 /opt/llama/
COPY decshim.c /src/decshim.c
RUN gcc -O2 -shared -fPIC -o /opt/llama/libdecshim.so /src/decshim.c -I/src/llama.cpp/include \
      -I/src/llama.cpp/ggml/include -L/opt/llama -lllama -lggml -Wl,-rpath,/opt/llama

FROM ${BASE}
COPY --from=build /opt/llama /opt/llama
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.9.1 --index-url ${TORCH_INDEX} >/dev/null \
 && pip install --no-cache-dir transformers==5.19.0 tokenizers==0.23.3 safetensors==0.8.0 numpy pydantic >/dev/null
COPY strands-decider /opt/strands-decider
RUN pip install --no-cache-dir --no-deps /opt/strands-decider >/dev/null
COPY ckpt /opt/ckpt
COPY lcpp_engine.py handler.py mvm_server.py eval_native.py ${LAMBDA_TASK_ROOT}/
ENV CKPT_DIR=/opt/ckpt GGUF=/tmp/model.gguf SHIM=/opt/llama/libdecshim.so DEC_GGML_DIR=/opt/llama HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH=/opt/llama
CMD ["handler.handler"]
