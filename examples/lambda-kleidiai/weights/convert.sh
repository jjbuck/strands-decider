#!/bin/bash
# merged HF checkpoint -> GGUF (f32) -> quantised variants; build the prefill shim against the same llama.cpp.
set -euxo pipefail
cd /opt/lam
L=/opt/lam/llama.cpp
/opt/lam/venv/bin/python $L/convert_hf_to_gguf.py /opt/lam/merged-2b --outtype f32 --outfile /opt/lam/gguf/s2b-f32.gguf 2>&1 | tail -15
for q in BF16 F16 Q8_0 Q4_0 Q4_K_M; do
  $L/build/bin/llama-quantize /opt/lam/gguf/s2b-f32.gguf /opt/lam/gguf/s2b-$q.gguf $q 8 2>&1 | tail -2
done
ls -la /opt/lam/gguf
gcc -O2 -shared -fPIC -o /opt/lam/libdecshim.so /opt/lam/engine/decshim.c -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -Wl,-rpath,$L/build/bin
echo CONVERT_DONE
