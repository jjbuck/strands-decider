#!/bin/bash
# Phase 2 zip: PR #77's function (handler.py, loader.py, strands_decider's torch-free modules, pydantic/numpy/tokenizers/
# safetensors, assets) with its engine.py and lib/ replaced by ours (snap_engine.py; the portable KleidiAI build and
# decshim, built in the Lambda python3.12 image). Checks, as PR #77's container_build.sh does, that every library
# resolves from the bundle and the modules import, in amazonlinux:2023 with python3.12. Then sizes by component.
set -x
S=/opt/h2h; T=$S/snap/task
rm -rf $S/snap && mkdir -p $T
cp -a $S/pr77/task/. $T/
rm -rf $T/lib $T/engine.py && mkdir $T/lib && cp -P $S/ours/lib/*.so* $T/lib/
cp $S/code/snap_engine.py $T/engine.py
docker run --rm --platform linux/arm64 -v $T:/task:ro amazonlinux:2023 bash -c '
  dnf -q -y install python3.12 binutils >/dev/null
  for so in /task/lib/*.so*; do [ -L "$so" ] && continue
    if LD_LIBRARY_PATH=/task/lib ldd "$so" | grep -q "not found"; then echo "UNRESOLVED $so"; LD_LIBRARY_PATH=/task/lib ldd "$so"; fi
  done
  cd /tmp && PYTHONPATH=/task LD_LIBRARY_PATH=/task/lib python3.12 -c "
import ast, engine, loader, pydantic, numpy, tokenizers, strands_decider.hidden_engine, ctypes
ast.parse(open(\"/task/handler.py\").read())
ctypes.CDLL(\"/task/lib/libdecshim.so\")
print(\"imports OK\", numpy.__version__, pydantic.__version__, tokenizers.__version__)"'
(cd $T && zip -q -r -X $S/snap/function.zip .)
echo "zip $(stat -c %s $S/snap/function.zip) bytes"
echo "unzipped total $(du -sb $T | cut -f1)"
echo "app (handler, engine, loader, strands_decider) $(du -scb $T/handler.py $T/engine.py $T/loader.py $T/strands_decider | tail -1 | cut -f1)"
echo "libs (lib/, zip follows symlinks) $(du -sbL $T/lib | cut -f1)"
echo "numpy (+numpy.libs) $(du -scb $T/numpy $T/numpy.libs 2>/dev/null | tail -1 | cut -f1)"
echo "tokenizers $(du -scb $T/tokenizers 2>/dev/null | tail -1 | cut -f1)"
echo "pydantic + pydantic_core + others $(du -scb $T/pydantic $T/pydantic_core $T/typing_extensions.py $T/annotated_types $T/typing_inspection $T/safetensors 2>/dev/null | tail -1 | cut -f1)"
echo "assets: tokenizer.json $(stat -c %s $T/assets/tokenizer.json), head.safetensors $(stat -c %s $T/assets/head.safetensors)"
ls $T; ls -la $T/lib
# the same for PR #77's own zip
P=$S/pr77/task
echo "PR77 zip $(stat -c %s $S/pr77/repo/examples/lambda/build/function.zip) bytes, unzipped $(du -sb $P | cut -f1), lib $(du -sbL $P/lib | cut -f1)"
echo BUILD_SNAP_DONE
