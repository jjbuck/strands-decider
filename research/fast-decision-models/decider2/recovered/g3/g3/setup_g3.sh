#!/bin/bash
source ~/venv/bin/activate; cd ~/work
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
from huggingface_hub import snapshot_download as s
for r in ["Tiiny/SmallThinker-4BA0.6B-Instruct", "ibm-granite/granite-3.1-3b-a800m-base", "ibm-granite/granite-3.1-1b-a400m-base", "utter-project/EuroMoE-2.6B-A0.6B-2512"]:
    p = s(r, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "tokenizer*"]); print(r, p, flush=True)
PY
echo setup_done
