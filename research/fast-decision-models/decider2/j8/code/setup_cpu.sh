set -e
cd ~/work
/opt/pytorch/bin/python -m venv --system-site-packages ~/venv >/dev/null
source ~/venv/bin/activate
python -m pip install -q --upgrade pip >/dev/null 2>&1
python -m pip install -q "transformers>=5.15,<6" accelerate safetensors "huggingface_hub>=1.5" numpy scipy scikit-learn pandas "peft>=0.21" typer rich pydantic fastapi uvicorn httpx pyyaml 2>&1 | tail -3
tar xzf work_bundle.tgz
cd ~/work/sd && python -m pip install -q --no-deps -e . 2>&1 | tail -2
python - <<PY
from huggingface_hub import snapshot_download
for r in ["StrandsAgents/strands-decider-2B-hobson-v19","Qwen/Qwen3.5-2B-Base"]:
    print(snapshot_download(r))
print("models ok")
PY
python -c "import torch,transformers,strands_decider;print('ok',torch.__version__,transformers.__version__)"
