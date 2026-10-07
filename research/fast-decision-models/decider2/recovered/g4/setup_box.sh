set -e
cd ~/work
/opt/pytorch/bin/python -m venv --system-site-packages ~/venv >/dev/null
source ~/venv/bin/activate
python -m pip install -q --upgrade pip >/dev/null 2>&1
python -m pip install -q "transformers>=5.1" accelerate safetensors huggingface_hub datasets numpy scipy scikit-learn pandas peft flash-linear-attention typer rich pydantic fastapi uvicorn httpx pyyaml >/dev/null 2>&1
tar xzf work_bundle.tgz
cd ~/work/sd && python -m pip install -q --no-deps -e . >/dev/null 2>&1
python - <<PY
from huggingface_hub import snapshot_download
for r in ["StrandsAgents/strands-decider-2B-hobson-v19","Qwen/Qwen3.5-0.8B","Qwen/Qwen3.5-0.8B-Base","Qwen/Qwen3.5-2B-Base","Qwen/Qwen3-0.6B","Qwen/Qwen3-0.6B-Base","answerdotai/ModernBERT-base"]:
    snapshot_download(r)
print("models ok")
PY
python -c "import torch,transformers,strands_decider;print('ok',torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"
