#!/usr/bin/env bash
# Verify the hobson host env. Run on the host (root) after setup-host.sh:
#   verify-host.sh [code-name]    -> prints evidence; exit 0 only if every check passes.
set -uo pipefail
. /opt/hobson/env.sh
CODE_NAME="${1:-aws-infra}"
fail=0
echo "== nvidia-smi"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || fail=1
echo "== mounts"; df -h /opt/hobson/scratch / | sed 1d; echo "HF_HOME=$HF_HOME"
echo "== versions + FLA smoke (Qwen/Qwen3.5-2B-Base, cuda:0, bf16)"
python - <<'EOF' || fail=1
import logging, warnings, io, sys, time, importlib.metadata as md
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.DEBUG)
logging.getLogger().addHandler(h)
import torch, transformers, peft, triton
transformers.logging.set_verbosity_warning()
transformers.logging.add_handler(h)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available(),
      "| device count", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| peft", peft.__version__, "| triton", triton.__version__,
      "| flash-linear-attention", md.version("flash-linear-attention"))
import fla  # noqa
try:
    import causal_conv1d  # noqa
    print("causal_conv1d: installed")
except ImportError:
    print("causal_conv1d: not installed (training/README.md reference env; conv stays on its PyTorch path)")
assert transformers.__version__ == "5.17.0" and peft.__version__ == "0.21.0" and torch.__version__.startswith("2.7.1")
assert torch.cuda.device_count() >= 1
mid = "Qwen/Qwen3.5-2B-Base"
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    cfg = transformers.AutoConfig.from_pretrained(mid)
    tok = transformers.AutoTokenizer.from_pretrained(mid)
    t0 = time.time()
    lm = transformers.Qwen3_5ForCausalLM.from_pretrained(mid, config=cfg.get_text_config(),
                                                         dtype=torch.bfloat16, device_map={"": "cuda:0"})
    print(f"loaded in {time.time()-t0:.1f}s")
    x = tok(["The quick brown fox jumps over the lazy dog. " * 40], return_tensors="pt").to("cuda:0")
    lm.train()
    out = lm(**x, labels=x["input_ids"])
    out.loss.backward()
    torch.cuda.synchronize()
    print(f"forward+backward ok: seq {x['input_ids'].shape[1]} tokens, loss {out.loss.item():.4f}, "
          f"peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    lm.eval()
    with torch.no_grad():
        t0 = time.time(); lm(**x); torch.cuda.synchronize(); print(f"eval forward {1000*(time.time()-t0):.0f} ms")
msgs = [str(w.message) for w in ws] + buf.getvalue().splitlines()
# transformers 5.17 (integrations/hub_kernels.py) logs "`<fn>` is falling back to its reference
# PyTorch implementation" the first time a torch-only kernel runs. The Gated DeltaNet delta-rule
# kernels must come from fla; causal_conv1d_fn falling back is expected (training/README.md#setup).
for m in msgs:
    if m.strip(): print("  log/warn:", m[:300])
gdn_fallback = [m for m in msgs if "gated_delta_rule" in m and "falling back" in m]
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa
print("GATED DELTANET FALLBACK WARNINGS:", gdn_fallback if gdn_fallback else "none (chunk/recurrent delta rule run on fla Triton kernels)")
sys.exit(1 if gdn_fallback else 0)
EOF
echo "== model cache"; du -sh "$HF_HOME"/hub/models--Qwen--Qwen3.5-* 2>/dev/null || fail=1
echo "== CPU unit tests (CUDA hidden)"
cd /opt/hobson/code/$CODE_NAME && CUDA_VISIBLE_DEVICES='' timeout 3000 python -m pytest -q tests -x -p no:cacheprovider 2>&1 | tail -25
rc=${PIPESTATUS[0]}; echo "pytest exit=$rc"; [[ $rc -eq 0 ]] || fail=1
echo "VERIFY RESULT: $([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit $fail
