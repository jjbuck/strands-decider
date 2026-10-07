"""Speed patches for the HF Qwen3.5 path (same function): fla causal conv in place of the torch fallback, torch.compile'd
RMSNorm / gated RMSNorm (dynamic shapes), PEFT adapters left in the base dtype (bf16; trainer keeps fp32 master copies)."""
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as QM
from fla.modules.convolution import causal_conv1d as _fconv


def _cfn(hidden_states, weight, bias=None, activation=None, **kw):
    o = _fconv(hidden_states.transpose(1, 2), weight, bias, activation=activation)
    o = o[0] if isinstance(o, tuple) else o
    return o.transpose(1, 2)


def apply(bf16_lora=True, compile_norms=True):
    QM.causal_conv1d_fn = _cfn
    if compile_norms:
        QM.Qwen3_5RMSNorm.forward = torch.compile(QM.Qwen3_5RMSNorm.forward, dynamic=True)
        QM.Qwen3_5RMSNormGated.forward = torch.compile(QM.Qwen3_5RMSNormGated.forward, dynamic=True)
    if bf16_lora:
        import peft
        _g = peft.get_peft_model
        peft.get_peft_model = lambda model, c, **kw: _g(model, c, autocast_adapter_dtype=False, **kw)
