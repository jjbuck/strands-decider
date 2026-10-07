from lib import *
eng = load(); tm = eng.model.torso
tot = 0
for i, l in enumerate(tm.layers):
    n = sum(p.numel() for p in l.parameters()); tot += n
    if i < 5 or i % 4 == 3 and i < 8: print(i, R_t := eng.model.torso.config.layer_types[i], n / 1e6)
print("layers total M", tot / 1e6, "embed M", tm.embed_tokens.weight.numel() / 1e6)
c = tm.config; print({k: getattr(c, k) for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim") if hasattr(c, k)})
