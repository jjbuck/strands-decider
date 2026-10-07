# coding=utf-8
from transformers.configuration_utils import PretrainedConfig

class SmallThinkerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`SmallThinkerModel`]. 
    It is used to instantiate a SmallThinker model according to the specified arguments, defining the model architecture. 
    The default values for each of the parameters are the same as the ones used in the original SmallThinker 4B model.

    General configs:
    - model_type: "smallthinker"
    - model_name
    - num_hidden_layers
    - hidden_size

    Tokenizer configs:
    - pad_token_id
    - bos_token_id
    - eos_token_id

    Embedding configs:
    - vocab_size

    RMSNorm configs:
    - rms_norm_eps

    Attention configs:
    - num_attention_heads
    - num_key_value_heads
    - head_dim
    - use_cache
    - rope_layout: array of 0 or 1s, 0 for nope, 1 for rope
    - rope_theta
    - max_position_embeddings
    - sliding_window_layout: array of 0 or 1s, 0 for normal attention, 1 for SWA
    - sliding_window_size

    MoE FFN configs:
    - moe_num_primary_experts
    - moe_ffn_hidden_size
    - moe_primary_router_apply_softmax: Use topk-softmax in routing instead of topk-sigmoid-normalize
    - moe_num_active_primary_experts

    LM Head configs:
    - tie_word_embeddings

    Other configs:
    - initializer_range
    """
    def __init__(self,
        model_type = "smallthinker",
        model_name="smallthinker_4b_base",
        num_hidden_layers=32,
        hidden_size=1536,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=[151643,151645],
        vocab_size=151936,
        rms_norm_eps=1e-6,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=128,
        use_cache=True,
        rope_layout=[1]*32,
        rope_theta=1e6,
        max_position_embeddings=4096 * 32,
        sliding_window_layout=[0]*32,
        sliding_window_size=4096,
        moe_num_primary_experts=32,
        moe_ffn_hidden_size=768,
        moe_primary_router_apply_softmax=False,
        moe_num_active_primary_experts=4,
        tie_word_embeddings=True,
        initializer_range=0.02,
        **kwargs,
    ):
        # Configuration sanitizers
        assert num_attention_heads % num_key_value_heads == 0,      "[SmallThinker config sanitizer] num_attention_heads must be divisible by num_key_value_heads"
        assert len(rope_layout) == num_hidden_layers,               "[SmallThinker config sanitizer] rope_layout must have the same length as num_hidden_layers"
        assert len(sliding_window_layout) == num_hidden_layers,     "[SmallThinker config sanitizer] sliding_window_layout must have the same length as num_hidden_layers"
        
        # General configs
        self.model_type = model_type
        self.model_name = model_name
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        # Tokenizer configs
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embedding configs
        self.vocab_size = vocab_size

        # RMSNorm configs
        self.rms_norm_eps = rms_norm_eps

        # Attention configs
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_cache = use_cache
        self.rope_layout = rope_layout
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window_layout = sliding_window_layout
        self.sliding_window_size = sliding_window_size

        # MoE FFN configs
        self.moe_num_primary_experts = moe_num_primary_experts
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.moe_primary_router_apply_softmax = moe_primary_router_apply_softmax
        self.moe_num_active_primary_experts = moe_num_active_primary_experts

        # Other configs
        self.initializer_range = initializer_range

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

        # VLLM config, not used in transformers, but VLLM requires these args to run correctly. DO NOT DELETE!
        self.sliding_window = sliding_window_size
        self.sliding_window_pattern = sliding_window_layout

        self._attn_implementation = "sdpa"
        
__all__ = ["SmallThinkerConfig"]
