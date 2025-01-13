import dataclasses


@dataclasses.dataclass
class LlamaConfig:
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 32
    num_hidden_layers: int = 22
    rms_norm_eps: float = 1e-05
    vocab_size: int = 32000
    rope_theta: int = 10000
    context_window_size: int = 2048
    prefill_chunk_size: int = 2048
    num_key_value_heads: int = 4
    head_dim: int = 64  # hidden_size // num_attention_heads
