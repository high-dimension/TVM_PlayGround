from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache

from LLM_Optimize.LlamaAttention import LlamaAttention
from LLM_Optimize.LlamaConfig import LlamaConfig
from LLM_Optimize.LlamaFFN import LlamaFFN


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaFFN(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), paged_kv_cache, layer_id
        )
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states
