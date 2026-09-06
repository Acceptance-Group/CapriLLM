from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GenerationConfig, ModelConfig

from .attention import GroupedQueryAttention, KVCache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_dim = config.d_model // config.num_heads
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            attention_backend=config.attention_backend,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.intermediate_size)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[List[KVCache]] = None) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, kv_cache, use_reentrant=False)
            else:
                x = layer(x, kv_cache)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        head_dim = self.config.d_model // self.config.num_heads

        kv_caches = [
            KVCache(batch_size, self.config.max_seq_len, self.config.num_kv_heads, head_dim, dtype, device)
            for _ in range(self.config.num_layers)
        ]
        output_ids = input_ids
        logits = self(input_ids, kv_caches)
        stop_ids = torch.tensor(generation_config.stop_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        for _ in range(generation_config.max_new_tokens):
            next_id = self._sample(logits[:, -1, :], output_ids, generation_config)
            output_ids = torch.cat([output_ids, next_id], dim=-1)
            if len(stop_ids) and (next_id == stop_ids).any():
                break
            logits = self(next_id, kv_caches)
        return output_ids

    def _sample(self, logits: torch.Tensor, output_ids: torch.Tensor, generation_config: GenerationConfig) -> torch.Tensor:
        if generation_config.repetition_penalty != 1.0:
            for batch_idx in range(logits.shape[0]):
                recent = output_ids[batch_idx, -64:]
                logits[batch_idx, recent] = logits[batch_idx, recent] / generation_config.repetition_penalty

        logits = logits / max(generation_config.temperature, 1e-5)
        if generation_config.top_k is not None:
            top_k = min(generation_config.top_k, logits.shape[-1])
            kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth_value, float("-inf"))

        if generation_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= generation_config.top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        return torch.multinomial(probs, num_samples=1)
