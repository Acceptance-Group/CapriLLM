import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device).unsqueeze(1).float()
    dims = torch.arange(0, head_dim, 2, device=device).float()
    freqs = positions / (theta ** (dims / head_dim))
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class KVCache:
    def __init__(self, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.k = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.position = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = k.shape[2]
        self.k[:, :, self.position : self.position + length] = k
        self.v[:, :, self.position : self.position + length] = v
        self.position += length
        return self.k[:, :, : self.position], self.v[:, :, : self.position]

    def slice(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, : self.position - length], self.v[:, :, : self.position - length]


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float, attention_backend: str):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.attention_backend = attention_backend

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

        cos, sin = build_rope_cache(max_seq_len, head_dim, rope_theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor, kv_cache: Optional[KVCache]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        offset = kv_cache.position if kv_cache is not None else 0

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_embeddings(q, self.cos[offset : offset + seq_len], self.sin[offset : offset + seq_len])
        k = apply_rotary_embeddings(k, self.cos[offset : offset + seq_len], self.sin[offset : offset + seq_len])

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        if self.attention_backend == "flash_attention_2":
            try:
                from flash_attn import flash_attn_func

                out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
            except ImportError:
                out = self._eager_attention(q, k, v, offset).transpose(1, 2)
        else:
            out = self._eager_attention(q, k, v, offset).transpose(1, 2)

        out = out.reshape(batch_size, seq_len, -1)
        return self.o_proj(out)

    def _eager_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offset: int) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_len = k.shape[2]
        query_len = q.shape[2]
        causal_mask = torch.ones(query_len, total_len, device=scores.device, dtype=torch.bool).tril(diagonal=offset)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
        return torch.matmul(attn, v)
