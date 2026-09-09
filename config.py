from dataclasses import dataclass
from typing import List, Literal, Optional

ROLE_TOKENS = {
    "system": "<system|>",
    "user": "<user|>",
    "assistant": "<assistant|>",
    "tool": "<tool|>",
}

SPECIAL_TOKENS = {
    "<pad>": "<pad>",
    "<unk>": "<unk>",
    "<bos>": "<bos>",
    "<eos>": "<eos>",
    "<system|>": "<system|>",
    "<user|>": "<user|>",
    "<assistant|>": "<assistant|>",
    "<tool|>": "<tool|>",
    "<tools|>": "<tools|>",
    "<call|>": "<call|>",
    "<end|>": "<end|>",
    "<think|>": "<think|>",
    "</think|>": "</think|>",
}


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1280
    num_layers: int = 24
    num_heads: int = 20
    num_kv_heads: int = 4
    max_seq_len: int = 512
    intermediate_size: int = 3584
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    attention_backend: Literal["flash_attention_2", "eager"] = "eager"
    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2


@dataclass
class TrainConfig:
    dataset_name: Optional[str] = None
    data_path: Optional[str] = None
    format_template: Optional[str] = None
    init_from: Optional[str] = None
    tokenizer_path: str = "tokenizer.json"
    output_dir: str = "outputs"
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    precision: Literal["bf16", "fp16", "fp32"] = "fp32"
    distributed_strategy: Literal["fsdp", "ddp", "single_gpu"] = "single_gpu"
    gradient_checkpointing: bool = True
    eval_interval: int = 100
    save_interval: int = 1000
    seed: int = 42
    num_workers: int = 0
    num_threads: int = 4


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    repetition_penalty: float = 1.1
    stop_token_ids: List[int] = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []
