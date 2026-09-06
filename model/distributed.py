from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def setup_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_training(model: torch.nn.Module, strategy: str, transformer_block_class: type, precision: str) -> torch.nn.Module:
    if strategy == "fsdp":
        from torch.distributed.fsdp import MixedPrecision

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype) if dtype else None
        auto_wrap_policy = transformer_auto_wrap_policy(module_cls={transformer_block_class})
        return FSDP(model, auto_wrap_policy=auto_wrap_policy, mixed_precision=mixed_precision)
    if strategy == "ddp":
        return torch.nn.parallel.DistributedDataParallel(model)
    return model
