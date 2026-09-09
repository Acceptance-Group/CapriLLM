import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from config import ModelConfig, TrainConfig
from data import ChatDataset, build_dataloader
from model.core import Transformer, TransformerBlock
from model.distributed import (
    cleanup_distributed,
    setup_distributed,
    wrap_model_for_training,
)
from utils import dist_initialized, get_device, is_main_process, set_seed


class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, tokenizer):
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.device = get_device()
        torch.set_num_threads(train_config.num_threads)
        set_seed(train_config.seed)

        self.model = Transformer(model_config).to(self.device)
        if train_config.init_from:
            checkpoint = torch.load(train_config.init_from, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"])
            print(f"Initialized weights from {train_config.init_from}")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        total_steps = train_config.max_steps or 10000
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config.warmup_steps,
            num_training_steps=total_steps,
        )
        self.scaler = torch.amp.GradScaler("cuda") if train_config.precision == "fp16" and self.device.type == "cuda" else None
        self.global_step = 0

    def train(self):
        if self.train_config.distributed_strategy != "single_gpu":
            setup_distributed()
        self.model = wrap_model_for_training(
            self.model,
            self.train_config.distributed_strategy,
            TransformerBlock,
            self.train_config.precision,
        )

        dataset = ChatDataset(
            tokenizer=self.tokenizer,
            max_seq_len=self.model_config.max_seq_len,
            dataset_name=self.train_config.dataset_name,
            data_path=self.train_config.data_path,
            format_template=self.train_config.format_template,
        )

        sampler = None
        if dist_initialized():
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(dataset)
        dataloader = build_dataloader(dataset, self.train_config.batch_size, self.train_config.num_workers, sampler)

        if self.train_config.gradient_checkpointing:
            self.model.gradient_checkpointing = True

        self.model.train()
        total_steps = self.train_config.max_steps or len(dataloader) * self.train_config.num_epochs
        train_start = time.time()
        for epoch in range(self.train_config.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in dataloader:
                self.global_step += 1
                loss = self._train_step(batch)
                if is_main_process() and self.global_step % self.train_config.eval_interval == 0:
                    elapsed = time.time() - train_start
                    speed = self.global_step / elapsed if elapsed > 0 else 0.0
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"step {self.global_step}/{total_steps}, loss {loss:.4f}, lr {lr:.2e}, {speed:.2f} steps/s")
                if is_main_process() and self.global_step % self.train_config.save_interval == 0:
                    self._save_checkpoint()
                if self.train_config.max_steps and self.global_step >= self.train_config.max_steps:
                    self._finish()
                    return
        self._finish()

    def _train_step(self, batch) -> float:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        precision = self.train_config.precision
        amp_enabled = precision in ("fp16", "bf16") and self.device.type in ("cuda", "mps")
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = self.model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=self.tokenizer.vocabulary.token_id("<pad>"),
            )

        loss = loss / self.train_config.gradient_accumulation_steps
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.global_step % self.train_config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item()

    def _save_checkpoint(self, name=None):
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / (name or f"checkpoint_{self.global_step}.pt")
        state = {
            "model": self.model.state_dict(),
            "model_config": vars(self.model_config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def _finish(self):
        self._save_checkpoint("checkpoint_final.pt")
        if dist_initialized():
            cleanup_distributed()
