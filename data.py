import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from config import ROLE_TOKENS


def format_tools(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    return "<tools|>\n" + json.dumps(tools, ensure_ascii=False, indent=2) + "\n<end|>\n"


def format_messages(messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    parts = [format_tools(tools)] if tools else []
    for message in messages:
        role_token = ROLE_TOKENS[message["role"]]
        parts.append(f"{role_token}\n{message['content']}\n<end|>")
    return "\n".join(parts)


def parse_function_call(text: str) -> Optional[Dict[str, Any]]:
    if "<call|>" not in text or "<end|>" not in text:
        return None
    raw = text.split("<call|>", 1)[1].split("<end|>", 1)[0].strip()
    try:
        call = json.loads(raw)
        if isinstance(call, dict) and "name" in call:
            return {"name": call["name"], "arguments": call.get("arguments", {})}
    except json.JSONDecodeError:
        return None
    return None


class ChatDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        dataset_name: Optional[str] = None,
        data_path: Optional[str] = None,
        format_template: Optional[str] = None,
        messages_column: Optional[str] = None,
        tools_column: Optional[str] = None,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.format_template = format_template
        self.messages_column = messages_column
        self.tools_column = tools_column

        if data_path:
            self.raw_data = load_dataset("json", data_files=data_path, split=split)
        elif dataset_name:
            self.raw_data = load_dataset(dataset_name, split=split)
        else:
            raise ValueError("Either dataset_name or data_path must be provided")

    def __len__(self):
        return len(self.raw_data)

    def _to_text(self, example) -> str:
        if self.format_template:
            return self.format_template.format(**example)
        if self.messages_column:
            return format_messages(example[self.messages_column], example.get(self.tools_column))
        return example["text"]

    def __getitem__(self, idx):
        text = self._to_text(self.raw_data[idx])
        input_ids = self.tokenizer.encode(text, add_bos=True)
        input_ids = input_ids[: self.max_seq_len]
        input_ids.append(self.tokenizer.vocabulary.token_id("<eos>"))
        pad_id = self.tokenizer.vocabulary.token_id("<pad>")
        input_ids = input_ids + [pad_id] * (self.max_seq_len - len(input_ids))
        return {"input_ids": input_ids, "labels": list(input_ids)}


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0, sampler=None) -> DataLoader:
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
