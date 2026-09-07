import json
from pathlib import Path
from typing import Optional

import torch

from config import GenerationConfig, ModelConfig
from data import format_messages, parse_function_call, split_reasoning
from model.core import Transformer
from tokenizer.core import BPETokenizer
from utils import get_device


def main(args):
    tokenizer = BPETokenizer.load(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = Transformer(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(get_device())
    model.eval()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    tools = None
    if args.tools:
        tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))

    think_prefix = "<think|>\n" if getattr(args, "think", False) else ""
    prompt = format_messages(messages, tools) + f"\n<assistant|>\n{think_prefix}"
    stop_token_ids = [
        tokenizer.vocabulary.token_id("<end|>"),
        tokenizer.vocabulary.token_id("<eos>"),
    ]
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=stop_token_ids,
    )

    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=next(model.parameters()).device)
    input_len = input_ids.shape[1]
    output_ids = model.generate(input_ids, generation_config)
    text = tokenizer.decode(output_ids[0, input_len:].tolist())
    assistant_text = text.split("<end|>")[0].split("<eos>")[0].strip()

    reasoning, answer = split_reasoning(assistant_text)
    function_call = parse_function_call(answer)
    if function_call:
        print(json.dumps({"function_call": function_call}, ensure_ascii=False, indent=2))
        return
    if reasoning:
        print(f"--- Reasoning ---\n{reasoning}\n--- Answer ---")
    print(answer)
