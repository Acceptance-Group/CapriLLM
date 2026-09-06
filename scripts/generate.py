import json
from pathlib import Path
from typing import Optional

import torch

from config import GenerationConfig, ModelConfig
from data import format_messages, parse_function_call
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

    prompt = format_messages(messages, tools)
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
    output_ids = model.generate(input_ids, generation_config)
    raw_output = tokenizer.decode(output_ids[0].tolist())
    assistant_text = raw_output.split("<assistant|>")[-1].split("<end|>")[0].strip() if "<assistant|>" in raw_output else raw_output

    function_call = parse_function_call(assistant_text)
    if function_call:
        print(json.dumps({"function_call": function_call}, ensure_ascii=False, indent=2))
    else:
        print(assistant_text)
