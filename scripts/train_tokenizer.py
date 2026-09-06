from pathlib import Path

from tokenizer.trainer import BPETrainer
from config import SPECIAL_TOKENS


def main(args):
    path = Path(args.input)
    if path.suffix == ".jsonl":
        import json

        with open(path, "r", encoding="utf-8") as f:
            texts = []
            for line in f:
                if line.strip():
                    texts.append(json.loads(line).get("text", ""))
    else:
        texts = path.read_text(encoding="utf-8").splitlines()

    trainer = BPETrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer = trainer.train(texts)
    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}, vocab size {tokenizer.vocabulary.size()}")
