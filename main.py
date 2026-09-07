import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.generate import main as generate_main
from scripts.train_model import main as train_model_main
from scripts.train_tokenizer import main as train_tokenizer_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm", description="Scalable instruct LLM platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenizer_parser = subparsers.add_parser("train-tokenizer", help="Train BPE tokenizer")
    tokenizer_parser.add_argument("--input", required=True, help="Path to raw text or jsonl")
    tokenizer_parser.add_argument("--output", default="tokenizer.json")
    tokenizer_parser.add_argument("--vocab-size", type=int, default=32000)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", required=True, help="Path to config YAML")
    train_parser.add_argument("--override", nargs="*", default=[], help="Override train config values, e.g. train.max_steps=100")

    generate_parser = subparsers.add_parser("generate", help="Chat completion with tools support")
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--tokenizer", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--system", default=None, help="System instruction")
    generate_parser.add_argument("--tools", default=None, help="Path to JSON with tool schemas")
    generate_parser.add_argument("--max-new-tokens", type=int, default=512)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("--top-k", type=int, default=None)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.1)
    generate_parser.add_argument("--think", action="store_true", help="Force reasoning with <think|>")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-tokenizer":
        train_tokenizer_main(args)
    elif args.command == "train":
        train_model_main(args)
    elif args.command == "generate":
        generate_main(args)


if __name__ == "__main__":
    main()
