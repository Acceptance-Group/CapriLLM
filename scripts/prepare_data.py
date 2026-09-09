import argparse
import json
from pathlib import Path

from datasets import load_dataset

EN_SYSTEM = "You are a helpful assistant."
RU_SYSTEM = "Ты полезный ассистент."
REASONING_SYSTEM = "You are a helpful assistant. Think step by step."

SFT_TEMPLATE = "<system|>\n{system}\n<end|>\n<user|>\n{instruction}\n{input}\n<end|>\n<assistant|>\n{output}\n<end|>"
REASONING_TEMPLATE = "<system|>\n{system}\n<end|>\n<user|>\n{question}\n<end|>\n<assistant|>\n<think|>\n{trace}\n</think|>\n{answer}\n<end|>"

CHUNK_CHARS = 3800
MIN_CHUNK_CHARS = 200


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")


def chunk_text(text):
    chunks = []
    for start in range(0, len(text), CHUNK_CHARS):
        chunk = text[start : start + CHUNK_CHARS].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
    return chunks


def dump_pretrain(en_articles, ru_articles):
    rows = []
    for lang, count in [("en", en_articles), ("ru", ru_articles)]:
        stream = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        taken = 0
        for example in stream:
            if taken >= count:
                break
            for chunk in chunk_text(example["text"]):
                rows.append({"text": chunk, "lang": lang})
            taken += 1
        print(f"pretrain: {taken} {lang} articles -> {len(rows)} chunks so far")
    return rows


def dump_ru_sft():
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    rows = []
    path = hf_hub_download(repo_id="IlyaGusev/ru_turbo_alpaca", filename="ru_turbo_alpaca.jsonl.zst", repo_type="dataset")
    with open(path, "rb") as fh:
        text = zstd.ZstdDecompressor().stream_reader(fh).read().decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        instruction = example.get("instruction") or example.get("question") or ""
        inp = example.get("input") or ""
        output = example.get("output") or example.get("answer") or ""
        rows.append({"text": SFT_TEMPLATE.format(system=RU_SYSTEM, instruction=instruction, input=inp, output=output)})
    return rows


def dump_sft():
    rows = []
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for example in alpaca:
        rows.append({"text": SFT_TEMPLATE.format(system=EN_SYSTEM, instruction=example["instruction"], input=example["input"], output=example["output"])})
    print(f"sft en: {len(rows)} examples")

    ru_rows = dump_ru_sft()
    print(f"sft ru: {len(ru_rows)} examples")
    return rows + ru_rows


def dump_reasoning():
    rows = []
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    total_budget = 15000
    for example in ds:
        trace = example.get("deepseek_thinking_trajectory") or ""
        answer = example.get("deepseek_attempt") or example.get("solution") or ""
        question = example["question"]
        if not trace or not answer:
            continue
        if len(trace) + len(answer) + len(question) > total_budget:
            limit = total_budget - len(answer) - len(question)
            trace = trace[:limit]
            cut = trace.rfind("\n\n")
            if cut > limit // 2:
                trace = trace[:cut]
        rows.append({"text": REASONING_TEMPLATE.format(system=REASONING_SYSTEM, question=question, trace=trace, answer=answer)})
    print(f"reasoning: {len(rows)} examples")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare RU+EN pretrain/SFT/reasoning datasets")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pretrain-en", type=int, default=50000)
    parser.add_argument("--pretrain-ru", type=int, default=50000)
    parser.add_argument("--tokenizer-pretrain-articles", type=int, default=10000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    pretrain_rows = dump_pretrain(args.pretrain_en, args.pretrain_ru)
    write_jsonl(data_dir / "pretrain_ru_en.jsonl", pretrain_rows)

    sft_rows = dump_sft()
    write_jsonl(data_dir / "sft_ru_en.jsonl", sft_rows)

    reasoning_rows = dump_reasoning()
    write_jsonl(data_dir / "reasoning_en.jsonl", reasoning_rows)

    tokenizer_rows = sft_rows + reasoning_rows + [
        {"text": row["text"]} for row in pretrain_rows[: args.tokenizer_pretrain_articles * 2]
    ]
    write_jsonl(data_dir / "tokenizer_corpus.jsonl", tokenizer_rows)
    print("Done. Train tokenizer on data/tokenizer_corpus.jsonl")


if __name__ == "__main__":
    main()
