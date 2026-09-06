# CapriLLM ♑

CapriLLM is a lean, scalable framework for building, training, and serving your own instruct-tuned Large Language Models — from a 0.5B model on a MacBook to trillion-parameter architectures on industrial GPU clusters. It ships with a custom BPE tokenizer, a modern decoder-only Transformer, and a single CLI entry point for the entire pipeline.

---

## ✨ Key Features & Technical Highlights

*   **🚀 Fast Autoregressive Generation with KV Caching:** A per-layer KV cache eliminates redundant computation during token generation, drastically reducing time-to-first-token (TTFT) and keeping inference fast even on long contexts.
*   **⚡ Modern Attention Core:** Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE), optional Flash Attention 2 on CUDA, and an eager fallback that runs anywhere — including Apple Silicon (MPS).
*   **🛠️ Custom BPE Tokenizer:** Train your own vocabulary from raw corpora or JSONL, with first-class special tokens for chat roles (`<system|>`, `<user|>`, `<assistant|>`, `<tool|>`), tool declarations (`<tools|>`), and function calls (`<call|>`).
*   **🌐 Scalable Distributed Training:** Single-GPU, DDP, and FSDP strategies selected via config — no code rewrites when moving from a laptop to a cluster.
*   **🎛️ Declarative Configuration:** Predefined `ARM-*` and `X64-*` profiles cover MacBook-scale 0.5B runs up to 1000B MoE setups, with CLI overrides (`--override key=value`) for every field.
*   **🧩 Instruct-First Design:** Chat templating, system instructions, sampling controls (temperature, top-p, top-k, repetition penalty), and built-in function-calling parsing come standard.
*   **🏎️ Efficient Trainer:** Mixed precision (bf16/fp16/fp32), gradient accumulation, gradient checkpointing, and thread capping (`num_threads`) so your machine stays responsive during local runs.

---

## 📁 Project Structure

```
instruct_llm/
├── config.py              # ModelConfig, TrainConfig, GenerationConfig, chat tokens
├── tokenizer/
│   ├── core.py            # Vocabulary + BPETokenizer (encode/decode/save/load)
│   └── trainer.py         # BPETrainer — learn merges from raw text
├── model/
│   ├── attention.py       # GQA + RoPE + KVCache + Flash Attention fallback
│   ├── core.py            # Transformer, TransformerBlock, RMSNorm, SwiGLU, generate()
│   └── distributed.py     # FSDP / DDP / single-GPU wrappers
├── data.py                # ChatDataset, chat formatting, function-call parsing
├── train.py               # Trainer: AMP, grad accum/checkpointing, checkpoints
├── utils.py               # Seed, device selection (CUDA/MPS/CPU), param counts
├── main.py                # CLI entry point
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_model.py
│   └── generate.py        # Chat completion with system/tools/sampling
├── configs/
│   ├── ARM-0.5b.yaml      # MacBook (M-series) presets
│   ├── ARM-1b.yaml
│   ├── ARM-7b.yaml
│   ├── X64-0.5b.yaml      # Single-GPU → cluster presets
│   ├── X64-1b.yaml
│   ├── X64-7b.yaml        # FSDP
│   ├── X64-70b.yaml       # FSDP
│   └── X64-1000b.yaml     # FSDP + MoE
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Train the tokenizer

```bash
python main.py train-tokenizer \
  --input data/alpaca_text.jsonl \
  --output tokenizer.json \
  --vocab-size 8000
```

### 2. Train a model

```bash
python main.py train --config configs/ARM-0.5b.yaml --override model.vocab_size=8000
```

Override any field without touching the file:

```bash
python main.py train --config configs/X64-1b.yaml \
  --override train.max_steps=5000 train.learning_rate=2.0e-4
```

On Apple Silicon, cap MPS memory to keep the system stable:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 python main.py train --config configs/ARM-0.5b.yaml
```

### 3. Generate with system prompt and tools

```bash
python main.py generate \
  --checkpoint outputs/arm_0.5b/checkpoint_1000.pt \
  --tokenizer tokenizer.json \
  --prompt "What is the weather in Paris?" \
  --system "You are a helpful assistant." \
  --tools tools.json \
  --temperature 0.7 --top-p 0.9
```

If the model decides to call a tool, it emits `<call|>{"name": ..., "arguments": ...}<end|>` and the CLI prints it as structured JSON:

```json
{"function_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}
```

---

## 🗂️ Configuration Profiles

| Profile | Target hardware | Strategy | Notes |
|---|---|---|---|
| `ARM-0.5b` / `ARM-1b` | MacBook (M-series, 16GB+) | single GPU (MPS) | fp32, thread-capped, checkpointing on |
| `ARM-7b` | Mac Studio / high-RAM ARM | single GPU (MPS) | Long context up to 4096 |
| `X64-0.5b` / `X64-1b` | Single CUDA GPU | single GPU | bf16, Flash Attention 2 |
| `X64-7b` | Multi-GPU node | FSDP | bf16, gradient checkpointing |
| `X64-70b` | Multi-node cluster | FSDP | 64K vocab, long context |
| `X64-1000b` | Industrial cluster | FSDP | MoE (64 experts, top-6), 100K vocab |

---

## 💡 Why CapriLLM?

### 🎯 Tailored for Proprietary and Domain-Specific Data
Train your own tokenizer and model directly on your corpus. Chat formatting, tool schemas, and system behavior are fully under your control.

### 🔬 Proven Architecture, Zero Bloat
Llama-style stack (RMSNorm, SwiGLU, GQA, RoPE) with an uncluttered codebase built for readability and extensibility — no comments, self-documenting code.

### 📉 Resource Efficiency
Gradient checkpointing, mixed precision, tied embeddings, and thread limits make local development practical while keeping the same code path ready for cluster scale.

---

## 🤝 Contributing & Community

Whether you want to optimize the distributed training logic, enhance the tokenizer performance, or fix a typo, your contributions are highly welcome! Please feel free to open issues, submit pull requests, or share your architectural benchmarks.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (GNU AGPLv3)**. 

Under this license, you are free to use, modify, and distribute this software for personal and research purposes. However, if you modify CapriLLM or integrate it into a network-accessible service (such as a cloud API or SaaS backend), you **must make your entire source code available to the public** under the same AGPLv3 terms. Commercial exploitation in closed-source proprietary environments is strictly prohibited by these copyleft provisions. See the [LICENSE](LICENSE) file for more details.
