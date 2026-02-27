# Clutch-AI

A GPT-style language model trained from scratch and instruction-tuned for conversational AI. Built on GPT-2 Medium (350M parameters) and fine-tuned on Alpaca-GPT4 data.

> Created by **Battu Ujwal Reddy**

---

## Quick Start

```bash
git clone https://github.com/ujwalreddybattu04/Clutch-AI
cd Clutch-AI
pip install -r requirements.txt
python scripts/chat.py
```

---

## Features

- **Chain-of-thought reasoning** — model thinks step-by-step inside `<think>...</think>` tags before answering
- **Streaming output** — tokens appear in real-time as they're generated
- **Conversation history** — multi-turn context for follow-up questions
- **Repetition penalty** — prevents repetitive loops in generation
- **Top-p (nucleus) sampling** — industry-standard sampling alongside top-k
- **Colored terminal UI** — distinct colors for user vs AI responses
- **Response stats** — token count, generation time, tokens/sec after each response
- **System prompt** — built-in identity and behavior guidelines

---

## Usage

**Interactive chat:**
```bash
python scripts/chat.py
```

**Sampling options:**
```bash
python scripts/chat.py --temp 0.8 --top_k 50          # adjust temperature & top-k
python scripts/chat.py --top_p 0.9 --rep_pen 1.3      # nucleus sampling + repetition penalty
python scripts/chat.py --no-stream                     # disable streaming (print all at once)
python scripts/chat.py --hide-thinking                 # hide reasoning, show only final answer
python scripts/chat.py --history-turns 5               # keep 5 turns of conversation context
```

**Single prompt (non-interactive):**
```bash
python scripts/chat.py --prompt "Explain quantum computing"
```

**Use a specific checkpoint:**
```bash
python scripts/chat.py --ckpt out-clutch-gpt2-alpacagpt4/ckpt.pt
python scripts/chat.py --base   # use base pretrained model
```

---

## Training

**Instruction tuning (recommended — starts from GPT-2 Medium):**
```bash
# Step 1: Prepare data (downloads Alpaca-GPT4 + adds identity examples)
python data/prepare_alpaca_gpt4.py

# Step 2: Fine-tune
python train.py config/train_alpaca_sft.py
```

**Override hyperparameters from CLI:**
```bash
python train.py config/train_alpaca_sft.py --learning_rate=3e-5 --max_iters=8000
```

Config files are plain Python — every hyperparameter is overridable.

---

## Project Layout

```
Clutch-AI/
├── config/
│   ├── train_fineweb.py            # pretraining config
│   └── train_alpaca_sft.py         # instruction tuning config
├── data/
│   ├── alpaca_gpt4/                # SFT data with label masking
│   └── prepare_alpaca_gpt4.py      # data preparation script
├── src/clutch_ai/
│   ├── models/gpt.py               # model architecture (GPT-2)
│   └── training/train.py           # training loop (DDP-ready)
├── scripts/
│   ├── chat.py                     # inference + chat script
│   └── test_ckpt.py                # checkpoint testing
├── out-clutch-gpt2-alpacagpt4/     # best checkpoint (default)
└── train.py                        # training entry point
```

---

## Model

| | |
|---|---|
| Architecture | GPT-2 (decoder-only transformer) |
| Base weights | GPT-2 Medium — 350M parameters |
| Context length | 512 tokens |
| Tokenizer | GPT-2 BPE via tiktoken |
| Fine-tuning data | Alpaca-GPT4 (52K instructions) |
| Sampling | Top-k, Top-p (nucleus), temperature, repetition penalty |
| Hardware | NVIDIA RTX 4050 (6GB) |

---

## Checkpoints

| Path | Description |
|------|-------------|
| `out-clutch-0.1/` | Pretrained from scratch on FineWeb |
| `out-clutch-sft-alpaca/` | SFT on scratch base (weak) |
| `out-clutch-gpt2-alpacagpt4/` | SFT on GPT-2 Medium — **best quality (default)** |

---

## License

MIT