# Clutch-AI v0.3

A powerful, instruction-tuned AI assistant built by upgrading from GPT-2 to **Meta's Llama 3.2 3B Instruct**. Fine-tuned on a world-class mix of ~2.4 million examples (general knowledge, coding, math, reasoning, and multi-turn conversations).

> Created by **Battu Ujwal Reddy**

---

## 🌟 What's New in v0.3 (The Llama Upgrade)

- **New Base Model**: Upgraded from GPT-2 Medium (350M) to Meta Llama 3.2 Instruct (3B).
- **Massive Dataset Mix**: Fine-tuned on ~2.4M high-quality examples instead of just 52K.
  - *SlimOrca* (General GPT-4 quality)
  - *OpenHermes 2.5* (Diverse tasks)
  - *UltraChat 200K* (Conversations)
  - *Open-Platypus* (Reasoning)
  - *Orca-Math & MetaMathQA* (Math)
  - *MagicCoder & Code-Alpaca* (Coding)
  - *ScienceQA* (Science)
  - *TruthfulQA & Custom Datasets* (Identity and Anti-hallucination)
- **Kaggle-Optimized Training**: Uses 4-bit QLoRA and Unsloth for blazing fast, low-memory training on a Kaggle free T4 GPU. Checkpoint auto-resume supported!

---

## 🚀 Quick Start (Inference)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Chat with the new Llama 3.2 model:**
```bash
python scripts/chat_llama.py
```

*Note: Make sure you have downloaded your trained model from Kaggle and placed it in the `out-clutch-llama3.2-final/merged` directory.*

### Chat Options:
```bash
# Custom sampling
python scripts/chat_llama.py --temp 0.8 --top_k 50 --top_p 0.9 --rep_pen 1.2

# Single-shot prompt
python scripts/chat_llama.py --prompt "Write a Python script to scrape a website."

# Disable streaming or hide <think> blocks
python scripts/chat_llama.py --no-stream --hide-thinking
```

> **For Legacy GPT-2 Models:**
> Use `python scripts/chat.py` (the old script) for any GPT-2 checkpoints.

---

## 🏋️‍♂️ Training on Kaggle

This project is configured to be trained on Kaggle using their free T4 GPUs.

1. **HuggingFace Setup:**
   - Create an account on HuggingFace.
   - Accept the license for `meta-llama/Llama-3.2-3B-Instruct`.
   - Create a Read access token.
2. **Kaggle Setup:**
   - Create a new notebook with GPU T4 x2 accelerator.
   - Add your HF token to Kaggle Secrets as `HF_TOKEN`.
   - Copy the contents of `notebooks/train_llama_kaggle.py` into the notebook.
   - Run all cells.
3. **Resume Training:**
   - The dataset is huge (~2.4M). Kaggle limits sessions to 30 hours. The notebook automatically saves checkpoints every 500 steps. If it stops, just Run All again to resume!

---

## 📁 Project Layout

```text
Clutch-AI/
├── config/
│   ├── model_config.json           # Model identity, settings, and full dataset config
│   ├── custom_examples.json        # Your custom identity & safety examples
│   └── train_alpaca_sft.py         # (Legacy) GPT-2 fine-tuning config
├── notebooks/
│   ├── train_llama_kaggle.py       # 🔥 NEW: v0.3 Llama 3.2 QLoRA training notebook
│   └── train_kaggle.py             # (Legacy) GPT-2 manual training
├── scripts/
│   ├── chat_llama.py               # 🔥 NEW: Llama 3.2 chat inference
│   ├── chat.py                     # (Legacy) GPT-2 chat inference
│   ├── test_ckpt_llama.py          # Quick checkpoint tester for Llama
│   └── test_ckpt.py                # Quick checkpoint tester for GPT-2
├── data/
│   └── prepare_alpaca_gpt4.py      # (Legacy) Data prep for GPT-2
├── src/clutch_ai/                  # (Legacy) Custom nanoGPT implementation
└── requirements.txt                # Updated dependencies
```

---

## 🧠 Model Specs (v0.3)

| Feature | Detail |
|---|---|
| Architecture | Meta Llama 3.2 |
| Size | 3 Billion Parameters |
| Context Length | 2048 tokens (training setup) |
| Training Method | SFT + 4-bit QLoRA via Unsloth |
| Hardware Used | Kaggle Dual T4 GPUs |
| Sampling | Top-k, Top-p, temperature, repetition penalty |

---

## 📄 License
MIT License (Refer to specific dataset and base model licenses for fair use policies).