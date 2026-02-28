# Clutch-AI v1.0.0

A powerful, instruction-tuned AI assistant built on **Meta's Llama 3.2 3B Instruct** architecture. Fine-tuned on a world-class mix of ~2.4 million examples spanning general knowledge, coding, math, reasoning, and multi-turn conversations.

> Created by **Clutch Group**

---

## 🌟 Key Features

- **Base Model**: Meta Llama 3.2 Instruct (3B parameters).
- **Massive Dataset Mix**: Fine-tuned on ~2.4M high-quality examples:
  - *SlimOrca* (General knowledge)
  - *OpenHermes 2.5* (Diverse tasks)
  - *UltraChat 200K* (Conversations)
  - *Open-Platypus* (Reasoning)
  - *Orca-Math & MetaMathQA* (Math)
  - *MagicCoder & Code-Alpaca* (Coding)
  - *ScienceQA* (Science)
  - *TruthfulQA & Custom Datasets* (Identity and Anti-hallucination)
- **Kaggle-Optimized Training**: Uses native 4-bit QLoRA via HuggingFace PEFT and TRL (optimized for P100 compatibility) for highly efficient, low-memory training on Kaggle GPUs. Checkpoint auto-resume supported!

---

## 🚀 Quick Start (Inference)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Chat with your model:**
```bash
python scripts/inference.py
```

*Note: Make sure you have downloaded your trained model from Kaggle and placed it in the appropriate output directory (e.g. `out-clutch-1.0.0-final/merged`).*

### Chat Options:
```bash
# Custom sampling
python scripts/inference.py --temp 0.8 --top_k 50 --top_p 0.9 --rep_pen 1.2

# Single-shot prompt
python scripts/inference.py --prompt "Write a Python script to scrape a website."

# Disable streaming or hide <think> blocks
python scripts/inference.py --no-stream --hide-thinking
```

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
   - Copy the contents of `notebooks/train.py` into the notebook.
   - Run all cells.
3. **Resume Training:**
   - The dataset is huge (~2.4M). Kaggle limits sessions to 30 hours. The notebook automatically saves checkpoints every 500 steps. If it times out, simply run all cells again to resume from the last checkpoint.

---

## 📁 Project Layout

```text
Clutch-AI/
├── config/
│   ├── model_config.json           # Model identity, settings, and full dataset config
│   └── custom_examples.json        # Custom identity & safety examples
├── notebooks/
│   └── train.py                    # Kaggle training notebook
├── scripts/
│   ├── inference.py                # Primary Inference and Chat CLI
│   ├── evaluate_model.py           # Merged checkpoint standalone tester
│   └── inference_lora.py           # Local LoRA adapter live testing utility
└── requirements.txt                # Python dependencies
```

---

## 🧠 Model Specs

| Feature | Detail |
|---|---|
| Architecture | Meta Llama 3.2 |
| Size | 3 Billion Parameters |
| Context Length | 2048 tokens |
| Training Method | SFT + 4-bit QLoRA (Native HuggingFace) |
| Hardware Used | Kaggle Dual T4 GPUs |
| Sampling | Top-k, Top-p, temperature, repetition penalty |

---

## 📄 License
MIT License (Refer to specific dataset and base model licenses for fair use policies).