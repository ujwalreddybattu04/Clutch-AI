<p align="center">
  <img src="https://img.shields.io/badge/Clutch--AI-v1.0.0-00D4FF?style=for-the-badge&logo=robot&logoColor=white" alt="Clutch-AI v1.0.0" />
  <img src="https://img.shields.io/badge/Llama_3.2-3B_Instruct-purple?style=for-the-badge&logo=meta&logoColor=white" alt="Llama 3.2" />
  <img src="https://img.shields.io/badge/QLoRA-4--bit-green?style=for-the-badge" alt="QLoRA" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License" />
</p>

<h1 align="center">🤖 Clutch-AI</h1>

<p align="center">
  <strong>An industry-grade AI assistant with real-time web intelligence, built on Meta's Llama architecture.</strong>
  <br />
  <em>Perplexity-style smart search · LLM-powered query routing · Config-driven architecture · Zero hardcoded logic</em>
</p>

<p align="center">
  Created by <strong>Clutch Group</strong>
</p>

---

## ✨ Features

### 🌐 Real-Time Web Intelligence (Perplexity-Style)
- **Silent web search** powered by [Tavily API](https://tavily.com/) — fetches real-time data from the internet
- **LLM-powered query router** — the model itself decides when web search is needed (no keyword matching)
- **Seamless synthesis** — web results are woven naturally into responses, never shown raw to the user
- No URLs, no source dumps — just clean, informed answers

### ⚙️ Fully Config-Driven Architecture
Every prompt, parameter, and setting lives in `config/model_config.json`:
- System prompt, behavior rules, web context instructions, router classification prompt
- Generation parameters (temperature, top_k, top_p, repetition_penalty, max_tokens)
- Web search settings (depth, max results, content length)
- Memory depth (conversation turns)

**Change any behavior by editing one JSON file — zero Python changes needed.**

### 🧠 Smart Query Router
- Uses the LLM itself for intent classification (SEARCH or SKIP)
- Few-shot examples loaded from config for reliable routing
- Greedy decoding with `max_new_tokens=2` for minimal latency
- Defaults to SEARCH when uncertain (Perplexity-style bias)

### 🕐 Real-Time Date & Time
- Injects live UTC timestamps with pre-computed timezone conversions (UTC, EST, GMT, IST, JST, PST)
- The model always knows the current date and time — no hallucinated dates

### 🏋️ Training Pipeline
- Fine-tuned on **~249K curated examples** across 11 datasets
- 4-bit QLoRA via HuggingFace PEFT + TRL
- Optimized for Kaggle free-tier GPUs (P100/T4, 16GB VRAM)
- Auto-checkpoint every 500 steps with seamless resume

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (4GB+ VRAM for inference)
- [HuggingFace](https://huggingface.co/) account with Llama 3.2 access

### 1. Clone & Install
```bash
git clone https://github.com/ujwalreddybattu04/Clutch-AI.git
cd Clutch-AI
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file in the project root:
```env
TAVILY_API_KEY=your_tavily_api_key_here
```
> Get a free Tavily API key at [tavily.com](https://tavily.com/)

### 3. Run Clutch-AI
```bash
python scripts/inference_lora.py
```

You'll be prompted for your HuggingFace token, then dropped into an interactive chat:

```
════════════════════════════════════════════════════════════
  🤖  Clutch-AI v1.0.0  —  by Clutch Group
════════════════════════════════════════════════════════════
  GPU           NVIDIA GeForce RTX 4050 Laptop GPU
  Web Search    ON (Perplexity mode — smart routing)
  Router        LLM-powered intent classifier
────────────────────────────────────────────────────────────
  Commands      /web on · /web off · /search <query> · quit
════════════════════════════════════════════════════════════

You ▶ what is quantum computing?

  Clutch-AI ▶ Quantum computing is a type of computation that uses quantum
  mechanical phenomena to process information...
  ⚡ 145 tokens · 18.2s · 8.0 tok/s · 🌐 web · router 642ms
```

### Chat Commands
| Command | Action |
|---|---|
| `/web on` | Enable smart web search (default) |
| `/web off` | Disable web search (faster, offline) |
| `/search <query>` | Force web search for a specific query |
| `quit` or `exit` | Exit the chat |

---

## 🏋️ Training on Kaggle

### Setup
1. **HuggingFace**: Accept the [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and create a read access token
2. **Kaggle**: Create a new notebook with **GPU T4 x2** accelerator
3. **Secrets**: Add your HF token to Kaggle Secrets as `HF_TOKEN`

### Run Training
```python
import os
import subprocess

os.chdir('/kaggle/working/')

if not os.path.exists('Clutch-AI'):
    subprocess.run(['git', 'clone', 'https://github.com/ujwalreddybattu04/Clutch-AI.git'])

os.chdir('/kaggle/working/Clutch-AI')

!python notebooks/train.py
```

### Training Configuration
| Parameter | Value |
|---|---|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | SFT + 4-bit QLoRA |
| LoRA Rank | 16 |
| Batch Size | 2 × 4 = 8 effective |
| Learning Rate | 2e-4 (cosine schedule) |
| Max Seq Length | 1024 tokens |
| Total Examples | ~249,031 |
| Total Steps | ~31,128 |
| Checkpoints | Every 500 steps (auto-resume) |

### Training Datasets

| # | Dataset | Examples | Purpose |
|---|---|---|---|
| 1 | Custom Examples (×10) | 540 | Identity & anti-hallucination |
| 2 | SlimOrca | 50,000 | General knowledge |
| 3 | OpenHermes 2.5 | 50,000 | Diverse instruction following |
| 4 | UltraChat 200K | 30,000 | Multi-turn conversations |
| 5 | Open-Platypus | 24,926 | Logical reasoning |
| 6 | Orca-Math | 20,000 | Mathematical problem solving |
| 7 | MetaMathQA | 20,000 | Advanced math |
| 8 | MagicCoder OSS | 20,000 | Code generation |
| 9 | Code-Alpaca | 20,022 | Code instruction following |
| 10 | ScienceQA | 12,726 | Science Q&A |
| 11 | TruthfulQA | 817 | Truthfulness & accuracy |

---

## 📁 Project Structure

```
Clutch-AI/
├── config/
│   ├── model_config.json        # All prompts, parameters, and settings (SINGLE SOURCE OF TRUTH)
│   └── custom_examples.json     # Custom training examples (identity & anti-hallucination)
├── src/
│   └── query_router.py          # LLM-powered SEARCH/SKIP classifier
├── scripts/
│   ├── inference_lora.py        # Main inference engine (LoRA adapter, Perplexity-style)
│   ├── inference.py             # Merged model inference
│   └── evaluate_model.py        # Model evaluation utility
├── notebooks/
│   └── train.py                 # Kaggle training script
├── .env                         # API keys (TAVILY_API_KEY)
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.safetensors    # Trained LoRA weights
└── requirements.txt             # Python dependencies
```

---

## 🔧 Configuration

All behavior is controlled via `config/model_config.json`:

```jsonc
{
  "name": "Clutch-AI",
  "version": "1.0.0",
  "creator": "Clutch Group",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",

  // Identity and behavior prompts
  "system_prompt": "...",
  "behavior_prompt": "...",
  "web_context_prompt": "...",
  "router_prompt": "...",

  // Generation parameters
  "generation": {
    "max_new_tokens": 300,
    "temperature": 0.6,
    "top_k": 30,
    "top_p": 0.8,
    "repetition_penalty": 1.1
  },

  // Web search configuration
  "web_search": {
    "max_results": 3,
    "search_depth": "basic",
    "max_content_length": 500
  },

  // Conversation memory
  "memory": { "max_turns": 4 }
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | Meta Llama 3.2 3B Instruct |
| Fine-Tuning | QLoRA (4-bit NF4) via PEFT + TRL |
| Quantization | BitsAndBytes 4-bit |
| Web Search | Tavily API |
| Query Routing | LLM-powered intent classification |
| Date/Time | Python `datetime` with UTC timezone support |
| Config | JSON-driven (zero hardcoded logic) |
| Framework | PyTorch + HuggingFace Transformers |
| Training Hardware | Kaggle P100/T4 (16GB VRAM) |
| Inference Hardware | Any NVIDIA GPU with 4GB+ VRAM |

---

## 📊 Model Specs

| Specification | Detail |
|---|---|
| Architecture | Llama 3.2 (Transformer, RoPE, GQA) |
| Parameters | 3B total · 24.3M trainable (LoRA) |
| Context Length | 1024 tokens (training) · 2048+ (inference) |
| Training Method | SFT + 4-bit QLoRA |
| LoRA Config | rank=16, alpha=32, dropout=0.05 |
| Training Data | ~249K examples across 11 datasets |
| Quantization | NF4 with double quantization |

---

## 🗺️ Roadmap

- [x] Custom fine-tuning pipeline with QLoRA
- [x] Real-time web search integration (Tavily)
- [x] LLM-powered query routing
- [x] Config-driven architecture (zero hardcoding)
- [x] UTC date/time injection with timezone conversions
- [x] Conversation memory
- [ ] Groq API integration (Llama 4 Scout cloud inference)
- [ ] Upgrade to Llama 3.1 8B for improved quality
- [ ] Streaming token generation
- [ ] Web UI dashboard

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> **Note**: The base model (Llama 3.2) is subject to Meta's [Llama Community License Agreement](https://ai.meta.com/llama/license/). Training datasets may have their own licenses.

---

<p align="center">
  Built with ❤️ by <strong>Clutch Group</strong>
</p>