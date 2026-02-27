"""
Alpaca-GPT4 Dataset Preparation for Clutch-AI
===============================================
Downloads vicgalle/alpaca-gpt4 from Hugging Face and converts it to
train.bin / train_labels.bin / val.bin / val_labels.bin

All identity data and system prompt are loaded from config files:
  - config/model_config.json    (name, creator, system prompt template)
  - config/custom_examples.json (identity, math, reasoning examples)

Label masking: prompt tokens are set to -1 so loss is only computed
on the response. This is standard SFT practice.

Usage:
    python data/prepare_alpaca_gpt4.py

Output (written to data/alpaca_gpt4/):
    train.bin         - tokenized inputs (uint16)
    train_labels.bin  - masked labels: -1 for prompt, token_id for response (int32)
    val.bin           - validation inputs
    val_labels.bin    - validation labels
"""

import os
import json
import random
import numpy as np
import tiktoken
from pathlib import Path
from datasets import load_dataset

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
CONFIG_DIR  = REPO_ROOT / "config"
OUT_DIR     = Path(__file__).resolve().parent / "alpaca_gpt4"
VAL_RATIO   = 0.05
SEED        = 42
REPEAT      = 20  # how many times to repeat custom examples in training set
# ─────────────────────────────────────────────────────────────────────────


def load_model_config() -> dict:
    """Load model identity and system prompt from config/model_config.json."""
    config_path = CONFIG_DIR / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Create config/model_config.json with name, creator, and system_prompt fields."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Resolve template variables in system_prompt
    cfg["system_prompt_resolved"] = cfg["system_prompt"].format(
        name=cfg["name"],
        creator=cfg["creator"],
    )
    return cfg


def load_custom_examples(cfg: dict) -> list[dict]:
    """Load custom training examples from config/custom_examples.json and resolve template vars."""
    examples_path = CONFIG_DIR / "custom_examples.json"
    if not examples_path.exists():
        print(f"  [warning] No custom examples found at {examples_path}, skipping.")
        return []

    with open(examples_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Resolve {name} and {creator} template variables
    resolved = []
    for ex in raw:
        resolved.append({
            "instruction": ex["instruction"].format(name=cfg["name"], creator=cfg["creator"]),
            "input": ex.get("input", ""),
            "output": ex["output"].format(name=cfg["name"], creator=cfg["creator"]),
        })
    return resolved


# ── Tokenization ─────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def format_prompt(example: dict, system_prompt: str) -> tuple[str, str]:
    """
    Returns (prompt, response) strings in Alpaca instruction format.
    Includes a ### System: block so the model learns its identity.
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if inp:
        prompt = (
            f"### System:\n{system_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### System:\n{system_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    return prompt, output


def tokenize_example(example: dict, system_prompt: str):
    """
    Tokenizes one example. Returns (input_ids, label_ids).
    label_ids has -1 for prompt tokens and the real token id for response tokens.
    """
    prompt, response = format_prompt(example, system_prompt)

    prompt_ids   = enc.encode(prompt)
    response_ids = enc.encode(response) + [EOT]

    input_ids  = prompt_ids + response_ids
    label_ids  = [-1] * len(prompt_ids) + response_ids   # mask prompt

    return input_ids, label_ids


def write_split(examples, split_name: str, system_prompt: str):
    inputs_path = os.path.join(OUT_DIR, f"{split_name}.bin")
    labels_path = os.path.join(OUT_DIR, f"{split_name}_labels.bin")

    all_inputs = []
    all_labels = []

    for ex in examples:
        try:
            ids, lids = tokenize_example(ex, system_prompt)
            all_inputs.extend(ids)
            all_labels.extend(lids)
        except Exception:
            continue   # skip malformed examples

    np.array(all_inputs, dtype=np.uint16).tofile(inputs_path)
    np.array(all_labels, dtype=np.int32).tofile(labels_path)

    print(f"  {split_name}: {len(all_inputs):,} tokens -> {inputs_path}")


def main():
    # ── Load config ──
    print("Loading config...")
    cfg = load_model_config()
    system_prompt = cfg["system_prompt_resolved"]
    print(f"  Model: {cfg['name']}")
    print(f"  Creator: {cfg['creator']}")
    print(f"  System prompt: {system_prompt[:80]}...")

    # ── Load custom examples ──
    custom = load_custom_examples(cfg)
    print(f"  Custom examples: {len(custom)} loaded from config/custom_examples.json")

    # ── Download dataset ──
    print("\nDownloading vicgalle/alpaca-gpt4 from Hugging Face...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    examples = list(ds)
    print(f"  Dataset examples: {len(examples):,}")

    # ── Augment with custom examples ──
    custom_augmented = custom * REPEAT
    examples.extend(custom_augmented)
    print(f"  Added {len(custom_augmented)} custom examples ({len(custom)} unique x {REPEAT} repeats)")
    print(f"  Total examples: {len(examples):,}")

    # ── Shuffle and split ──
    random.seed(SEED)
    random.shuffle(examples)

    val_size   = int(len(examples) * VAL_RATIO)
    val_data   = examples[:val_size]
    train_data = examples[val_size:]

    print(f"\n  Train: {len(train_data):,}  |  Val: {len(val_data):,}")
    print("  Tokenizing and writing...")

    write_split(train_data, "train", system_prompt)
    write_split(val_data,   "val",   system_prompt)

    print(f"\nDone! Data saved to: {OUT_DIR}")
    print("\nNext step — start training:")
    print("  python train.py config/train_alpaca_sft.py")


if __name__ == "__main__":
    main()
