import os
import random
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

"""
Alpaca SFT dataset prep for Clutch-AI.

Creates 4 files in this folder:
  - train.bin (uint16 tokens)
  - val.bin   (uint16 tokens)
  - train_labels.bin (int32 targets, -1 = ignore)
  - val_labels.bin   (int32 targets, -1 = ignore)

Targets are next-token labels aligned with tokens (same length).
We compute loss only on assistant response tokens by using -1 for prompt tokens.
"""

DATASET_NAME = os.environ.get("ALPACA_DATASET", "tatsu-lab/alpaca")
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.01"))  # 1% validation
RANDOM_SEED = int(os.environ.get("SEED", "1337"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "0"))  # 0 = no limit

THIS_DIR = Path(__file__).resolve().parent


def format_alpaca_prompt(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()

    if inp:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    rng = random.Random(RANDOM_SEED)

    train_tokens = []
    train_labels = []
    val_tokens = []
    val_labels = []

    def add_example(target_tokens, target_labels, prompt: str, output: str):
        prompt_ids = enc.encode_ordinary(prompt)
        out_ids = enc.encode_ordinary((output or "").strip())

        full = prompt_ids + out_ids + [eot]

        L = len(full)
        labels = [-1] * L
        response_start = len(prompt_ids)

        for i in range(L - 1):
            next_pos = i + 1
            if next_pos >= response_start:
                labels[i] = full[next_pos]
        labels[-1] = -1

        target_tokens.extend(full)
        target_labels.extend(labels)

    total_tok = 0

    for ex in ds:
        prompt = format_alpaca_prompt(ex.get("instruction", ""), ex.get("input", ""))
        output = ex.get("output", "")

        to_val = (rng.random() < VAL_FRACTION)
        if to_val:
            add_example(val_tokens, val_labels, prompt, output)
        else:
            add_example(train_tokens, train_labels, prompt, output)

        total_tok = len(train_tokens) + len(val_tokens)
        if MAX_TOKENS and total_tok >= MAX_TOKENS:
            print(f"Reached MAX_TOKENS={MAX_TOKENS:,}. Stopping early.")
            break

    def write_bin(path: Path, arr, dtype):
        path.write_bytes(np.array(arr, dtype=dtype).tobytes())

    train_bin = THIS_DIR / "train.bin"
    val_bin = THIS_DIR / "val.bin"
    train_lbl = THIS_DIR / "train_labels.bin"
    val_lbl = THIS_DIR / "val_labels.bin"

    print("Writing files...")
    write_bin(train_bin, train_tokens, np.uint16)
    write_bin(val_bin, val_tokens, np.uint16)
    write_bin(train_lbl, train_labels, np.int32)
    write_bin(val_lbl, val_labels, np.int32)

    def gb(path: Path) -> float:
        return path.stat().st_size / (1024 ** 3)

    print("Done.")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")
    print(f"train.bin:        {gb(train_bin):.3f} GB")
    print(f"train_labels.bin: {gb(train_lbl):.3f} GB")
    print(f"val.bin:          {gb(val_bin):.3f} GB")
    print(f"val_labels.bin:   {gb(val_lbl):.3f} GB")


if __name__ == "__main__":
    main()
