"""
Quick sanity check for a Clutch-AI checkpoint.
Loads config from config/model_config.json, then tests the checkpoint.

Usage:
    python scripts/test_ckpt.py
    python scripts/test_ckpt.py --ckpt path/to/ckpt.pt
"""
import sys
import json
import argparse
from pathlib import Path

import torch
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clutch_ai.models.gpt import GPT, GPTConfig

# ── Load config (no hardcoding) ──
config_path = REPO_ROOT / "config" / "legacy_gpt2" / "model_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = json.load(f)

SYSTEM_PROMPT = CFG["system_prompt"].format(name=CFG["name"], creator=CFG["creator"])
DEFAULT_CKPT  = REPO_ROOT / CFG["default_checkpoint"] / "ckpt.pt"

TEST_PROMPTS = [
    "Who are you?",
    "What is 300 + 300?",
    "Explain machine learning in simple terms.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    print(f"Model: {CFG['name']} by {CFG['creator']}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"Loaded! iter_num={ckpt.get('iter_num', '?')}")

    enc = tiktoken.get_encoding(CFG.get("tokenizer", "gpt2"))
    eot = enc.eot_token

    for i, instruction in enumerate(TEST_PROMPTS, 1):
        prompt = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
        idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=256,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                stop_idx=eot,
            )

        full_text = enc.decode(out[0].tolist())
        generated = full_text[len(prompt):]
        generated = generated.split(enc.decode([eot]))[0].strip()

        print(f"\n{'='*60}")
        print(f"Test {i}: {instruction}")
        print(f"{'='*60}")
        print(f"{CFG['name']}: {generated}")

    print(f"\n{'='*60}")
    print("All tests complete.")


if __name__ == "__main__":
    main()
