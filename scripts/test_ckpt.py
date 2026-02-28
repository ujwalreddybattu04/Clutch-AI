"""
Quick sanity check for the Clutch-AI checkpoint.

Usage:
    python scripts/test_ckpt.py
    python scripts/test_ckpt.py --ckpt path/to/merged_model
"""
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Load config ──
config_path = REPO_ROOT / "config" / "llama_3b_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = json.load(f)

SYSTEM_PROMPT = CFG["system_prompt"].format(name=CFG["name"], creator=CFG["creator"])
DEFAULT_CKPT  = REPO_ROOT / CFG["default_checkpoint"] / "merged"

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

    print(f"Model: {CFG.get('name', 'Clutch-AI')} by {CFG.get('creator', 'Clutch Group')} (v{CFG.get('version', '1.0.0')})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    if not ckpt_path.exists():
        print(f"\n❌ Error: Model not found at {ckpt_path}. Did you extract the Kaggle output?")
        sys.exit(1)

    print("\nLoading tokenizer and model (this might take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)
    model.eval()

    for i, instruction in enumerate(TEST_PROMPTS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

        print(f"\n{'='*60}")
        print(f"Test {i}: {instruction}")
        print(f"{'='*60}")
        print(f"{CFG['name']}: {response.strip()}")

    print(f"\n{'='*60}")
    print("All tests complete.")

if __name__ == "__main__":
    main()
