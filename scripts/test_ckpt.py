import sys
from pathlib import Path
import torch
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clutch_ai.models.gpt import GPT, GPTConfig

CKPT_PATH = REPO_ROOT / "out-clutch-sft-alpaca" / "ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# make results repeatable
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

print("Checkpoint:", CKPT_PATH)
ckpt = torch.load(CKPT_PATH, map_location=device)

model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print("Loaded! iter_num =", ckpt.get("iter_num", "unknown"))

enc = tiktoken.get_encoding("gpt2")

instruction = "Explain the theory of relativity in simple terms."
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    out = model.generate(
        idx,
        max_new_tokens=200,
        temperature=0.7,   # a bit higher so it actually answers
        top_k=200,
        stop_idx=enc.eot_token
    )

full_text = enc.decode(out[0].tolist())

# ✅ print ONLY the new generated part (after the prompt)
generated = full_text[len(prompt):]
generated = generated.split("<|endoftext|>")[0].strip()

print("\n--- GENERATED TEXT ---")
print(prompt + generated)
