import sys
from pathlib import Path
import torch
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clutch_ai.models.gpt import GPT, GPTConfig

CKPT_PATH = REPO_ROOT / "out-clutch-0.1" / "ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Checkpoint:", CKPT_PATH)
ckpt = torch.load(CKPT_PATH, map_location=device)

model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print("Loaded! iter_num =", ckpt.get("iter_num", "unknown"))

enc = tiktoken.get_encoding("gpt2")
prompt = "The future of AI is"
idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    out = model.generate(
        idx,
        max_new_tokens=200,
        temperature=0.8,
        top_k=200,
        stop_idx=enc.eot_token
    )

print("\n--- GENERATED TEXT ---")
print(enc.decode(out[0].tolist()))
