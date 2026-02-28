"""
Instruction tuning config (Alpaca) for Clutch-AI.

Run:
  python train.py config/train_alpaca_sft.py
"""

# Save SFT checkpoints here (new folder for GPT-2 based run)
out_dir = "out-clutch-gpt2-alpacagpt4"

# Start from real GPT-2 weights (trained by OpenAI on 40B tokens!)
# Downloads automatically the first time.
init_from = "gpt2-medium"  # 350M params — better quality, small batch needed for 6GB GPU

# dataset folder under ./data/
dataset = "alpaca_gpt4"   # GPT-4 quality answers — better than standard Alpaca

# Small batch to fit in 6GB GPU VRAM
batch_size = 1
block_size = 512
gradient_accumulation_steps = 64  # compensates for small batch, same effective size

# Keep model same (checkpoint decides architecture anyway)
n_layer = 12
n_head = 12
n_embd = 768
bias = False
dropout = 0.0

# Train steps — 5000 gives better quality for SFT (was 2000)
max_iters = 5000

# eval + logging
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

# LR schedule — cosine decay for smoother convergence
learning_rate = 5e-5
warmup_iters = 200
decay_lr = True
min_lr = 5e-6
lr_decay_iters = max_iters

device = "cuda"
dtype = "float16"
compile = False

# Early stopping — stop if val loss doesn't improve for 5 evals
early_stop_patience = 5
