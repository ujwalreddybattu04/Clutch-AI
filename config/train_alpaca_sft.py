"""
Instruction tuning config (Alpaca) for Clutch-AI.

Run:
  python train.py config/train_alpaca_sft.py
"""

# Save SFT checkpoints here
out_dir = "out-clutch-sft-alpaca"

# Load weights from your base pretrain checkpoint folder
init_from = "resume"
init_from_dir = "out-clutch-0.1"   # <-- base run folder (must contain ckpt.pt)
reset_optimizer = True             # fresh optimizer + fresh schedule

# dataset folder under ./data/
dataset = "alpaca_sft"

# SFT setup (lighter than pretraining)
batch_size = 4
block_size = 512
gradient_accumulation_steps = 16

# Keep model same (checkpoint decides architecture anyway)
n_layer = 12
n_head = 12
n_embd = 768
bias = False
dropout = 0.0

# Train steps (SFT is small dataset)
max_iters = 3000

# eval + logging
eval_interval = 200
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# Smaller LR for SFT
learning_rate = 5e-5
warmup_iters = 100
decay_lr = False
min_lr = 5e-5
lr_decay_iters = max_iters

device = "cuda"
dtype = "float16"
compile = False
