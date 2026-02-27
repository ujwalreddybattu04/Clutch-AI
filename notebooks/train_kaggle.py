"""
Clutch-AI Training on Kaggle
==============================
Upload this as a Kaggle notebook with GPU accelerator enabled.

Steps:
1. Go to kaggle.com -> New Notebook
2. Copy-paste this entire script
3. Settings -> Accelerator -> GPU T4 x2 (or P100)
4. Click "Run All"
5. After training, download the checkpoint from the output

The notebook will:
- Clone your GitHub repo
- Install dependencies
- Prepare the training data
- Train the model (5000 iterations)
- Save the checkpoint for download
"""

# ============================================================
# Step 1: Clone your repo
# ============================================================
import os
os.chdir('/kaggle/working')

!git clone https://github.com/ujwalreddybattu04/Clutch-AI.git
os.chdir('/kaggle/working/Clutch-AI')

# ============================================================
# Step 2: Install dependencies
# ============================================================
!pip install -q tiktoken datasets transformers

# ============================================================
# Step 3: Verify GPU
# ============================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    raise RuntimeError("No GPU found! Enable GPU in Kaggle: Settings -> Accelerator -> GPU")

# ============================================================
# Step 4: Prepare training data
# ============================================================
!python data/prepare_alpaca_gpt4.py

# ============================================================
# Step 5: Train!
# ============================================================
# Kaggle T4 has 16GB VRAM — we can use larger batch size for faster training
# Override batch_size and gradient_accumulation for Kaggle GPU
!python train.py config/train_alpaca_sft.py \
    --batch_size=4 \
    --gradient_accumulation_steps=2 \
    --device=cuda \
    --dtype=float16

# ============================================================
# Step 6: Test the trained model
# ============================================================
print("\n" + "="*60)
print("Training complete! Testing the model...")
print("="*60 + "\n")

!python scripts/test_ckpt.py

# ============================================================
# Step 7: Package checkpoint for download
# ============================================================
import shutil

ckpt_dir = '/kaggle/working/Clutch-AI/out-clutch-gpt2-alpacagpt4'
output_zip = '/kaggle/working/clutch-ai-checkpoint.zip'

if os.path.exists(ckpt_dir):
    shutil.make_archive('/kaggle/working/clutch-ai-checkpoint', 'zip', ckpt_dir)
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Checkpoint saved: {output_zip} ({size_mb:.1f} MB)")
    print("Download it from: Kaggle Output -> clutch-ai-checkpoint.zip")
    print("Then place ckpt.pt in your local out-clutch-gpt2-alpacagpt4/ folder")
else:
    print("❌ Checkpoint directory not found!")
