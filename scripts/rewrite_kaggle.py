with open("notebooks/train_llama_kaggle.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Dependency replacement
old_deps = """def install_deps():
    print("Installing dependencies...")
    # Force uninstall any cached broken versions
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "unsloth", "unsloth-zoo"])
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "packaging", "ninja", "einops", "flash-attn", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth", "unsloth-zoo"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "sentencepiece", "protobuf"])"""

new_deps = """def install_deps():
    print("Installing dependencies...")
    # Unsloth is removed to ensure compatibility with older GPUs like P100
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "peft", "bitsandbytes", "trl", "accelerate", "transformers", "sentencepiece", "protobuf"])"""
text = text.replace(old_deps, new_deps)

# 2. GPU Warning removal
old_gpu = """if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    raise RuntimeError("❌ No GPU found! Enable GPU in Settings → Accelerator → GPU T4 x2")"""

new_gpu = """if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    raise RuntimeError("❌ No GPU found! Enable GPU in Settings")"""
text = text.replace(old_gpu, new_gpu)


# 3. Model Loading
old_load = """from unsloth import FastLanguageModel

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024  # Reduced from 2048 to prevent CUDA OOM
DTYPE = None  # auto-detect (float16 for T4)
LOAD_IN_4BIT = True

print(f"📥 Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    token=hf_token,
)
print(f"✅ Model loaded! Parameters: {model.num_parameters() / 1e6:.1f}M")

# ════════════════════════════════════════════════════════════════
# CELL 5: Apply LoRA Adapters
# ════════════════════════════════════════════════════════════════
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank (higher = more capacity)
    target_modules=[               # Which layers to fine-tune
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,                # Unsloth optimized: 0 dropout
    bias="none",
    use_gradient_checkpointing="unsloth",  # 60% less VRAM
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())"""

new_load = """from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024

print(f"📥 Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)
model = prepare_model_for_kbit_training(model)
print(f"✅ Model loaded!")

# ════════════════════════════════════════════════════════════════
# CELL 5: Apply LoRA Adapters
# ════════════════════════════════════════════════════════════════
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())"""
text = text.replace(old_load, new_load)


# 4. Formatting Func
old_fmt = """from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1", # Llama 3.2 uses the same template
)

def formatting_func(examples):"""
new_fmt = """def formatting_func(examples):"""
text = text.replace(old_fmt, new_fmt)

# 5. Trainer Setup
old_trainer = """from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

OUTPUT_DIR = "/kaggle/working/Clutch-AI/out-clutch-1.0.0"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Training duration ──
    num_train_epochs=1,                    # 1 full pass over 2.4M examples
    max_steps=-1,                          # -1 = use num_train_epochs

    # ── Batch size ──
    per_device_train_batch_size=1,         # Reduced to 1 to prevent CUDA OOM
    gradient_accumulation_steps=8,         # Effective batch = 1 * 8 = 8

    # ── Learning rate ──
    learning_rate=2e-4,                    # Standard for QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # ── Precision ──
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    # ── Logging ──
    logging_steps=25,
    logging_first_step=True,

    # ── Checkpointing (for resume across Kaggle sessions) ──
    save_strategy="steps",
    save_steps=500,                        # Save every 500 steps
    save_total_limit=3,                    # Keep last 3 checkpoints

    # ── Optimization ──
    optim="adamw_8bit",                    # Memory-efficient optimizer
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,

    # ── Performance ──
    dataloader_num_workers=2,
    group_by_length=True,                  # Faster training
    report_to="none",                      # No wandb/tensorboard
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=training_args,
)"""

new_trainer = """from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

OUTPUT_DIR = "/kaggle/working/Clutch-AI/out-clutch-1.0.0"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True, # Ensure P100 compatibility
    bf16=False,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    dataloader_num_workers=2,
    group_by_length=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)"""
text = text.replace(old_trainer, new_trainer)

# 6. Model Saving
old_save = """print("💾 Saving merged full model (for easy inference)...")
model.save_pretrained_merged(
    f"{SAVE_DIR}/merged",
    tokenizer,
    save_method="merged_16bit",
)

print(f"✅ Model saved to {SAVE_DIR}")

# Also save to the Clutch-AI output dir
import shutil
merged_ckpt_dir = "/kaggle/working/Clutch-AI/out-clutch-1.0.0-final"
if os.path.exists(merged_ckpt_dir):
    shutil.rmtree(merged_ckpt_dir)
shutil.copytree(f"{SAVE_DIR}/merged", merged_ckpt_dir)
print(f"✅ Also copied to {merged_ckpt_dir}")"""

new_save = """# We skip merging here because we aren't using Unsloth.
# Users can merge locally if they wish using PEFT.
print(f"✅ Model LoRA saved to {SAVE_DIR}/lora-adapter")

import shutil
merged_ckpt_dir = "/kaggle/working/Clutch-AI/out-clutch-1.0.0-final"
if os.path.exists(merged_ckpt_dir):
    shutil.rmtree(merged_ckpt_dir)
shutil.copytree(f"{SAVE_DIR}/lora-adapter", merged_ckpt_dir)
print(f"✅ Also copied to {merged_ckpt_dir}")"""
text = text.replace(old_save, new_save)

# 7. Remove FastLanguageModel.for_inference
text = text.replace("FastLanguageModel.for_inference(model)", "model.eval()")

with open("notebooks/train_llama_kaggle.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Done rewriting.")
