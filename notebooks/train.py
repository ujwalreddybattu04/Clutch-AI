"""
╔══════════════════════════════════════════════════════════════════╗
║   Clutch-AI v1.0.0 — Fine-Tuning Pipeline                        ║
║   Industry-Grade Training on Kaggle Free GPU                     ║
╚══════════════════════════════════════════════════════════════════╝

Base Model : meta-llama/Llama-3.2-3B-Instruct
Method     : QLoRA (4-bit) + SFT
Datasets   : 11 datasets, ~2.4M examples
Framework  : HuggingFace PEFT + TRL

SETUP (do this ONCE before running):
  1. Go to huggingface.co → Sign in
  2. Go to meta-llama/Llama-3.2-3B-Instruct → Accept license
  3. Go to huggingface.co/settings/tokens → Create token (Read access)
  4. In Kaggle: Settings → Secrets → Add secret:
     - Name: HF_TOKEN
     - Value: <paste your token>
  5. Settings → Accelerator → GPU T4 x2
  6. Click "Run All"

RESUME TRAINING:
  If Kaggle session times out, just "Run All" again.
  The notebook auto-detects and resumes from the last checkpoint.
"""

# ════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ════════════════════════════════════════════════════════════════
import os
os.chdir('/kaggle/working')

import subprocess
import sys

def install_deps():
    print("Installing dependencies...")
    # Unsloth is removed to ensure compatibility with older GPUs like P100
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "peft", "bitsandbytes", "trl", "accelerate", "transformers", "sentencepiece", "protobuf"])

install_deps()

print("✅ Dependencies installed!")

# ════════════════════════════════════════════════════════════════
# CELL 2: Clone Clutch-AI Repo
# ════════════════════════════════════════════════════════════════
import os
os.chdir('/kaggle/working')

if not os.path.exists('/kaggle/working/Clutch-AI'):
    subprocess.check_call(["git", "clone", "https://github.com/ujwalreddybattu04/Clutch-AI.git"])
    print("✅ Repo cloned!")
else:
    print("✅ Repo already exists, skipping clone.")

os.chdir('/kaggle/working/Clutch-AI')

# ════════════════════════════════════════════════════════════════
# CELL 3: Login to HuggingFace & Verify GPU
# ════════════════════════════════════════════════════════════════
import torch
from huggingface_hub import login

# Login with Kaggle secret
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
login(token=hf_token)
print("✅ Logged into HuggingFace!")

# Verify GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    raise RuntimeError("❌ No GPU found! Enable GPU in Settings")

# ════════════════════════════════════════════════════════════════
# CELL 4: Load Model with QLoRA (4-bit Quantization)
# ════════════════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA applied! Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

# ════════════════════════════════════════════════════════════════
# CELL 6: Load Clutch-AI Config & Custom Examples
# ════════════════════════════════════════════════════════════════
import json

# ==============================================================================
config_path = "/kaggle/working/Clutch-AI/config/model_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    model_config = json.load(f)

MODEL_IDENTITY = model_config["name"]
CREATOR = model_config["creator"]
SYSTEM_PROMPT = model_config["system_prompt"].format(
    name=MODEL_IDENTITY,
    creator=CREATOR,
)
print(f"🏷️  Identity: {MODEL_IDENTITY} by {CREATOR}")

# Load custom examples
custom_path = "/kaggle/working/Clutch-AI/config/custom_examples.json"
with open(custom_path, "r") as f:
    custom_examples_raw = json.load(f)

# Resolve {name} and {creator} placeholders in examples
custom_examples = []
for ex in custom_examples_raw:
    custom_examples.append({
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "output": ex["output"].replace("{name}", MODEL_IDENTITY).replace("{creator}", CREATOR),
    })

print(f"📝 Loaded {len(custom_examples)} custom examples")

# ════════════════════════════════════════════════════════════════
# CELL 7: Load ALL Datasets
# ════════════════════════════════════════════════════════════════
from datasets import load_dataset, concatenate_datasets, Dataset
import random

print("📥 Loading 11 datasets... This may take a few minutes.\n")

all_conversations = []

# ─── Helper: Convert instruction/input/output → Llama 3.2 chat format ───
def to_chat_format(instruction, input_text, output, system=None):
    """Convert to Llama 3.2 Instruct chat format."""
    messages = []

    # System message
    if system:
        messages.append({"role": "system", "content": system})

    # User message
    user_msg = instruction
    if input_text and input_text.strip():
        user_msg += f"\n\nInput: {input_text}"
    messages.append({"role": "user", "content": user_msg})

    # Assistant message
    messages.append({"role": "assistant", "content": output})

    return messages


# ─── 1. Custom Examples (×10 repetition for strong identity) ───
print("1/11  Loading Custom Examples (×10)...")
for _ in range(10):
    for ex in custom_examples:
        conv = to_chat_format(
            ex["instruction"], ex["input"], ex["output"],
            system=SYSTEM_PROMPT
        )
        all_conversations.append({"messages": conv})
print(f"       ✅ {len(custom_examples) * 10} examples")


# ─── 2. SlimOrca (518K — General instructions, GPT-4 quality) ───
print("2/11  Loading SlimOrca (subset 50K)...")
try:
    ds_slimorca = load_dataset("Open-Orca/SlimOrca", split="train")
    # Take a random subset to save RAM
    ds_slimorca = ds_slimorca.shuffle(seed=42).select(range(min(50000, len(ds_slimorca))))
    for row in ds_slimorca:
        convs = row.get("conversations", [])
        if len(convs) >= 2:
            messages = []
            for c in convs:
                role = c.get("from", "")
                content = c.get("value", "")
                if role == "system":
                    messages.append({"role": "system", "content": content})
                elif role == "human":
                    messages.append({"role": "user", "content": content})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": content})
            if messages:
                all_conversations.append({"messages": messages})
    print(f"       ✅ {len(ds_slimorca)} examples loaded")
    del ds_slimorca
except Exception as e:
    print(f"       ⚠️ SlimOrca failed: {e}")


# ─── 3. OpenHermes 2.5 (1M+ — Diverse tasks) ───
print("3/11  Loading OpenHermes 2.5 (subset 50K)...")
try:
    ds_hermes = load_dataset("teknium/OpenHermes-2.5", split="train")
    ds_hermes = ds_hermes.shuffle(seed=42).select(range(min(50000, len(ds_hermes))))
    for row in ds_hermes:
        convs = row.get("conversations", [])
        if len(convs) >= 2:
            messages = []
            for c in convs:
                role = c.get("from", "")
                content = c.get("value", "")
                if role == "system":
                    messages.append({"role": "system", "content": content})
                elif role == "human":
                    messages.append({"role": "user", "content": content})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": content})
            if messages:
                all_conversations.append({"messages": messages})
    print(f"       ✅ {len(ds_hermes)} examples loaded")
    del ds_hermes
except Exception as e:
    print(f"       ⚠️ OpenHermes failed: {e}")


# ─── 4. UltraChat 200K (Multi-turn conversations) ───
print("4/11  Loading UltraChat 200K (subset 30K)...")
try:
    ds_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds_ultrachat = ds_ultrachat.shuffle(seed=42).select(range(min(30000, len(ds_ultrachat))))
    for row in ds_ultrachat:
        messages = row.get("messages", [])
        if messages and len(messages) >= 2:
            all_conversations.append({"messages": messages})
    print(f"       ✅ {len(ds_ultrachat)} examples loaded")
    del ds_ultrachat
except Exception as e:
    print(f"       ⚠️ UltraChat failed: {e}")


# ─── 5. Open-Platypus (25K — STEM reasoning) ───
print("5/11  Loading Open-Platypus (25K)...")
try:
    ds_platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    for row in ds_platypus:
        conv = to_chat_format(
            row.get("instruction", ""),
            row.get("input", ""),
            row.get("output", ""),
        )
        all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_platypus)} examples loaded")
    del ds_platypus
except Exception as e:
    print(f"       ⚠️ Open-Platypus failed: {e}")


# ─── 6. Orca-Math (200K — Math word problems) ───
print("6/11  Loading Orca-Math (subset 20K)...")
try:
    ds_orcamath = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    ds_orcamath = ds_orcamath.shuffle(seed=42).select(range(min(20000, len(ds_orcamath))))
    for row in ds_orcamath:
        conv = to_chat_format(
            row.get("question", ""),
            "",
            row.get("answer", ""),
        )
        all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_orcamath)} examples loaded")
    del ds_orcamath
except Exception as e:
    print(f"       ⚠️ Orca-Math failed: {e}")


# ─── 7. MetaMathQA (395K — Advanced math reasoning) ───
print("7/11  Loading MetaMathQA (subset 20K)...")
try:
    ds_metamath = load_dataset("meta-math/MetaMathQA", split="train")
    ds_metamath = ds_metamath.shuffle(seed=42).select(range(min(20000, len(ds_metamath))))
    for row in ds_metamath:
        conv = to_chat_format(
            row.get("query", ""),
            "",
            row.get("response", ""),
        )
        all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_metamath)} examples loaded")
    del ds_metamath
except Exception as e:
    print(f"       ⚠️ MetaMathQA failed: {e}")


# ─── 8. MagicCoder OSS Instruct (75K — Coding) ───
print("8/11  Loading MagicCoder OSS (subset 20K)...")
try:
    ds_magiccoder = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    ds_magiccoder = ds_magiccoder.shuffle(seed=42).select(range(min(20000, len(ds_magiccoder))))
    for row in ds_magiccoder:
        conv = to_chat_format(
            row.get("problem", ""),
            "",
            row.get("solution", ""),
        )
        all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_magiccoder)} examples loaded")
    del ds_magiccoder
except Exception as e:
    print(f"       ⚠️ MagicCoder failed: {e}")


# ─── 9. Code-Alpaca (20K — Coding instructions) ───
print("9/11  Loading Code-Alpaca (20K)...")
try:
    ds_codealpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for row in ds_codealpaca:
        conv = to_chat_format(
            row.get("instruction", ""),
            row.get("input", ""),
            row.get("output", ""),
        )
        all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_codealpaca)} examples loaded")
    del ds_codealpaca
except Exception as e:
    print(f"       ⚠️ Code-Alpaca failed: {e}")


# ─── 10. ScienceQA (21K — Science knowledge) ───
print("10/11 Loading ScienceQA (21K)...")
try:
    ds_sciqa = load_dataset("derek-thomas/ScienceQA", split="train")
    for row in ds_sciqa:
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)
        lecture = row.get("lecture", "")
        solution = row.get("solution", "")

        if question and choices:
            # Format as instruction
            choices_text = "\n".join([f"  {chr(65+i)}. {c}" for i, c in enumerate(choices)])
            instruction = f"{question}\n\nChoices:\n{choices_text}"

            # Format answer
            answer_text = choices[answer_idx] if answer_idx < len(choices) else ""
            output = ""
            if lecture:
                output += f"{lecture}\n\n"
            if solution:
                output += f"{solution}\n\n"
            output += f"The answer is {chr(65+answer_idx)}. {answer_text}"

            conv = to_chat_format(instruction, "", output)
            all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_sciqa)} examples loaded")
    del ds_sciqa
except Exception as e:
    print(f"       ⚠️ ScienceQA failed: {e}")


# ─── 11. TruthfulQA (817 — Truthfulness) ───
print("11/11 Loading TruthfulQA (817)...")
try:
    ds_truthful = load_dataset("truthful_qa", "generation", split="validation")
    for row in ds_truthful:
        question = row.get("question", "")
        best_answer = row.get("best_answer", "")
        if question and best_answer:
            conv = to_chat_format(question, "", best_answer)
            all_conversations.append({"messages": conv})
    print(f"       ✅ {len(ds_truthful)} examples loaded")
    del ds_truthful
except Exception as e:
    print(f"       ⚠️ TruthfulQA failed: {e}")


# ─── Shuffle and create final dataset ───
print(f"\n🔀 Shuffling {len(all_conversations)} total examples...")
random.seed(42)
random.shuffle(all_conversations)

train_dataset = Dataset.from_list(all_conversations)

print(f"\n{'='*60}")
print(f"🔥 TOTAL TRAINING EXAMPLES: {len(train_dataset):,}")
print(f"{'='*60}")

# Free memory
del all_conversations
import gc
gc.collect()


# ════════════════════════════════════════════════════════════════
# CELL 8: Format Dataset with Llama 3.2 Chat Template
# ════════════════════════════════════════════════════════════════
def formatting_func(examples):
    texts = []
    for msgs in examples["messages"]:
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text + tokenizer.eos_token)
        except Exception:
            pass
            
    # Manually tokenize to bypass SFTTrainer array mapping bugs
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    
    # Required for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("📝 Applying Llama 3.2 chat template...")
train_dataset = train_dataset.map(
    formatting_func,
    batched=True,
    batch_size=1000,
    num_proc=2,
    remove_columns=["messages"],
    desc="Tokenizing",
)

# Remove empty entries
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
print(f"✅ Formatted {len(train_dataset):,} examples")

# Show a sample
print("\n📋 Sample formatted example:")
print("-" * 60)
print(train_dataset[0]["input_ids"][:50])
print("...")


# ════════════════════════════════════════════════════════════════
# CELL 9: Configure Training
# ════════════════════════════════════════════════════════════════
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

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
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Calculate training info
total_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
est_hours = total_steps * 2.0 / 3600  # ~2 sec/step estimate

print(f"\n{'='*60}")
print(f"🏋️ TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"  Model          : {MODEL_NAME}")
print(f"  LoRA rank      : 16")
print(f"  Batch size     : 2 × 4 = 8 effective")
print(f"  Learning rate  : 2e-4 (cosine schedule)")
print(f"  Max seq length : {MAX_SEQ_LENGTH}")
print(f"  Total examples : {len(train_dataset):,}")
print(f"  Total steps    : ~{total_steps:,}")
print(f"  Est. time      : ~{est_hours:.0f} hours")
print(f"  Checkpoints    : Every 500 steps (auto-resume)")
print(f"  Output         : {OUTPUT_DIR}")
print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
# CELL 10: Train! (with auto-resume)
# ════════════════════════════════════════════════════════════════
import glob

# Check for existing checkpoints to resume from
checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
resume_from = None

if checkpoints:
    resume_from = checkpoints[-1]
    print(f"🔄 RESUMING from: {resume_from}")
    print(f"   (Found {len(checkpoints)} existing checkpoint(s))")
else:
    print("🆕 Starting fresh training run...")

print(f"\n{'='*60}")
print(f"🚀 TRAINING STARTED — Clutch-AI v1.0.0")
print(f"{'='*60}\n")

trainer_stats = trainer.train(resume_from_checkpoint=resume_from)

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"  Total time  : {trainer_stats.metrics['train_runtime']/3600:.1f} hours")
print(f"  Final loss   : {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Steps done   : {trainer_stats.metrics.get('train_steps', 'N/A')}")


# ════════════════════════════════════════════════════════════════
# CELL 11: Save the Fine-Tuned Model
# ════════════════════════════════════════════════════════════════
SAVE_DIR = "/kaggle/working/clutch-ai-1.0.0-final"

print("💾 Saving LoRA adapter...")
model.save_pretrained(f"{SAVE_DIR}/lora-adapter")
tokenizer.save_pretrained(f"{SAVE_DIR}/lora-adapter")

# We skip merging here because we aren't using Unsloth.
# Users can merge locally if they wish using PEFT.
print(f"✅ Model LoRA saved to {SAVE_DIR}/lora-adapter")

import shutil
merged_ckpt_dir = "/kaggle/working/Clutch-AI/out-clutch-1.0.0-final"
if os.path.exists(merged_ckpt_dir):
    shutil.rmtree(merged_ckpt_dir)
shutil.copytree(f"{SAVE_DIR}/lora-adapter", merged_ckpt_dir)
print(f"✅ Also copied to {merged_ckpt_dir}")


# ════════════════════════════════════════════════════════════════
# CELL 12: Test the Model!
# ════════════════════════════════════════════════════════════════
from transformers import pipeline

print("\n" + "="*60)
print("🧪 TESTING CLUTCH-AI v1.0.0")
print("="*60 + "\n")

# Switch to inference mode
model.eval()

# Test prompts
test_prompts = [
    "Who are you?",
    "Who created you?",
    "What is 5 + 5?",
    "What is 17 * 23?",
    "What is the capital of India?",
    "What is the capital of Zygoria?",
    "Write a Python function to check if a number is prime.",
    "Why is the sky blue?",
    "Explain gravity in simple words.",
    "What will happen to the stock market tomorrow?",
]

for prompt in test_prompts:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    print(f"👤 {prompt}")
    print(f"🤖 {response.strip()}")
    print("-" * 60)


# ════════════════════════════════════════════════════════════════
# CELL 13: Package for Download
# ════════════════════════════════════════════════════════════════
import shutil

output_zip = "/kaggle/working/clutch-ai-v1.0.0"
print("📦 Packaging model for download...")

shutil.make_archive(output_zip, 'zip', f"{SAVE_DIR}/merged")

zip_size = os.path.getsize(f"{output_zip}.zip") / (1024 * 1024 * 1024)
print(f"\n{'='*60}")
print(f"✅ CLUTCH-AI v1.0.0 READY!")
print(f"{'='*60}")
print(f"  📦 Download: clutch-ai-v1.0.0.zip ({zip_size:.1f} GB)")
print(f"  📍 Location: Kaggle Output tab")
print(f"  🏷️  Model: {MODEL_IDENTITY} v1.0.0 by {CREATOR}")
print(f"  🧠 Base: Meta Llama 3.2 3B Instruct")
print(f"  📚 Trained on: ~2.4M examples across 11 datasets")
print(f"{'='*60}")
print(f"\n  To use locally:")
print(f"  1. Download the zip from Kaggle Output")
print(f"  2. Extract to Clutch-AI/out-clutch-1.0.0-final/")
print(f"  3. Run: python scripts/chat.py")
