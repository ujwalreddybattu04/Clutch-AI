"""
Clutch-AI | Pretraining runner (nanoGPT-style, config-driven)

Single GPU (Kaggle):
  python train.py --config configs/train_fineweb_clutch_0_1.py

DDP (multi-GPU):
  torchrun --standalone --nproc_per_node=4 train.py --config configs/train_fineweb_clutch_0_1.py
"""

import os
import time
import math
import pickle
import argparse
import importlib.util
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# -----------------------------
# Config loading
# -----------------------------
def load_py_config(path: str):
    path = str(path)
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_cli_overrides(cfg_dict: dict, unknown_args: list[str]):
    """
    Allow overrides like:
      --batch_size=4 --learning_rate=3e-4 --compile=False
    """
    for a in unknown_args:
        if not a.startswith("--"):
            continue
        a = a[2:]
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        if k not in cfg_dict:
            continue

        # auto-cast
        if v.lower() in ("true", "false"):
            vv = (v.lower() == "true")
        else:
            try:
                if "." in v or "e" in v.lower():
                    vv = float(v)
                else:
                    vv = int(v)
            except Exception:
                vv = v
        cfg_dict[k] = vv


# -----------------------------
# DDP setup
# -----------------------------
def setup_ddp(backend: str, device: str):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        return {
            "ddp": False,
            "master_process": True,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "seed_offset": 0,
        }

    init_process_group(backend=backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    return {
        "ddp": True,
        "master_process": (rank == 0),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "seed_offset": rank,
    }


# -----------------------------
# Data: memmap loader (train.bin / val.bin)
# -----------------------------
def build_memmap_batcher(data_dir: Path, block_size: int, batch_size: int, device: str):
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if not train_bin.exists() or not val_bin.exists():
        return None

    device_type = "cuda" if "cuda" in device else "cpu"

    def get_batch(split: str):
        bin_path = train_bin if split == "train" else val_bin
        # recreate to avoid memmap growth issues on some environments
        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    return get_batch


# -----------------------------
# LR schedule (cosine + warmup)
# -----------------------------
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / max(1, (warmup_iters + 1))
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(out_dir: Path, raw_model, optimizer, model_args, iter_num, best_val_loss, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(ckpt, out_dir / "ckpt.pt")


def load_checkpoint(out_dir: Path, device: str):
    ckpt_path = out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        return None
    return torch.load(ckpt_path, map_location=device)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_fineweb.py")
    args, unknown = parser.parse_known_args()

    # Defaults (safe-ish baseline)
    cfg = {
        # I/O
        "out_dir": "out-clutch-0.1",
        "eval_interval": 500,
        "log_interval": 10,
        "eval_iters": 50,
        "eval_only": False,
        "always_save_checkpoint": True,
        "init_from": "scratch",  # scratch | resume | gpt2*
        # wandb
        "wandb_log": False,
        "wandb_project": "clutch-0.1",
        "wandb_run_name": "clutch-0.1",
        # data
        "dataset": "fineweb",
        "batch_size": 2,
        "block_size": 1024,
        "gradient_accumulation_steps": 8,
        # model
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0,
        "bias": False,
        # optim
        "learning_rate": 6e-4,
        "max_iters": 20000,
        "weight_decay": 1e-1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        # lr decay
        "decay_lr": True,
        "warmup_iters": 2000,
        "lr_decay_iters": 20000,
        "min_lr": 6e-5,
        # system
        "backend": "nccl",
        "device": "cuda",
        "dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        "compile": False,
        # early stopping (optional)
        "early_stop_patience": 0,  # 0 disables
    }

    # Load your config file and override defaults
    user_cfg = load_py_config(args.config)
    for k in list(cfg.keys()):
        if hasattr(user_cfg, k):
            cfg[k] = getattr(user_cfg, k)

    # Allow quick CLI overrides too: --batch_size=1 etc.
    apply_cli_overrides(cfg, unknown)

    # DDP
    ddp_info = setup_ddp(cfg["backend"], cfg["device"])
    ddp = ddp_info["ddp"]
    master_process = ddp_info["master_process"]
    device = ddp_info["device"]
    world_size = ddp_info["world_size"]

    # scale grad accumulation across ranks
    if ddp:
        assert cfg["gradient_accumulation_steps"] % world_size == 0
        cfg["gradient_accumulation_steps"] //= world_size

    # deterministic-ish
    torch.manual_seed(1337 + ddp_info["seed_offset"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # tokens/iter info
    tokens_per_iter = cfg["gradient_accumulation_steps"] * world_size * cfg["batch_size"] * cfg["block_size"]
    if master_process:
        print("---- Clutch-AI train ----")
        print("config:", cfg)
        print(f"tokens per iter: {tokens_per_iter:,}")
        Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)

    # Import model (your project)
    # Expecting: src/clutch_ai/models/gpt.py defines GPTConfig and GPT
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    os.sys.path.insert(0, str(src_dir))
    try:
        from clutch_ai.models.gpt import GPTConfig, GPT
    except Exception as e:
        raise RuntimeError(
            "âŒ Could not import your model.\n"
            "Expected: src/clutch_ai/models/gpt.py with GPTConfig and GPT.\n"
            f"Import error: {e}"
        )

    # Data (expects preprocessed bins)
    data_dir = repo_root / "data" / cfg["dataset"]
    get_batch = build_memmap_batcher(data_dir, cfg["block_size"], cfg["batch_size"], device)
    if get_batch is None:
        raise RuntimeError(
            "âŒ Dataset files not found.\n"
            f"Expected these files:\n"
            f"  {data_dir / 'train.bin'}\n"
            f"  {data_dir / 'val.bin'}\n\n"
            "You must run a dataset preparation step to create train.bin/val.bin.\n"
            "Once they exist, run training again."
        )

    # vocab size from meta.pkl if exists
    meta_path = data_dir / "meta.pkl"
    meta_vocab_size = None
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get("vocab_size", None)
        if master_process:
            print(f"found vocab_size={meta_vocab_size} in {meta_path}")

    # Build model args
    model_args = dict(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        bias=cfg["bias"],
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,  # safe default
        dropout=cfg["dropout"],
    )

    # Init / resume
    iter_num = 0
    best_val_loss = 1e9
    out_dir = Path(cfg["out_dir"])
    init_from = cfg["init_from"]

    if init_from == "scratch":
        if master_process:
            print("Initializing model from scratch")
        gconf = GPTConfig(**model_args)
        model = GPT(gconf)
    elif init_from == "resume":
        if master_process:
            print(f"Resuming from: {out_dir}")
        ckpt = load_checkpoint(out_dir, device)
        if ckpt is None:
            raise RuntimeError(f"âŒ init_from='resume' but no checkpoint found at {out_dir / 'ckpt.pt'}")
        # lock architecture to checkpoint
        ckpt_args = ckpt["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = ckpt_args[k]
        gconf = GPTConfig(**model_args)
        model = GPT(gconf)
        model.load_state_dict(ckpt["model"])
        iter_num = ckpt.get("iter_num", 0)
        best_val_loss = ckpt.get("best_val_loss", 1e9)
    elif isinstance(init_from, str) and init_from.startswith("gpt2"):
        if not hasattr(GPT, "from_pretrained"):
            raise RuntimeError("âŒ Your GPT class has no from_pretrained(). Use init_from='scratch' or 'resume'.")
        if master_process:
            print(f"Initializing from pretrained weights: {init_from}")
        model = GPT.from_pretrained(init_from, override_args={"dropout": cfg["dropout"]})
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    model.to(device)

    # Optimizer (works with both nanoGPT-like models and simple models)
    if hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(
            cfg["weight_decay"],
            cfg["learning_rate"],
            (cfg["beta1"], cfg["beta2"]),
            device_type
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            betas=(cfg["beta1"], cfg["beta2"]),
            weight_decay=cfg["weight_decay"],
        )

    # restore optimizer if resume
    if init_from == "resume":
        optimizer.load_state_dict(ckpt["optimizer"])

    # AMP scaler (float16 only)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["dtype"] == "float16"))

    # compile (optional)
    if cfg["compile"]:
        if master_process:
            print("Compiling model (torch.compile)...")
        model = torch.compile(model)

    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[ddp_info["local_rank"]])

    raw_model = model.module if ddp else model

    # wandb
    if cfg["wandb_log"] and master_process:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        raw_model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg["eval_iters"])
            for k in range(cfg["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = raw_model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        raw_model.train()
        return out

    # Training loop
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter = 0
    running_mfu = -1.0
    patience = 0
    early_patience = int(cfg.get("early_stop_patience", 0))

    while True:
        # LR step
        lr = get_lr(
            iter_num,
            cfg["warmup_iters"],
            cfg["lr_decay_iters"],
            cfg["learning_rate"],
            cfg["min_lr"],
        ) if cfg["decay_lr"] else cfg["learning_rate"]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint
        if master_process and (iter_num % cfg["eval_interval"] == 0):
            losses = estimate_loss()
            print(f"step {iter_num}: train={losses['train']:.4f}, val={losses['val']:.4f}, lr={lr:.2e}")

            if cfg["wandb_log"]:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100 if running_mfu >= 0 else 0.0,
                })

            improved = losses["val"] < best_val_loss
            if improved:
                best_val_loss = losses["val"]
                patience = 0
                if cfg["always_save_checkpoint"] or iter_num > 0:
                    save_checkpoint(out_dir, raw_model, optimizer, model_args, iter_num, best_val_loss, cfg)
                    print(f"ðŸ’¾ saved checkpoint: {out_dir / 'ckpt.pt'}")
            else:
                patience += 1
                if early_patience > 0:
                    print(f"no improvement, patience {patience}/{early_patience}")
                    if patience >= early_patience:
                        print("ðŸ›‘ early stopping triggered")
                        break

        if iter_num == 0 and cfg["eval_only"]:
            break

        # grad accumulation
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro in range(cfg["gradient_accumulation_steps"]):
            if ddp:
                model.require_backward_grad_sync = (micro == cfg["gradient_accumulation_steps"] - 1)

            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg["gradient_accumulation_steps"]

            # prefetch next batch
            X, Y = get_batch("train")

            scaler.scale(loss).backward()
            total_loss += loss.item()

        # grad clip
        if cfg["grad_clip"] and cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        # logging
        dt = time.time() - t0
        t0 = time.time()

        if master_process and (iter_num % cfg["log_interval"] == 0):
            # approximate unscaled loss
            lossf = total_loss * cfg["gradient_accumulation_steps"]

            # optional MFU if your model supports it
            if hasattr(raw_model, "estimate_mfu") and local_iter >= 5:
                mfu = raw_model.estimate_mfu(cfg["batch_size"] * cfg["gradient_accumulation_steps"], dt)
                running_mfu = mfu if running_mfu < 0 else (0.9 * running_mfu + 0.1 * mfu)

            mfu_txt = f"{running_mfu*100:.2f}%" if running_mfu >= 0 else "n/a"
            print(f"iter {iter_num:>6} | loss {lossf:.4f} | {dt*1000:.1f} ms | mfu {mfu_txt}")

        iter_num += 1
        local_iter += 1

        if iter_num > cfg["max_iters"]:
            break

    # final save
    if master_process:
        save_checkpoint(out_dir, raw_model, optimizer, model_args, iter_num, best_val_loss, cfg)
        print(f"âœ… finished. final checkpoint: {out_dir / 'ckpt.pt'}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
