import os, pickle, argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
import tiktoken

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset_config", default="sample-10BT")
    ap.add_argument("--out_dir", default="data/fineweb")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=2_000_000)  # keep small first!
    ap.add_argument("--val_ratio", type=float, default=0.001)     # ~0.1% to val
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "meta.pkl"

    # reset files
    if train_path.exists(): train_path.unlink()
    if val_path.exists(): val_path.unlink()

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    vocab_size = enc.n_vocab  # 50257

    ds = load_dataset(args.dataset_name, args.dataset_config, split="train", streaming=True)

    train_f = open(train_path, "ab")
    val_f = open(val_path, "ab")

    total = 0
    ex_i = 0

    print("Tokenizing + writing .bin files...")
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue

        ids = enc.encode(text)
        ids.append(eot)

        arr = np.array(ids, dtype=np.uint16)

        # simple split rule: every Nth example goes to val
        if np.random.random() < args.val_ratio:
            arr.tofile(val_f)
        else:
            arr.tofile(train_f)

        total += len(arr)
        ex_i += 1

        if ex_i % 200 == 0:
            print(f"processed examples={ex_i}, tokens={total:,}")

        if total >= args.max_tokens:
            break

    train_f.close()
    val_f.close()

    with open(meta_path, "wb") as f:
        pickle.dump({"vocab_size": vocab_size}, f)

    print("✅ Done")
    print("train.bin:", train_path)
    print("val.bin:", val_path)
    print("meta.pkl:", meta_path)
    print("total tokens:", total)

if __name__ == "__main__":
    main()
