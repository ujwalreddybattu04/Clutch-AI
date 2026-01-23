import os
import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =========================
# Configuration
# =========================
DATA_CACHE_DIR = os.path.dirname(__file__)   # data/fineweb/
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_config = "sample-10BT"
split = "train"

VAL_RATIO = 0.01
MAX_TOKENS = 7_000_000_000   # stop early to keep Kaggle disk for checkpoints

BATCH_SIZE = 1000            # docs per sub-batch
HUGE_CHUNK = 10000           # docs collected before tokenizing
DTYPE = np.uint16            # GPT-2 vocab fits in uint16 (<= 65535)

enc = None


def init_worker():
    """Initialize tokenizer in each worker process."""
    global enc
    enc = tiktoken.get_encoding("gpt2")


def tokenize_batch(texts):
    """Tokenize a batch of documents."""
    global enc
    eot = enc.eot_token
    out = []
    for text in texts:
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        out.append(tokens)
    return out


def process():
    print(f"Loading {dataset_name} ({dataset_config}) streaming...")
    ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)

    train_filename = os.path.join(DATA_CACHE_DIR, "train.bin")
    val_filename = os.path.join(DATA_CACHE_DIR, "val.bin")

    # Use all available cores (Kaggle often has few anyway)
    num_cores = multiprocessing.cpu_count()
    NUM_WORKERS = max(1, num_cores - 1)

    print(f"Writing to:\n  {train_filename}\n  {val_filename}")
    print(f"Tokenizing with {NUM_WORKERS} workers...")

    total_tokens = 0
    pbar = tqdm.tqdm(desc="Tokenizing", unit="tok")

    current_texts = []

    def write_tokens(tokens, f_train, f_val):
        """Write one document tokens to train or val, update progress, stop at MAX_TOKENS."""
        nonlocal total_tokens
        arr = np.array(tokens, dtype=DTYPE)

        if np.random.rand() < VAL_RATIO:
            f_val.write(arr.tobytes())
        else:
            f_train.write(arr.tobytes())

        count = len(tokens)
        total_tokens += count
        pbar.update(count)

        # STOP condition
        if total_tokens >= MAX_TOKENS:
            pbar.close()
            print(f"Stopped early at {total_tokens} tokens (limit {MAX_TOKENS}).")
            return True  # signal stop
        return False

    with open(train_filename, "wb") as f_train, open(val_filename, "wb") as f_val, \
         ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:

        for example in ds:
            current_texts.append(example["text"])

            # Once we have enough docs, tokenize them
            if len(current_texts) >= HUGE_CHUNK:
                sub_batches = [
                    current_texts[j:j + BATCH_SIZE]
                    for j in range(0, len(current_texts), BATCH_SIZE)
                ]

                # IMPORTANT: iterate directly (don’t build list) to reduce RAM
                for batch_results in executor.map(tokenize_batch, sub_batches):
                    for tokens in batch_results:
                        if write_tokens(tokens, f_train, f_val):
                            return

                current_texts = []

        # Flush remaining docs
        if current_texts:
            sub_batches = [
                current_texts[j:j + BATCH_SIZE]
                for j in range(0, len(current_texts), BATCH_SIZE)
            ]
            for batch_results in executor.map(tokenize_batch, sub_batches):
                for tokens in batch_results:
                    if write_tokens(tokens, f_train, f_val):
                        return

    pbar.close()
    print(f"Done. Total tokens: {total_tokens}")


if __name__ == "__main__":
    process()
