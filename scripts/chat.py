"""
Clutch-AI v0.3 — Llama 3.2 Interactive Chat & Inference
==========================================================
Usage:
  python scripts/chat_llama.py                            # interactive chat
  python scripts/chat_llama.py --prompt "your question"   # single-shot mode
  python scripts/chat_llama.py --temp 0.8 --top_k 50     # custom sampling
  python scripts/chat_llama.py --no-stream                # disable streaming

Requirements:
  pip install transformers accelerate torch sentencepiece protobuf
"""

import sys
import time
import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ─── Config Loader ───────────────────────────────────────────────
import json

def load_config():
    """Load model identity from config/llama_3b_config.json."""
    config_path = REPO_ROOT / "config" / "llama_3b_config.json"
    if not config_path.exists():
        return {
            "name": "Clutch-AI",
            "version": "0.3.0",
            "creator": "Unknown",
            "system_prompt_resolved": "You are a helpful AI assistant.",
            "default_checkpoint": "out-clutch-llama3.2-final",
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["system_prompt_resolved"] = cfg["system_prompt"].format(
        name=cfg["name"],
        creator=cfg["creator"],
    )
    return cfg

_CFG = load_config()
VERSION       = _CFG.get("version", "0.3.0")
CREATOR       = _CFG.get("creator", "Unknown")
MODEL_NAME    = _CFG.get("name", "Clutch-AI")
SYSTEM_PROMPT = _CFG.get("system_prompt_resolved", "You are a helpful AI assistant.")


# ─── ANSI Colors ─────────────────────────────────────────────────
class Colors:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# ─── Model Loading ───────────────────────────────────────────────
def load_model(model_path: Path, device: str):
    """Load the fine-tuned Llama 3.2 model."""
    print(colored(f"  Loading model: {model_path}", Colors.DIM))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(colored(f"  Parameters: {n_params:.1f}M", Colors.DIM))
    return model, tokenizer


# ─── Generation ──────────────────────────────────────────────────
def generate_response(
    model, tokenizer, messages: list,
    max_new_tokens: int, temperature: float,
    top_k: int, top_p: float, rep_penalty: float,
    stream: bool = True,
) -> tuple:
    """Generate a response. Returns (text, elapsed_seconds, token_count)."""

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.perf_counter()

    with torch.no_grad():
        if stream:
            # Streaming generation
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                do_sample=True,
                streamer=streamer,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                generated_text += new_text

            thread.join()
        else:
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                do_sample=True,
            )
            generated_text = tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True,
            )

    elapsed = time.perf_counter() - t0
    n_tokens = len(tokenizer.encode(generated_text))

    return generated_text.strip(), elapsed, n_tokens


# ─── Reasoning Parser ────────────────────────────────────────────
def parse_reasoning(text: str) -> tuple:
    """Extract <think>...</think> blocks from the response."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return thinking, answer
    return "", text.strip()


def display_response(text: str, show_thinking: bool, stream_was_on: bool):
    """Display the response with optional thinking section."""
    thinking, answer = parse_reasoning(text)

    if thinking and show_thinking:
        print(colored("\n  💭 Thinking:", Colors.DIM + Colors.YELLOW))
        for line in thinking.split('\n'):
            print(colored(f"    {line}", Colors.DIM))
        print()

    if not stream_was_on:
        print(colored(f"{MODEL_NAME}: ", Colors.GREEN + Colors.BOLD) + answer)
    elif thinking:
        print(colored(f"\n  Answer: ", Colors.GREEN + Colors.BOLD) + answer)


# ─── Prompt Building ─────────────────────────────────────────────
def build_messages(instruction: str, history: list) -> list:
    """Build Llama 3.2 chat messages with system prompt and history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": instruction})
    return messages


# ─── Banner ──────────────────────────────────────────────────────
def print_banner(args):
    print()
    print(colored("  ╔══════════════════════════════════════════╗", Colors.CYAN))
    print(colored("  ║         ", Colors.CYAN) + colored("Clutch-AI", Colors.BOLD + Colors.CYAN) + colored(f" v{VERSION}", Colors.DIM) + colored("                  ║", Colors.CYAN))
    print(colored(f"  ║     Created by {CREATOR}      ║", Colors.CYAN))
    print(colored("  ║     Powered by Llama 3.2 3B 🦙           ║", Colors.CYAN))
    print(colored("  ╚══════════════════════════════════════════╝", Colors.CYAN))
    print()
    show_think = not getattr(args, 'hide_thinking', False)
    print(colored(f"  temp={args.temp}  top_k={args.top_k}  top_p={args.top_p}  rep_pen={args.rep_pen}", Colors.DIM))
    print(colored(f"  max_tokens={args.max_tokens}  stream={'on' if not args.no_stream else 'off'}  thinking={'on' if show_think else 'off'}  history={args.history_turns}", Colors.DIM))
    print(colored("  ─" * 22, Colors.DIM))
    print()


# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clutch-AI v0.3 — Llama 3.2 Chat")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature (default 0.7)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (default 40)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling (default 0.9)")
    parser.add_argument("--rep_pen", type=float, default=1.15,
                        help="Repetition penalty (default 1.15)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate (default 512)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom model path")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming")
    parser.add_argument("--show-thinking", action="store_true", default=True,
                        help="Show <think> reasoning blocks")
    parser.add_argument("--hide-thinking", action="store_true",
                        help="Hide <think> reasoning blocks")
    parser.add_argument("--history-turns", type=int, default=3,
                        help="Number of previous turns to keep (default 3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(colored(f"\n  Device: {device}", Colors.DIM))

    # Pick model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = REPO_ROOT / "out-clutch-llama3.2-final"

    if not model_path.exists():
        print(colored(f"\n  ERROR: Model not found at {model_path}", Colors.YELLOW))
        print(colored(f"  Run the training notebook first, then extract the model here.", Colors.DIM))
        sys.exit(1)

    model, tokenizer = load_model(model_path, device)
    print_banner(args)

    show_thinking = not args.hide_thinking

    # Single-shot mode
    if args.prompt:
        messages = build_messages(args.prompt, [])
        stream = not args.no_stream
        if stream:
            sys.stdout.write(colored(f"{MODEL_NAME}: ", Colors.GREEN))
        response, elapsed, n_tok = generate_response(
            model, tokenizer, messages, args.max_tokens,
            args.temp, args.top_k, args.top_p, args.rep_pen,
            stream=stream,
        )
        if stream:
            print()
        display_response(response, show_thinking, stream)
        tps = n_tok / elapsed if elapsed > 0 else 0
        print(colored(f"  ({n_tok} tokens in {elapsed:.1f}s — {tps:.1f} tok/s)", Colors.DIM))
        return

    # Interactive mode
    print(f"  Type your message and press Enter. Type {colored('quit', Colors.YELLOW)} to exit.\n")
    history: list = []

    while True:
        try:
            instruction = input(colored("You: ", Colors.MAGENTA + Colors.BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print(colored("\n\n  Goodbye! 👋\n", Colors.CYAN))
            break

        if not instruction or instruction.lower() in ("quit", "exit", "q"):
            print(colored("\n  Goodbye! 👋\n", Colors.CYAN))
            break

        recent_history = history[-args.history_turns:] if args.history_turns > 0 else []
        messages = build_messages(instruction, recent_history)

        stream = not args.no_stream
        if stream:
            sys.stdout.write(colored(f"\n{MODEL_NAME}: ", Colors.GREEN + Colors.BOLD))

        response, elapsed, n_tok = generate_response(
            model, tokenizer, messages, args.max_tokens,
            args.temp, args.top_k, args.top_p, args.rep_pen,
            stream=stream,
        )

        if stream:
            print()

        display_response(response, show_thinking, stream)

        tps = n_tok / elapsed if elapsed > 0 else 0
        print(colored(f"  ({n_tok} tokens · {elapsed:.1f}s · {tps:.1f} tok/s)", Colors.DIM))
        print()

        history.append({"user": instruction, "assistant": response})


if __name__ == "__main__":
    main()
