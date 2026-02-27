"""
Clutch-AI — Interactive Chat & Inference
=========================================
Usage:
  python scripts/chat.py                              # interactive chat (best model)
  python scripts/chat.py --base                       # base pretrained model
  python scripts/chat.py --prompt "your prompt"       # single-shot mode
  python scripts/chat.py --temp 0.8 --top_k 50       # custom sampling
  python scripts/chat.py --top_p 0.9 --rep_pen 1.2   # nucleus + repetition penalty
  python scripts/chat.py --no-stream                  # disable streaming
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clutch_ai.models.gpt import GPT, GPTConfig

# ─── Config Loader ───────────────────────────────────────────────────────
import json

def load_config():
    """Load model identity from config/model_config.json."""
    config_path = REPO_ROOT / "config" / "model_config.json"
    if not config_path.exists():
        # Fallback defaults if config is missing
        return {
            "name": "Clutch-AI",
            "version": "0.2.0",
            "creator": "Unknown",
            "system_prompt_resolved": "You are a helpful AI assistant.",
            "default_checkpoint": "out-clutch-gpt2-alpacagpt4",
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["system_prompt_resolved"] = cfg["system_prompt"].format(
        name=cfg["name"],
        creator=cfg["creator"],
    )
    return cfg

_CFG = load_config()
VERSION       = _CFG.get("version", "0.2.0")
CREATOR       = _CFG.get("creator", "Unknown")
MODEL_NAME    = _CFG.get("name", "Clutch-AI")
SYSTEM_PROMPT = _CFG.get("system_prompt_resolved", "You are a helpful AI assistant.")

# ─── ANSI Colors ─────────────────────────────────────────────────────────
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


# ─── Model Loading ───────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: str):
    print(colored(f"  Loading checkpoint: {ckpt_path}", Colors.DIM))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    val_loss = ckpt.get("best_val_loss", float("nan"))
    iter_num = ckpt.get("iter_num", "?")
    n_params = model.get_num_params() / 1e6
    print(colored(f"  Parameters: {n_params:.1f}M | Trained: {iter_num} iters | Val loss: {val_loss:.4f}", Colors.DIM))
    return model


# ─── Generation ──────────────────────────────────────────────────────────
def generate_response(
    model, enc, prompt: str,
    max_new_tokens: int, temperature: float,
    top_k: int, top_p: float, rep_penalty: float,
    device: str, stream: bool = True,
) -> tuple[str, float, int]:
    """Generate a response. Returns (text, elapsed_seconds, token_count)."""
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    eot = enc.eot_token

    generated_tokens = []
    t0 = time.perf_counter()

    def on_token(token_id: int):
        generated_tokens.append(token_id)
        if stream and token_id != eot:
            sys.stdout.write(enc.decode([token_id]))
            sys.stdout.flush()

    with torch.no_grad():
        model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            stop_idx=eot,
            stream_callback=on_token,
        )

    elapsed = time.perf_counter() - t0

    # decode full response (for non-streaming or return value)
    text = enc.decode(generated_tokens)
    eot_str = enc.decode([eot])
    text = text.split(eot_str)[0].strip()

    return text, elapsed, len(generated_tokens)


# ─── Reasoning Parser ────────────────────────────────────────────────────
import re

def parse_reasoning(text: str) -> tuple[str, str]:
    """
    Extract <think>...</think> blocks from the response.
    Returns (thinking, answer) where thinking may be empty.
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return thinking, answer
    return "", text.strip()


def display_response(text: str, show_thinking: bool, stream_was_on: bool):
    """
    Display the response with optional thinking section.
    If streaming was on, the raw text was already printed — we re-format it.
    """
    thinking, answer = parse_reasoning(text)

    if stream_was_on:
        # Clear the streamed output and reprint with formatting
        sys.stdout.write('\r' + ' ' * 120 + '\r')  # crude line clear
        # For streaming, we printed raw — now reformat
        pass  # streamed output already shown; we print thinking below

    if thinking and show_thinking:
        print(colored("\n  Thinking:", Colors.DIM + Colors.YELLOW))
        for line in thinking.split('\n'):
            print(colored(f"    {line}", Colors.DIM))
        print()

    if not stream_was_on:
        print(colored("Clutch-AI: ", Colors.GREEN + Colors.BOLD) + answer)
    elif thinking:
        # Reprint the clean answer after showing thinking
        print(colored("  Answer: ", Colors.GREEN + Colors.BOLD) + answer)
    # If streamed and no thinking, output was already shown


# ─── Prompt Building ────────────────────────────────────────────────────
def build_prompt(instruction: str, history: list[dict], use_sft: bool) -> str:
    """Build the full prompt with system context and conversation history."""
    if not use_sft:
        return instruction

    parts = [f"### System:\n{SYSTEM_PROMPT}\n"]

    # Include recent conversation history for multi-turn context
    for turn in history:
        parts.append(f"### Instruction:\n{turn['user']}\n")
        parts.append(f"### Response:\n{turn['assistant']}\n")

    # Current instruction
    parts.append(f"### Instruction:\n{instruction}\n")
    parts.append("### Response:\n")

    return "\n".join(parts)


# ─── Banner ──────────────────────────────────────────────────────────────
def print_banner(args):
    print()
    print(colored("  ╔══════════════════════════════════════════╗", Colors.CYAN))
    print(colored("  ║         ", Colors.CYAN) + colored("Clutch-AI", Colors.BOLD + Colors.CYAN) + colored(f" v{VERSION}", Colors.DIM) + colored("                  ║", Colors.CYAN))
    print(colored(f"  ║     Created by {CREATOR}      ║", Colors.CYAN))
    print(colored("  ╚══════════════════════════════════════════╝", Colors.CYAN))
    print()
    show_think = not getattr(args, 'hide_thinking', False)
    print(colored(f"  temp={args.temp}  top_k={args.top_k}  top_p={args.top_p}  rep_pen={args.rep_pen}", Colors.DIM))
    print(colored(f"  max_tokens={args.max_tokens}  stream={'on' if not args.no_stream else 'off'}  thinking={'on' if show_think else 'off'}  history={args.history_turns}", Colors.DIM))
    print(colored("  ─" * 22, Colors.DIM))
    print()


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clutch-AI chat inference")
    parser.add_argument("--base", action="store_true",
                        help="Use base pretrained model instead of SFT model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature (default 0.7)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (default 40)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p / nucleus sampling (default 0.9)")
    parser.add_argument("--rep_pen", type=float, default=1.15,
                        help="Repetition penalty (default 1.15, 1.0=off)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate (default 512)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Custom checkpoint path")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (print all at once)")
    parser.add_argument("--show-thinking", action="store_true", default=True,
                        help="Show <think> reasoning blocks (default: on)")
    parser.add_argument("--hide-thinking", action="store_true",
                        help="Hide <think> reasoning blocks")
    parser.add_argument("--history-turns", type=int, default=3,
                        help="Number of previous turns to keep as context (default 3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(colored(f"\n  Device: {device}", Colors.DIM))

    # --- Pick checkpoint ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif args.base:
        ckpt_path = REPO_ROOT / "out-clutch-0.1" / "ckpt.pt"
    else:
        ckpt_path = REPO_ROOT / "out-clutch-gpt2-alpacagpt4" / "ckpt.pt"

    if not ckpt_path.exists():
        print(colored(f"\n  ERROR: Checkpoint not found at {ckpt_path}", Colors.YELLOW))
        sys.exit(1)

    use_sft = not args.base
    model = load_model(ckpt_path, device)
    enc = tiktoken.get_encoding("gpt2")

    print_banner(args)

    show_thinking = not args.hide_thinking

    # --- Single-shot mode ---
    if args.prompt:
        prompt = build_prompt(args.prompt, [], use_sft)
        stream = not args.no_stream
        if stream:
            sys.stdout.write(colored("Clutch-AI: ", Colors.GREEN))
        response, elapsed, n_tok = generate_response(
            model, enc, prompt, args.max_tokens,
            args.temp, args.top_k, args.top_p, args.rep_pen,
            device, stream=stream,
        )
        if stream:
            print()  # newline after streamed output
        display_response(response, show_thinking, stream)
        tps = n_tok / elapsed if elapsed > 0 else 0
        print(colored(f"  ({n_tok} tokens in {elapsed:.1f}s \u2014 {tps:.1f} tok/s)", Colors.DIM))
        return

    # --- Interactive mode ---
    print(f"  Type your message and press Enter. Type {colored('quit', Colors.YELLOW)} to exit.\n")
    history: list[dict] = []

    while True:
        try:
            instruction = input(colored("You: ", Colors.MAGENTA + Colors.BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print(colored("\n\n  Goodbye! 👋\n", Colors.CYAN))
            break

        if not instruction or instruction.lower() in ("quit", "exit", "q"):
            print(colored("\n  Goodbye! 👋\n", Colors.CYAN))
            break

        # Build prompt with conversation history
        recent_history = history[-args.history_turns:] if args.history_turns > 0 else []
        prompt = build_prompt(instruction, recent_history, use_sft)

        # Generate
        stream = not args.no_stream
        if stream:
            sys.stdout.write(colored("\nClutch-AI: ", Colors.GREEN + Colors.BOLD))

        response, elapsed, n_tok = generate_response(
            model, enc, prompt, args.max_tokens,
            args.temp, args.top_k, args.top_p, args.rep_pen,
            device, stream=stream,
        )

        if stream:
            print()  # newline after streamed output

        display_response(response, show_thinking, stream)

        # Stats line
        tps = n_tok / elapsed if elapsed > 0 else 0
        print(colored(f"  ({n_tok} tokens \u00b7 {elapsed:.1f}s \u00b7 {tps:.1f} tok/s)", Colors.DIM))
        print()

        # Save to history
        history.append({"user": instruction, "assistant": response})


if __name__ == "__main__":
    main()
