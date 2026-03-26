"""
Clutch-AI v1.0.0 — Industry-Grade Local Inference (LoRA)
==========================================================
Perplexity-style AI assistant. Everything is config-driven:
pythn script 
"""

import os
import sys
import re
import time
import json
import warnings
import logging
from datetime import datetime, timezone
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

# ── Suppress noisy framework warnings ──
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Project paths ──
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from query_router import QueryRouter, RouterDecision


# ─── ANSI Colors ─────────────────────────────────────────────────
class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    CLEAR_LINE = "\r\033[K"


# ─── Config Loader ───────────────────────────────────────────────
def load_config() -> dict:
    """
    Load everything from config/model_config.json.
    This is the SINGLE source of truth for all prompts,
    generation parameters, and settings.
    """
    config_path = REPO_ROOT / "config" / "model_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Resolve identity placeholders in system prompt
        cfg["system_prompt_resolved"] = cfg.get("system_prompt", "").format(
            name=cfg.get("name", "Clutch-AI"),
            creator=cfg.get("creator", "Clutch Group"),
        )
        return cfg
    except Exception as e:
        print(f"{C.RED}  ❌ Failed to load config: {e}{C.RESET}")
        sys.exit(1)


# ─── Web Search (Silent — Perplexity Style) ──────────────────────
def fetch_web_context(query: str, tavily_client, cfg: dict) -> str:
    """
    Silently fetch web context. User NEVER sees raw results.
    Settings loaded from config. URLs excluded from context
    so the model can't regurgitate them.
    """
    web_cfg = cfg.get("web_search", {})
    max_results = web_cfg.get("max_results", 3)
    search_depth = web_cfg.get("search_depth", "basic")
    max_content_len = web_cfg.get("max_content_length", 300)

    try:
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
        )
        results = response.get("results", [])
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            content = r.get("content", "")
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."
            parts.append(f"[Source {i}: {title}]\n{content}")

        return "\n\n".join(parts)
    except Exception:
        return ""


# ─── Model Loading ───────────────────────────────────────────────
def load_model(model_name: str, adapter_path: str):
    """Load base model in optimized 4-bit and attach LoRA adapter."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"{C.DIM}  Loading tokenizer...{C.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"{C.DIM}  Loading base model (4-bit NF4)...{C.RESET}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"{C.DIM}  Attaching LoRA adapter...{C.RESET}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


# ─── Response Parser ─────────────────────────────────────────────
def parse_response(text: str) -> tuple[str, str]:
    """
    Extract <think> reasoning from the model's response.
    This is legitimate parsing of a structured model feature.

    Returns:
        (thinking, answer) — thinking may be empty.
    """
    thinking = ""
    answer = text

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()

    if text.strip().startswith("<think>") and "</think>" not in text:
        thinking = text.replace("<think>", "").strip()
        answer = thinking

    return thinking, answer


# ─── Response Generation ────────────────────────────────────────
@torch.inference_mode()
def generate(model, tokenizer, messages: list, cfg: dict) -> tuple[str, float, int]:
    """Generate a response using params from config."""

    gen_cfg = cfg.get("generation", {})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Resolve Llama 3 stopping tokens
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        temperature=gen_cfg.get("temperature", 0.7),
        top_k=gen_cfg.get("top_k", 30),
        top_p=gen_cfg.get("top_p", 0.85),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.15),
        eos_token_id=terminators,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    elapsed = time.perf_counter() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    token_count = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed, token_count


# ─── Banner ──────────────────────────────────────────────────────
def print_banner(cfg: dict, gpu_name: str, web_enabled: bool):
    name = cfg.get("name", "Clutch-AI")
    version = cfg.get("version", "1.0.0")
    creator = cfg.get("creator", "Clutch Group")

    print(f"\n{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🤖  {name} v{version}  —  by {creator}{C.RESET}")
    print(f"{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"  {C.DIM}GPU           {gpu_name}{C.RESET}")
    print(f"  {C.DIM}Web Search    {'ON (Perplexity mode — smart routing)' if web_enabled else 'OFF'}{C.RESET}")
    print(f"  {C.DIM}Router        LLM-powered intent classifier{C.RESET}")
    print(f"{C.CYAN}{'─' * 60}{C.RESET}")
    print(f"  {C.DIM}Commands      /web on · /web off · /search <query> · quit{C.RESET}")
    print(f"{C.CYAN}{'═' * 60}{C.RESET}\n")


# ─── System Prompt Builder ───────────────────────────────────────
def build_system_prompt(cfg: dict, web_context: str = "") -> str:
    """
    Construct the full system prompt from config.
    All prompt text comes from model_config.json — nothing hardcoded.
    """
    # Identity prompt (from config "system_prompt")
    prompt = cfg.get("system_prompt_resolved", "You are Clutch-AI.")

    # Inject real-time date/time with pre-computed timezone conversions
    # The model can't do timezone math, so Python does it
    from datetime import timedelta
    utc_now = datetime.now(timezone.utc)
    timezones = {
        "UTC": utc_now,
        "EST (New York)": utc_now + timedelta(hours=-5),
        "GMT (London)": utc_now,
        "IST (India)": utc_now + timedelta(hours=5, minutes=30),
        "JST (Tokyo)": utc_now + timedelta(hours=9),
        "PST (Los Angeles)": utc_now + timedelta(hours=-8),
    }
    time_lines = [f"- {tz}: {dt.strftime('%A, %B %d, %Y at %I:%M %p')}" for tz, dt in timezones.items()]
    prompt += "\n\nCurrent date and time:\n" + "\n".join(time_lines)

    # Behavior instructions (from config "behavior_prompt")
    behavior = cfg.get("behavior_prompt", "")
    if behavior:
        prompt += "\n\n" + behavior

    # Web context injection (from config "web_context_prompt")
    if web_context:
        web_instruction = cfg.get("web_context_prompt", "")
        prompt += f"\n\n## Web-Informed Response\n\n{web_instruction}\n\n{web_context}"

    return prompt


# ─── Main ────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    model_name = cfg.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
    max_turns = cfg.get("memory", {}).get("max_turns", 4)

    # ── Auth ──
    print(f"\n{C.CYAN}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🤖  {cfg.get('name', 'Clutch-AI')} — Local Inference{C.RESET}")
    print(f"{C.CYAN}{'═' * 60}{C.RESET}")

    hf_token = input(f"\n{C.YELLOW}  🔑 HuggingFace Token: {C.RESET}").strip()
    if hf_token:
        login(hf_token, add_to_git_credential=False)

    # ── Tavily (web search) ──
    load_dotenv()
    tavily_client = None
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=tavily_key)
        except ImportError:
            print(f"{C.YELLOW}  ⚠️  tavily-python not installed — web search disabled{C.RESET}")

    # ── Adapter check ──
    adapter_path = str(REPO_ROOT)
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        print(f"\n{C.RED}  ❌ adapter_config.json not found in project root.{C.RESET}")
        print(f"{C.DIM}     Place adapter_model.safetensors & adapter_config.json in:{C.RESET}")
        print(f"{C.DIM}     {REPO_ROOT}{C.RESET}")
        return

    # ── Load model ──
    print(f"\n{C.DIM}  📥 Loading {model_name}...{C.RESET}")
    model, tokenizer = load_model(model_name, adapter_path)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    web_enabled = tavily_client is not None
    auto_search = web_enabled

    print_banner(cfg, gpu_name, web_enabled)

    # ── Router (prompt from config) ──
    router_prompt = cfg.get("router_prompt", "")
    router = QueryRouter(model, tokenizer, router_prompt)
    print(f"{C.DIM}  🧠 LLM-powered query router ready{C.RESET}\n")

    history: list[dict] = []

    # ── Chat loop ──
    while True:
        try:
            user_input = input(f"{C.GREEN}{C.BOLD}You ▶ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}  👋 Goodbye!{C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{C.DIM}  👋 Goodbye!{C.RESET}")
            break

        # ── Commands ──
        if user_input.lower() == "/web on":
            auto_search = True
            print(f"{C.GREEN}  ✅ Smart web search ON{C.RESET}\n")
            continue
        if user_input.lower() == "/web off":
            auto_search = False
            print(f"{C.YELLOW}  🔇 Web search OFF (faster, model-only){C.RESET}\n")
            continue
        if user_input.lower().startswith("/search "):
            search_query = user_input[8:].strip()
            if tavily_client and search_query:
                sys.stdout.write(f"{C.DIM}  🌐 searching...{C.RESET}")
                sys.stdout.flush()
                web_context = fetch_web_context(search_query, tavily_client, cfg)
                sys.stdout.write(C.CLEAR_LINE)
                messages = [
                    {"role": "system", "content": build_system_prompt(cfg, web_context)},
                    *history[-(max_turns * 2):],
                    {"role": "user", "content": search_query},
                ]
                response, elapsed, tok_count = generate(model, tokenizer, messages, cfg)
                thinking, answer = parse_response(response)
                if thinking and thinking != answer:
                    print(f"\n{C.DIM}  💭 {thinking[:150]}{'...' if len(thinking) > 150 else ''}{C.RESET}")
                print(f"\n{C.CYAN}{C.BOLD}  Clutch-AI ▶ {C.RESET}{answer}")
                tps = tok_count / elapsed if elapsed > 0 else 0
                print(f"{C.DIM}  ⚡ {tok_count} tokens · {elapsed:.1f}s · {tps:.1f} tok/s · 🌐 forced search{C.RESET}\n")
                history.append({"role": "user", "content": search_query})
                history.append({"role": "assistant", "content": answer})
            continue

        # ── Route the query ──
        decision: RouterDecision = router.route(user_input, has_history=len(history) > 0)

        web_context = ""
        if auto_search and tavily_client and decision.should_search:
            sys.stdout.write(f"{C.DIM}  🌐 searching...{C.RESET}")
            sys.stdout.flush()
            web_context = fetch_web_context(user_input, tavily_client, cfg)
            sys.stdout.write(C.CLEAR_LINE)
            sys.stdout.flush()

        # ── Build messages ──
        system_content = build_system_prompt(cfg, web_context)
        messages = [{"role": "system", "content": system_content}]

        for turn in history[-(max_turns * 2):]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_input})

        # ── Generate ──
        response, elapsed, tok_count = generate(model, tokenizer, messages, cfg)

        # ── Parse ──
        thinking, answer = parse_response(response)

        if thinking and thinking != answer:
            print(f"\n{C.DIM}  💭 {thinking[:150]}{'...' if len(thinking) > 150 else ''}{C.RESET}")

        print(f"\n{C.CYAN}{C.BOLD}  Clutch-AI ▶ {C.RESET}{answer}")

        tps = tok_count / elapsed if elapsed > 0 else 0
        search_tag = ""
        if auto_search and tavily_client:
            if decision.should_search:
                search_tag = f" · 🌐 web" if web_context else " · 🌐 no results"
            else:
                search_tag = f" · 💭 skipped search"
            search_tag += f" · router {decision.latency_ms:.0f}ms"

        print(f"{C.DIM}  ⚡ {tok_count} tokens · {elapsed:.1f}s · {tps:.1f} tok/s{search_tag}{C.RESET}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
