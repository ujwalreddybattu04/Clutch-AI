"""
Clutch-AI — LLM-Powered Query Router
========================================
Uses the loaded Llama model itself to intelligently decide
whether a query needs a web search — just like Perplexity.

No hardcoded keywords. No hardcoded prompts.
The classification prompt is loaded from config/model_config.json
so it can be tuned, updated, or completely rewritten without
touching a single line of Python.

Usage:
    router = QueryRouter(model, tokenizer, config)
    decision = router.route("what is quantum computing?")
    if decision.should_search:
        # fetch web context
"""

from __future__ import annotations

import time
import warnings
import torch
from dataclasses import dataclass


@dataclass
class RouterDecision:
    """Result of the routing classification."""
    should_search: bool
    reason: str
    latency_ms: float = 0.0


class QueryRouter:
    """
    LLM-powered query router.

    Uses the loaded model to classify queries via a classification
    prompt loaded entirely from config. Zero hardcoded logic.

    Fast because:
      - max_new_tokens=2 (one word only)
      - Greedy decoding (deterministic, no sampling)
      - Same model already in GPU memory
    """

    def __init__(self, model, tokenizer, router_prompt: str):
        """
        Args:
            model: The loaded language model.
            tokenizer: The tokenizer for the model.
            router_prompt: The classification prompt (from config).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.router_prompt = router_prompt

    @torch.inference_mode()
    def route(self, query: str, has_history: bool = False) -> RouterDecision:
        """
        Classify a query using the LLM.

        Args:
            query: The user's input text.
            has_history: Whether there is prior conversation context.

        Returns:
            RouterDecision with should_search, reason, and latency.
        """
        q = query.strip()
        if not q:
            return RouterDecision(False, "empty query", 0.0)

        # Build the classification message
        if has_history:
            user_content = f"(This is a follow-up in an ongoing chat)\n\"{q}\" →"
        else:
            user_content = f"\"{q}\" →"

        messages = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": user_content},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode the classification
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

        # Parse — default to SEARCH if unclear (Perplexity bias)
        if "SKIP" in result:
            return RouterDecision(False, "router: skip", latency_ms)
        else:
            return RouterDecision(True, "router: search", latency_ms)
