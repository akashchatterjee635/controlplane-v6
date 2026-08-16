"""Cost tracking and estimation utilities."""

from __future__ import annotations

import time
from typing import Any

import yaml

from app.state import CostRecord


def load_cost_config(policies_path: str = "app/policies/policies.yaml") -> dict:
    """Load cost-per-token configuration from policies."""
    with open(policies_path, "r") as f:
        policies = yaml.safe_load(f)
    return policies.get("cost", {})


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_config: dict | None = None,
) -> float:
    """Estimate USD cost for a given LLM call."""
    if cost_config is None:
        cost_config = load_cost_config()

    # Normalize model name to match config keys
    model_key = model.replace("-", "_").replace(".", "_")
    config = cost_config.get(model_key, cost_config.get("gpt_4o_mini", {}))

    prompt_cost = (prompt_tokens / 1000) * config.get("prompt_per_1k", 0.00015)
    completion_cost = (completion_tokens / 1000) * config.get("completion_per_1k", 0.0006)
    return round(prompt_cost + completion_cost, 8)


def new_cost_record() -> dict[str, Any]:
    """Create a fresh cost record as a serializable dict."""
    record = CostRecord()
    return {
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "llm_calls": record.llm_calls,
        "retrieval_latency_ms": record.retrieval_latency_ms,
        "total_latency_ms": record.total_latency_ms,
        "estimated_cost_usd": record.estimated_cost_usd,
        "start_time": record.start_time,
    }


def update_cost_record(
    tracker: dict[str, Any],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str = "gpt-4o-mini",
    cost_config: dict | None = None,
) -> dict[str, Any]:
    """Update a cost tracker dict with new token counts."""
    tracker = dict(tracker)  # shallow copy
    tracker["prompt_tokens"] = tracker.get("prompt_tokens", 0) + prompt_tokens
    tracker["completion_tokens"] = tracker.get("completion_tokens", 0) + completion_tokens
    tracker["llm_calls"] = tracker.get("llm_calls", 0) + 1
    tracker["estimated_cost_usd"] = tracker.get("estimated_cost_usd", 0.0) + estimate_cost(
        model, prompt_tokens, completion_tokens, cost_config
    )
    tracker["total_latency_ms"] = (time.time() - tracker.get("start_time", time.time())) * 1000
    return tracker


def format_cost_report(tracker: dict[str, Any]) -> str:
    """Format a cost tracker as a human-readable report."""
    return (
        f"Cost Report:\n"
        f"  LLM Calls:         {tracker.get('llm_calls', 0)}\n"
        f"  Prompt Tokens:     {tracker.get('prompt_tokens', 0):,}\n"
        f"  Completion Tokens: {tracker.get('completion_tokens', 0):,}\n"
        f"  Retrieval Latency: {tracker.get('retrieval_latency_ms', 0):.1f}ms\n"
        f"  Total Latency:     {tracker.get('total_latency_ms', 0):.1f}ms\n"
        f"  Estimated Cost:    ${tracker.get('estimated_cost_usd', 0):.6f}\n"
    )
