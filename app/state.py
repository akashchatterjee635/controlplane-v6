"""ControlPlane v6 — Central State Schema.

Defines the shared state TypedDict used across all LangGraph nodes.
All nodes read from and write to this state.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
@dataclass
class CostRecord:
    """Tracks resource consumption for a single graph execution."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Output of the validation pipeline."""
    passed: bool = True
    grounded: Optional[bool] = None
    safe: Optional[bool] = None
    compliant: Optional[bool] = None
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    details: str = ""


# ---------------------------------------------------------------------------
# Human review decision
# ---------------------------------------------------------------------------
@dataclass
class HumanDecision:
    """Captures the outcome of a human review."""
    decision: Literal["approve", "redact", "deny"] = "approve"
    redacted_response: Optional[str] = None
    reason: str = ""
    reviewer: str = "unknown"


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class ControlPlaneState(TypedDict, total=False):
    """Central state shared by every node in the ControlPlane graph.

    Fields marked with Annotated[..., operator.add] are *appended*
    rather than overwritten when a node returns them.
    """
    # ---- Input ----
    query: str

    # ---- Router ----
    complexity_score: int
    risk_score: int
    route: Literal["fast", "verified"]

    # ---- Retrieval ----
    documents: list[Document]

    # ---- Grading (verified path) ----
    graded_documents: list[Document]
    web_search_needed: bool

    # ---- Generation ----
    generation: str

    # ---- Validation ----
    validation_result: dict[str, Any]  # serialized ValidationResult
    human_review_needed: bool

    # ---- HITL ----
    human_decision: dict[str, Any]  # serialized HumanDecision

    # ---- Observability ----
    cost_tracker: dict[str, Any]  # serialized CostRecord
    audit_log: Annotated[list[str], operator.add]
