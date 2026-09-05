"""ControlPlane — Central State Schema.

Defines the shared state TypedDict used across all LangGraph nodes.
All nodes read from and write to this state.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

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
# Risk assessment (multi-label)
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    """Multi-label risk assessment for a query/response pair."""
    labels: list[str] = field(default_factory=list)        # e.g. ["hallucination", "pii_leak"]
    severity: str = "low"                                  # low | medium | high | critical
    scores: dict[str, float] = field(default_factory=dict) # per-label scores
    details: str = ""


# ---------------------------------------------------------------------------
# Validator result (per-validator)
# ---------------------------------------------------------------------------
@dataclass
class ValidatorResult:
    """Output of a single validator in the parallel validation layer."""
    name: str = ""               # e.g. "pii_detector"
    passed: bool = True
    risk_labels: list[str] = field(default_factory=list)
    confidence: float = 1.0
    details: str = ""
    edit_suggestion: str | None = None  # optional auto-edit for EDIT decision


# ---------------------------------------------------------------------------
# Validation result (combined)
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Combined output of the validation pipeline."""
    passed: bool = True
    grounded: bool | None = None
    safe: bool | None = None
    compliant: bool | None = None
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    details: str = ""
    validator_results: list[dict] = field(default_factory=list)  # serialized ValidatorResults


# ---------------------------------------------------------------------------
# Human review decision
# ---------------------------------------------------------------------------
@dataclass
class HumanDecision:
    """Captures the outcome of a human review."""
    decision: Literal["approve", "redact", "deny"] = "approve"
    redacted_response: str | None = None
    reason: str = ""
    reviewer: str = "unknown"


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------
@dataclass
class AuditRecord:
    """Structured audit log entry for every graph execution."""
    query_id: str = ""
    request_id: str = ""
    timestamp: str = ""
    use_case: str = "default"
    profile_name: str = "default"
    route: str = ""
    complexity_score: int = 0
    risk_score: int = 0
    risk_labels: list[str] = field(default_factory=list)
    validators_triggered: list[str] = field(default_factory=list)
    validator_versions: dict[str, str] = field(default_factory=dict)
    decision: str = ""           # allow/edit/flag/review/block
    decision_reasoning: str = ""
    reviewer_action: str | None = None
    authenticated_reviewer_id: str | None = None
    final_output_hash: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model_id: str = ""
    model_revision: str = ""
    policy_version: str = "1.0"
    prompt_version: str = "1.0"
    retrieval_index_version: str = "1.0"
    trace_id: str = ""


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
    use_case: str                                    # NEW: policy profile selector

    # ---- Profile ----
    active_profile: dict[str, Any]                   # NEW: resolved profile settings

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

    # ---- Multi-label risk ----
    risk_labels: list[str]                           # NEW: overlapping risk labels
    risk_assessment: dict[str, Any]                  # NEW: serialized RiskAssessment

    # ---- Validation ----
    validation_result: dict[str, Any]                # serialized ValidationResult
    validator_results: list[dict[str, Any]]          # NEW: per-validator results
    human_review_needed: bool

    # ---- Decision ----
    decision: str                                    # NEW: allow/edit/flag/review/block
    decision_reasoning: str                          # NEW: why this decision

    # ---- HITL ----
    human_decision: dict[str, Any]                   # serialized HumanDecision

    # ---- Observability ----
    cost_tracker: dict[str, Any]                     # serialized CostRecord
    audit_log: Annotated[list[str], operator.add]
    audit_record: dict[str, Any]                     # NEW: serialized AuditRecord
