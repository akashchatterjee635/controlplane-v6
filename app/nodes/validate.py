"""Output validation node.

Performs Layer 1 deterministic checks (PII, injection patterns) and
Layer 2 LLM-based checks (groundedness, safety, compliance).
"""

import os
from typing import Any, Literal
import yaml

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from app.state import ControlPlaneState, ValidationResult
from app.utils.security import detect_pii, check_prompt_injection, sanitize_output
from app.utils.cost import update_cost_record


# ---------------------------------------------------------------------------
# Structured output for Layer 2 LLM validation
# ---------------------------------------------------------------------------
class LLMValidationGrade(BaseModel):
    """LLM-generated validation assessment."""
    grounded: bool = Field(
        description="True if the response is supported by the provided context documents."
    )
    safe: bool = Field(
        description="True if the response is safe and does not contain harmful content."
    )
    compliant: bool = Field(
        description="True if the response follows policy guidelines and does not leak sensitive info."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence score (0.0 to 1.0) in the response quality."
    )
    reasoning: str = Field(
        description="Brief explanation of the validation assessment."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_policies() -> dict:
    """Load grading/validation policies."""
    policies_path = os.path.join(os.path.dirname(__file__), "..", "policies", "policies.yaml")
    if os.path.exists(policies_path):
        with open(policies_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _format_context_for_validation(documents: list[Document]) -> str:
    """Format documents for the validation prompt."""
    if not documents:
        return "No source documents available."
    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"[Document {i}]: {doc.page_content[:500]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 1 — Deterministic checks
# ---------------------------------------------------------------------------
def _run_layer1(generation: str, policies: dict) -> tuple[str, bool, list[str], list[str]]:
    """Run Layer 1 deterministic checks.

    Returns:
        (sanitized_generation, passed, flags, audit_entries)
    """
    pii_patterns = policies.get("pii_patterns", {})
    prohibited_keywords = policies.get("prohibited_keywords", [])

    audit_entries: list[str] = []
    flags: list[str] = []
    passed = True

    # PII detection
    pii_matches = detect_pii(generation, pii_patterns)
    if pii_matches:
        flags.append(f"PII detected: {len(pii_matches)} instances.")
        passed = False
        audit_entries.append("[VALIDATE] Layer 1: PII detected in output.")
        generation = sanitize_output(generation, pii_matches)
        audit_entries.append("[VALIDATE] Layer 1: Output sanitized.")

    # Prohibited keyword detection
    if check_prompt_injection(generation, prohibited_keywords):
        flags.append("Prohibited keywords detected in output.")
        passed = False
        audit_entries.append("[VALIDATE] Layer 1: Prohibited keywords detected in output.")

    return generation, passed, flags, audit_entries


# ---------------------------------------------------------------------------
# Node: validate_basic (fast path — Layer 1 only)
# ---------------------------------------------------------------------------
def validate_basic_node(state: ControlPlaneState) -> dict[str, Any]:
    """Layer 1 deterministic validation for the fast path.

    Checks for PII and basic prohibited patterns in the generated output.
    Automatically sanitizes PII to maintain fast-path speed.
    """
    generation = state.get("generation", "")
    policies = _load_policies()

    generation, passed, flags, audit_entries = _run_layer1(generation, policies)

    result = ValidationResult(
        passed=passed,
        confidence=1.0,
        flags=flags,
        details="Layer 1 deterministic checks completed.",
    )

    return {
        "generation": generation,
        "validation_result": {
            "passed": result.passed,
            "grounded": None,
            "safe": None,
            "compliant": None,
            "confidence": result.confidence,
            "flags": result.flags,
            "details": result.details,
        },
        "audit_log": audit_entries if audit_entries else ["[VALIDATE_BASIC] All checks passed."],
    }


# ---------------------------------------------------------------------------
# Node: validate_full (verified path — Layer 1 + Layer 2)
# ---------------------------------------------------------------------------
def validate_full_node(state: ControlPlaneState) -> dict[str, Any]:
    """Layer 1 + Layer 2 validation for the verified path.

    Layer 1: Deterministic PII/injection checks.
    Layer 2: LLM-based groundedness, safety, and compliance assessment.

    If the overall confidence is below the policy threshold, or if any
    check fails, the response is flagged for human review.
    """
    generation = state.get("generation", "")
    policies = _load_policies()
    cost_tracker = state.get("cost_tracker", {})

    # ---- Layer 1 ----
    generation, l1_passed, flags, audit_entries = _run_layer1(generation, policies)

    # ---- Layer 2: LLM validation ----
    documents = state.get("graded_documents", state.get("documents", []))
    context = _format_context_for_validation(documents)
    query = state.get("query", "")

    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    validation_prompt = f"""You are a quality assurance reviewer. Evaluate the following AI-generated response.

Source Documents:
{context}

User Question: {query}

AI Response: {generation}

Evaluate the response on these criteria:
1. **Grounded**: Is the response factually supported by the source documents? (true/false)
2. **Safe**: Does the response avoid harmful, dangerous, or inappropriate content? (true/false)
3. **Compliant**: Does the response follow responsible AI guidelines and avoid leaking sensitive information? (true/false)
4. **Confidence**: How confident are you (0.0-1.0) in the overall quality and correctness of the response?

Provide your assessment."""

    try:
        llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(
            LLMValidationGrade
        )
        llm_result: LLMValidationGrade = llm.invoke([
            SystemMessage(content="You are a strict AI quality assurance validator."),
            HumanMessage(content=validation_prompt),
        ])

        grounded = llm_result.grounded
        safe = llm_result.safe
        compliant = llm_result.compliant
        confidence = llm_result.confidence
        reasoning = llm_result.reasoning

        audit_entries.append(
            f"[VALIDATE] Layer 2: grounded={grounded}, safe={safe}, "
            f"compliant={compliant}, confidence={confidence:.2f}"
        )
        audit_entries.append(f"[VALIDATE] Layer 2 reasoning: {reasoning}")

        # Estimate token usage for cost tracking
        prompt_tokens_est = len(validation_prompt.split()) + 20
        comp_tokens_est = 60
        cost_tracker = update_cost_record(
            cost_tracker,
            prompt_tokens=prompt_tokens_est,
            completion_tokens=comp_tokens_est,
            model=model_name,
        )

    except Exception as e:
        # If LLM validation fails, default to conservative assessment
        grounded = None
        safe = None
        compliant = None
        confidence = 0.5  # low confidence triggers human review
        reasoning = f"LLM validation failed: {str(e)}"
        audit_entries.append(f"[VALIDATE] Layer 2 failed: {str(e)}")

    # ---- Combine results ----
    confidence_threshold = policies.get("thresholds", {}).get(
        "validation_confidence_min", 0.85
    )

    l2_passed = all([
        grounded is not False,
        safe is not False,
        compliant is not False,
    ])
    overall_passed = l1_passed and l2_passed

    # Human review needed if: any check failed, or confidence is below threshold
    human_review_needed = (
        not overall_passed
        or confidence < confidence_threshold
    )

    if not grounded:
        flags.append("Response may not be grounded in source documents.")
    if not safe:
        flags.append("Response may contain unsafe content.")
    if not compliant:
        flags.append("Response may not comply with policy guidelines.")
    if confidence < confidence_threshold:
        flags.append(
            f"Confidence ({confidence:.2f}) below threshold ({confidence_threshold})."
        )

    result = ValidationResult(
        passed=overall_passed,
        grounded=grounded,
        safe=safe,
        compliant=compliant,
        confidence=confidence,
        flags=flags,
        details=reasoning if isinstance(reasoning, str) else "Validation complete.",
    )

    if human_review_needed:
        audit_entries.append("[VALIDATE_FULL] Output flagged. Human review needed.")
    else:
        audit_entries.append("[VALIDATE_FULL] Output passed all checks.")

    return {
        "generation": generation,
        "validation_result": {
            "passed": result.passed,
            "grounded": result.grounded,
            "safe": result.safe,
            "compliant": result.compliant,
            "confidence": result.confidence,
            "flags": result.flags,
            "details": result.details,
        },
        "human_review_needed": human_review_needed,
        "cost_tracker": cost_tracker,
        "audit_log": audit_entries,
    }


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------
def triage_decision(state: ControlPlaneState) -> Literal["end", "human_review"]:
    """Conditional edge after validation to route to HITL if necessary."""
    if state.get("human_review_needed", False):
        return "human_review"
    return "end"
