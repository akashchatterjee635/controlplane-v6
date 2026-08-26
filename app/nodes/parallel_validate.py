"""Parallel validation layer.

Runs 5 independent validators concurrently via ThreadPoolExecutor:
  1. PII detector
  2. Grounding verifier
  3. Policy checker
  4. Bias / sensitive-topic detector
  5. Claim confidence scorer

Each validator returns a ValidatorResult.  Results are combined into a
single validation_result dict for the decision layer.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

import yaml

from app.state import ControlPlaneState, ValidatorResult
from app.utils.security import detect_pii, check_prompt_injection, sanitize_output
from app.utils.risk_classifier import (
    _check_pii,
    _check_bias,
    _check_compliance,
    _check_policy_violation,
    _check_unsupported_claims,
    PII_PATTERNS,
)


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def run_pii_check(state: dict, profile: dict) -> ValidatorResult:
    """Detect PII in the generated response and query."""
    query = state.get("query", "")
    response = state.get("generation", "")
    
    # Check both query and response for PII
    found_q, score_q, detail_q = _check_pii(query)
    found_r, score_r, detail_r = _check_pii(response)
    
    found = found_q or found_r
    score = max(score_q, score_r)
    detail_parts = []
    if found_q: detail_parts.append(f"Query: {detail_q}")
    if found_r: detail_parts.append(f"Response: {detail_r}")
    detail = " | ".join(detail_parts)

    pii_policy = profile.get("pii_policy", "redact")
    edit_suggestion = None

    if found and pii_policy == "redact":
        # If redact, suggest a sanitized response
        if response:
            from app.utils.security import detect_pii
            matches = detect_pii(response, PII_PATTERNS)
            edit_suggestion = sanitize_output(response, matches)
        else:
            edit_suggestion = ""
        
        if found_q and not found_r:
            # If PII was only in query, the response might be fine but we still trigger EDIT outcome
            # to demonstrate redaction capability (e.g. redacting it from logs or context)
            pass

    return ValidatorResult(
        name="pii_detector",
        passed=not found,
        risk_labels=["pii_leak"] if found else [],
        confidence=1.0 - score,
        details=detail or "No PII detected",
        edit_suggestion=edit_suggestion,
    )


def run_grounding_check(state: dict, profile: dict) -> ValidatorResult:
    """Check if the response is grounded in retrieved documents."""
    response = state.get("generation", "")
    documents = state.get("documents", [])

    if not response:
        return ValidatorResult(
            name="grounding_verifier", passed=True,
            confidence=1.0, details="No response to verify",
        )

    if not documents:
        return ValidatorResult(
            name="grounding_verifier", passed=False,
            risk_labels=["hallucination"],
            confidence=0.2, details="No source documents for grounding",
        )

    # Simple keyword overlap heuristic: check if response keywords appear in docs
    doc_text = " ".join(
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in documents
    ).lower()

    response_words = set(response.lower().split())
    # Filter to substantive words (>4 chars)
    substantive = {w for w in response_words if len(w) > 4}

    if not substantive:
        return ValidatorResult(
            name="grounding_verifier", passed=True,
            confidence=0.8, details="Response too short for grounding check",
        )

    overlap = sum(1 for w in substantive if w in doc_text)
    overlap_ratio = overlap / len(substantive)

    hallucination_check = profile.get("hallucination_check", "medium")
    thresholds = {"strict": 0.3, "medium": 0.2, "relaxed": 0.1}
    threshold = thresholds.get(hallucination_check, 0.2)

    passed = overlap_ratio >= threshold
    labels = [] if passed else ["hallucination"]

    return ValidatorResult(
        name="grounding_verifier",
        passed=passed,
        risk_labels=labels,
        confidence=min(1.0, overlap_ratio + 0.3),
        details=f"Grounding overlap: {overlap_ratio:.2f} (threshold: {threshold})",
    )


def run_policy_check(state: dict, profile: dict) -> ValidatorResult:
    """Check for prompt injection and policy violations."""
    query = state.get("query", "")
    response = state.get("generation", "")
    combined = f"{query} {response}"

    found, score, detail = _check_policy_violation(combined)

    # Also check the query for injection patterns
    prohibited = profile.get("prohibited_keywords", [])
    injection_found = check_prompt_injection(query, prohibited)

    if injection_found and not found:
        found = True
        score = 0.8
        detail = "Prompt injection pattern detected in query"

    return ValidatorResult(
        name="policy_checker",
        passed=not found,
        risk_labels=["policy_violation"] if found else [],
        confidence=1.0 - score,
        details=detail or "No policy violations",
    )


def run_bias_check(state: dict, profile: dict) -> ValidatorResult:
    """Detect bias and sensitive-topic indicators (keyword-based)."""
    response = state.get("generation", "")
    found, score, detail = _check_bias(response)

    return ValidatorResult(
        name="bias_detector",
        passed=not found,
        risk_labels=["bias"] if found else [],
        confidence=1.0 - score,
        details=detail or "No bias indicators detected",
    )


def run_claim_confidence(state: dict, profile: dict) -> ValidatorResult:
    """Score confidence in claims made by the response."""
    response = state.get("generation", "")
    found, score, detail = _check_unsupported_claims(response)

    return ValidatorResult(
        name="claim_scorer",
        passed=not found,
        risk_labels=["unsupported_claim"] if found else [],
        confidence=1.0 - score,
        details=detail or "No unsupported claims detected",
    )


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

VALIDATORS = [
    ("pii_detector", run_pii_check),
    ("grounding_verifier", run_grounding_check),
    ("policy_checker", run_policy_check),
    ("bias_detector", run_bias_check),
    ("claim_scorer", run_claim_confidence),
]


def run_parallel_validators(
    state: dict,
    profile: dict,
    max_workers: int = 5,
) -> list[ValidatorResult]:
    """Run all validators concurrently.

    Args:
        state: The current graph state (as dict)
        profile: The active use-case profile
        max_workers: Thread pool size

    Returns:
        List of ValidatorResult from each validator
    """
    results: list[ValidatorResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(fn, state, profile): name
            for name, fn in VALIDATORS
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(ValidatorResult(
                    name=name,
                    passed=False,
                    risk_labels=["validator_error"],
                    confidence=0.0,
                    details=f"Validator error: {str(e)}",
                ))

    return results


def combine_validator_results(results: list[ValidatorResult]) -> dict[str, Any]:
    """Combine individual validator results into a unified validation dict.

    Returns a dict compatible with the state's validation_result field.
    """
    all_labels: list[str] = []
    all_details: list[str] = []
    min_confidence = 1.0
    all_passed = True

    for r in results:
        if not r.passed:
            all_passed = False
        all_labels.extend(r.risk_labels)
        if r.details:
            all_details.append(f"[{r.name}] {r.details}")
        min_confidence = min(min_confidence, r.confidence)

    return {
        "passed": all_passed,
        "confidence": min_confidence,
        "flags": list(set(all_labels)),
        "details": "; ".join(all_details),
        "validator_results": [asdict(r) for r in results],
        "grounded": not any("hallucination" in r.risk_labels for r in results),
        "safe": not any(l in ["policy_violation", "bias"] for r in results for l in r.risk_labels),
        "compliant": not any("compliance" in r.risk_labels for r in results),
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def parallel_validate_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: run all validators in parallel on the verified path."""
    profile = state.get("active_profile", {})
    results = run_parallel_validators(dict(state), profile)
    combined = combine_validator_results(results)

    # Collect all risk labels
    all_labels = list(set(
        label
        for r in results
        for label in r.risk_labels
    ))

    return {
        "validation_result": combined,
        "validator_results": [asdict(r) for r in results],
        "risk_labels": all_labels,
        "audit_log": [
            f"[PARALLEL_VALIDATE] {len(results)} validators, "
            f"passed={combined['passed']}, confidence={combined['confidence']:.2f}, "
            f"labels={all_labels}"
        ],
    }


def validate_fast_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: lightweight PII-only check for the fast path."""
    profile = state.get("active_profile", {})
    pii_result = run_pii_check(dict(state), profile)

    passed = pii_result.passed
    generation = state.get("generation", "")

    # Auto-sanitize PII on fast path if policy is "redact"
    if not passed and profile.get("pii_policy", "redact") == "redact":
        if pii_result.edit_suggestion is not None:
            generation = pii_result.edit_suggestion

    combined = {
        "passed": passed or profile.get("pii_policy") == "allow_internal",
        "confidence": pii_result.confidence,
        "flags": pii_result.risk_labels,
        "details": pii_result.details,
        "validator_results": [asdict(pii_result)],
        "grounded": None,
        "safe": True,
        "compliant": True,
    }

    result: dict[str, Any] = {
        "validation_result": combined,
        "validator_results": [asdict(pii_result)],
        "risk_labels": pii_result.risk_labels,
        "audit_log": [
            f"[VALIDATE_FAST] PII check: passed={pii_result.passed}, "
            f"details={pii_result.details}"
        ],
    }

    if not passed and generation != state.get("generation", ""):
        result["generation"] = generation

    return result
