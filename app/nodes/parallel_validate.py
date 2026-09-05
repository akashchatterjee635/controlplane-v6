"""Parallel validation layer with cascade architecture.

Cascade Design:
  Layer 0: Regex/keyword checks (cheap, deterministic) — always runs
  Layer 1: LLM semantic judge (called only when Layer 0 flags) — confirms/overrides
  Layer 2: Human review (via HITL node) — only when ambiguous

Validators:
  1. PII detector           — regex patterns for SSN, email, credit card, phone
  2. Grounding verifier     — claim-level attribution via LLM entailment
  3. Policy checker         — injection patterns + prohibited keywords
  4. Bias detector          — keyword + LLM semantic override
  5. Compliance checker     — keyword + LLM semantic override
  6. Claim confidence       — keyword + LLM semantic override

Each validator returns a ValidatorResult. Results are combined for the decision layer.
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
# LLM Semantic Judge (Layer 1)
# ---------------------------------------------------------------------------

def _llm_judge(text: str, question: str) -> bool:
    """Layer 1 semantic judge — confirms or overrides Layer 0 keyword flags.

    Args:
        text: The text that Layer 0 flagged.
        question: The yes/no question to ask the LLM about the text.

    Returns:
        True if the LLM confirms the flag is a true positive.
        False if the LLM says it's a false positive (override).
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        judge = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=10,
        )
        sys_msg = SystemMessage(content=(
            "You are a content safety judge in an AI governance system. "
            "Answer the following question about the given text with EXACTLY "
            "YES or NO. Nothing else."
        ))
        user_msg = HumanMessage(content=f"Text: {text}\n\nQuestion: {question}")
        response = judge.invoke([sys_msg, user_msg]).content.strip().upper()
        return "YES" in response
    except Exception:
        # If LLM fails, treat Layer 0 flag as true positive (fail-safe)
        return True


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
    """Check if the response is grounded in retrieved documents.

    Uses claim-level attribution via LLM entailment:
      1. Extract atomic claims from the response.
      2. For each claim, check if source documents support it.
      3. Compute: Groundedness = supported_claims / total_claims
    """
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

    # Combine document texts for entailment checking
    doc_text = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in documents[:5]  # limit to top 5 for cost
    )

    # Use LLM to extract claims and check support
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    hallucination_check = profile.get("hallucination_check", "medium")

    try:
        judge = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=200,
        )

        # Step 1+2 combined: extract claims and check support in one call
        sys_msg = SystemMessage(content=(
            "You are a grounding verifier. Given source documents and an AI response, "
            "determine what fraction of factual claims in the response are supported by "
            "the source documents.\n\n"
            "Reply with a JSON object (nothing else):\n"
            '{"supported": <int>, "total": <int>, "unsupported_claims": [<string>, ...]}\n\n'
            "Only count substantive factual claims, not hedges or meta-statements."
        ))
        user_msg = HumanMessage(content=(
            f"Source Documents:\n{doc_text[:3000]}\n\n"
            f"AI Response:\n{response[:2000]}\n\n"
            "Analyze grounding:"
        ))

        result_text = judge.invoke([sys_msg, user_msg]).content.strip()

        # Parse the JSON response
        import json
        # Handle potential markdown code blocks
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_data = json.loads(result_text)

        supported = result_data.get("supported", 0)
        total = result_data.get("total", 1)
        unsupported = result_data.get("unsupported_claims", [])

        if total == 0:
            groundedness = 1.0
        else:
            groundedness = supported / total

        # Apply threshold based on hallucination_check level
        thresholds = {"strict": 0.7, "medium": 0.5, "relaxed": 0.3}
        threshold = thresholds.get(hallucination_check, 0.5)

        passed = groundedness >= threshold
        labels = [] if passed else ["hallucination"]

        detail = (
            f"Claim-level grounding: {supported}/{total} claims supported "
            f"(groundedness={groundedness:.2f}, threshold={threshold})"
        )
        if unsupported:
            detail += f" | Unsupported: {'; '.join(unsupported[:3])}"

        return ValidatorResult(
            name="grounding_verifier",
            passed=passed,
            risk_labels=labels,
            confidence=groundedness,
            details=detail,
        )

    except Exception as e:
        # Fallback to simple word-overlap if LLM fails
        doc_text_lower = " ".join(
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in documents
        ).lower()
        response_words = set(response.lower().split())
        substantive = {w for w in response_words if len(w) > 4}
        if not substantive:
            return ValidatorResult(
                name="grounding_verifier", passed=True,
                confidence=0.8, details="Response too short for grounding check",
            )
        overlap = sum(1 for w in substantive if w in doc_text_lower)
        overlap_ratio = overlap / len(substantive)
        thresholds = {"strict": 0.3, "medium": 0.2, "relaxed": 0.1}
        threshold = thresholds.get(hallucination_check, 0.2)
        passed = overlap_ratio >= threshold
        labels = [] if passed else ["hallucination"]
        return ValidatorResult(
            name="grounding_verifier", passed=passed,
            risk_labels=labels,
            confidence=min(1.0, overlap_ratio + 0.3),
            details=f"Fallback word-overlap: {overlap_ratio:.2f} (LLM error: {e})",
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
    """Detect bias — cascade: Layer 0 keyword → Layer 1 LLM semantic judge.

    Layer 0 catches keywords like 'race', 'gender', 'disability'.
    Layer 1 confirms whether the usage is actually biased or merely
    discusses bias-related topics neutrally (e.g., anti-discrimination policies).
    """
    response = state.get("generation", "")
    found, score, detail = _check_bias(response)

    if not found:
        return ValidatorResult(
            name="bias_detector", passed=True,
            confidence=1.0, details="No bias indicators detected",
        )

    # Layer 1: LLM semantic judge to confirm/override
    is_true_positive = _llm_judge(
        response[:2000],
        "Does this text express actual bias, prejudice, stereotyping, or "
        "discriminatory views? Answer NO if it merely discusses bias-related "
        "topics in a neutral, educational, or anti-discrimination context."
    )

    if not is_true_positive:
        # Layer 1 override: false positive
        return ValidatorResult(
            name="bias_detector", passed=True,
            risk_labels=[],
            confidence=0.85,
            details=f"Layer 0 flagged ({detail}), Layer 1 override: neutral/educational context",
        )

    return ValidatorResult(
        name="bias_detector",
        passed=False,
        risk_labels=["bias"],
        confidence=1.0 - score,
        details=f"Confirmed bias: {detail}",
    )


def run_claim_confidence(state: dict, profile: dict) -> ValidatorResult:
    """Score unsupported claims — cascade: Layer 0 keyword → Layer 1 LLM judge.

    Layer 0 catches phrases like 'studies show', 'guaranteed', '100%'.
    Layer 1 confirms whether the claims are actually unsupported or
    properly cited/hedged.
    """
    response = state.get("generation", "")
    found, score, detail = _check_unsupported_claims(response)

    if not found:
        return ValidatorResult(
            name="claim_scorer", passed=True,
            confidence=1.0, details="No unsupported claims detected",
        )

    # Layer 1: LLM semantic judge
    is_true_positive = _llm_judge(
        response[:2000],
        "Does this text contain factual claims that are presented as definitive "
        "facts but lack proper citation, sourcing, or hedging? Answer NO if "
        "the claims are properly hedged, cited, or clearly stated as opinions."
    )

    if not is_true_positive:
        return ValidatorResult(
            name="claim_scorer", passed=True,
            risk_labels=[],
            confidence=0.85,
            details=f"Layer 0 flagged ({detail}), Layer 1 override: claims properly hedged/cited",
        )

    return ValidatorResult(
        name="claim_scorer",
        passed=False,
        risk_labels=["unsupported_claim"],
        confidence=1.0 - score,
        details=f"Confirmed unsupported claims: {detail}",
    )


def run_compliance_check(state: dict, profile: dict) -> ValidatorResult:
    """Check compliance — cascade: Layer 0 keyword → Layer 1 LLM judge.

    Layer 0 catches phrases like 'medical advice', 'medication dosage', 'HIPAA'.
    Layer 1 confirms whether the text actually provides regulated advice or
    merely discusses these topics educationally.
    """
    query = state.get("query", "")
    response = state.get("generation", "")
    combined = f"{query} {response}"
    
    from app.utils.risk_classifier import _check_compliance
    found, score, detail = _check_compliance(combined)

    if not found:
        return ValidatorResult(
            name="compliance_checker", passed=True,
            confidence=1.0, details="No compliance concerns detected",
        )

    # Layer 1: LLM semantic judge
    is_true_positive = _llm_judge(
        combined[:2000],
        "Does this text provide specific medical, legal, or financial advice "
        "that should only come from a licensed professional? Answer NO if it "
        "merely discusses these topics in an educational, general, or policy context."
    )

    if not is_true_positive:
        return ValidatorResult(
            name="compliance_checker", passed=True,
            risk_labels=[],
            confidence=0.85,
            details=f"Layer 0 flagged ({detail}), Layer 1 override: educational/general context",
        )

    return ValidatorResult(
        name="compliance_checker",
        passed=False,
        risk_labels=["compliance"],
        confidence=1.0 - score,
        details=f"Confirmed compliance concern: {detail}",
    )


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

VALIDATORS = [
    ("pii_detector", run_pii_check),
    ("grounding_verifier", run_grounding_check),
    ("policy_checker", run_policy_check),
    ("bias_detector", run_bias_check),
    ("compliance_checker", run_compliance_check),
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
