"""Multi-label risk classifier.

Produces overlapping risk labels for a query/response pair.
Categories: hallucination, pii_leak, bias, policy_violation,
            unsupported_claim, compliance.

Uses deterministic keyword/pattern matching (fast, no LLM calls).
"""

import re
from dataclasses import asdict
from typing import Any

from app.state import RiskAssessment

# ---------------------------------------------------------------------------
# Keyword / pattern banks
# ---------------------------------------------------------------------------

BIAS_KEYWORDS = [
    "race", "gender", "ethnicity", "religion", "disability",
    "sexual orientation", "age discrimination", "stereotype",
    "demographic", "minority", "marginalized",
]

COMPLIANCE_TOPICS = [
    "medical advice", "legal counsel", "financial recommendation",
    "medication dosage", "diagnosis", "treatment plan",
    "investment advice", "legal liability", "hipaa", "gdpr",
    "personal health", "prescription",
]

UNSUPPORTED_CLAIM_INDICATORS = [
    "studies show", "research proves", "according to experts",
    "it is well known", "everyone knows", "statistics show",
    "scientifically proven", "100%", "guaranteed",
]

INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|your)\s+(instructions|rules|prompt)",
    r"act\s+as\s+(root|admin|god)",
    r"bypass\s+safety",
    r"disregard\s+all\s+rules",
    r"pretend\s+you\s+are",
    r"jailbreak",
    r"system\s+prompt",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}


# ---------------------------------------------------------------------------
# Individual classifiers
# ---------------------------------------------------------------------------

def _check_pii(text: str) -> tuple[bool, float, str]:
    """Check for PII patterns in text."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pii_type)
    if found:
        return True, min(1.0, len(found) * 0.4), f"PII detected: {', '.join(found)}"
    return False, 0.0, ""


def _check_bias(text: str) -> tuple[bool, float, str]:
    """Check for bias/sensitive-topic indicators."""
    text_lower = text.lower()
    hits = [kw for kw in BIAS_KEYWORDS if kw in text_lower]
    if hits:
        score = min(1.0, len(hits) * 0.3)
        return True, score, f"Bias indicators: {', '.join(hits[:3])}"
    return False, 0.0, ""


def _check_compliance(text: str) -> tuple[bool, float, str]:
    """Check for compliance/regulatory concerns."""
    text_lower = text.lower()
    hits = [topic for topic in COMPLIANCE_TOPICS if topic in text_lower]
    if hits:
        score = min(1.0, len(hits) * 0.4)
        return True, score, f"Compliance topics: {', '.join(hits[:3])}"
    return False, 0.0, ""


def _check_policy_violation(text: str) -> tuple[bool, float, str]:
    """Check for injection patterns / policy violations."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, 0.9, "Policy violation: injection pattern detected"
    return False, 0.0, ""


def _check_unsupported_claims(text: str) -> tuple[bool, float, str]:
    """Check for unsupported/unverifiable claims."""
    text_lower = text.lower()
    hits = [ind for ind in UNSUPPORTED_CLAIM_INDICATORS if ind in text_lower]
    if hits:
        score = min(1.0, len(hits) * 0.3)
        return True, score, f"Unsupported claims: {', '.join(hits[:3])}"
    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_risk(
    query: str,
    response: str = "",
    context_docs: list[str] | None = None,
) -> RiskAssessment:
    """Produce multi-label risk assessment for a query/response pair.

    Checks both the query and the response for overlapping risk categories.

    Args:
        query: The user's query
        response: The generated response (empty during pre-generation checks)
        context_docs: Optional list of context document texts for grounding checks

    Returns:
        RiskAssessment with labels, severity, per-label scores, and details
    """
    combined_text = f"{query} {response}"
    labels: list[str] = []
    scores: dict[str, float] = {}
    details_parts: list[str] = []

    # Check each category
    checkers = [
        ("pii_leak", _check_pii),
        ("bias", _check_bias),
        ("compliance", _check_compliance),
        ("policy_violation", _check_policy_violation),
        ("unsupported_claim", _check_unsupported_claims),
    ]

    for label, checker in checkers:
        found, score, detail = checker(combined_text)
        if found:
            labels.append(label)
            scores[label] = score
            if detail:
                details_parts.append(detail)

    # Simple grounding check: if response exists and no docs, flag hallucination risk
    if response and context_docs is not None and len(context_docs) == 0:
        labels.append("hallucination")
        scores["hallucination"] = 0.8
        details_parts.append("No context documents for grounding")

    # Determine overall severity
    if not labels:
        severity = "low"
    elif any(scores.get(l, 0) >= 0.8 for l in labels):
        severity = "critical" if "policy_violation" in labels else "high"
    elif len(labels) >= 3:
        severity = "high"
    elif len(labels) >= 2:
        severity = "medium"
    else:
        severity = "medium" if max(scores.values(), default=0) >= 0.5 else "low"

    return RiskAssessment(
        labels=labels,
        severity=severity,
        scores=scores,
        details="; ".join(details_parts) if details_parts else "No risks detected",
    )


def risk_assessment_to_dict(assessment: RiskAssessment) -> dict[str, Any]:
    """Serialize a RiskAssessment to a dict for state storage."""
    return asdict(assessment)
