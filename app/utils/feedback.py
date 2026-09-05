"""Feedback loop and threshold tuning analytics.

Analyzes the audit log to compute human override rates and suggest
adjustments to profile thresholds.
"""

from typing import Any

from app.policies.profile_loader import load_profile
from app.utils.audit import load_audit_log


def compute_override_rate(use_case: str | None = None) -> dict[str, Any]:
    """Compute the human override rate from the audit log.
    
    An override is when the decision layer triggers REVIEW, but the human
    reviewer takes an action other than "approve" (e.g., "redact" or "deny").
    Or, conversely, when they approve something that was flagged.
    
    Args:
        use_case: Optional filter by use case.
        
    Returns:
        Dict with total reviews, overrides, and override rate.
    """
    records = load_audit_log()
    
    if use_case:
        records = [r for r in records if r.get("use_case") == use_case]
        
    reviews = [r for r in records if r.get("decision") == "review"]
    total_reviews = len(reviews)
    
    if total_reviews == 0:
        return {"total_reviews": 0, "overrides": 0, "override_rate": 0.0}
        
    overrides = sum(1 for r in reviews if r.get("reviewer_action") in ("redact", "deny"))
    
    return {
        "total_reviews": total_reviews,
        "overrides": overrides,
        "override_rate": overrides / total_reviews
    }

def suggest_threshold_adjustments(use_case: str) -> list[str]:
    """Suggest policy threshold adjustments based on audit feedback."""
    stats = compute_override_rate(use_case)
    rate = stats["override_rate"]
    total = stats["total_reviews"]
    
    if total < 10:
        return ["Not enough data to suggest threshold adjustments (need at least 10 reviews)."]
        
    suggestions = []
    profile = load_profile(use_case)
    threshold = profile.get("human_review_threshold", 0.65)
    
    if rate > 0.4:
        suggestions.append(
            f"High override rate ({rate:.1%}). Reviewers are frequently modifying or denying flagged responses. "
            f"Consider RAISE human_review_threshold (current: {threshold}) to route more queries to humans."
        )
    elif rate < 0.05:
        suggestions.append(
            f"Low override rate ({rate:.1%}). Reviewers are almost always approving flagged responses. "
            f"Consider LOWER human_review_threshold (current: {threshold}) to reduce human bottleneck."
        )
    else:
        suggestions.append(
            f"Override rate ({rate:.1%}) is within normal bounds. Current threshold ({threshold}) appears well-calibrated."
        )
        
    # Could also add checks for fast/verified routing overrides
    
    return suggestions
