"""Decision layer.

Maps validator results and risk profiles to explicit outcomes:
  ALLOW  : Safe, grounded, low risk → return as-is
  EDIT   : Redact PII or soften unsupported claims
  FLAG   : Add warning/confidence note to response
  REVIEW : Send to human reviewer
  BLOCK  : Unsafe or prohibited → block entirely
"""

from typing import Any, Literal
from app.state import ControlPlaneState
from app.utils.security import sanitize_output

def decision_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: evaluate validator results and output a decision."""
    profile = state.get("active_profile", {})
    val_results = state.get("validator_results", [])
    
    # Collect data from validators
    all_labels = set(label for r in val_results for label in r.get("risk_labels", []))
    min_confidence = min((r.get("confidence", 1.0) for r in val_results), default=1.0)
    
    # Defaults
    decision = "allow"
    reasoning = "All checks passed"
    generation = state.get("generation", "")
    
    # BLOCK rules
    if "policy_violation" in all_labels:
        violation_action = profile.get("policy_violation_action", "block")
        if violation_action == "review":
            decision = "review"
            reasoning = "Policy violation detected (routed to review for internal testing)"
            updates = {
                "decision": decision,
                "decision_reasoning": reasoning,
                "audit_log": ["[DECISION] Outcome: REVIEW (Policy violation)"]
            }
            if generation != state.get("generation", ""):
                updates["generation"] = generation
            return updates
        else:
            return {
                "decision": "block",
                "decision_reasoning": "Policy violation detected",
                "audit_log": ["[DECISION] Outcome: BLOCK (Policy violation)"]
            }
        
    # PII rules
    if "pii_leak" in all_labels:
        pii_policy = profile.get("pii_policy", "redact")
        if pii_policy == "block":
            return {
                "decision": "block",
                "decision_reasoning": "PII detected (policy=block)",
                "audit_log": ["[DECISION] Outcome: BLOCK (PII detected)"]
            }
        elif pii_policy == "redact":
            decision = "edit"
            reasoning = "PII detected (policy=redact)"
            # Find the PII validator's suggestion
            pii_res = next((r for r in val_results if r.get("name") == "pii_detector"), None)
            if pii_res and pii_res.get("edit_suggestion") is not None:
                generation = pii_res["edit_suggestion"]

    # REVIEW rules
    review_threshold = profile.get("human_review_threshold", 0.65)
    if decision not in ("block", "edit"): # Don't override block/edit with review yet, or maybe do?
        # Typically REVIEW takes precedence if confidence is low, unless we blocked.
        if min_confidence < review_threshold:
            decision = "review"
            reasoning = f"Confidence ({min_confidence:.2f}) below threshold ({review_threshold})"
            
    # FLAG rules
    if decision == "allow":
        if "unsupported_claim" in all_labels and profile.get("hallucination_check") == "strict":
            decision = "flag"
            reasoning = "Unsupported claim detected (strict policy)"
            generation = f"[WARNING: Contains unsupported claims]\n\n{generation}"
        elif "bias" in all_labels:
            decision = "flag"
            reasoning = "Bias or sensitive topic detected"
            generation = f"[WARNING: Contains sensitive demographic topics]\n\n{generation}"

    updates: dict[str, Any] = {
        "decision": decision,
        "decision_reasoning": reasoning,
        "audit_log": [f"[DECISION] Outcome: {decision.upper()} ({reasoning})"]
    }
    
    # If the response was modified, update it in state
    if generation != state.get("generation", ""):
        updates["generation"] = generation
        
    return updates


def decision_routing(state: ControlPlaneState) -> str:
    """Conditional edge routing based on decision outcome."""
    decision = state.get("decision", "allow")
    
    if decision == "block":
        return "block_response"
    elif decision == "review":
        return "human_review"
    else:
        # allow, edit, flag all continue to end. Their edits are already applied.
        return "end"

def block_response_node(state: ControlPlaneState) -> dict[str, Any]:
    """Replaces the generation with a canned block message."""
    return {
        "generation": "This request was blocked due to safety or policy violations.",
        "audit_log": ["[BLOCK] Replaced generation with block message"]
    }
