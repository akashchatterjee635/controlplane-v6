"""Human-in-the-loop review node.

Uses LangGraph's interrupt() to pause graph execution when the validation
pipeline flags a response for human review. A reviewer can then approve,
redact, or deny the response via the API or Streamlit dashboard.
"""

from typing import Any

from langgraph.types import interrupt

from app.state import ControlPlaneState, HumanDecision


def human_review_node(state: ControlPlaneState) -> dict[str, Any]:
    """Pause execution and wait for a human reviewer.

    This node uses LangGraph's ``interrupt()`` to surface the current
    state to a reviewer. The reviewer responds with a decision dict:

        {
            "decision": "approve" | "redact" | "deny",
            "redacted_response": "...",   # only for "redact"
            "reason": "...",
            "reviewer": "reviewer_name"
        }

    After ``resume()``, the graph continues with the updated state.
    """
    query = state.get("query", "")
    generation = state.get("generation", "")
    validation_result = state.get("validation_result", {})
    risk_score = state.get("risk_score", 0)
    complexity_score = state.get("complexity_score", 0)

    # Build the review payload shown to the human reviewer
    review_payload = {
        "query": query,
        "generated_response": generation,
        "validation_flags": validation_result.get("flags", []),
        "validation_details": validation_result.get("details", ""),
        "confidence": validation_result.get("confidence", 0),
        "risk_score": risk_score,
        "complexity_score": complexity_score,
    }

    # ---- INTERRUPT: execution pauses here ----
    human_input = interrupt(review_payload)
    # ---- RESUME: execution continues here with the reviewer's decision ----

    # Parse the reviewer's decision
    decision = human_input.get("decision", "approve")
    redacted_response = human_input.get("redacted_response")
    reason = human_input.get("reason", "")
    reviewer = human_input.get("reviewer", "unknown")

    human_decision = HumanDecision(
        decision=decision,
        redacted_response=redacted_response,
        reason=reason,
        reviewer=reviewer,
    )

    result: dict[str, Any] = {
        "human_decision": {
            "decision": human_decision.decision,
            "redacted_response": human_decision.redacted_response,
            "reason": human_decision.reason,
            "reviewer": human_decision.reviewer,
        },
        "audit_log": [
            f"[HITL] Reviewer={reviewer}, decision={decision}, reason={reason}"
        ],
    }

    # Apply the decision
    if decision == "redact" and redacted_response:
        result["generation"] = redacted_response
        result["audit_log"].append("[HITL] Response replaced with redacted version.")
    elif decision == "deny":
        result["generation"] = (
            "This response has been blocked by a human reviewer. "
            f"Reason: {reason}"
        )
        result["audit_log"].append("[HITL] Response denied by reviewer.")
    else:
        # "approve" — keep generation as-is
        result["audit_log"].append("[HITL] Response approved by reviewer.")

    return result
