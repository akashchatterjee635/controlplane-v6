"""Audit logging utility.

Writes structured JSONL audit logs for every graph execution.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from app.state import AuditRecord

_AUDIT_LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "data/audit_log.jsonl"))

def _hash_output(text: str) -> str:
    """Create a short hash of the final output for verification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def create_audit_record(state: dict[str, Any]) -> AuditRecord:
    """Create an AuditRecord from the final graph state."""
    
    # Cost handling
    cost_tracker = state.get("cost_tracker", {})
    cost_usd = cost_tracker.get("estimated_cost_usd", 0.0)
    latency_ms = cost_tracker.get("total_latency_ms", 0.0)
    
    # HITL handling
    human_decision = state.get("human_decision", {})
    reviewer_action = human_decision.get("decision")
    
    # Risk handling
    val_results = state.get("validator_results", [])
    validators_triggered = [r.get("name", "unknown") for r in val_results if not r.get("passed", True)]
    
    return AuditRecord(
        query_id=state.get("configurable", {}).get("thread_id", "unknown"),
        timestamp=cost_tracker.get("start_time", 0.0), # Simplification
        use_case=state.get("use_case", "default"),
        route=state.get("route", "unknown"),
        complexity_score=state.get("complexity_score", 0),
        risk_score=state.get("risk_score", 0),
        risk_labels=state.get("risk_labels", []),
        validators_triggered=validators_triggered,
        decision=state.get("decision", "unknown"),
        decision_reasoning=state.get("decision_reasoning", ""),
        reviewer_action=reviewer_action,
        final_output_hash=_hash_output(state.get("generation", "")),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_id=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        policy_version='1.0',
        trace_id=state.get('configurable', {}).get('thread_id', '') + '_' + str(int(cost_tracker.get('start_time', 0))),
    )

def log_audit(record: AuditRecord) -> None:
    """Append an AuditRecord to the JSONL log file."""
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict, handling any non-serializable fields if necessary
    import dataclasses
    record_dict = dataclasses.asdict(record)
    
    with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict) + "\n")

def load_audit_log() -> list[dict[str, Any]]:
    """Load all records from the audit log."""
    if not _AUDIT_LOG_PATH.exists():
        return []
    
    records = []
    with open(_AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
