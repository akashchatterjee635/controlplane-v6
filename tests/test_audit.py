import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from app.utils.audit import create_audit_record, log_audit, load_audit_log
from app.utils.feedback import compute_override_rate

@pytest.fixture
def mock_audit_path(tmp_path):
    test_path = tmp_path / "test_audit_log.jsonl"
    with patch("app.utils.audit._AUDIT_LOG_PATH", test_path):
        yield test_path

def test_create_and_log_audit(mock_audit_path):
    state = {
        "configurable": {"thread_id": "test-123"},
        "use_case": "customer_support",
        "route": "verified",
        "decision": "allow",
        "generation": "Test response"
    }
    
    record = create_audit_record(state)
    assert record.query_id == "test-123"
    assert record.use_case == "customer_support"
    assert record.decision == "allow"
    
    log_audit(record)
    
    loaded = load_audit_log()
    assert len(loaded) == 1
    assert loaded[0]["query_id"] == "test-123"

from unittest.mock import patch

def test_compute_override_rate():
    # Write some dummy records
    records = [
        {"decision": "review", "reviewer_action": "approve", "use_case": "default"},
        {"decision": "review", "reviewer_action": "redact", "use_case": "default"},
        {"decision": "review", "reviewer_action": "deny", "use_case": "default"},
        {"decision": "allow", "reviewer_action": None, "use_case": "default"},
    ]
    
    with patch("app.utils.feedback.load_audit_log", return_value=records):
        stats = compute_override_rate("default")
        assert stats["total_reviews"] == 3
        assert stats["overrides"] == 2
        assert stats["override_rate"] == 2.0 / 3.0
