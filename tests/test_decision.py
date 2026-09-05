from app.nodes.decision import decision_node


def test_decision_allow():
    state = {
        "active_profile": {"human_review_threshold": 0.5, "pii_policy": "redact"},
        "validator_results": [{"passed": True, "confidence": 0.9, "risk_labels": []}],
        "generation": "All good."
    }
    result = decision_node(state)
    assert result["decision"] == "allow"

def test_decision_block_on_policy_violation():
    state = {
        "active_profile": {"human_review_threshold": 0.5},
        "validator_results": [{"passed": False, "confidence": 0.1, "risk_labels": ["policy_violation"]}],
        "generation": "Bad stuff."
    }
    result = decision_node(state)
    assert result["decision"] == "block"

def test_decision_edit_pii_redact():
    state = {
        "active_profile": {"pii_policy": "redact"},
        "validator_results": [{"name": "pii_detector", "passed": False, "confidence": 0.1, "risk_labels": ["pii_leak"], "edit_suggestion": "[REDACTED]"}],
        "generation": "My SSN is 123-45-6789."
    }
    result = decision_node(state)
    assert result["decision"] == "edit"
    assert result["generation"] == "[REDACTED]"

def test_decision_block_pii():
    state = {
        "active_profile": {"pii_policy": "block"},
        "validator_results": [{"name": "pii_detector", "passed": False, "confidence": 0.1, "risk_labels": ["pii_leak"]}],
        "generation": "My SSN is 123-45-6789."
    }
    result = decision_node(state)
    assert result["decision"] == "block"

def test_decision_review_low_confidence():
    state = {
        "active_profile": {"human_review_threshold": 0.8},
        "validator_results": [{"passed": True, "confidence": 0.5, "risk_labels": []}],
        "generation": "Maybe?"
    }
    result = decision_node(state)
    assert result["decision"] == "review"

def test_decision_flag_bias():
    state = {
        "active_profile": {"human_review_threshold": 0.5},
        "validator_results": [{"passed": False, "confidence": 0.9, "risk_labels": ["bias"]}],
        "generation": "Some bias."
    }
    result = decision_node(state)
    assert result["decision"] == "flag"
    assert "[WARNING:" in result["generation"]
