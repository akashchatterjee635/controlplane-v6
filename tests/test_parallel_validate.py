import pytest
from app.nodes.parallel_validate import run_parallel_validators, parallel_validate_node

def test_run_parallel_validators_clean():
    state = {"query": "Hello", "generation": "Hi there.", "documents": []}
    profile = {"pii_policy": "redact"}
    results = run_parallel_validators(state, profile)
    
    assert len(results) == 5
    for r in results:
        # grounding will fail due to no docs, but others should pass
        if r.name == "grounding_verifier":
            continue
        assert r.passed

def test_parallel_validate_node():
    state = {"query": "Hello", "generation": "Hi there.", "documents": [], "active_profile": {}}
    result = parallel_validate_node(state)
    
    assert "validation_result" in result
    assert "validator_results" in result
    assert "risk_labels" in result
    
    # Hallucination expected since documents is empty
    assert "hallucination" in result["risk_labels"]
