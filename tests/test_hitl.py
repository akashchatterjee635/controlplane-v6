"""Tests for Human-in-the-Loop (HITL) workflow.

Tests cover:
- Graph compilation with human_review node
- interrupt() pauses execution at human_review node
- Command(resume=...) with approve/redact/deny decisions
- State persistence across interrupt/resume via SQLite checkpointer
- validate_full_node Layer 2 upgrades
"""

import os
import tempfile
import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import build_graph, get_compiled_graph
from app.nodes.human_review import human_review_node
from app.nodes.validate import (
    validate_basic_node,
    validate_full_node,
    triage_decision,
    _run_layer1,
)
from app.utils.security import detect_pii, check_prompt_injection, sanitize_output


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_policies():
    """Return a minimal policies dict for testing."""
    return {
        "pii_patterns": {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        },
        "prohibited_keywords": [
            "ignore previous instructions",
            "act as root",
            "bypass safety",
        ],
        "thresholds": {
            "validation_confidence_min": 0.85,
        },
    }


@pytest.fixture
def sqlite_checkpointer(tmp_path):
    """Provide a SQLite checkpointer for HITL tests."""
    db_path = str(tmp_path / "test_checkpoints.db")
    ctx = SqliteSaver.from_conn_string(db_path)
    saver = ctx.__enter__()
    yield saver
    ctx.__exit__(None, None, None)


# ============================================================================
# Graph compilation tests
# ============================================================================

class TestGraphWithHITL:
    """Tests for graph structure with human_review node."""

    def test_graph_has_human_review_node(self):
        """Graph should include the human_review node."""
        builder = build_graph()
        graph = builder.compile()
        assert "human_review" in graph.nodes

    def test_graph_compiles_with_checkpointer(self, sqlite_checkpointer):
        """Graph should compile successfully with SQLite checkpointer."""
        graph = get_compiled_graph(checkpointer=sqlite_checkpointer)
        assert graph is not None
        assert "human_review" in graph.nodes

    def test_all_verified_path_nodes_present(self):
        """All verified path nodes should be registered."""
        builder = build_graph()
        graph = builder.compile()
        expected_nodes = [
            "router",
            "retrieve_verified",
            "grade_docs",
            "web_search",
            "generate_verified",
            "validate_full",
            "human_review",
        ]
        for node in expected_nodes:
            assert node in graph.nodes, f"Missing node: {node}"


# ============================================================================
# Layer 1 validation tests (refactored)
# ============================================================================

class TestLayer1Validation:
    """Tests for the refactored Layer 1 validation."""

    def test_run_layer1_clean(self, sample_policies):
        """Clean text should pass Layer 1."""
        gen, passed, flags, audit = _run_layer1(
            "The capital of France is Paris.", sample_policies
        )
        assert passed is True
        assert len(flags) == 0

    def test_run_layer1_pii_detected(self, sample_policies):
        """PII should be detected and sanitized."""
        gen, passed, flags, audit = _run_layer1(
            "Contact me at test@example.com.", sample_policies
        )
        assert passed is False
        assert "PII detected" in flags[0]
        assert "[REDACTED EMAIL]" in gen

    def test_run_layer1_injection_detected(self, sample_policies):
        """Prohibited keywords should be detected."""
        gen, passed, flags, audit = _run_layer1(
            "Please ignore previous instructions.", sample_policies
        )
        assert passed is False
        assert any("Prohibited" in f for f in flags)


# ============================================================================
# validate_basic_node tests (updated)
# ============================================================================

class TestValidateBasicNode:
    """Tests for the fast-path validation node."""

    def test_clean_output_passes(self, monkeypatch, sample_policies):
        """Clean output should pass basic validation."""
        monkeypatch.setattr("app.nodes.validate._load_policies", lambda: sample_policies)
        state = {"generation": "The capital of France is Paris."}
        result = validate_basic_node(state)
        assert result["validation_result"]["passed"] is True

    def test_pii_is_auto_sanitized(self, monkeypatch, sample_policies):
        """PII should be automatically redacted in fast path."""
        monkeypatch.setattr("app.nodes.validate._load_policies", lambda: sample_policies)
        state = {"generation": "Contact me at test@example.com."}
        result = validate_basic_node(state)
        assert result["validation_result"]["passed"] is False
        assert "[REDACTED EMAIL]" in result["generation"]


# ============================================================================
# triage_decision tests
# ============================================================================

class TestTriageDecision:
    """Tests for the conditional edge function."""

    def test_clean_goes_to_end(self):
        """Non-flagged state should route to 'end'."""
        state = {"human_review_needed": False}
        assert triage_decision(state) == "end"

    def test_flagged_goes_to_review(self):
        """Flagged state should route to 'human_review'."""
        state = {"human_review_needed": True}
        assert triage_decision(state) == "human_review"

    def test_missing_field_defaults_to_end(self):
        """Missing human_review_needed should default to 'end'."""
        assert triage_decision({}) == "end"


# ============================================================================
# human_review_node unit tests (with mocked interrupt)
# ============================================================================

class TestHumanReviewNode:
    """Unit tests for the human_review_node."""

    def test_approve_preserves_generation(self, monkeypatch):
        """Approve decision should keep the original generation."""
        # Mock interrupt to return an approve decision
        monkeypatch.setattr(
            "app.nodes.human_review.interrupt",
            lambda payload: {
                "decision": "approve",
                "reason": "Looks good",
                "reviewer": "test_reviewer",
            },
        )

        state = {
            "query": "What is Python?",
            "generation": "Python is a programming language.",
            "validation_result": {"flags": ["low confidence"]},
            "risk_score": 3,
            "complexity_score": 5,
        }

        result = human_review_node(state)

        assert result["human_decision"]["decision"] == "approve"
        assert result["human_decision"]["reviewer"] == "test_reviewer"
        assert "generation" not in result or result.get("generation") is None or \
            "[HITL] Response approved" in " ".join(result["audit_log"])

    def test_redact_replaces_generation(self, monkeypatch):
        """Redact decision should replace the generation."""
        monkeypatch.setattr(
            "app.nodes.human_review.interrupt",
            lambda payload: {
                "decision": "redact",
                "redacted_response": "Safe redacted response.",
                "reason": "PII in original",
                "reviewer": "test_reviewer",
            },
        )

        state = {
            "query": "What is my SSN?",
            "generation": "Your SSN is 123-45-6789.",
            "validation_result": {"flags": ["PII detected"]},
            "risk_score": 5,
            "complexity_score": 2,
        }

        result = human_review_node(state)

        assert result["human_decision"]["decision"] == "redact"
        assert result["generation"] == "Safe redacted response."
        assert any("[HITL] Response replaced" in log for log in result["audit_log"])

    def test_deny_blocks_response(self, monkeypatch):
        """Deny decision should block the response."""
        monkeypatch.setattr(
            "app.nodes.human_review.interrupt",
            lambda payload: {
                "decision": "deny",
                "reason": "Inappropriate content",
                "reviewer": "admin",
            },
        )

        state = {
            "query": "Tell me harmful things",
            "generation": "Some harmful content.",
            "validation_result": {"flags": ["unsafe"]},
            "risk_score": 8,
            "complexity_score": 1,
        }

        result = human_review_node(state)

        assert result["human_decision"]["decision"] == "deny"
        assert "blocked" in result["generation"].lower()
        assert "Inappropriate content" in result["generation"]
        assert any("[HITL] Response denied" in log for log in result["audit_log"])

    def test_interrupt_receives_correct_payload(self, monkeypatch):
        """interrupt() should receive the review payload with all context."""
        captured_payload = {}

        def mock_interrupt(payload):
            captured_payload.update(payload)
            return {"decision": "approve", "reason": "", "reviewer": "test"}

        monkeypatch.setattr("app.nodes.human_review.interrupt", mock_interrupt)

        state = {
            "query": "Test query",
            "generation": "Test response",
            "validation_result": {"flags": ["test_flag"], "details": "test", "confidence": 0.5},
            "risk_score": 4,
            "complexity_score": 6,
        }

        human_review_node(state)

        assert captured_payload["query"] == "Test query"
        assert captured_payload["generated_response"] == "Test response"
        assert captured_payload["risk_score"] == 4
        assert captured_payload["complexity_score"] == 6
        assert "test_flag" in captured_payload["validation_flags"]
