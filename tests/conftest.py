"""Shared test fixtures for ControlPlane v6."""

import os
import sys
import tempfile

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_policies() -> dict:
    """Return a minimal policies dict for testing without file I/O."""
    return {
        "thresholds": {
            "complexity_fast_max": 4,
            "risk_fast_max": 2,
            "validation_confidence_min": 0.85,
            "grading_relevance_min": 0.7,
            "retrieval_top_k": 5,
        },
        "scoring": {
            "query_length_buckets": {
                "short": 50,
                "medium": 150,
                "long": 300,
            },
            "constraint_keywords": [
                "must",
                "should not",
                "exactly",
                "at least",
                "no more than",
                "between",
                "only if",
                "unless",
                "require",
                "mandatory",
            ],
            "reasoning_indicators": [
                "compare",
                "analyze",
                "evaluate",
                "trade-off",
                "pros and cons",
                "implications",
                "recommend",
                "why",
                "how does",
                "what if",
            ],
        },
        "sensitive_topics": [
            "medical advice",
            "legal counsel",
            "financial recommendation",
            "personal health",
            "medication dosage",
            "investment advice",
            "legal liability",
            "diagnosis",
            "treatment plan",
        ],
        "prohibited_keywords": [
            "ignore previous instructions",
            "act as root",
            "bypass safety",
            "disregard all rules",
            "pretend you are",
            "ignore your system prompt",
            "jailbreak",
        ],
        "pii_patterns": {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        },
        "cost": {
            "gpt_4o_mini": {
                "prompt_per_1k": 0.00015,
                "completion_per_1k": 0.0006,
            },
        },
    }


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Provide a temporary directory for ChromaDB."""
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir()
    return str(chroma_dir)
