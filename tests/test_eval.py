"""Tests for evaluation metrics."""

import pytest
from eval.metrics import (
    answer_similarity,
    keyword_recall,
    risk_routing_accuracy,
    pii_leak_check,
    groundedness_score,
    score_single_query,
    aggregate_report,
    load_dataset,
)


# ============================================================================
# answer_similarity
# ============================================================================

class TestAnswerSimilarity:
    def test_identical_strings(self):
        assert answer_similarity("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        score = answer_similarity("Hello World", "hello world")
        assert score == 1.0

    def test_completely_different(self):
        score = answer_similarity("abc", "xyz")
        assert score < 0.5

    def test_partial_match(self):
        score = answer_similarity(
            "Docker is a containerization platform",
            "Docker is a platform for containerization",
        )
        assert score > 0.6

    def test_empty_strings(self):
        assert answer_similarity("", "something") == 0.0
        assert answer_similarity("something", "") == 0.0
        assert answer_similarity("", "") == 0.0


# ============================================================================
# keyword_recall
# ============================================================================

class TestKeywordRecall:
    def test_full_recall(self):
        expected = "Docker containers platform orchestration"
        generated = "Docker is a containers platform for orchestration."
        assert keyword_recall(generated, expected) == 1.0

    def test_partial_recall(self):
        expected = "Docker containers platform orchestration"
        generated = "Docker is a platform."
        score = keyword_recall(generated, expected)
        assert 0.3 < score < 0.8

    def test_zero_recall(self):
        expected = "Kubernetes pods services deployment"
        generated = "The weather is nice today."
        score = keyword_recall(generated, expected)
        assert score == 0.0

    def test_empty_expected(self):
        assert keyword_recall("anything", "") == 1.0

    def test_empty_generated(self):
        assert keyword_recall("", "important keywords here") == 0.0


# ============================================================================
# risk_routing_accuracy
# ============================================================================

class TestRiskRoutingAccuracy:
    def test_low_low_should_be_fast(self):
        assert risk_routing_accuracy("fast", "low", "low") is True
        assert risk_routing_accuracy("verified", "low", "low") is False

    def test_high_risk_should_be_verified(self):
        assert risk_routing_accuracy("verified", "high", "low") is True
        assert risk_routing_accuracy("fast", "high", "low") is False

    def test_high_complexity_should_be_verified(self):
        assert risk_routing_accuracy("verified", "low", "high") is True

    def test_medium_risk_should_be_verified(self):
        assert risk_routing_accuracy("verified", "medium", "low") is True


# ============================================================================
# pii_leak_check
# ============================================================================

class TestPiiLeakCheck:
    def test_clean_response(self):
        assert pii_leak_check("The capital of France is Paris.") == []

    def test_ssn_detected(self):
        leaks = pii_leak_check("Your SSN is 123-45-6789.")
        assert "ssn" in leaks

    def test_email_detected(self):
        leaks = pii_leak_check("Contact test@example.com for help.")
        assert "email" in leaks

    def test_credit_card_detected(self):
        leaks = pii_leak_check("Card: 4111-1111-1111-1111")
        assert "credit_card" in leaks

    def test_redacted_not_detected(self):
        assert pii_leak_check("Your info is [REDACTED SSN].") == []


# ============================================================================
# groundedness_score
# ============================================================================

class TestGroundednessScore:
    def test_grounded_returns_confidence(self):
        val = {"grounded": True, "confidence": 0.95}
        assert groundedness_score(val) == 0.95

    def test_ungrounded_returns_zero(self):
        val = {"grounded": False, "confidence": 0.8}
        assert groundedness_score(val) == 0.0

    def test_none_input(self):
        assert groundedness_score(None) is None


# ============================================================================
# score_single_query
# ============================================================================

class TestScoreSingleQuery:
    def test_basic_scoring(self):
        query_data = {
            "id": "q01",
            "query": "What is Docker?",
            "expected_answer": "Docker is a containerization platform.",
            "expected_complexity": "low",
            "expected_risk": "low",
            "category": "factual",
        }
        result = {
            "generation": "Docker is a platform for containerization.",
            "route": "fast",
            "cost_tracker": {"estimated_cost_usd": 0.001},
        }
        score = score_single_query(query_data, result)

        assert score["id"] == "q01"
        assert score["route"] == "fast"
        assert score["routing_correct"] is True
        assert score["similarity"] > 0.5
        assert score["pii_clean"] is True

    def test_pii_in_response(self):
        query_data = {
            "id": "q12",
            "query": "Check balance",
            "expected_answer": "Cannot process.",
            "expected_complexity": "low",
            "expected_risk": "high",
            "category": "pii",
        }
        result = {
            "generation": "Your card 4111-1111-1111-1111 has balance $100.",
            "route": "verified",
        }
        score = score_single_query(query_data, result)
        assert score["pii_clean"] is False
        assert "credit_card" in score["pii_leaks"]


# ============================================================================
# aggregate_report
# ============================================================================

class TestAggregateReport:
    def test_basic_aggregation(self):
        scores = [
            {
                "id": "q01", "query": "test", "category": "factual",
                "route": "fast", "routing_correct": True,
                "similarity": 0.8, "keyword_recall": 0.7,
                "groundedness": 0.9, "pii_leaks": [], "pii_clean": True,
                "cost_usd": 0.001, "status": "complete",
            },
            {
                "id": "q02", "query": "test2", "category": "factual",
                "route": "verified", "routing_correct": True,
                "similarity": 0.6, "keyword_recall": 0.5,
                "groundedness": None, "pii_leaks": [], "pii_clean": True,
                "cost_usd": 0.002, "status": "complete",
            },
        ]
        report = aggregate_report(scores)

        assert report["total_queries"] == 2
        assert report["avg_similarity"] == 0.7
        assert report["routing_accuracy"] == 1.0
        assert report["pii_clean_rate"] == 1.0
        assert report["total_cost_usd"] == 0.003
        assert "factual" in report["category_breakdown"]

    def test_empty_scores(self):
        report = aggregate_report([])
        assert "error" in report


# ============================================================================
# load_dataset
# ============================================================================

class TestLoadDataset:
    def test_loads_eval_dataset(self):
        dataset = load_dataset()
        assert len(dataset) == 20
        assert all("id" in q for q in dataset)
        assert all("query" in q for q in dataset)
        assert all("expected_answer" in q for q in dataset)

    def test_dataset_has_diversity(self):
        dataset = load_dataset()
        complexities = {q["expected_complexity"] for q in dataset}
        risks = {q["expected_risk"] for q in dataset}
        assert "low" in complexities
        assert "high" in complexities
        assert "low" in risks
        assert "high" in risks
