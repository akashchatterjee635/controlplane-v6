"""ControlPlane v6 — Evaluation Metrics.

Scoring functions for evaluating RAG response quality:
- answer_similarity: Fuzzy string similarity between generated and expected answers
- keyword_recall: Fraction of expected keywords present in the response
- risk_routing_accuracy: Whether the router made the correct path decision
- pii_leak_check: Whether PII appears in the final output
- aggregate_report: Combine all metrics into a summary table
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Individual Scoring Functions
# ---------------------------------------------------------------------------

def answer_similarity(generated: str, expected: str) -> float:
    """Compute fuzzy string similarity between generated and expected answers.

    Uses SequenceMatcher ratio (0.0 to 1.0). Case-insensitive.
    """
    if not generated or not expected:
        return 0.0
    return SequenceMatcher(
        None, generated.lower().strip(), expected.lower().strip()
    ).ratio()


def keyword_recall(generated: str, expected: str, min_word_len: int = 4) -> float:
    """Fraction of 'important' words from the expected answer found in the generated answer.

    Filters out short common words (< min_word_len characters).
    """
    if not expected:
        return 1.0
    if not generated:
        return 0.0

    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "that",
        "this", "with", "they", "from", "been", "were", "more", "when",
        "will", "each", "than", "them", "some", "also", "into", "such",
        "most", "only", "other", "which", "their", "about",
    }

    expected_words = {
        w.lower()
        for w in re.findall(r"\b\w+\b", expected)
        if len(w) >= min_word_len and w.lower() not in stop_words
    }

    if not expected_words:
        return 1.0

    generated_lower = generated.lower()
    hits = sum(1 for w in expected_words if w in generated_lower)
    return hits / len(expected_words)


def risk_routing_accuracy(
    route: str,
    expected_risk: str,
    expected_complexity: str,
) -> bool:
    """Check whether the router made the correct path decision.

    Rules:
    - expected_risk == 'high' → should route to 'verified'
    - expected_complexity == 'high' → should route to 'verified'
    - Both 'low' → should route to 'fast'
    """
    should_verify = expected_risk in ("high", "medium") or expected_complexity == "high"
    if should_verify:
        return route == "verified"
    else:
        return route == "fast"


def pii_leak_check(response: str) -> list[str]:
    """Check if any PII patterns appear in the final response.

    Returns a list of PII types found (empty = clean).
    """
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    }
    leaks = []
    for pii_type, pattern in patterns.items():
        if re.search(pattern, response):
            leaks.append(pii_type)
    return leaks


def groundedness_score(
    validation_result: dict[str, Any] | None,
) -> float | None:
    """Extract the groundedness assessment from the validation result.

    Returns the confidence score if available, None otherwise.
    """
    if not validation_result:
        return None
    confidence = validation_result.get("confidence")
    grounded = validation_result.get("grounded")
    if grounded is False:
        return 0.0
    return confidence


# ---------------------------------------------------------------------------
# Single Query Scoring
# ---------------------------------------------------------------------------

def score_single_query(
    query_data: dict,
    result: dict,
) -> dict[str, Any]:
    """Score a single query result against the expected data.

    Args:
        query_data: The evaluation query (from dataset.jsonl)
        result: The graph execution result

    Returns:
        A dict of metric scores for this query.
    """
    generated = result.get("generation", result.get("response", ""))
    expected = query_data.get("expected_answer", "")
    route = result.get("route", "unknown")

    similarity = answer_similarity(generated, expected)
    recall = keyword_recall(generated, expected)
    routing_correct = risk_routing_accuracy(
        route,
        query_data.get("expected_risk", "low"),
        query_data.get("expected_complexity", "low"),
    )
    pii_leaks = pii_leak_check(generated)

    val_result = result.get("validation_result")
    grounded = groundedness_score(val_result)

    cost = result.get("cost_tracker", result.get("cost", {}))
    cost_usd = cost.get("estimated_cost_usd", 0) if isinstance(cost, dict) else 0

    return {
        "id": query_data.get("id", "unknown"),
        "query": query_data["query"][:80],
        "category": query_data.get("category", "unknown"),
        "route": route,
        "routing_correct": routing_correct,
        "similarity": round(similarity, 3),
        "keyword_recall": round(recall, 3),
        "groundedness": round(grounded, 3) if grounded is not None else None,
        "pii_leaks": pii_leaks,
        "pii_clean": len(pii_leaks) == 0,
        "cost_usd": round(cost_usd, 6),
        "status": result.get("status", "complete"),
    }


# ---------------------------------------------------------------------------
# Aggregate Report
# ---------------------------------------------------------------------------

def aggregate_report(scores: list[dict]) -> dict[str, Any]:
    """Aggregate individual query scores into a summary report.

    Returns a dict with mean metrics and per-category breakdowns.
    """
    if not scores:
        return {"error": "No scores to aggregate"}

    n = len(scores)

    # Overall metrics
    avg_similarity = sum(s["similarity"] for s in scores) / n
    avg_recall = sum(s["keyword_recall"] for s in scores) / n
    routing_accuracy = sum(1 for s in scores if s["routing_correct"]) / n
    pii_clean_rate = sum(1 for s in scores if s["pii_clean"]) / n
    total_cost = sum(s["cost_usd"] for s in scores)

    grounded_scores = [s["groundedness"] for s in scores if s["groundedness"] is not None]
    avg_groundedness = (
        sum(grounded_scores) / len(grounded_scores) if grounded_scores else None
    )

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for s in scores:
        cat = s["category"]
        categories.setdefault(cat, []).append(s)

    category_summary = {}
    for cat, cat_scores in categories.items():
        cn = len(cat_scores)
        category_summary[cat] = {
            "count": cn,
            "avg_similarity": round(sum(s["similarity"] for s in cat_scores) / cn, 3),
            "avg_keyword_recall": round(
                sum(s["keyword_recall"] for s in cat_scores) / cn, 3
            ),
            "routing_accuracy": round(
                sum(1 for s in cat_scores if s["routing_correct"]) / cn, 3
            ),
        }

    # Route distribution
    route_counts: dict[str, int] = {}
    for s in scores:
        route_counts[s["route"]] = route_counts.get(s["route"], 0) + 1

    return {
        "total_queries": n,
        "avg_similarity": round(avg_similarity, 3),
        "avg_keyword_recall": round(avg_recall, 3),
        "avg_groundedness": round(avg_groundedness, 3) if avg_groundedness else None,
        "routing_accuracy": round(routing_accuracy, 3),
        "pii_clean_rate": round(pii_clean_rate, 3),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query": round(total_cost / n, 6) if n > 0 else 0,
        "route_distribution": route_counts,
        "category_breakdown": category_summary,
    }


def print_report(report: dict, mode_name: str = "ControlPlane") -> None:
    """Pretty-print an aggregate report to stdout."""
    print(f"\n{'='*70}")
    print(f"  {mode_name} — Evaluation Report")
    print(f"{'='*70}")
    print(f"  Queries evaluated:    {report['total_queries']}")
    print(f"  Avg similarity:       {report['avg_similarity']:.3f}")
    print(f"  Avg keyword recall:   {report['avg_keyword_recall']:.3f}")
    if report.get("avg_groundedness") is not None:
        print(f"  Avg groundedness:     {report['avg_groundedness']:.3f}")
    print(f"  Routing accuracy:     {report['routing_accuracy']:.3f}")
    print(f"  PII clean rate:       {report['pii_clean_rate']:.3f}")
    print(f"  Total cost (USD):     ${report['total_cost_usd']:.6f}")
    print(f"  Avg cost/query:       ${report['avg_cost_per_query']:.6f}")
    print(f"  Route distribution:   {report['route_distribution']}")
    print(f"\n  Category Breakdown:")
    for cat, data in report.get("category_breakdown", {}).items():
        print(
            f"    {cat:20s}  n={data['count']:2d}  "
            f"sim={data['avg_similarity']:.3f}  "
            f"recall={data['avg_keyword_recall']:.3f}  "
            f"routing={data['routing_accuracy']:.3f}"
        )
    print(f"{'='*70}\n")


def load_dataset(path: str | Path | None = None) -> list[dict]:
    """Load the evaluation dataset from JSONL."""
    if path is None:
        path = Path(__file__).parent / "dataset.jsonl"
    path = Path(path)

    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def save_results(
    scores: list[dict],
    report: dict,
    output_path: str | Path,
) -> None:
    """Save individual scores and aggregate report to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"report": report, "individual_scores": scores},
            f,
            indent=2,
            default=str,
        )
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def compare_reports(*report_paths: str) -> None:
    """Load and compare multiple evaluation reports side-by-side."""
    reports = []
    for rp in report_paths:
        with open(rp, "r") as f:
            data = json.load(f)
        reports.append(data)

    if not reports:
        print("No reports to compare.")
        return

    # Header
    names = [Path(rp).stem for rp in report_paths]
    header = f"{'Metric':30s}" + "".join(f"{n:>18s}" for n in names)
    print(f"\n{'='*len(header)}")
    print("  Comparison Report")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    metrics = [
        ("Avg Similarity", "avg_similarity"),
        ("Avg Keyword Recall", "avg_keyword_recall"),
        ("Avg Groundedness", "avg_groundedness"),
        ("Routing Accuracy", "routing_accuracy"),
        ("PII Clean Rate", "pii_clean_rate"),
        ("Total Cost (USD)", "total_cost_usd"),
        ("Avg Cost/Query", "avg_cost_per_query"),
    ]

    for label, key in metrics:
        row = f"  {label:28s}"
        for data in reports:
            r = data.get("report", data)
            val = r.get(key)
            if val is None:
                row += f"{'N/A':>18s}"
            elif "cost" in key.lower():
                row += f"${val:>16.6f}"
            else:
                row += f"{val:>18.3f}"
        print(row)

    print(f"{'='*len(header)}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_reports(*sys.argv[2:])
    else:
        dataset = load_dataset()
        print(f"Loaded {len(dataset)} evaluation queries.")
        for q in dataset:
            print(f"  [{q['id']}] ({q['expected_complexity']}/{q['expected_risk']}) {q['query'][:60]}...")
