"""ControlPlane — Routing Strategy Comparison.

Compares three routing strategies to demonstrate the value of adaptive routing:
  1. Always Fast — all queries use fast path (minimal validation)
  2. Always Verified — all queries use verified path (full validation)
  3. Adaptive — router decides based on complexity/risk scoring

Usage:
    python eval/routing_comparison.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from app.graph import build_graph
from eval.metrics import load_dataset


def _make_forced_router(force_route: str):
    """Create a router node that always returns the specified route."""
    from app.policies.profile_loader import load_profile
    from app.utils.cost import new_cost_record

    def forced_router_node(state):
        query = state.get("query", "")
        use_case = state.get("use_case", "default")
        profile = load_profile(use_case)
        return {
            "active_profile": profile,
            "complexity_score": 0,
            "risk_score": 0,
            "route": force_route,
            "cost_tracker": new_cost_record(),
            "audit_log": [f"[ROUTER] FORCED route={force_route}"],
        }
    return forced_router_node


def build_forced_graph(force_route: str):
    """Build a graph with a forced routing strategy."""
    from langgraph.graph import END, START, StateGraph

    from app.graph import audit_logger_node
    from app.nodes.decision import block_response_node, decision_node, decision_routing
    from app.nodes.generate import generate_node
    from app.nodes.grade import decide_to_generate, grade_documents_node
    from app.nodes.human_review import human_review_node
    from app.nodes.parallel_validate import parallel_validate_node, validate_fast_node
    from app.nodes.retrieve import retrieve_node
    from app.nodes.router import route_decision
    from app.nodes.web_search import web_search_node
    from app.state import ControlPlaneState

    builder = StateGraph(ControlPlaneState)

    # Register nodes — use forced router
    builder.add_node("router", _make_forced_router(force_route))
    builder.add_node("retrieve_fast", retrieve_node)
    builder.add_node("generate_fast", generate_node)
    builder.add_node("validate_fast", validate_fast_node)
    builder.add_node("retrieve_verified", retrieve_node)
    builder.add_node("grade_docs", grade_documents_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate_verified", generate_node)
    builder.add_node("parallel_validate", parallel_validate_node)
    builder.add_node("decision", decision_node)
    builder.add_node("block_response", block_response_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("audit_logger", audit_logger_node)

    # Wire edges (same as main graph)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_decision, {
        "retrieve_fast": "retrieve_fast",
        "retrieve_verified": "retrieve_verified",
    })
    builder.add_edge("retrieve_fast", "generate_fast")
    builder.add_edge("generate_fast", "validate_fast")
    builder.add_edge("validate_fast", "decision")
    builder.add_edge("retrieve_verified", "grade_docs")
    builder.add_conditional_edges("grade_docs", decide_to_generate, {
        "generate_verified": "generate_verified",
        "web_search": "web_search",
    })
    builder.add_edge("web_search", "generate_verified")
    builder.add_edge("generate_verified", "parallel_validate")
    builder.add_edge("parallel_validate", "decision")
    builder.add_conditional_edges("decision", decision_routing, {
        "end": "audit_logger",
        "human_review": "human_review",
        "block_response": "block_response",
    })
    builder.add_edge("human_review", "audit_logger")
    builder.add_edge("block_response", "audit_logger")
    builder.add_edge("audit_logger", END)

    return builder.compile()


def run_query(graph, query: str, use_case: str, thread_id: str) -> dict:
    """Run a single query and handle HITL interrupts."""
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    try:
        result = graph.invoke({"query": query, "use_case": use_case}, config)
    except Exception:
        try:
            state = graph.get_state(config)
            if state and state.next:
                values = state.values
                return {
                    "decision": values.get("decision", "review").upper(),
                    "route": values.get("route", "verified"),
                    "risk_labels": values.get("risk_labels", []),
                    "llm_calls": values.get("cost_tracker", {}).get("llm_calls", 0),
                    "cost_usd": values.get("cost_tracker", {}).get("estimated_cost_usd", 0),
                    "latency_ms": (time.time() - start) * 1000,
                    "hitl": True,
                }
        except Exception:
            pass
        return {"decision": "ERROR", "route": "unknown", "risk_labels": [],
                "llm_calls": 0, "cost_usd": 0, "latency_ms": 0, "hitl": False}

    elapsed = (time.time() - start) * 1000
    cost = result.get("cost_tracker", {})
    return {
        "decision": result.get("decision", "allow").upper(),
        "route": result.get("route", "fast"),
        "risk_labels": result.get("risk_labels", []),
        "llm_calls": cost.get("llm_calls", 0),
        "cost_usd": cost.get("estimated_cost_usd", 0),
        "latency_ms": elapsed,
        "hitl": False,
    }


def main():
    dataset = load_dataset()
    strategies = {
        "Always Fast": build_forced_graph("fast"),
        "Always Verified": build_forced_graph("verified"),
        "Adaptive": build_graph().compile(),
    }

    print("\n🔬 ControlPlane — Routing Strategy Comparison")
    print(f"   Queries: {len(dataset)}")
    print(f"   Model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   Strategies: {', '.join(strategies.keys())}\n")

    all_results = {name: [] for name in strategies}

    for i, item in enumerate(dataset):
        qid = item["id"]
        query = item["query"]
        expected_risk = item.get("expected_risk", "low")
        print(f"\n[{i+1:2d}/{len(dataset)}] {qid} (expected_risk={expected_risk})")
        print(f"  Q: {query[:80]}")

        for name, graph in strategies.items():
            thread_id = f"cmp_{name.lower().replace(' ', '_')}_{qid}_{int(time.time())}"
            res = run_query(graph, query, "customer_support", thread_id)
            all_results[name].append({**res, "id": qid, "expected_risk": expected_risk})
            print(f"  [{name:18s}] {res['decision']:8s} | {res['latency_ms']:8.0f}ms | ${res['cost_usd']:.6f} | hitl={res['hitl']}")

    # Aggregate metrics
    print("\n" + "=" * 80)
    print("ROUTING STRATEGY COMPARISON RESULTS")
    print("=" * 80)

    header = f"{'Metric':<30s} | {'Always Fast':>14s} | {'Always Verified':>16s} | {'Adaptive':>14s}"
    print(header)
    print("-" * len(header))

    for metric_name, metric_fn in [
        ("Safety Recall", lambda results: sum(1 for r in results if r['expected_risk'] == 'high' and r['decision'] in ('BLOCK', 'REVIEW', 'EDIT')) / max(1, sum(1 for r in results if r['expected_risk'] == 'high'))),
        ("False Escalation Rate", lambda results: sum(1 for r in results if r['expected_risk'] == 'low' and r['decision'] in ('BLOCK', 'REVIEW')) / max(1, sum(1 for r in results if r['expected_risk'] == 'low'))),
        ("HITL Rate", lambda results: sum(1 for r in results if r['hitl'] or r['decision'] == 'REVIEW') / max(1, len(results))),
        ("Avg Latency (ms)", lambda results: sum(r['latency_ms'] for r in results) / max(1, len(results))),
        ("Avg Cost/Query ($)", lambda results: sum(r['cost_usd'] for r in results) / max(1, len(results))),
        ("Block Rate", lambda results: sum(1 for r in results if r['decision'] == 'BLOCK') / max(1, len(results))),
    ]:
        values = []
        for name in strategies:
            val = metric_fn(all_results[name])
            values.append(val)
        if "Latency" in metric_name:
            print(f"{metric_name:<30s} | {values[0]:>13.0f}ms | {values[1]:>15.0f}ms | {values[2]:>13.0f}ms")
        elif "Cost" in metric_name:
            print(f"{metric_name:<30s} | ${values[0]:>12.6f} | ${values[1]:>14.6f} | ${values[2]:>12.6f}")
        else:
            print(f"{metric_name:<30s} | {values[0]:>13.1%} | {values[1]:>15.1%} | {values[2]:>13.1%}")

    # Save results
    output_path = Path(__file__).parent / "results" / "routing_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
