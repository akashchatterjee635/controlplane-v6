"""ControlPlane — Use-Case Profile Comparison.

Runs the evaluation dataset through the ControlPlane pipeline across
three different policy profiles:
  1. customer_support
  2. internal_knowledge
  3. decision_support

Outputs a comparison report showing how the decision outcomes
change based on the active profile's tolerances.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from app.graph import build_graph
from eval.metrics import load_dataset


def run_single_query(graph, query: str, use_case: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke({"query": query, "use_case": use_case}, config)
    except Exception as e:
        # Check if paused (HITL)
        try:
            state = graph.get_state(config)
            if state and state.next:
                # Get the state values before resuming to see the decision
                values = state.values
                return {
                    "decision": values.get("decision", "review").upper(),
                    "labels": values.get("risk_labels", []),
                    "route": values.get("route", "verified"),
                    "reasoning": values.get("decision_reasoning", "")
                }
        except Exception:
            pass
        return {"decision": "ERROR", "labels": [str(e)], "route": "unknown", "reasoning": ""}

    return {
        "decision": result.get("decision", "ALLOW").upper(),
        "labels": result.get("risk_labels", []),
        "route": result.get("route", "fast"),
        "reasoning": result.get("decision_reasoning", "")
    }

def main():
    dataset = load_dataset()
    profiles = ["customer_support", "internal_knowledge", "decision_support"]
    
    print("\n🚀 ControlPlane — Policy Profile Comparison")
    print(f"   Queries: {len(dataset)}")
    print(f"   Profiles: {', '.join(profiles)}")
    print(f"   Model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}\n")

    builder = build_graph()
    graph = builder.compile()

    results = []

    for i, item in enumerate(dataset):
        qid = item["id"]
        query = item["query"]
        print(f"\n--- [{i+1:2d}/{len(dataset)}] {qid} ---")
        print(f"Q: {query[:100]}")
        
        query_results = {"id": qid, "query": query, "outcomes": {}}
        
        for profile in profiles:
            thread_id = f"eval_{profile}_{qid}_{int(time.time())}"
            res = run_single_query(graph, query, profile, thread_id)
            
            query_results["outcomes"][profile] = res
            
            labels_str = ",".join(res["labels"]) if res["labels"] else "none"
            print(f"  [{profile:18s}] -> {res['decision']:8s} | Route: {res['route']:8s} | Labels: {labels_str}")
            
        results.append(query_results)
        
    # Generate summary report
    print("\n================ SUMMARY ================")
    summary = {p: {"ALLOW": 0, "EDIT": 0, "FLAG": 0, "REVIEW": 0, "BLOCK": 0, "ERROR": 0} for p in profiles}
    
    for r in results:
        for p in profiles:
            decision = r["outcomes"][p]["decision"]
            if decision in summary[p]:
                summary[p][decision] += 1
            else:
                summary[p][decision] = 1
                
    for p in profiles:
        print(f"\nProfile: {p}")
        for outcome in ["ALLOW", "EDIT", "REVIEW", "BLOCK", "ERROR"]:
            count = summary[p].get(outcome, 0)
            if count > 0 or outcome in ["EDIT", "BLOCK"]:
                print(f"  {outcome:8s}: {count:2d}")
                
    output_path = Path(__file__).parent / "results" / "profile_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
