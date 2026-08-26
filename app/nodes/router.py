"""Deterministic query router with complexity and risk scoring.

Scoring rubric:
  complexity_score =
      query_length_bucket     (0-3)
    + number_of_constraints   (0-3, capped)
    + retrieval_requirement   (0-2)
    + reasoning_requirement   (0-2)
    → range: 0-10

  risk_score =
      policy_keyword_hits     (0-3, capped)
    + sensitive_topic_match   (0-2)
    + prompt_injection_risk   (0-3)
    → range: 0-8

Routing decision:
  if complexity <= complexity_fast_max AND risk <= risk_fast_max:
      route = "fast"
  else:
      route = "verified"
"""

import re
import os
from typing import Any, Literal

import yaml

from app.state import ControlPlaneState
from app.utils.cost import new_cost_record

# Load policies at module level
_POLICIES_PATH = os.path.join(os.path.dirname(__file__), "..", "policies", "policies.yaml")

def _load_policies() -> dict:
    with open(_POLICIES_PATH, "r") as f:
        return yaml.safe_load(f)


def compute_complexity(query: str, policies: dict | None = None) -> int:
    """Score query complexity from 0-10."""
    if policies is None:
        policies = _load_policies()
    
    scoring = policies.get("scoring", {})
    score = 0
    
    # 1. Query length bucket (0-3)
    length = len(query)
    buckets = scoring.get("query_length_buckets", {})
    if length > buckets.get("long", 300):
        score += 3
    elif length > buckets.get("medium", 150):
        score += 2
    elif length > buckets.get("short", 50):
        score += 1
    # else: 0

    # 2. Number of constraints (0-3, capped)
    constraint_keywords = scoring.get("constraint_keywords", [])
    query_lower = query.lower()
    constraint_count = sum(1 for kw in constraint_keywords if kw.lower() in query_lower)
    score += min(constraint_count, 3)

    # 3. Retrieval requirement (0-2)
    # Questions and reference requests suggest retrieval need
    retrieval_indicators = ["?", "what is", "how to", "explain", "describe", "tell me about", "find", "search", "look up"]
    retrieval_hits = sum(1 for ind in retrieval_indicators if ind.lower() in query_lower)
    score += min(retrieval_hits, 2)

    # 4. Reasoning requirement (0-2)
    reasoning_indicators = scoring.get("reasoning_indicators", [])
    reasoning_hits = sum(1 for ind in reasoning_indicators if ind.lower() in query_lower)
    score += min(reasoning_hits, 2)

    return min(score, 10)  # cap at 10


def compute_risk(query: str, policies: dict | None = None) -> int:
    """Score query risk from 0-8."""
    if policies is None:
        policies = _load_policies()
    
    score = 0
    query_lower = query.lower()

    # 1. Policy keyword / prohibited patterns (0-3)
    prohibited = policies.get("prohibited_keywords", [])
    prohibited_hits = sum(1 for kw in prohibited if kw.lower() in query_lower)
    if prohibited_hits > 0:
        # Prompt injections / prohibited words are instantly high risk
        score += 4
    else:
        score += min(prohibited_hits, 3)

    # 2. Sensitive topic match (0-4)
    sensitive_topics = policies.get("sensitive_topics", [])
    topic_hits = sum(1 for topic in sensitive_topics if topic.lower() in query_lower)
    if topic_hits > 0:
        score += 2 * topic_hits

    # 3. Prompt injection risk (0-3)
    # Check PII patterns in the query itself (could indicate data exfil attempt)
    pii_patterns = policies.get("pii_patterns", {})
    pii_hits = 0
    for pattern_name, pattern in pii_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            pii_hits += 1
    score += min(pii_hits, 3)

    return min(score, 8)  # cap at 8


def router_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: scores the query and determines the execution path."""
    query = state.get("query", "")
    use_case = state.get("use_case", "default")
    
    from app.policies.profile_loader import load_profile
    profile = load_profile(use_case)
    
    # We still use the base policies for global scoring rules, 
    # but thresholds and overrides come from the profile.
    policies = _load_policies()
    # Override global lists with profile-specific ones if present
    for key in ["prohibited_keywords", "sensitive_topics", "pii_patterns"]:
        if key in profile:
            policies[key] = profile[key]

    complexity = compute_complexity(query, policies)
    risk = compute_risk(query, policies)

    complexity_max = profile.get("complexity_fast_max", 4)
    risk_max = profile.get("risk_fast_max", 2)

    route: Literal["fast", "verified"] = (
        "fast" if complexity <= complexity_max and risk <= risk_max else "verified"
    )

    return {
        "active_profile": profile,
        "complexity_score": complexity,
        "risk_score": risk,
        "route": route,
        "cost_tracker": new_cost_record(),
        "audit_log": [
            f"[ROUTER] use_case={use_case}, complexity={complexity}, risk={risk}, route={route}"
        ],
    }


def route_decision(state: ControlPlaneState) -> Literal["retrieve_fast", "retrieve_verified"]:
    """Conditional edge function: directs to the appropriate retrieval path."""
    if state.get("route") == "fast":
        return "retrieve_fast"
    return "retrieve_verified"
