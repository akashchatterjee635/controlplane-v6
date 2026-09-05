"""Document grading node for the verified path.

Uses an LLM to assess the relevance of retrieved documents to the query.
If documents are irrelevant, they are filtered out. If all are irrelevant,
web search is triggered.
"""

import os
from typing import Any, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.state import ControlPlaneState


class DocumentGrade(BaseModel):
    """Binary score for relevance check."""
    relevant: bool = Field(description="True if the document is relevant to the question, False otherwise.")
    reason: str = Field(description="Brief reason for the grading decision.")


def _load_policies() -> dict:
    """Load grading policies."""
    policies_path = os.path.join(os.path.dirname(__file__), "..", "policies", "policies.yaml")
    if os.path.exists(policies_path):
        with open(policies_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def grade_documents_node(state: ControlPlaneState) -> dict[str, Any]:
    """Grade retrieved documents for relevance.

    Filters irrelevant documents out of the context.
    Sets web_search_needed=True if not enough relevant documents remain.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    cost_tracker = state.get("cost_tracker", {})

    if not documents:
        return {
            "graded_documents": [],
            "web_search_needed": True,
            "audit_log": ["[GRADE] No documents to grade. Fallback to web search."],
        }

    import json

    from app.utils.llm_gateway import LLMGateway

    gateway = LLMGateway(cost_tracker)
    
    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\n"
        "Reply with a JSON object (nothing else):\n"
        '{"relevant": <bool>, "reason": <string>}'
    )

    graded_docs = []
    audit_entries = []

    for idx, doc in enumerate(documents):
        human_prompt = f"Retrieved document: \n\n{doc.page_content}\n\nUser question: {query}"
        
        result_text = gateway.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
            purpose="grade-document"
        ).strip()
        
        if "```" in result_text:
            result_text = result_text.split("```")[1].removeprefix("json").strip()
            
        try:
            result_data = json.loads(result_text)
            relevant = result_data.get("relevant", False)
            reason = result_data.get("reason", "No reason provided")
        except json.JSONDecodeError:
            relevant = False
            reason = "Failed to parse LLM response"

        if relevant:
            graded_docs.append(doc)
            audit_entries.append(f"[GRADE] Document {idx+1} relevant: {reason}")
        else:
            audit_entries.append(f"[GRADE] Document {idx+1} irrelevant: {reason}")

    # Load policies to check threshold, default to 1 relevant doc
    # policies = _load_policies()
    
    web_search_needed = len(graded_docs) == 0
    
    if web_search_needed:
        audit_entries.append("[GRADE] Insufficient relevant documents. Triggering web search.")

    return {
        "graded_documents": graded_docs,
        "web_search_needed": web_search_needed,
        "cost_tracker": gateway.cost_tracker,
        "audit_log": audit_entries,
    }

def decide_to_generate(state: ControlPlaneState) -> Literal["generate_verified", "web_search"]:
    """Conditional edge after grading documents."""
    web_search_needed = state.get("web_search_needed", False)
    if web_search_needed:
        return "web_search"
    return "generate_verified"
