"""Document grading node for the verified path.

Uses an LLM to assess the relevance of retrieved documents to the query.
If documents are irrelevant, they are filtered out. If all are irrelevant,
web search is triggered.
"""

import os
from typing import Any, Literal
import yaml

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from app.state import ControlPlaneState
from app.utils.cost import update_cost_record


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

    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(DocumentGrade)

    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "It does not need to be a stringent test. The goal is to filter out erroneous retrievals."
    )

    graded_docs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    audit_entries = []

    for idx, doc in enumerate(documents):
        human_prompt = f"Retrieved document: \n\n{doc.page_content}\n\nUser question: {query}"
        
        # In a real app we'd get token usage from the LLM callback, but for now we'll mock token counts
        # or rely on langchain's metadata if available. Using standard LLM call.
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # We roughly estimate tokens since with_structured_output doesn't easily expose token usage in all langchain versions
        # A more precise implementation would use get_openai_callback
        prompt_tokens_est = len(system_prompt.split()) + len(human_prompt.split())
        comp_tokens_est = 20 # small JSON output
        total_prompt_tokens += prompt_tokens_est
        total_completion_tokens += comp_tokens_est

        if response.relevant:
            graded_docs.append(doc)
            audit_entries.append(f"[GRADE] Document {idx+1} relevant: {response.reason}")
        else:
            audit_entries.append(f"[GRADE] Document {idx+1} irrelevant: {response.reason}")

    # Load policies to check threshold, default to 1 relevant doc
    # policies = _load_policies()
    
    web_search_needed = len(graded_docs) == 0
    
    if web_search_needed:
        audit_entries.append("[GRADE] Insufficient relevant documents. Triggering web search.")

    updated_cost = update_cost_record(
        cost_tracker,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        model=model_name
    )

    return {
        "graded_documents": graded_docs,
        "web_search_needed": web_search_needed,
        "cost_tracker": updated_cost,
        "audit_log": audit_entries,
    }

def decide_to_generate(state: ControlPlaneState) -> Literal["generate_verified", "web_search"]:
    """Conditional edge after grading documents."""
    web_search_needed = state.get("web_search_needed", False)
    if web_search_needed:
        return "web_search"
    return "generate_verified"
