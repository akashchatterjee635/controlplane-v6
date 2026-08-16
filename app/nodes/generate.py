"""RAG response generation using retrieved documents as context."""

import os
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ControlPlaneState
from app.utils.cost import update_cost_record


RAG_SYSTEM_PROMPT = """You are ControlPlane, an AI assistant that answers questions based on provided evidence.

Rules:
1. Answer ONLY based on the provided context documents.
2. If the context does not contain enough information, say so clearly.
3. Cite which documents support your answer when possible.
4. Be concise and precise.
5. Never fabricate information not present in the context."""


def _format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a numbered context block."""
    if not documents:
        return "No documents retrieved."
    
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: generates a response using retrieved documents as context."""
    query = state["query"]
    documents = state.get("documents", [])
    
    # Use graded documents if available (verified path), otherwise raw documents
    context_docs = state.get("graded_documents", documents)
    if not context_docs:
        context_docs = documents
    
    context = _format_context(context_docs)
    
    # Build the prompt
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        ),
    ]

    response = llm.invoke(messages)
    
    # Extract token usage from response metadata
    usage = response.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    # Update cost tracking
    cost_tracker = update_cost_record(
        state.get("cost_tracker", {}),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model_name,
    )

    return {
        "generation": response.content,
        "cost_tracker": cost_tracker,
        "audit_log": [
            f"[GENERATE] model={model_name}, "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"docs_used={len(context_docs)}"
        ],
    }
