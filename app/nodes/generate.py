"""RAG response generation using retrieved documents as context."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import ControlPlaneState
from app.utils.retrieval_security import sanitize_documents

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
    from app.utils.llm_gateway import LLMGateway
    
    query = state["query"]
    documents = state.get("documents", [])
    
    # Use graded documents if available (verified path), otherwise raw documents
    context_docs = state.get("graded_documents", documents)
    if not context_docs:
        context_docs = documents
    
    # Sanitize retrieved documents against indirect prompt injection
    context_docs = sanitize_documents(context_docs)
    
    context = _format_context(context_docs)
    
    # Build the prompt
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        ),
    ]

    gateway = LLMGateway(state.get("cost_tracker", {}))
    content = gateway.invoke(messages, purpose="generation")
    
    return {
        "generation": content,
        "cost_tracker": gateway.cost_tracker,
        "audit_log": [
            f"[GENERATE] docs_used={len(context_docs)}"
        ],
    }
