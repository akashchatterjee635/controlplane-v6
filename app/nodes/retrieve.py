"""Document retrieval from ChromaDB vector store."""

import os
import time
from typing import Any

import chromadb
from langchain_core.documents import Document

from app.state import ControlPlaneState
from app.utils.embeddings import get_embedding_service

# ChromaDB client (lazy init)
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (lazy singleton)."""
    global _chroma_client, _collection
    if _collection is None:
        persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=persist_path)
        _collection = _chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def retrieve_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: retrieves top-k documents from the vector store."""
    query = state["query"]
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    embedding_service = get_embedding_service()
    collection = _get_collection()

    start = time.time()
    query_embedding = embedding_service.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    retrieval_ms = (time.time() - start) * 1000

    # Convert to LangChain Document objects
    documents: list[Document] = []
    if results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            metadata["retrieval_distance"] = distance
            documents.append(Document(page_content=doc_text, metadata=metadata))

    # Update cost tracker
    cost_tracker = dict(state.get("cost_tracker", {}))
    cost_tracker["retrieval_latency_ms"] = retrieval_ms

    return {
        "documents": documents,
        "cost_tracker": cost_tracker,
        "audit_log": [
            f"[RETRIEVE] Found {len(documents)} documents in {retrieval_ms:.1f}ms"
        ],
    }


def reset_collection() -> None:
    """Reset the cached collection (useful for testing)."""
    global _chroma_client, _collection
    _chroma_client = None
    _collection = None
