"""Unit tests for the document retrieval node.

Tests cover:
- ChromaDB collection creation and document ingestion
- Embedding generation via EmbeddingService
- Top-k retrieval with correct Document conversion
- Retrieval latency tracking in cost_tracker
- Edge cases: empty collection, no results
"""

import os
import json

import pytest
import chromadb
from langchain_core.documents import Document

from app.utils.embeddings import EmbeddingService, get_embedding_service
from app.nodes.retrieve import retrieve_node, reset_collection


# ============================================================================
# EmbeddingService tests
# ============================================================================

class TestEmbeddingService:
    """Tests for the singleton embedding service."""

    def test_singleton_pattern(self):
        """EmbeddingService should return the same instance."""
        svc1 = EmbeddingService()
        svc2 = EmbeddingService()
        assert svc1 is svc2

    def test_embed_query_returns_list(self):
        """embed_query should return a list of floats."""
        svc = get_embedding_service()
        result = svc.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_returns_list_of_lists(self):
        """embed_documents should return a list of embedding lists."""
        svc = get_embedding_service()
        docs = ["First document", "Second document"]
        result = svc.embed_documents(docs)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert all(isinstance(x, float) for x in result[0])

    def test_embedding_dimension_consistency(self):
        """All embeddings should have the same dimension."""
        svc = get_embedding_service()
        emb1 = svc.embed_query("query one")
        emb2 = svc.embed_query("query two")
        assert len(emb1) == len(emb2)


# ============================================================================
# ChromaDB integration tests
# ============================================================================

class TestChromaDBIntegration:
    """Tests for ChromaDB operations."""

    def test_collection_creation(self, temp_chroma_path):
        """Should create a ChromaDB collection successfully."""
        client = chromadb.PersistentClient(path=temp_chroma_path)
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )
        assert collection.name == "test_collection"
        assert collection.count() == 0

    def test_add_and_query_documents(self, temp_chroma_path):
        """Should add documents and retrieve them via similarity search."""
        client = chromadb.PersistentClient(path=temp_chroma_path)
        collection = client.get_or_create_collection(
            name="test_retrieval",
            metadata={"hnsw:space": "cosine"},
        )

        svc = get_embedding_service()

        # Add documents
        texts = [
            "Python is a programming language.",
            "Machine learning uses statistical models.",
            "Docker containers package applications.",
        ]
        ids = ["d1", "d2", "d3"]
        embeddings = svc.embed_documents(texts)
        metadatas = [
            {"source": "test", "category": "programming"},
            {"source": "test", "category": "ai"},
            {"source": "test", "category": "devops"},
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        assert collection.count() == 3

        # Query
        query_embedding = svc.embed_query("What programming language is popular?")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        assert len(results["documents"][0]) == 2
        # The Python document should be the top result
        assert "Python" in results["documents"][0][0]

    def test_empty_collection_query(self, temp_chroma_path):
        """Querying an empty collection should return empty results."""
        client = chromadb.PersistentClient(path=temp_chroma_path)
        collection = client.get_or_create_collection(name="empty_test")

        svc = get_embedding_service()
        query_embedding = svc.embed_query("anything")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        assert len(results["documents"][0]) == 0


# ============================================================================
# retrieve_node() tests
# ============================================================================

class TestRetrieveNode:
    """Tests for the retrieve_node graph node."""

    @pytest.fixture(autouse=True)
    def _setup_chroma(self, temp_chroma_path, monkeypatch):
        """Set up a temp ChromaDB with sample data for each test."""
        # Reset the module-level singleton
        reset_collection()

        # Point to temp path
        monkeypatch.setenv("CHROMA_PERSIST_PATH", temp_chroma_path)
        monkeypatch.setenv("RETRIEVAL_TOP_K", "3")

        # Seed some documents
        client = chromadb.PersistentClient(path=temp_chroma_path)
        collection = client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

        svc = get_embedding_service()
        texts = [
            "LangGraph is a library for building stateful multi-actor applications.",
            "Kubernetes orchestrates container deployment and scaling.",
            "RAG combines retrieval with generation for better LLM responses.",
            "OAuth 2.0 is a protocol for authorization using access tokens.",
            "Transformers are neural network architectures using self-attention.",
        ]
        ids = [f"test_{i}" for i in range(len(texts))]
        embeddings = svc.embed_documents(texts)
        metadatas = [{"source": f"test_source_{i}"} for i in range(len(texts))]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        yield

        # Cleanup
        reset_collection()

    def test_retrieve_returns_documents(self):
        """retrieve_node should return a list of Document objects."""
        state = {"query": "How does LangGraph work?", "cost_tracker": {}}
        result = retrieve_node(state)

        assert "documents" in result
        assert len(result["documents"]) > 0
        assert all(isinstance(d, Document) for d in result["documents"])

    def test_retrieve_respects_top_k(self):
        """Should return at most top_k documents."""
        state = {"query": "Tell me about AI", "cost_tracker": {}}
        result = retrieve_node(state)

        # RETRIEVAL_TOP_K is set to 3
        assert len(result["documents"]) <= 3

    def test_retrieve_documents_have_metadata(self):
        """Retrieved documents should include metadata with retrieval_distance."""
        state = {"query": "What is Kubernetes?", "cost_tracker": {}}
        result = retrieve_node(state)

        for doc in result["documents"]:
            assert "retrieval_distance" in doc.metadata

    def test_retrieve_updates_cost_tracker(self):
        """Cost tracker should be updated with retrieval latency."""
        state = {"query": "Authorization protocols", "cost_tracker": {"start_time": 1000}}
        result = retrieve_node(state)

        assert "cost_tracker" in result
        assert result["cost_tracker"]["retrieval_latency_ms"] > 0

    def test_retrieve_appends_audit_log(self):
        """Should append a retrieval audit log entry."""
        state = {"query": "Neural networks", "cost_tracker": {}}
        result = retrieve_node(state)

        assert "audit_log" in result
        assert len(result["audit_log"]) == 1
        assert "[RETRIEVE]" in result["audit_log"][0]

    def test_retrieve_semantic_relevance(self):
        """Top result should be semantically closest to the query."""
        state = {"query": "How to deploy containers with Kubernetes?", "cost_tracker": {}}
        result = retrieve_node(state)

        # Kubernetes doc should be the top or near-top result
        top_doc_text = result["documents"][0].page_content
        assert "Kubernetes" in top_doc_text or "container" in top_doc_text.lower()
