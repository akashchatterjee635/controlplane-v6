"""Seed the ChromaDB knowledge base with sample documents.

Usage:
    python -m data.knowledge_base.seed_data
    # or
    python data/knowledge_base/seed_data.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chromadb
from app.utils.embeddings import get_embedding_service


def seed_knowledge_base(
    docs_path: str | None = None,
    chroma_path: str | None = None,
    collection_name: str = "knowledge_base",
) -> int:
    """Load documents from JSONL into ChromaDB.
    
    Returns the number of documents loaded.
    """
    if docs_path is None:
        docs_path = os.path.join(os.path.dirname(__file__), "sample_docs.jsonl")
    if chroma_path is None:
        chroma_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

    # Read documents
    documents = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    if not documents:
        print("No documents found in", docs_path)
        return 0

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Delete existing collection if it exists to avoid duplicates
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Generate embeddings
    embedding_service = get_embedding_service()
    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = embedding_service.embed_documents(texts)

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Successfully loaded {len(documents)} documents into '{collection_name}'")
    return len(documents)


if __name__ == "__main__":
    seed_knowledge_base()
