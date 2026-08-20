"""ControlPlane v6 — Vanilla RAG Baseline Evaluation.

Runs every query in the evaluation dataset through a simple
retrieve-and-generate pipeline (no routing, no grading, no validation)
to establish a performance baseline for comparison.

Usage:
    python eval/baseline.py
    python eval/baseline.py --dataset eval/dataset.jsonl --output eval/results/baseline.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from app.utils.embeddings import EmbeddingService
from app.utils.cost import update_cost_record
from eval.metrics import (
    load_dataset,
    score_single_query,
    aggregate_report,
    print_report,
    save_results,
)


RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context.
Answer ONLY based on the provided context documents.
If the context does not contain enough information, say so clearly.
Be concise and precise."""


def vanilla_rag(query: str, top_k: int = 5) -> dict:
    """Run a simple retrieve-and-generate pipeline.

    No routing, no grading, no validation, no HITL.
    Returns a result dict compatible with the scoring functions.
    """
    start = time.time()

    # Retrieve
    embedding_service = EmbeddingService()
    import chromadb

    chroma_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection(
            name="controlplane_kb",
            embedding_function=embedding_service.get_chroma_ef(),
        )
    except Exception:
        return {
            "generation": "Knowledge base not found. Run data/knowledge_base/seed_data.py first.",
            "route": "baseline",
            "cost_tracker": {},
            "status": "error",
        }

    results = collection.query(query_texts=[query], n_results=top_k)

    documents = []
    if results and results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            documents.append(Document(page_content=doc_text, metadata=metadata))

    # Format context
    if not documents:
        context = "No documents retrieved."
    else:
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Document {i}] (source: {source})\n{doc.page_content}")
        context = "\n\n".join(parts)

    # Generate
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
    ]

    response = llm.invoke(messages)

    # Cost tracking
    usage = response.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    cost_tracker = update_cost_record(
        {},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model_name,
    )

    elapsed = (time.time() - start) * 1000

    return {
        "generation": response.content,
        "route": "baseline",
        "cost_tracker": cost_tracker,
        "latency_ms": round(elapsed, 1),
        "status": "complete",
        "documents_used": len(documents),
    }


def run_baseline_eval(dataset_path: str | None = None, output_path: str | None = None):
    """Run the full baseline evaluation."""
    dataset = load_dataset(dataset_path)
    print(f"\n🔬 Baseline Evaluation — {len(dataset)} queries")
    print(f"   Model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   Mode: Vanilla RAG (no routing, no validation)\n")

    scores = []
    for i, query_data in enumerate(dataset):
        qid = query_data["id"]
        query = query_data["query"]
        print(f"  [{i+1:2d}/{len(dataset)}] {qid}: {query[:60]}...", end=" ", flush=True)

        try:
            result = vanilla_rag(query)
            score = score_single_query(query_data, result)
            score["latency_ms"] = result.get("latency_ms", 0)
            scores.append(score)
            print(
                f"✓ sim={score['similarity']:.2f} "
                f"recall={score['keyword_recall']:.2f} "
                f"pii_clean={score['pii_clean']}"
            )
        except Exception as e:
            print(f"✗ Error: {e}")
            scores.append({
                "id": qid,
                "query": query[:80],
                "category": query_data.get("category", "unknown"),
                "route": "baseline",
                "routing_correct": False,
                "similarity": 0.0,
                "keyword_recall": 0.0,
                "groundedness": None,
                "pii_leaks": [],
                "pii_clean": True,
                "cost_usd": 0,
                "status": "error",
            })

    report = aggregate_report(scores)
    print_report(report, mode_name="Baseline (Vanilla RAG)")

    if output_path is None:
        output_path = Path(__file__).parent / "results" / "baseline.json"
    save_results(scores, report, output_path)

    return scores, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline RAG evaluation")
    parser.add_argument("--dataset", default=None, help="Path to dataset.jsonl")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    run_baseline_eval(args.dataset, args.output)
