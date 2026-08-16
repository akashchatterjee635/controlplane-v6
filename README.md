# 🛡️ ControlPlane v6 — Adaptive RAG with Risk-Aware Routing

> **Not every query deserves the same amount of computation or oversight.**
> ControlPlane dynamically decides when to retrieve more evidence, validate more aggressively, or involve a human.

A self-governing RAG agent built on [LangGraph](https://github.com/langchain-ai/langgraph) that scores each query for **complexity** and **risk**, then routes it through the appropriate execution path — from a fast, low-overhead pipeline to a fully validated, human-reviewed flow.

---

## Architecture

```
                    USER QUERY
                        │
                        ▼
                ┌──────────────┐
                │ Query Router │
                │ complexity + │
                │ risk scoring │
                └──────┬───────┘
                       │
             ┌─────────┴─────────┐
             │                   │
         LOW RISK             HIGH RISK
             │                   │
             ▼                   ▼
       ┌──────────┐       ┌──────────────┐
       │ Retrieve │       │   Retrieve   │
       └────┬─────┘       └──────┬───────┘
            │                    │
            ▼                    ▼
       ┌──────────┐       ┌──────────────┐
       │ Generate │       │ Grade Docs   │──── Irrelevant ──▶ Web Search
       └────┬─────┘       └──────┬───────┘                       │
            │                    │◀──────────────────────────────┘
            ▼                    ▼
       ┌──────────┐       ┌──────────────┐
       │ Validate │       │   Generate   │
       │ (Layer 1)│       └──────┬───────┘
       └────┬─────┘              │
            │                    ▼
            │              ┌──────────────┐
            │              │   Validate   │
            │              │(Layer 1 + 2) │
            │              └──────┬───────┘
            │                     │
            │              ┌──────┴──────┐
            │              │             │
            │            PASS          FAIL
            │              │             │
            │              │             ▼
            │              │      ┌─────────────┐
            │              │      │ ⏸ HUMAN     │
            │              │      │   REVIEW    │
            │              │      └──────┬──────┘
            │              │             │
            ▼              ▼             ▼
                      RESPONSE
```

### Two Execution Paths

| Path | Triggers When | Steps | Overhead |
|------|--------------|-------|----------|
| **🏎️ Fast** | `complexity ≤ 4` AND `risk ≤ 2` | Retrieve → Generate → Basic Validate | ~1 LLM call |
| **🔍 Verified** | Everything else | Retrieve → Grade → (Web Search?) → Generate → Full Validate → (HITL?) | 3-5 LLM calls |

### Deterministic Router Scoring

```
complexity_score (0-10) =
    query_length_bucket      (0-3)
  + constraint_keywords      (0-3, capped)
  + retrieval_requirement    (0-2)
  + reasoning_requirement    (0-2)

risk_score (0-8) =
    prohibited_keyword_hits  (0-3, capped)
  + sensitive_topic_match    (0-2)
  + pii_detection            (0-3)
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- An OpenAI API key (for LLM generation + validation)
- A Tavily API key (optional, for web search fallback)

### 1. Clone & Install

```bash
cd controlplane
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here    # optional
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Seed the Knowledge Base

```bash
python data/knowledge_base/seed_data.py
```

This loads 30 sample documents (AI/ML, Software Engineering, Cloud, Security) into ChromaDB with sentence-transformer embeddings.

### 4. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

### 5. Launch the Dashboard (Optional)

```bash
streamlit run ui/dashboard.py
```

### 6. Query

```bash
# Simple query → Fast path
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Docker?"}'

# Complex query → Verified path
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare and analyze the trade-offs between Kubernetes and Docker Swarm for production. You must include at least three differences."}'

# Risky query → Verified + HITL
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What medication dosage should I take? My SSN is 123-45-6789"}'
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/query` | Submit a query (returns response or `pending_review` status) |
| `GET` | `/api/v1/status/{thread_id}` | Check thread status |
| `GET` | `/api/v1/pending-reviews` | List all threads awaiting human review |
| `POST` | `/api/v1/review/{thread_id}` | Submit review decision (approve/redact/deny) and resume graph |

### Query Request

```json
{
  "query": "Your question here",
  "thread_id": "optional-uuid"
}
```

### Query Response

```json
{
  "thread_id": "abc-123",
  "query": "What is Docker?",
  "response": "Docker is a containerization platform...",
  "route": "fast",
  "complexity_score": 2,
  "risk_score": 0,
  "cost": {
    "llm_calls": 1,
    "prompt_tokens": 450,
    "completion_tokens": 120,
    "estimated_cost_usd": 0.000139
  },
  "audit_log": ["[ROUTER] ...", "[RETRIEVE] ...", "[GENERATE] ..."],
  "status": "complete"
}
```

### Review Request

```json
{
  "decision": "approve | redact | deny",
  "redacted_response": "Safe replacement text (for redact only)",
  "reason": "Why this decision was made",
  "reviewer": "reviewer_name"
}
```

---

## Project Structure

```
controlplane/
├── .env.example              # Environment variable template
├── .env                      # Your API keys (git-ignored)
├── requirements.txt          # Python dependencies
│
├── app/
│   ├── __init__.py
│   ├── api.py                # FastAPI endpoints (query, review, status)
│   ├── graph.py              # LangGraph StateGraph construction
│   ├── state.py              # Central ControlPlaneState schema
│   │
│   ├── nodes/
│   │   ├── __init__.py       # Node exports
│   │   ├── router.py         # Deterministic complexity + risk scoring
│   │   ├── retrieve.py       # ChromaDB vector retrieval
│   │   ├── grade.py          # LLM document relevance grading
│   │   ├── generate.py       # RAG response generation
│   │   ├── web_search.py     # Tavily web search fallback
│   │   ├── validate.py       # Layer 1 (deterministic) + Layer 2 (LLM) validation
│   │   └── human_review.py   # HITL interrupt/resume node
│   │
│   ├── policies/
│   │   ├── __init__.py
│   │   └── policies.yaml     # Thresholds, keywords, PII patterns, cost config
│   │
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py     # Singleton sentence-transformer service
│       ├── cost.py           # Token cost tracking & reporting
│       └── security.py       # PII detection, injection checks, sanitization
│
├── data/
│   └── knowledge_base/
│       ├── sample_docs.jsonl  # 30 sample documents
│       └── seed_data.py       # ChromaDB seeding script
│
├── ui/
│   └── dashboard.py           # Streamlit review dashboard
│
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures
    ├── test_router.py         # 26 tests — routing logic
    ├── test_retrieval.py      # 13 tests — embeddings + ChromaDB
    ├── test_validation.py     # 11 tests — security + Layer 1
    ├── test_graph.py          #  2 tests — graph structure
    └── test_hitl.py           # 14 tests — HITL workflow
```

---

## Configuration

All thresholds are configurable in [`app/policies/policies.yaml`](app/policies/policies.yaml):

```yaml
thresholds:
  complexity_fast_max: 4          # Queries scoring ≤ this use fast path
  risk_fast_max: 2                # Queries scoring ≤ this use fast path
  validation_confidence_min: 0.85 # Below this triggers human review
  grading_relevance_min: 0.7      # Document relevance threshold
  retrieval_top_k: 5              # Number of documents to retrieve
```

### PII Patterns

The system detects and auto-redacts:
- Social Security Numbers (`123-45-6789`)
- Email addresses (`user@example.com`)
- Credit card numbers (`4111-1111-1111-1111`)
- Phone numbers (`555-123-4567`)

### Sensitive Topics

Queries about medical advice, legal counsel, financial recommendations, medication dosage, and diagnosis are automatically flagged as high-risk.

---

## Two-Layer Validation

| Layer | Path | Method | Cost |
|-------|------|--------|------|
| **Layer 1** | Both | Deterministic regex (PII, injection, prohibited keywords) | Zero |
| **Layer 2** | Verified only | LLM self-assessment (grounded? safe? compliant?) | ~1 LLM call |

Layer 2 uses structured output to ask the LLM:

```json
{
  "grounded": true,       // Is the response supported by source documents?
  "safe": true,           // Is the response free of harmful content?
  "compliant": true,      // Does it follow policy guidelines?
  "confidence": 0.92,     // Overall quality confidence (0.0 - 1.0)
  "reasoning": "..."      // Explanation of the assessment
}
```

Human review is triggered when:
- Any Layer 1 check fails (PII detected, injection patterns)
- Any Layer 2 check returns `false`
- Confidence drops below `validation_confidence_min` (default: 0.85)

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific suites
python -m pytest tests/test_router.py -v       # Router scoring
python -m pytest tests/test_retrieval.py -v     # Embeddings + ChromaDB
python -m pytest tests/test_validation.py -v    # Security + validation
python -m pytest tests/test_hitl.py -v          # Human-in-the-loop
python -m pytest tests/test_graph.py -v         # Graph structure
```

**Current status: 67 tests passing ✅**

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `test_router.py` | 26 | Complexity scoring, risk scoring, routing decisions, edge cases |
| `test_retrieval.py` | 13 | Embedding service, ChromaDB integration, semantic relevance |
| `test_validation.py` | 11 | PII detection, sanitization, prompt injection, Layer 1 node |
| `test_hitl.py` | 14 | Graph structure, Layer 1 refactor, triage, approve/redact/deny |
| `test_graph.py` | 2 | Full graph compilation, node count verification |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Graph engine | [LangGraph](https://github.com/langchain-ai/langgraph) with `StateGraph` |
| LLM | OpenAI GPT-4o-mini (via `langchain-openai`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Persistence | SQLite (via `langgraph-checkpoint-sqlite`) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Web search | Tavily |

---

## Roadmap

- [x] **Phase 1** — Foundation (router, retrieve, generate, fast path)
- [x] **Phase 2** — Corrective RAG (grade, web search, Layer 1 validation)
- [x] **Phase 3** — Governance + HITL (Layer 2 validation, interrupt/resume, dashboard)
- [ ] **Phase 4** — Evaluation + Packaging (benchmarks, Docker, metrics comparison)

---

## License

MIT
