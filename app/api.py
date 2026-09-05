"""ControlPlane — FastAPI application.

Provides REST endpoints for query processing, human review,
and status checking.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import get_compiled_graph

load_dotenv()

# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------
_checkpointer: SqliteSaver | None = None
_compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _checkpointer, _compiled_graph
    
    import sqlite3
    db_path = os.getenv("SQLITE_CHECKPOINT_PATH", "./checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _checkpointer = SqliteSaver(conn)
    _compiled_graph = get_compiled_graph(checkpointer=_checkpointer)
    
    yield
    
    if conn:
        conn.close()


app = FastAPI(
    title="ControlPlane",
    description="Adaptive RAG with Risk-Aware Routing and Human Oversight",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
# --- V2 Models ---
class QueryRequestV2(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    use_case: str = Field(default="default", description="The policy profile to use")
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional thread ID for conversation continuity",
    )

class QueryResponseV2(BaseModel):
    thread_id: str
    query: str
    use_case: str
    response: str
    route: str
    decision: str
    decision_reasoning: str
    risk_labels: list[str]
    complexity_score: int
    risk_score: int
    cost: dict[str, Any]
    audit_log: list[str]
    status: str  # "complete" | "pending_review"

# --- V1 Models (Legacy) ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional thread ID for conversation continuity",
    )


class QueryResponse(BaseModel):
    thread_id: str
    query: str
    response: str
    route: str
    complexity_score: int
    risk_score: int
    cost: dict[str, Any]
    audit_log: list[str]
    status: str  # "complete" | "pending_review"


class ReviewRequest(BaseModel):
    decision: Literal["approve", "redact", "deny"] = Field(
        ..., description="The reviewer's decision"
    )
    redacted_response: Optional[str] = Field(
        default=None,
        description="Replacement text when decision is 'redact'",
    )
    reason: str = Field(default="", description="Reason for the decision")
    reviewer: str = Field(default="api_reviewer", description="Name of the reviewer")


class ReviewResponse(BaseModel):
    thread_id: str
    decision: str
    response: str
    status: str


class PendingReviewItem(BaseModel):
    thread_id: str
    query: str
    use_case: str = "default"
    generated_response: str
    validation_flags: list[str]
    confidence: float
    decision_reasoning: str = ""
    risk_score: int
    complexity_score: int


class HealthResponse(BaseModel):
    status: str
    version: str


class ThreadStatusResponse(BaseModel):
    """Explicit DTO for thread status — never expose raw LangGraph state."""
    thread_id: str
    status: str          # "complete" | "pending_review"
    decision: str        # allow/edit/flag/review/block
    route: str           # fast/verified
    risk_labels: list[str]
    complexity_score: int
    risk_score: int


# ---------------------------------------------------------------------------
# API Key Authentication Stub
# ---------------------------------------------------------------------------
# NOTE: In production, replace with JWT/OAuth + RBAC with roles:
#   viewer, reviewer, policy_admin, system_admin
# The server should derive reviewer_id, tenant_id, roles from
# authenticated identity — not from user-controlled JSON.

from fastapi import Depends, Header

_API_KEY = os.getenv("CONTROLPLANE_API_KEY")  # None = auth disabled


async def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Optional API key verification. Enabled when CONTROLPLANE_API_KEY is set."""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.4.0")


@app.post("/api/v2/query", response_model=QueryResponseV2)
async def submit_query_v2(request: QueryRequestV2):
    """Submit a query for adaptive RAG processing with use-case governance."""
    if _compiled_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialized",
        )

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await run_in_threadpool(
            _compiled_graph.invoke,
            {"query": request.query, "use_case": request.use_case},
            config,
        )
    except Exception as e:
        # Check if this was an interrupt (HITL pause)
        try:
            graph_state = await run_in_threadpool(
                _compiled_graph.get_state, config
            )
            if graph_state and graph_state.next:
                # Graph is paused at human_review
                values = graph_state.values
                return QueryResponseV2(
                    thread_id=thread_id,
                    query=request.query,
                    use_case=request.use_case,
                    response=values.get("generation", ""),
                    route=values.get("route", "verified"),
                    decision=values.get("decision", "review"),
                    decision_reasoning=values.get("decision_reasoning", ""),
                    risk_labels=values.get("risk_labels", []),
                    complexity_score=values.get("complexity_score", 0),
                    risk_score=values.get("risk_score", 0),
                    cost=values.get("cost_tracker", {}),
                    audit_log=values.get("audit_log", []),
                    status="pending_review",
                )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution failed: {str(e)}",
        )

    return QueryResponseV2(
        thread_id=thread_id,
        query=request.query,
        use_case=request.use_case,
        response=result.get("generation", ""),
        route=result.get("route", "unknown"),
        decision=result.get("decision", "allow"),
        decision_reasoning=result.get("decision_reasoning", ""),
        risk_labels=result.get("risk_labels", []),
        complexity_score=result.get("complexity_score", 0),
        risk_score=result.get("risk_score", 0),
        cost=result.get("cost_tracker", {}),
        audit_log=result.get("audit_log", []),
        status="complete",
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Legacy v1 endpoint for backward compatibility."""
    if _compiled_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialized",
        )

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await run_in_threadpool(
            _compiled_graph.invoke,
            {"query": request.query},
            config,
        )
    except Exception as e:
        # Check if this was an interrupt (HITL pause)
        try:
            graph_state = await run_in_threadpool(
                _compiled_graph.get_state, config
            )
            if graph_state and graph_state.next:
                # Graph is paused at human_review
                values = graph_state.values
                return QueryResponse(
                    thread_id=thread_id,
                    query=request.query,
                    response=values.get("generation", ""),
                    route=values.get("route", "verified"),
                    complexity_score=values.get("complexity_score", 0),
                    risk_score=values.get("risk_score", 0),
                    cost=values.get("cost_tracker", {}),
                    audit_log=values.get("audit_log", []),
                    status="pending_review",
                )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution failed: {str(e)}",
        )

    return QueryResponse(
        thread_id=thread_id,
        query=request.query,
        response=result.get("generation", ""),
        route=result.get("route", "unknown"),
        complexity_score=result.get("complexity_score", 0),
        risk_score=result.get("risk_score", 0),
        cost=result.get("cost_tracker", {}),
        audit_log=result.get("audit_log", []),
        status="complete",
    )


@app.post("/api/v1/review/{thread_id}", response_model=ReviewResponse)
async def submit_review(thread_id: str, request: ReviewRequest):
    """Submit a human review decision and resume the paused graph.

    The graph must be paused at the human_review node (i.e., status
    is 'pending_review') for this endpoint to work.
    """
    if _compiled_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialized",
        )

    config = {"configurable": {"thread_id": thread_id}}

    # Verify the thread is actually paused
    try:
        graph_state = await run_in_threadpool(
            _compiled_graph.get_state, config
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    if not graph_state or not graph_state.next:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Thread {thread_id} is not pending review",
        )

    # Resume the graph with the reviewer's decision
    resume_value = {
        "decision": request.decision,
        "redacted_response": request.redacted_response,
        "reason": request.reason,
        "reviewer": request.reviewer,
    }

    try:
        result = await run_in_threadpool(
            _compiled_graph.invoke,
            Command(resume=resume_value),
            config,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume failed: {str(e)}",
        )

    return ReviewResponse(
        thread_id=thread_id,
        decision=request.decision,
        response=result.get("generation", ""),
        status="complete",
    )


@app.get("/api/v1/pending-reviews", response_model=list[PendingReviewItem])
async def get_pending_reviews():
    """List all threads currently awaiting human review.

    Scans the checkpointer for threads paused at the human_review node.
    """
    if _compiled_graph is None or _checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialized",
        )

    pending: list[PendingReviewItem] = []

    try:
        # List all checkpoints and find paused ones
        # Note: SqliteSaver.list() returns checkpoints; we iterate threads
        configs = await run_in_threadpool(
            lambda: list(_checkpointer.list(None))
        )

        seen_threads: set[str] = set()
        for checkpoint_tuple in configs:
            thread_id = checkpoint_tuple.config.get("configurable", {}).get(
                "thread_id", ""
            )
            if thread_id in seen_threads:
                continue
            seen_threads.add(thread_id)

            try:
                state = await run_in_threadpool(
                    _compiled_graph.get_state,
                    {"configurable": {"thread_id": thread_id}},
                )
                if state and state.next:
                    values = state.values
                    val_result = values.get("validation_result", {})
                    pending.append(
                        PendingReviewItem(
                            thread_id=thread_id,
                            query=values.get("query", ""),
                            generated_response=values.get("generation", ""),
                            validation_flags=val_result.get("flags", []),
                            confidence=val_result.get("confidence", 0),
                            risk_score=values.get("risk_score", 0),
                            complexity_score=values.get("complexity_score", 0),
                        )
                    )
            except Exception:
                continue

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pending reviews: {str(e)}",
        )

    return pending


@app.get("/api/v1/status/{thread_id}", response_model=ThreadStatusResponse)
async def get_thread_status(thread_id: str):
    """Check the status of a thread.

    Returns an explicit DTO — never exposes raw LangGraph state
    (which could contain prompts, retrieved documents, PII, etc.).
    """
    if _compiled_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialized",
        )

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await run_in_threadpool(_compiled_graph.get_state, config)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    values = state.values or {}
    return ThreadStatusResponse(
        thread_id=thread_id,
        status="pending_review" if state.next else "complete",
        decision=values.get("decision", "unknown"),
        route=values.get("route", "unknown"),
        risk_labels=values.get("risk_labels", []),
        complexity_score=values.get("complexity_score", 0),
        risk_score=values.get("risk_score", 0),
    )
