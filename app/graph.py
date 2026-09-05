"""ControlPlane — LangGraph state graph construction.

Full pipeline:
  Fast path:     retrieve → generate → validate_fast → decision → (outcomes)
  Verified path: retrieve → grade → (generate | web_search) → parallel_validate 
                   → decision → (END | human_review | block) → audit_logger
"""

import os
from typing import Literal, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import ControlPlaneState
from app.nodes.router import router_node, route_decision
from app.nodes.retrieve import retrieve_node
from app.nodes.generate import generate_node
from app.utils.audit import create_audit_record, log_audit


def audit_logger_node(state: ControlPlaneState) -> dict[str, Any]:
    """LangGraph node: logs the final state to the audit log."""
    record = create_audit_record(dict(state))
    log_audit(record)
    import dataclasses
    return {"audit_record": dataclasses.asdict(record)}


def build_graph() -> StateGraph:
    """Construct the ControlPlane StateGraph.
    
    Returns an uncompiled StateGraph builder.
    """
    builder = StateGraph(ControlPlaneState)

    # ---- Register nodes ----
    builder.add_node("router", router_node)
    from app.nodes.grade import grade_documents_node, decide_to_generate
    from app.nodes.web_search import web_search_node
    from app.nodes.parallel_validate import validate_fast_node, parallel_validate_node
    from app.nodes.decision import decision_node, decision_routing, block_response_node
    from app.nodes.human_review import human_review_node
    
    # Fast path nodes
    builder.add_node("retrieve_fast", retrieve_node)
    builder.add_node("generate_fast", generate_node)
    builder.add_node("validate_fast", validate_fast_node)
    
    # Verified path nodes
    builder.add_node("retrieve_verified", retrieve_node)
    builder.add_node("grade_docs", grade_documents_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate_verified", generate_node)
    builder.add_node("parallel_validate", parallel_validate_node)
    
    # Shared decision/outcome nodes
    builder.add_node("decision", decision_node)
    builder.add_node("block_response", block_response_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("audit_logger", audit_logger_node)

    # ---- Wire edges ----
    builder.add_edge(START, "router")
    
    # Conditional routing based on complexity/risk scores
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve_fast": "retrieve_fast",
            "retrieve_verified": "retrieve_verified",
        },
    )

    # Fast path: retrieve → generate → validate_fast → decision
    builder.add_edge("retrieve_fast", "generate_fast")
    builder.add_edge("generate_fast", "validate_fast")
    builder.add_edge("validate_fast", "decision")

    # Verified path: retrieve → grade → (generate | web_search) → parallel_validate → decision
    builder.add_edge("retrieve_verified", "grade_docs")
    
    builder.add_conditional_edges(
        "grade_docs",
        decide_to_generate,
        {
            "generate_verified": "generate_verified",
            "web_search": "web_search",
        },
    )
    
    builder.add_edge("web_search", "generate_verified")
    builder.add_edge("generate_verified", "parallel_validate")
    builder.add_edge("parallel_validate", "decision")
    
    # Conditional edge: Decision Layer outcomes
    builder.add_conditional_edges(
        "decision",
        decision_routing,
        {
            "end": "audit_logger",
            "human_review": "human_review",
            "block_response": "block_response",
        },
    )
    
    # Re-converge to audit logger
    builder.add_edge("human_review", "audit_logger")
    builder.add_edge("block_response", "audit_logger")
    
    builder.add_edge("audit_logger", END)

    return builder


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, optionally with a checkpointer."""
    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)


def get_graph_with_sqlite(db_path: str | None = None):
    """Build and compile the graph with SQLite persistence."""
    if db_path is None:
        db_path = os.getenv("SQLITE_CHECKPOINT_PATH", "./checkpoints.db")
    
    class _GraphContext:
        def __init__(self, path: str):
            self._path = path
            self._saver = None
            self._graph = None
        
        def __enter__(self):
            self._saver = SqliteSaver.from_conn_string(self._path)
            self._saver.__enter__()
            self._graph = get_compiled_graph(checkpointer=self._saver)
            return self._graph
        
        def __exit__(self, *args):
            if self._saver:
                self._saver.__exit__(*args)
    
    return _GraphContext(db_path)
