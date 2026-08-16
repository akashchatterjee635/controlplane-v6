"""ControlPlane v6 — LangGraph state graph construction.

Full pipeline:
  Fast path:     retrieve → generate → validate_basic → END
  Verified path: retrieve → grade → (generate | web_search) → validate_full
                   → (END | human_review → END)
"""

import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import ControlPlaneState
from app.nodes.router import router_node, route_decision
from app.nodes.retrieve import retrieve_node
from app.nodes.generate import generate_node


def build_graph() -> StateGraph:
    """Construct the ControlPlane StateGraph.
    
    Returns an uncompiled StateGraph builder.
    """
    builder = StateGraph(ControlPlaneState)

    # ---- Register nodes ----
    builder.add_node("router", router_node)
    from app.nodes.grade import grade_documents_node, decide_to_generate
    from app.nodes.web_search import web_search_node
    from app.nodes.validate import validate_basic_node, validate_full_node, triage_decision
    from app.nodes.human_review import human_review_node
    
    # Fast path nodes (same functions, different node names)
    builder.add_node("retrieve_fast", retrieve_node)
    builder.add_node("generate_fast", generate_node)
    builder.add_node("validate_basic", validate_basic_node)
    
    # Verified path nodes (same functions for now; Phase 2 will differentiate)
    builder.add_node("retrieve_verified", retrieve_node)
    builder.add_node("grade_docs", grade_documents_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate_verified", generate_node)
    builder.add_node("validate_full", validate_full_node)
    builder.add_node("human_review", human_review_node)

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

    # Fast path: retrieve → generate → validate_basic → END
    builder.add_edge("retrieve_fast", "generate_fast")
    builder.add_edge("generate_fast", "validate_basic")
    builder.add_edge("validate_basic", END)

    # Verified path: retrieve → grade → (generate | web_search) → validate → END
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
    builder.add_edge("generate_verified", "validate_full")
    
    # Conditional edge: if human review needed, pause; otherwise end
    builder.add_conditional_edges(
        "validate_full",
        triage_decision,
        {"end": END, "human_review": "human_review"},
    )
    builder.add_edge("human_review", END)

    return builder


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, optionally with a checkpointer.

    Args:
        checkpointer: A LangGraph checkpointer instance. If None,
                      graph runs without persistence (no HITL support).
    
    Returns:
        A compiled LangGraph runnable.
    """
    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)


def get_graph_with_sqlite(db_path: str | None = None):
    """Build and compile the graph with SQLite persistence.
    
    Returns a context manager that yields the compiled graph.
    Use with `with` statement:
        with get_graph_with_sqlite() as graph:
            result = graph.invoke({...}, config={...})
    """
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
