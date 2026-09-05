"""Tests for the StateGraph construction and execution."""

from app.graph import build_graph


def test_build_graph():
    """Test that the graph compiles with all required nodes."""
    builder = build_graph()
    
    # Check that nodes were added
    nodes = builder.nodes.keys()
    assert "router" in nodes
    assert "retrieve_fast" in nodes
    assert "generate_fast" in nodes
    assert "validate_fast" in nodes
    assert "retrieve_verified" in nodes
    assert "grade_docs" in nodes
    assert "web_search" in nodes
    assert "generate_verified" in nodes
    assert "parallel_validate" in nodes
    assert "decision" in nodes
    assert "human_review" in nodes
    assert "block_response" in nodes
    assert "audit_logger" in nodes


def test_graph_node_count():
    """Verify the exact number of nodes to ensure no unexpected additions."""
    builder = build_graph()
    # Subtract 1 for the __start__ node that langgraph adds internally
    custom_nodes = [n for n in builder.nodes if not n.startswith("__")]
    
    # 13 custom nodes in Phase 5:
    # router, retrieve_fast, generate_fast, validate_fast,
    # retrieve_verified, grade_docs, web_search, generate_verified, parallel_validate,
    # decision, human_review, block_response, audit_logger
    assert len(custom_nodes) == 13
