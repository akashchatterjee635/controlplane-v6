"""Tests for the StateGraph construction and execution."""

import pytest
from app.graph import build_graph


def test_build_graph():
    """Test that the graph compiles successfully with the correct nodes and edges."""
    builder = build_graph()
    graph = builder.compile()
    
    nodes = graph.nodes
    
    # Fast path nodes
    assert "router" in nodes
    assert "retrieve_fast" in nodes
    assert "generate_fast" in nodes
    assert "validate_basic" in nodes
    
    # Verified path nodes
    assert "retrieve_verified" in nodes
    assert "grade_docs" in nodes
    assert "web_search" in nodes
    assert "generate_verified" in nodes
    assert "validate_full" in nodes
    
    # HITL node
    assert "human_review" in nodes


def test_graph_node_count():
    """Graph should have exactly 10 custom nodes (excluding __start__/__end__)."""
    builder = build_graph()
    graph = builder.compile()
    
    custom_nodes = [n for n in graph.nodes if not n.startswith("__")]
    assert len(custom_nodes) == 10
