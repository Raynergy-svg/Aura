"""Unit tests for Aura SelfModelGraph — US-234.

Tests cover:
  1. Node CRUD (add, get, get_by_type, delete)
  2. update_node (US-230)
  3. Edge operations (add, get_edges_from, get_edges_to US-230)
  4. Search nodes
  5. Connected nodes (bidirectional)
  6. Conversation logging and retrieval
  7. Readiness history logging
  8. Graph stats
  9. Delete node cascades edges
  10. Override event validation (US-233 via ReadinessComputer)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.core.self_model import (
    SelfModelGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)


@pytest.fixture
def graph(tmp_path):
    """Create a fresh in-memory-like SQLite graph in tmp_path."""
    db_path = tmp_path / "test_graph.db"
    g = SelfModelGraph(db_path=db_path)
    return g


# ── 1. Node CRUD ─────────────────────────────────────────────────────────

def test_add_and_get_node(graph):
    node = GraphNode(id="p-1", node_type=NodeType.PERSON, label="David", confidence=0.9)
    result = graph.add_node(node)
    assert result.id == "p-1"

    fetched = graph.get_node("p-1")
    assert fetched is not None
    assert fetched.label == "David"
    assert fetched.confidence == 0.9


def test_get_node_not_found(graph):
    assert graph.get_node("nonexistent") is None


def test_get_nodes_by_type(graph):
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Profitable trading"))
    graph.add_node(GraphNode(id="g-2", node_type=NodeType.GOAL, label="Work-life balance"))
    graph.add_node(GraphNode(id="p-1", node_type=NodeType.PERSON, label="David"))

    goals = graph.get_nodes_by_type(NodeType.GOAL)
    assert len(goals) == 2
    persons = graph.get_nodes_by_type(NodeType.PERSON)
    assert len(persons) == 1


def test_delete_node(graph):
    graph.add_node(GraphNode(id="d-1", node_type=NodeType.DECISION, label="Test decision"))
    assert graph.delete_node("d-1") is True
    assert graph.get_node("d-1") is None


def test_delete_nonexistent_node(graph):
    assert graph.delete_node("nope") is False


# ── 2. update_node (US-230) ──────────────────────────────────────────────

def test_update_node_properties(graph):
    graph.add_node(GraphNode(id="p-1", node_type=NodeType.PERSON, label="David", confidence=0.8))

    updated = graph.update_node("p-1", properties={"mood": "calm"}, confidence=0.95)
    assert updated is not None
    assert updated.confidence == 0.95
    assert updated.properties["mood"] == "calm"

    # Verify persistence
    fetched = graph.get_node("p-1")
    assert fetched.confidence == 0.95
    assert fetched.properties["mood"] == "calm"


def test_update_node_label_only(graph):
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Old label"))
    updated = graph.update_node("g-1", label="New label")
    assert updated.label == "New label"


def test_update_node_not_found(graph):
    result = graph.update_node("nonexistent", confidence=0.5)
    assert result is None


# ── 3. Edge operations ───────────────────────────────────────────────────

def test_add_and_get_edges_from(graph):
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Goal"))
    graph.add_node(GraphNode(id="e-1", node_type=NodeType.EMOTION, label="Anxiety"))
    graph.add_edge(GraphEdge(
        source_id="e-1", target_id="g-1",
        edge_type=EdgeType.INFLUENCES, weight=0.7,
    ))

    edges = graph.get_edges_from("e-1")
    assert len(edges) == 1
    assert edges[0].target_id == "g-1"
    assert edges[0].weight == 0.7


def test_get_edges_from_filtered(graph):
    graph.add_node(GraphNode(id="a", node_type=NodeType.GOAL, label="A"))
    graph.add_node(GraphNode(id="b", node_type=NodeType.GOAL, label="B"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.INFLUENCES))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.SUPPORTS))

    filtered = graph.get_edges_from("a", edge_type=EdgeType.SUPPORTS)
    assert len(filtered) == 1
    assert filtered[0].edge_type == EdgeType.SUPPORTS


def test_get_edges_to(graph):
    """US-230: Reverse edge lookup."""
    graph.add_node(GraphNode(id="src", node_type=NodeType.DECISION, label="Src"))
    graph.add_node(GraphNode(id="tgt", node_type=NodeType.GOAL, label="Tgt"))
    graph.add_edge(GraphEdge(source_id="src", target_id="tgt", edge_type=EdgeType.INFLUENCES))

    # Forward
    fwd = graph.get_edges_from("src")
    assert len(fwd) == 1

    # Reverse
    rev = graph.get_edges_to("tgt")
    assert len(rev) == 1
    assert rev[0].source_id == "src"


def test_get_edges_to_filtered(graph):
    graph.add_node(GraphNode(id="a", node_type=NodeType.GOAL, label="A"))
    graph.add_node(GraphNode(id="b", node_type=NodeType.GOAL, label="B"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.INFLUENCES))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.TRIGGERS))

    filtered = graph.get_edges_to("b", edge_type=EdgeType.TRIGGERS)
    assert len(filtered) == 1


# ── 4. Search ────────────────────────────────────────────────────────────

def test_search_nodes(graph):
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Profitable trading"))
    graph.add_node(GraphNode(id="g-2", node_type=NodeType.GOAL, label="Work-life balance"))

    results = graph.search_nodes("trading")
    assert len(results) == 1
    assert results[0].label == "Profitable trading"


def test_search_nodes_by_type(graph):
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Trading"))
    graph.add_node(GraphNode(id="p-1", node_type=NodeType.PROJECT, label="Trading bot"))

    results = graph.search_nodes("Trading", node_type=NodeType.GOAL)
    assert len(results) == 1
    assert results[0].id == "g-1"


# ── 5. Connected nodes ──────────────────────────────────────────────────

def test_get_connected_nodes(graph):
    graph.add_node(GraphNode(id="center", node_type=NodeType.PERSON, label="Center"))
    graph.add_node(GraphNode(id="left", node_type=NodeType.GOAL, label="Left"))
    graph.add_node(GraphNode(id="right", node_type=NodeType.EMOTION, label="Right"))
    graph.add_edge(GraphEdge(source_id="center", target_id="left", edge_type=EdgeType.INFLUENCES))
    graph.add_edge(GraphEdge(source_id="right", target_id="center", edge_type=EdgeType.TRIGGERS))

    connected = graph.get_connected_nodes("center")
    assert len(connected) == 2
    labels = {node.label for node, edge in connected}
    assert labels == {"Left", "Right"}


# ── 6. Conversation logging ─────────────────────────────────────────────

def test_log_and_get_conversations(graph):
    graph.log_conversation(
        conversation_id="c-1",
        summary="Discussed trading anxiety",
        emotional_state="anxious",
        topics=["trading", "emotions"],
        readiness_impact=-5.0,
    )
    convos = graph.get_recent_conversations(limit=5)
    assert len(convos) == 1
    assert convos[0]["summary"] == "Discussed trading anxiety"


# ── 7. Readiness history ────────────────────────────────────────────────

def test_log_readiness_history(graph):
    graph.log_readiness(score=75.0, components={"emotional": 0.8}, trigger="scheduled")
    graph.log_readiness(score=60.0, components={"emotional": 0.6}, trigger="conversation")
    history = graph.get_readiness_history(limit=10)
    assert len(history) == 2
    # Most recent first
    assert history[0]["score"] == 60.0


# ── 8. Graph stats ──────────────────────────────────────────────────────

def test_graph_stats(graph):
    graph.add_node(GraphNode(id="p-1", node_type=NodeType.PERSON, label="David"))
    graph.add_node(GraphNode(id="g-1", node_type=NodeType.GOAL, label="Goal"))
    graph.add_edge(GraphEdge(source_id="p-1", target_id="g-1", edge_type=EdgeType.INFLUENCES))

    stats = graph.get_stats()
    assert stats["total_nodes"] == 2
    assert stats["total_edges"] == 1


# ── 9. Delete node cascades edges ───────────────────────────────────────

def test_delete_node_cascades_edges(graph):
    graph.add_node(GraphNode(id="a", node_type=NodeType.GOAL, label="A"))
    graph.add_node(GraphNode(id="b", node_type=NodeType.GOAL, label="B"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.INFLUENCES))

    graph.delete_node("a")
    edges = graph.get_edges_from("a")
    assert len(edges) == 0
    # Also reverse
    edges_to = graph.get_edges_to("b")
    assert len(edges_to) == 0


# ── 10. Override event validation (US-233 integration) ───────────────────

def test_override_validation_malformed_events(tmp_path):
    """Malformed override events should be skipped, not crash."""
    from src.aura.core.readiness import ReadinessComputer

    rc = ReadinessComputer(signal_path=tmp_path / "bridge" / "sig.json", circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(
        recent_override_events=[
            {"trade_won": False},
            {"bad_key": "no trade_won"},   # Missing key — skipped
            "not a dict",                  # Not a dict — skipped
            {"trade_won": True},
        ],
    )
    # Only 2 valid events: 1 loss, 1 win → 50% loss rate
    assert signal.override_loss_rate_7d == 0.5
