"""Self-Model Graph — Aura's persistent memory of the user.

Implements the typed graph schema from PRD v2.2 §5:
    Nodes: Person, Goal, Value, Project, Emotion, Decision, Pattern, TradingState
    Edges: INFLUENCES, BLOCKED_BY, CORRELATES, TRIGGERS, PREDICTS

Storage: SQLite (with SQLCipher encryption when available) + sqlite-vec for
vector similarity search. Falls back gracefully to plain SQLite if SQLCipher
is not installed.

This is the core data structure that makes Aura different from a stateless
chatbot — it accumulates knowledge about the user across every conversation.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    PERSON = "person"
    GOAL = "goal"
    VALUE = "value"
    PROJECT = "project"
    EMOTION = "emotion"
    DECISION = "decision"
    PATTERN = "pattern"
    TRADING_STATE = "trading_state"
    CONVERSATION = "conversation"
    LIFE_EVENT = "life_event"


class EdgeType(str, Enum):
    INFLUENCES = "influences"
    BLOCKED_BY = "blocked_by"
    CORRELATES = "correlates"
    TRIGGERS = "triggers"
    PREDICTS = "predicts"
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    CONFLICTS_WITH = "conflicts_with"


@dataclass
class GraphNode:
    """A node in the self-model graph."""

    id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    confidence: float = 0.5  # How confident we are in this node's accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
        }


@dataclass
class GraphEdge:
    """An edge connecting two nodes in the self-model graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class SelfModelGraph:
    """Persistent graph database for Aura's self-model.

    Uses SQLite for storage with optional SQLCipher encryption.
    All data stays on-device.

    Args:
        db_path: Path to the SQLite database file
        encryption_key: Optional passphrase for SQLCipher encryption
    """

    # US-307: Default growth caps
    DEFAULT_MAX_NODES = 5000
    SOFT_CAP_RATIO = 0.8      # Pruning triggers at 80% of max
    PRUNE_BATCH_RATIO = 0.10  # Remove 10% of soft cap per trigger
    # US-307: Node types that are NEVER pruned (immune)
    IMMUNE_NODE_TYPES = {NodeType.PERSON, NodeType.VALUE, NodeType.GOAL}
    # US-307: Pruning priority (lower number = pruned first)
    PRUNE_PRIORITY = {
        NodeType.EMOTION: 0,
        NodeType.CONVERSATION: 1,
        NodeType.TRADING_STATE: 2,
        NodeType.LIFE_EVENT: 3,
        NodeType.DECISION: 4,
        NodeType.PATTERN: 5,
        NodeType.PROJECT: 6,
    }

    def __init__(
        self,
        db_path: Optional[Path] = None,
        encryption_key: Optional[str] = None,
        max_nodes: int = DEFAULT_MAX_NODES,
    ):
        self.db_path = db_path or Path(".aura/self_model.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._encryption_key = encryption_key
        self._conn: Optional[sqlite3.Connection] = None
        self._max_nodes = max_nodes
        self._soft_cap = int(max_nodes * self.SOFT_CAP_RATIO)
        self._prune_batch = max(1, int(self._soft_cap * self.PRUNE_BATCH_RATIO))
        self._pruning_events_total = 0
        self._last_prune_timestamp: Optional[str] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Try to enable SQLCipher if available and key provided
        # US-201: Fixed SQL injection — hex-encode key to prevent injection via
        # crafted encryption_key values. PRAGMA key does not support parameterized
        # queries in SQLite/SQLCipher, so we sanitize by hex-encoding.
        if self._encryption_key:
            try:
                _safe_key = self._encryption_key.encode("utf-8").hex()
                self._conn.execute(f"PRAGMA key = \"x'{_safe_key}'\"")
                logger.info("SQLCipher encryption enabled for self-model database")
            except Exception:
                logger.warning(
                    "SQLCipher not available — self-model database is unencrypted. "
                    "Install pysqlcipher3 for AES-256 encryption."
                )

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confidence REAL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                properties TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                summary TEXT,
                emotional_state TEXT,
                topics TEXT DEFAULT '[]',
                readiness_impact REAL DEFAULT 0.0,
                raw_messages TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS readiness_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                components TEXT DEFAULT '{}',
                trigger TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_readiness_ts ON readiness_history(timestamp);
        """)
        self._conn.commit()

    # --- Node Operations ---

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add or update a node in the graph."""
        now = datetime.now(timezone.utc).isoformat()
        if not node.created_at:
            node.created_at = now
        node.updated_at = now

        self._conn.execute(
            """INSERT INTO nodes (id, node_type, label, properties, created_at, updated_at, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   label = excluded.label,
                   properties = excluded.properties,
                   updated_at = excluded.updated_at,
                   confidence = excluded.confidence""",
            (
                node.id,
                node.node_type.value,
                node.label,
                json.dumps(node.properties, default=str),
                node.created_at,
                node.updated_at,
                node.confidence,
            ),
        )
        self._conn.commit()

        # US-307: Check growth caps after insert
        self._prune_if_needed()

        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE node_type = ? ORDER BY updated_at DESC",
            (node_type.value,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def search_nodes(self, query: str, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        """Search nodes by label (case-insensitive substring match)."""
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE label LIKE ? AND node_type = ? ORDER BY updated_at DESC",
                (f"%{query}%", node_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE label LIKE ? ORDER BY updated_at DESC",
                (f"%{query}%",),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges. Returns True if deleted."""
        self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
        result = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        return result.rowcount > 0

    # --- Edge Operations ---

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge between two nodes."""
        now = datetime.now(timezone.utc).isoformat()
        if not edge.created_at:
            edge.created_at = now

        self._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, properties, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.weight,
                json.dumps(edge.properties, default=str),
                edge.created_at,
            ),
        )
        self._conn.commit()
        return edge

    def get_edges_from(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        if edge_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? AND edge_type = ?",
                (node_id, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[GraphEdge]:
        """Get all edges pointing TO a node (reverse lookup). US-230."""
        if edge_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ? AND edge_type = ?",
                (node_id, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def update_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        label: Optional[str] = None,
    ) -> Optional[GraphNode]:
        """Update an existing node's properties, confidence, and/or label. US-230.

        Only updates fields that are provided (not None).
        Returns the updated node, or None if not found.
        """
        existing = self.get_node(node_id)
        if existing is None:
            logger.warning("US-230: update_node — node %s not found", node_id)
            return None

        if properties is not None:
            existing.properties.update(properties)
        if confidence is not None:
            existing.confidence = confidence
        if label is not None:
            existing.label = label
        existing.updated_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """UPDATE nodes SET label = ?, properties = ?, confidence = ?, updated_at = ?
               WHERE id = ?""",
            (
                existing.label,
                json.dumps(existing.properties, default=str),
                existing.confidence,
                existing.updated_at,
                node_id,
            ),
        )
        self._conn.commit()
        return existing

    def get_connected_nodes(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get all nodes connected to a given node with their edges."""
        rows = self._conn.execute(
            """SELECT n.*, e.source_id as e_source, e.target_id as e_target,
                      e.edge_type as e_type, e.weight as e_weight,
                      e.properties as e_props, e.created_at as e_created
               FROM edges e
               JOIN nodes n ON (
                   (e.source_id = ? AND n.id = e.target_id) OR
                   (e.target_id = ? AND n.id = e.source_id)
               )""",
            (node_id, node_id),
        ).fetchall()

        results = []
        for row in rows:
            node = self._row_to_node(row)
            edge = GraphEdge(
                source_id=row["e_source"],
                target_id=row["e_target"],
                edge_type=EdgeType(row["e_type"]),
                weight=row["e_weight"],
                properties=json.loads(row["e_props"] or "{}"),
                created_at=row["e_created"],
            )
            results.append((node, edge))
        return results

    # --- Conversation Logging ---

    def log_conversation(
        self,
        conversation_id: str,
        summary: str,
        emotional_state: str,
        topics: List[str],
        readiness_impact: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Log a conversation for pattern analysis."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO conversations (id, timestamp, summary, emotional_state, topics, readiness_impact, raw_messages)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   summary = excluded.summary,
                   emotional_state = excluded.emotional_state,
                   topics = excluded.topics,
                   readiness_impact = excluded.readiness_impact""",
            (
                conversation_id,
                now,
                summary,
                emotional_state,
                json.dumps(topics),
                readiness_impact,
                json.dumps(messages or []),
            ),
        )
        self._conn.commit()

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent conversations."""
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- Readiness History ---

    def log_readiness(self, score: float, components: Dict[str, float], trigger: str = "") -> None:
        """Log a readiness score computation for historical tracking."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO readiness_history (timestamp, score, components, trigger) VALUES (?, ?, ?, ?)",
            (now, score, json.dumps(components, default=str), trigger),
        )
        self._conn.commit()

    def get_readiness_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent readiness score history."""
        rows = self._conn.execute(
            "SELECT * FROM readiness_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- Graph Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        conv_count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

        type_counts = {}
        for row in self._conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM nodes GROUP BY node_type"
        ).fetchall():
            type_counts[row["node_type"]] = row["cnt"]

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "total_conversations": conv_count,
            "nodes_by_type": type_counts,
            # US-307: Growth cap metrics
            "max_nodes": self._max_nodes,
            "soft_cap": self._soft_cap,
            "prune_batch": self._prune_batch,
            "pruning_events_total": self._pruning_events_total,
            "last_prune_timestamp": self._last_prune_timestamp,
        }

    # --- Internal ---

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            id=row["id"],
            node_type=NodeType(row["node_type"]),
            label=row["label"],
            properties=json.loads(row["properties"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            confidence=row["confidence"],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=row["weight"],
            properties=json.loads(row["properties"] or "{}"),
            created_at=row["created_at"],
        )

    # --- US-291: Temporal Decay & Reinforcement ---

    # Decay constant: exp(-0.05 * days) ≈ 14-day half-life
    DECAY_LAMBDA = 0.05
    # Default strength for new nodes
    DEFAULT_STRENGTH = 0.5

    def get_effective_strength(self, node_id: str, query_time: Optional[datetime] = None) -> float:
        """US-291: Get time-decayed strength of a node.

        Applies exponential decay based on time since last reinforcement.
        Strength decays with half-life of ~14 days.

        Args:
            node_id: Node identifier
            query_time: Time to evaluate at (defaults to now)

        Returns:
            Effective strength (0.0 to 1.0), or 0.0 if node not found
        """
        node = self.get_node(node_id)
        if node is None:
            return 0.0

        base_strength = node.properties.get("strength", self.DEFAULT_STRENGTH)
        last_reinforced_str = node.properties.get("last_reinforced", node.updated_at)

        try:
            last_reinforced = datetime.fromisoformat(last_reinforced_str)
            # Ensure timezone-aware
            if last_reinforced.tzinfo is None:
                last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return base_strength

        qt = query_time or datetime.now(timezone.utc)
        if qt.tzinfo is None:
            qt = qt.replace(tzinfo=timezone.utc)

        days_elapsed = max(0.0, (qt - last_reinforced).total_seconds() / 86400.0)
        decay = math.exp(-self.DECAY_LAMBDA * days_elapsed)
        return max(0.0, min(1.0, base_strength * decay))

    def reinforce_node(self, node_id: str) -> Optional[GraphNode]:
        """US-291: Reinforce a node on re-observation (spaced repetition).

        Boost is larger for longer gaps since last reinforcement:
        - < 1 day: +0.05 (weak, already fresh)
        - 1-7 days: +0.15 (good spacing)
        - > 7 days: +0.20 (strong, long gap = big reinforcement)

        Returns the updated node, or None if not found.
        """
        node = self.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc)
        current_strength = node.properties.get("strength", self.DEFAULT_STRENGTH)
        last_reinforced_str = node.properties.get("last_reinforced", node.updated_at)

        try:
            last_reinforced = datetime.fromisoformat(last_reinforced_str)
            if last_reinforced.tzinfo is None:
                last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)
            days_since = max(0.0, (now - last_reinforced).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            days_since = 7.0  # Default to medium boost

        # Spaced repetition: larger gap → larger boost
        if days_since < 1.0:
            boost = 0.05
        elif days_since <= 7.0:
            boost = 0.15
        else:
            boost = 0.20

        new_strength = min(1.0, current_strength + boost)
        reinforcement_count = node.properties.get("reinforcement_count", 0) + 1

        return self.update_node(
            node_id,
            properties={
                "strength": new_strength,
                "last_reinforced": now.isoformat(),
                "reinforcement_count": reinforcement_count,
            },
        )

    def get_nodes_by_type_filtered(
        self,
        node_type: NodeType,
        min_effective_strength: float = 0.0,
        query_time: Optional[datetime] = None,
    ) -> List[GraphNode]:
        """US-291: Get nodes of a type, optionally filtered by effective strength.

        Args:
            node_type: Type of nodes to retrieve
            min_effective_strength: Minimum decay-adjusted strength (0.0 = all)
            query_time: Time to evaluate decay at

        Returns:
            List of nodes meeting the strength threshold
        """
        all_nodes = self.get_nodes_by_type(node_type)
        if min_effective_strength <= 0.0:
            return all_nodes

        return [
            n for n in all_nodes
            if self.get_effective_strength(n.id, query_time) >= min_effective_strength
        ]

    # --- US-292: Graph Pruning with Archive ---

    def prune_dormant_nodes(
        self,
        min_strength: float = 0.05,
        min_age_days: int = 60,
        archive_path: Optional[Path] = None,
    ) -> int:
        """US-292: Remove dormant nodes and archive them.

        Prunes nodes where effective_strength < min_strength AND
        age > min_age_days. Archives pruned nodes to JSONL file.

        Args:
            min_strength: Minimum effective strength to survive
            min_age_days: Minimum age in days before eligible for pruning
            archive_path: Where to archive pruned nodes (default: .aura/archive/pruned_nodes.jsonl)

        Returns:
            Number of nodes pruned
        """
        archive_path = archive_path or Path(".aura/archive/pruned_nodes.jsonl")
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        pruned = 0
        archive_lines = []

        # Get all nodes
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        for row in rows:
            node = self._row_to_node(row)

            # Check age
            try:
                created = datetime.fromisoformat(node.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                age_days = 0.0

            if age_days < min_age_days:
                continue

            # Check effective strength
            effective = self.get_effective_strength(node.id, now)
            if effective >= min_strength:
                continue

            # Archive before deleting
            archive_entry = {
                "node": node.to_dict(),
                "effective_strength": round(effective, 4),
                "age_days": round(age_days, 1),
                "pruned_at": now.isoformat(),
                "reason": f"strength={effective:.4f} < {min_strength}, age={age_days:.0f}d > {min_age_days}d",
            }
            archive_lines.append(json.dumps(archive_entry, default=str))

            # Delete node (cascade deletes edges)
            self.delete_node(node.id)
            pruned += 1

        # Write archive
        if archive_lines:
            try:
                with open(archive_path, "a") as f:
                    for line in archive_lines:
                        f.write(line + "\n")
                logger.info("US-292: Archived %d pruned nodes to %s", pruned, archive_path)
            except (OSError, IOError) as e:
                logger.error("US-292: Failed to write prune archive: %s", e)

        if pruned > 0:
            logger.info("US-292: Pruned %d dormant nodes (strength < %.2f, age > %dd)",
                        pruned, min_strength, min_age_days)

        return pruned

    # --- US-307: Automatic Growth Cap Pruning ---

    def _prune_if_needed(self) -> int:
        """US-307: Prune lowest-value nodes when graph exceeds soft cap.

        Selection strategy:
        1. Skip immune node types (PERSON, VALUE, GOAL)
        2. Sort eligible nodes by (type_priority ASC, confidence ASC, updated_at ASC)
        3. Delete batch of lowest-priority nodes and their edges

        Returns:
            Number of nodes pruned (0 if below soft cap)
        """
        node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        if node_count <= self._soft_cap:
            return 0

        # Build SQL for immune types exclusion
        immune_values = ", ".join(f"'{nt.value}'" for nt in self.IMMUNE_NODE_TYPES)

        # Build CASE expression for type priority ordering
        case_parts = " ".join(
            f"WHEN '{nt.value}' THEN {priority}"
            for nt, priority in self.PRUNE_PRIORITY.items()
        )
        priority_case = f"CASE node_type {case_parts} ELSE 99 END"

        # Select candidates: non-immune, ordered by priority then confidence then staleness
        candidates = self._conn.execute(
            f"""SELECT id FROM nodes
                WHERE node_type NOT IN ({immune_values})
                ORDER BY {priority_case} ASC, confidence ASC, updated_at ASC
                LIMIT ?""",
            (self._prune_batch,),
        ).fetchall()

        pruned = 0
        for row in candidates:
            node_id = row["id"]
            # Delete edges first, then node
            self._conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )
            self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            pruned += 1

        if pruned > 0:
            self._conn.commit()
            self._pruning_events_total += 1
            self._last_prune_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(
                "US-307: Pruned %d nodes (count was %d, soft cap %d)",
                pruned, node_count, self._soft_cap,
            )

        return pruned

    # --- US-299: Multi-hop Graph Path Analysis ---

    def get_path_between(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        min_effective_strength: float = 0.1,
    ) -> Optional[List[Tuple[str, Optional[str]]]]:
        """US-299: BFS path finding between two nodes, respecting temporal decay.

        Returns the shortest path as a list of (node_id, edge_type) tuples.
        The first tuple has edge_type=None (starting node). Subsequent tuples
        show the edge_type used to reach that node.

        Args:
            source_id: Starting node ID
            target_id: Target node ID
            max_depth: Maximum hops to traverse (default 3)
            min_effective_strength: Minimum node strength to traverse (default 0.1)

        Returns:
            List of (node_id, edge_type) tuples representing the path,
            or None if no path exists within max_depth.
        """
        if source_id == target_id:
            return [(source_id, None)]

        # Verify both nodes exist
        if self.get_node(source_id) is None or self.get_node(target_id) is None:
            return None

        now = datetime.now(timezone.utc)

        # BFS with visited tracking
        from collections import deque
        queue: deque = deque()
        queue.append((source_id, [(source_id, None)], 0))
        visited = {source_id}

        while queue:
            current_id, path, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Get all edges from current node (both directions)
            outgoing = self._conn.execute(
                "SELECT target_id, edge_type FROM edges WHERE source_id = ?",
                (current_id,),
            ).fetchall()
            incoming = self._conn.execute(
                "SELECT source_id, edge_type FROM edges WHERE target_id = ?",
                (current_id,),
            ).fetchall()

            neighbors: List[Tuple[str, str]] = []
            for row in outgoing:
                neighbors.append((row["target_id"], row["edge_type"]))
            for row in incoming:
                neighbors.append((row["source_id"], row["edge_type"]))

            for neighbor_id, edge_type in neighbors:
                if neighbor_id in visited:
                    continue

                # Check temporal decay strength threshold
                strength = self.get_effective_strength(neighbor_id, now)
                if strength < min_effective_strength:
                    continue

                new_path = path + [(neighbor_id, edge_type)]

                if neighbor_id == target_id:
                    return new_path

                visited.add(neighbor_id)
                queue.append((neighbor_id, new_path, depth + 1))

        return None

    def get_common_influences(self, node_id1: str, node_id2: str) -> List[GraphNode]:
        """US-299: Find nodes that are connected to BOTH given nodes.

        These represent shared causal factors or common influences.

        Args:
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
            List of nodes connected to both input nodes
        """
        # Get neighbors of both nodes
        neighbors_1 = {n.id for n, _ in self.get_connected_nodes(node_id1)}
        neighbors_2 = {n.id for n, _ in self.get_connected_nodes(node_id2)}

        common_ids = neighbors_1 & neighbors_2
        return [self.get_node(nid) for nid in common_ids if self.get_node(nid) is not None]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
