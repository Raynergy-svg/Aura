"""Graph Topology Statistics — US-335.

Analyzes the structural properties of the self-model graph using networkx:
- Clustering coefficient: local triadic closure
- Betweenness centrality: node importance in information flow
- Density: ratio of actual to possible edges
- Community detection: Louvain modularity
- Modularity score: strength of community structure
- Largest component ratio: network fragmentation

All metrics are normalized to 0-1 range. Graceful fallback for small or
empty graphs. Handles networkx import failures gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.aura.core.self_model import SelfModelGraph


@dataclass
class GraphTopologyFeatures:
    """Structural features of the self-model graph (0-1 normalized)."""

    clustering_coefficient: float
    """Average clustering coefficient (0-1): local transitivity."""

    avg_betweenness: float
    """Mean betweenness centrality (0-1): node importance."""

    density: float
    """Graph density (0-1): ratio of actual to possible edges."""

    num_communities: float
    """Number of communities normalized (0-1, via min(len/10, 1))."""

    modularity: float
    """Modularity score (0-1): strength of community structure."""

    largest_component_ratio: float
    """Largest connected component ratio (0-1): fragmentation."""

    graph_context_score: float
    """Mean of all 6 features (0-1): overall graph health."""


class GraphTopologyAnalyzer:
    """Computes structural topology metrics for the self-model graph.

    Uses networkx (with graceful fallback if unavailable) to compute:
    - clustering coefficient
    - betweenness centrality
    - density
    - community detection (Louvain)
    - modularity
    - connected component analysis

    All metrics normalized to [0, 1] with sensible defaults for small/empty graphs.
    """

    # Try to import networkx at module load time (soft fail)
    _nx_available = False
    _nx = None

    def __init__(self):
        """Initialize the analyzer."""
        self._ensure_networkx()

    @classmethod
    def _ensure_networkx(cls) -> bool:
        """Lazy-load networkx. Returns True if available, False otherwise."""
        if cls._nx_available:
            return True

        try:
            import networkx as nx

            cls._nx = nx
            cls._nx_available = True
            logger.debug("networkx imported successfully for graph topology analysis")
            return True
        except ImportError:
            logger.warning(
                "networkx not available — graph topology analysis will return neutral defaults. "
                "Install with: pip install networkx"
            )
            return False

    def analyze(self, graph: SelfModelGraph) -> GraphTopologyFeatures:
        """Compute graph topology features from a SelfModelGraph.

        Args:
            graph: SelfModelGraph instance from src.aura.core.self_model

        Returns:
            GraphTopologyFeatures with 6 normalized metrics + composite score
        """
        if not self._nx_available:
            logger.warning("graphx unavailable, returning neutral topology features")
            return self._neutral_features()

        if graph is None or graph._conn is None:
            logger.warning("graph is None or closed, returning neutral features")
            return self._neutral_features()

        try:
            # Extract nodes and edges from SelfModelGraph
            nodes = self._get_nodes(graph)
            edges = self._get_edges(graph)

            if len(nodes) == 0:
                logger.debug("empty graph (0 nodes), returning neutral features")
                return self._neutral_features()

            # Build networkx Graph
            nx_graph = self._nx.Graph()
            nx_graph.add_nodes_from(nodes)
            nx_graph.add_weighted_edges_from(edges)

            logger.debug(
                f"built networkx graph: {len(nodes)} nodes, {len(edges)} edges"
            )

            # Compute features
            features = GraphTopologyFeatures(
                clustering_coefficient=self._compute_clustering(nx_graph),
                avg_betweenness=self._compute_avg_betweenness(nx_graph),
                density=self._compute_density(nx_graph),
                num_communities=self._compute_num_communities(nx_graph),
                modularity=self._compute_modularity(nx_graph),
                largest_component_ratio=self._compute_largest_component_ratio(
                    nx_graph
                ),
                graph_context_score=0.0,  # Computed after all others
            )

            # Compute composite score (mean of all 6)
            features.graph_context_score = (
                features.clustering_coefficient
                + features.avg_betweenness
                + features.density
                + features.num_communities
                + features.modularity
                + features.largest_component_ratio
            ) / 6.0

            logger.debug(
                f"computed topology features: clustering={features.clustering_coefficient:.3f}, "
                f"betweenness={features.avg_betweenness:.3f}, density={features.density:.3f}, "
                f"communities={features.num_communities:.3f}, modularity={features.modularity:.3f}, "
                f"largest_component={features.largest_component_ratio:.3f}, "
                f"context_score={features.graph_context_score:.3f}"
            )

            return features

        except Exception as e:
            logger.error(
                f"error computing graph topology: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return self._neutral_features()

    def _get_nodes(self, graph: SelfModelGraph) -> list:
        """Extract node IDs from SelfModelGraph.

        Args:
            graph: SelfModelGraph instance

        Returns:
            List of node IDs
        """
        try:
            rows = graph._conn.execute("SELECT id FROM nodes").fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"failed to extract nodes: {e}")
            return []

    def _get_edges(self, graph: SelfModelGraph) -> list:
        """Extract edges with weights from SelfModelGraph.

        Args:
            graph: SelfModelGraph instance

        Returns:
            List of (source_id, target_id, weight) tuples
        """
        try:
            rows = graph._conn.execute(
                "SELECT source_id, target_id, weight FROM edges"
            ).fetchall()
            return [(row[0], row[1], row[2]) for row in rows]
        except Exception as e:
            logger.error(f"failed to extract edges: {e}")
            return []

    def _compute_clustering(self, nx_graph) -> float:
        """Compute average clustering coefficient.

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float in [0, 1], default 0.0 if < 3 nodes
        """
        try:
            if nx_graph.number_of_nodes() < 3:
                return 0.0

            coeff = self._nx.average_clustering(nx_graph)
            return max(0.0, min(1.0, coeff))
        except Exception as e:
            logger.debug(f"clustering coefficient failed: {e}")
            return 0.0

    def _compute_avg_betweenness(self, nx_graph) -> float:
        """Compute mean betweenness centrality.

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float in [0, 1], default 0.0 if < 2 nodes
        """
        try:
            if nx_graph.number_of_nodes() < 2:
                return 0.0

            centrality = self._nx.betweenness_centrality(nx_graph)
            if not centrality:
                return 0.0

            mean_bc = sum(centrality.values()) / len(centrality)
            return max(0.0, min(1.0, mean_bc))
        except Exception as e:
            logger.debug(f"betweenness centrality failed: {e}")
            return 0.0

    def _compute_density(self, nx_graph) -> float:
        """Compute graph density.

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float in [0, 1], default 0.0 if empty
        """
        try:
            if nx_graph.number_of_nodes() == 0:
                return 0.0

            density = self._nx.density(nx_graph)
            return max(0.0, min(1.0, density))
        except Exception as e:
            logger.debug(f"density computation failed: {e}")
            return 0.0

    def _compute_num_communities(self, nx_graph) -> float:
        """Detect communities and return normalized count.

        Uses Louvain method with seed=42 for reproducibility.
        Normalized via min(count / 10, 1.0) to keep in [0, 1].

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float in [0, 1], default 0.0 if < 2 nodes or error
        """
        try:
            if nx_graph.number_of_nodes() < 2:
                return 0.0

            # Use Louvain community detection
            communities = self._nx.community.louvain_communities(
                nx_graph, seed=42
            )
            num_communities = len(communities)

            # Normalize: 1 community → 0.1, 10 communities → 1.0, 20+ → 1.0
            normalized = min(num_communities / 10.0, 1.0)
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            logger.debug(f"community detection failed: {e}")
            return 0.0

    def _compute_modularity(self, nx_graph) -> float:
        """Compute modularity of detected communities.

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float clamped to [0, 1], default 0.0 if error or < 2 nodes
        """
        try:
            if nx_graph.number_of_nodes() < 2:
                return 0.0

            # Detect communities first
            communities = self._nx.community.louvain_communities(
                nx_graph, seed=42
            )
            if not communities:
                return 0.0

            # Compute modularity (returns -0.5 to 1.0 range)
            mod = self._nx.community.modularity(nx_graph, communities)
            # Clamp to [0, 1]
            return max(0.0, min(1.0, mod))
        except Exception as e:
            logger.debug(f"modularity computation failed: {e}")
            return 0.0

    def _compute_largest_component_ratio(self, nx_graph) -> float:
        """Compute ratio of largest connected component.

        Args:
            nx_graph: networkx.Graph

        Returns:
            Float in [0, 1], default 1.0 for empty graph
        """
        try:
            if nx_graph.number_of_nodes() == 0:
                return 1.0

            # Get all connected components
            components = list(self._nx.connected_components(nx_graph))
            if not components:
                return 1.0

            largest_size = max(len(c) for c in components)
            ratio = largest_size / nx_graph.number_of_nodes()
            return max(0.0, min(1.0, ratio))
        except Exception as e:
            logger.debug(f"largest component ratio failed: {e}")
            return 1.0

    @staticmethod
    def _neutral_features() -> GraphTopologyFeatures:
        """Return neutral features (0.5 for all) when analysis unavailable.

        Returns:
            GraphTopologyFeatures with all values = 0.5
        """
        return GraphTopologyFeatures(
            clustering_coefficient=0.5,
            avg_betweenness=0.5,
            density=0.5,
            num_communities=0.5,
            modularity=0.5,
            largest_component_ratio=0.5,
            graph_context_score=0.5,
        )
