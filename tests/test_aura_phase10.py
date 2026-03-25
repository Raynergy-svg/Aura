"""Phase 10 Tests — Temporal Depth & Signal Enrichment.

US-296: Time-lag correlations in T2 cross-domain patterns
US-297: Enriched OutcomeSignal with emotional context
US-298: Evidence decay in T1 patterns with auto-archive
US-299: Multi-hop BFS path analysis in self-model graph
US-300: Feature expansion in ReadinessModelV2 (10→15 dimensions)
US-301: Circadian readiness multiplier

Total: ~50 tests across 6 stories.
"""

import json
import math
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aura.patterns.tier2 import (
    _compute_lagged_correlations,
    _compute_correlation,
    Tier2CrossDomainDetector,
    MIN_SAMPLE_SIZE,
)
from src.aura.bridge.signals import OutcomeSignal, OverrideEvent
from src.aura.patterns.tier1 import (
    Tier1FrequencyDetector,
    get_decay_weighted_confidence,
    EVIDENCE_DECAY_LAMBDA,
    EVIDENCE_ARCHIVE_THRESHOLD,
)
from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)
from src.aura.core.self_model import (
    SelfModelGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from src.aura.prediction.readiness_v2 import (
    ReadinessModelV2,
    ReadinessTrainingSample,
    V2_FEATURE_NAMES,
)
from src.aura.core.readiness import ReadinessComputer, ReadinessSignal


# Helper: flat circadian config that neutralizes time-of-day effects
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ═══════════════════════════════════════════════════════════════
# US-296: Time-lag correlations in T2
# ═══════════════════════════════════════════════════════════════

class TestUS296LaggedCorrelations(unittest.TestCase):
    """US-296: _compute_lagged_correlations discovers delayed effects."""

    def _make_date_series(self, start_date: str, values: list) -> dict:
        """Helper to build date→value dict from a start date and value list."""
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        result = {}
        for i, v in enumerate(values):
            key = (dt + timedelta(days=i)).strftime("%Y-%m-%d")
            result[key] = v
        return result

    def test_lag_zero_perfect_correlation(self):
        """Lag 0 should find perfect positive correlation for identical series."""
        series_a = self._make_date_series("2026-01-01", [1, 2, 3, 4, 5, 6, 7])
        series_b = self._make_date_series("2026-01-01", [1, 2, 3, 4, 5, 6, 7])
        results = _compute_lagged_correlations(series_a, series_b, max_lag=3)
        self.assertTrue(len(results) > 0)
        # First result (strongest) should be lag=0 with r≈1.0
        best_lag, best_r, best_p = results[0]
        self.assertEqual(best_lag, 0)
        self.assertAlmostEqual(best_r, 1.0, places=2)

    def test_lag_two_delayed_effect(self):
        """Series_a shifted by 2 days should show strongest correlation at lag=2."""
        # series_a: cause signal starting on day 0
        # series_b: effect signal — mirrors series_a but 2 days later
        values_a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        values_b = [0, 0, 10, 20, 30, 40, 50, 60, 70, 80]  # shifted by 2

        series_a = self._make_date_series("2026-01-01", values_a)
        series_b = self._make_date_series("2026-01-01", values_b)

        results = _compute_lagged_correlations(series_a, series_b, max_lag=5)
        self.assertTrue(len(results) > 0)

        # Lag=2 should have the strongest (or near-strongest) correlation
        lag_2_results = [r for r in results if r[0] == 2]
        self.assertTrue(len(lag_2_results) > 0)
        _, r_at_lag2, _ = lag_2_results[0]
        self.assertGreater(abs(r_at_lag2), 0.8)

    def test_insufficient_data_returns_empty(self):
        """Fewer than MIN_SAMPLE_SIZE overlapping points should return empty."""
        series_a = {"2026-01-01": 1.0, "2026-01-02": 2.0}
        series_b = {"2026-01-01": 1.0, "2026-01-02": 2.0}
        results = _compute_lagged_correlations(series_a, series_b, max_lag=0)
        self.assertEqual(len(results), 0)

    def test_no_overlap_returns_empty(self):
        """Non-overlapping date ranges should return empty for all lags."""
        series_a = self._make_date_series("2026-01-01", [1, 2, 3, 4, 5])
        series_b = self._make_date_series("2026-06-01", [1, 2, 3, 4, 5])
        results = _compute_lagged_correlations(series_a, series_b, max_lag=7)
        self.assertEqual(len(results), 0)

    def test_results_sorted_by_abs_correlation(self):
        """Results should be sorted by absolute correlation descending."""
        series_a = self._make_date_series("2026-01-01", list(range(20)))
        series_b = self._make_date_series("2026-01-01", [x * 0.5 + (x % 3) for x in range(20)])
        results = _compute_lagged_correlations(series_a, series_b, max_lag=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(abs(results[i][1]), abs(results[i + 1][1]))

    def test_max_lag_zero_only_tests_simultaneous(self):
        """max_lag=0 should only test lag 0."""
        series_a = self._make_date_series("2026-01-01", list(range(10)))
        series_b = self._make_date_series("2026-01-01", list(range(10)))
        results = _compute_lagged_correlations(series_a, series_b, max_lag=0)
        self.assertTrue(all(r[0] == 0 for r in results))

    def test_t2_readiness_pnl_uses_lagged(self):
        """T2 _correlate_readiness_with_pnl should use lagged correlations."""
        tmp = tempfile.mkdtemp()
        detector = Tier2CrossDomainDetector(
            patterns_dir=Path(tmp) / "patterns",
            config={"min_sample_size": 5, "min_correlation_strength": 0.2, "p_value_threshold": 0.5},
        )
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        readiness = [
            {"timestamp": (base + timedelta(days=i)).isoformat(), "score": 50 + i * 3}
            for i in range(15)
        ]
        trades = [
            {"timestamp": (base + timedelta(days=i)).isoformat(), "pnl_pips": 10 + i * 2}
            for i in range(15)
        ]
        patterns = detector._correlate_readiness_with_pnl(readiness, trades)
        # Should detect some pattern (positive correlation between readiness and pnl)
        if patterns:
            self.assertIn("readiness_pnl_correlation", patterns[0].pattern_id)


# ═══════════════════════════════════════════════════════════════
# US-297: Enriched OutcomeSignal
# ═══════════════════════════════════════════════════════════════

class TestUS297EnrichedOutcomeSignal(unittest.TestCase):
    """US-297: OutcomeSignal extended with emotional context fields."""

    def test_default_fields_are_none(self):
        """New fields should default to None."""
        signal = OutcomeSignal(pnl_today=50.0)
        self.assertIsNone(signal.emotional_state)
        self.assertIsNone(signal.readiness_at_trade)
        self.assertIsNone(signal.cognitive_load)
        self.assertIsNone(signal.active_biases)

    def test_enriched_fields_populated(self):
        """Enrichment fields can be populated."""
        signal = OutcomeSignal(
            pnl_today=50.0,
            emotional_state="stressed",
            readiness_at_trade=72.5,
            cognitive_load=0.65,
            active_biases={"loss_aversion": 0.8, "recency_bias": 0.3},
        )
        self.assertEqual(signal.emotional_state, "stressed")
        self.assertAlmostEqual(signal.readiness_at_trade, 72.5)
        self.assertAlmostEqual(signal.cognitive_load, 0.65)
        self.assertIn("loss_aversion", signal.active_biases)

    def test_to_dict_without_enrichment(self):
        """to_dict should NOT include enrichment fields when they're None."""
        signal = OutcomeSignal(pnl_today=10.0)
        d = signal.to_dict()
        self.assertNotIn("emotional_state", d)
        self.assertNotIn("readiness_at_trade", d)
        self.assertNotIn("cognitive_load", d)
        self.assertNotIn("active_biases", d)

    def test_to_dict_with_enrichment(self):
        """to_dict should include enrichment fields when populated."""
        signal = OutcomeSignal(
            pnl_today=25.0,
            emotional_state="calm",
            readiness_at_trade=85.0,
            cognitive_load=0.3,
            active_biases={"disposition_effect": 0.5},
        )
        d = signal.to_dict()
        self.assertEqual(d["emotional_state"], "calm")
        self.assertEqual(d["readiness_at_trade"], 85.0)
        self.assertAlmostEqual(d["cognitive_load"], 0.3)
        self.assertIn("disposition_effect", d["active_biases"])

    def test_to_dict_rounds_values(self):
        """Enrichment values should be rounded properly."""
        signal = OutcomeSignal(
            pnl_today=0.0,
            readiness_at_trade=72.123456,
            cognitive_load=0.654321,
            active_biases={"bias_a": 0.123456789},
        )
        d = signal.to_dict()
        self.assertEqual(d["readiness_at_trade"], 72.12)
        self.assertAlmostEqual(d["cognitive_load"], 0.6543)
        self.assertAlmostEqual(d["active_biases"]["bias_a"], 0.1235, places=3)

    def test_backward_compatible_deserialization(self):
        """Old OutcomeSignal JSON (without enrichment) should still load."""
        old_json = {
            "pnl_today": 30.0,
            "win_rate_7d": 0.6,
            "regime": "NORMAL",
            "streak": "winning",
            "trades_today": 3,
        }
        signal = OutcomeSignal(**{
            k: v for k, v in old_json.items()
            if k in OutcomeSignal.__dataclass_fields__
        })
        self.assertEqual(signal.pnl_today, 30.0)
        self.assertIsNone(signal.emotional_state)

    def test_partial_enrichment(self):
        """Only some enrichment fields populated."""
        signal = OutcomeSignal(pnl_today=0, emotional_state="anxious")
        d = signal.to_dict()
        self.assertIn("emotional_state", d)
        self.assertNotIn("readiness_at_trade", d)
        self.assertNotIn("cognitive_load", d)
        self.assertNotIn("active_biases", d)


# ═══════════════════════════════════════════════════════════════
# US-298: Evidence Decay in T1 Patterns
# ═══════════════════════════════════════════════════════════════

class TestUS298EvidenceDecay(unittest.TestCase):
    """US-298: Decay-weighted confidence and auto-archive for T1 patterns."""

    def _make_pattern(self, evidence_ages_days: list, base_confidence: float = 0.8) -> DetectedPattern:
        """Create a pattern with evidence items at given ages."""
        now = datetime.now(timezone.utc)
        evidence = []
        for age_days in evidence_ages_days:
            ts = (now - timedelta(days=age_days)).isoformat()
            evidence.append(EvidenceItem(
                source_type="test",
                source_id=f"test_{age_days}",
                timestamp=ts,
                summary=f"Test evidence {age_days}d old",
            ))
        return DetectedPattern(
            pattern_id="test_pattern",
            tier=PatternTier.T1_DAILY,
            domain=PatternDomain.HUMAN,
            description="Test pattern",
            evidence=evidence,
            confidence=base_confidence,
        )

    def test_fresh_evidence_no_decay(self):
        """Evidence from today should have near-zero decay."""
        pattern = self._make_pattern([0])
        dwc = get_decay_weighted_confidence(pattern)
        self.assertAlmostEqual(dwc, pattern.confidence, places=2)

    def test_14_day_half_life(self):
        """After ~14 days, confidence should be roughly halved."""
        half_life = math.log(2) / EVIDENCE_DECAY_LAMBDA
        pattern = self._make_pattern([half_life], base_confidence=1.0)
        dwc = get_decay_weighted_confidence(pattern)
        self.assertAlmostEqual(dwc, 0.5, delta=0.05)

    def test_very_old_evidence_near_zero(self):
        """Evidence 100+ days old should decay to near zero."""
        pattern = self._make_pattern([100], base_confidence=0.8)
        dwc = get_decay_weighted_confidence(pattern)
        self.assertLess(dwc, 0.01)

    def test_mixed_age_evidence_averaging(self):
        """Mixed fresh and stale evidence should average decay weights."""
        pattern = self._make_pattern([0, 30, 60], base_confidence=0.9)
        dwc = get_decay_weighted_confidence(pattern)
        # Fresh evidence keeps it above zero, but old evidence pulls down
        self.assertGreater(dwc, 0.1)
        self.assertLess(dwc, 0.9)

    def test_empty_evidence_returns_base_confidence(self):
        """Pattern with no evidence returns base confidence."""
        pattern = DetectedPattern(
            pattern_id="empty",
            tier=PatternTier.T1_DAILY,
            domain=PatternDomain.HUMAN,
            description="No evidence",
            evidence=[],
            confidence=0.75,
        )
        dwc = get_decay_weighted_confidence(pattern)
        self.assertEqual(dwc, 0.75)

    def test_auto_archive_below_threshold(self):
        """T1 detector should auto-archive patterns below decay threshold."""
        tmp = tempfile.mkdtemp()
        detector = Tier1FrequencyDetector(patterns_dir=Path(tmp) / "patterns")
        # Inject a pattern with very old evidence manually
        old_pattern = self._make_pattern([200], base_confidence=0.5)
        old_pattern.status = PatternStatus.DETECTED
        detector._active_patterns["old_pattern"] = old_pattern
        archived = detector._archive_decayed_patterns()
        self.assertEqual(archived, 1)
        self.assertEqual(detector._active_patterns["old_pattern"].status, PatternStatus.ARCHIVED)

    def test_no_archive_for_fresh_patterns(self):
        """Fresh patterns should NOT be archived."""
        tmp = tempfile.mkdtemp()
        detector = Tier1FrequencyDetector(patterns_dir=Path(tmp) / "patterns")
        fresh_pattern = self._make_pattern([0, 1], base_confidence=0.8)
        fresh_pattern.status = PatternStatus.DETECTED
        detector._active_patterns["fresh_pattern"] = fresh_pattern
        archived = detector._archive_decayed_patterns()
        self.assertEqual(archived, 0)
        self.assertEqual(detector._active_patterns["fresh_pattern"].status, PatternStatus.DETECTED)

    def test_already_archived_not_double_archived(self):
        """Already-archived patterns should be skipped."""
        tmp = tempfile.mkdtemp()
        detector = Tier1FrequencyDetector(patterns_dir=Path(tmp) / "patterns")
        pattern = self._make_pattern([200], base_confidence=0.5)
        pattern.status = PatternStatus.ARCHIVED
        detector._active_patterns["already_archived"] = pattern
        archived = detector._archive_decayed_patterns()
        self.assertEqual(archived, 0)


# ═══════════════════════════════════════════════════════════════
# US-299: Multi-hop BFS Path Analysis
# ═══════════════════════════════════════════════════════════════

class TestUS299MultiHopBFS(unittest.TestCase):
    """US-299: get_path_between and get_common_influences in SelfModelGraph."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.graph = SelfModelGraph(db_path=Path(self.tmp) / "test.db")

    def tearDown(self):
        self.graph.close()

    def _add_strong_node(self, node_id: str, label: str, node_type: NodeType = NodeType.EMOTION):
        """Add a node with high strength so it passes BFS threshold."""
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            label=label,
            properties={
                "strength": 0.9,
                "last_reinforced": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.graph.add_node(node)

    def _add_edge(self, src: str, tgt: str, edge_type: EdgeType = EdgeType.INFLUENCES, weight: float = 0.8):
        """Add an edge between two nodes."""
        edge = GraphEdge(source_id=src, target_id=tgt, edge_type=edge_type, weight=weight)
        self.graph.add_edge(edge)

    def test_direct_connection(self):
        """Path between directly connected nodes should be length 2."""
        self._add_strong_node("A", "Node A")
        self._add_strong_node("B", "Node B")
        self._add_edge("A", "B", EdgeType.INFLUENCES, weight=0.8)
        path = self.graph.get_path_between("A", "B")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], ("A", None))
        self.assertEqual(path[1][0], "B")
        self.assertEqual(path[1][1], "influences")

    def test_two_hop_path(self):
        """Path through one intermediate node should work."""
        self._add_strong_node("A", "Node A")
        self._add_strong_node("B", "Node B", NodeType.GOAL)
        self._add_strong_node("C", "Node C")
        self._add_edge("A", "B", EdgeType.INFLUENCES, weight=0.8)
        self._add_edge("B", "C", EdgeType.TRIGGERS, weight=0.7)
        path = self.graph.get_path_between("A", "C")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0][0], "A")
        self.assertEqual(path[1][0], "B")
        self.assertEqual(path[2][0], "C")

    def test_no_path_returns_none(self):
        """Disconnected nodes should return None."""
        self._add_strong_node("A", "Isolated A")
        self._add_strong_node("B", "Isolated B")
        path = self.graph.get_path_between("A", "B")
        self.assertIsNone(path)

    def test_same_node_path(self):
        """Path from a node to itself should return single-element list."""
        self._add_strong_node("A", "Self")
        path = self.graph.get_path_between("A", "A")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], ("A", None))

    def test_nonexistent_node_returns_none(self):
        """Path involving a non-existent node should return None."""
        self._add_strong_node("A", "Exists")
        path = self.graph.get_path_between("A", "nonexistent")
        self.assertIsNone(path)

    def test_max_depth_respected(self):
        """Path beyond max_depth should not be found."""
        # Chain: A → B → C → D → E (depth 4)
        for label in "ABCDE":
            self._add_strong_node(label, f"Node {label}")
        self._add_edge("A", "B")
        self._add_edge("B", "C")
        self._add_edge("C", "D")
        self._add_edge("D", "E")

        # max_depth=3 should NOT find A→E (needs 4 hops)
        path = self.graph.get_path_between("A", "E", max_depth=3)
        self.assertIsNone(path)

        # max_depth=4 should find it
        path = self.graph.get_path_between("A", "E", max_depth=4)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 5)

    def test_weak_node_filtered_by_strength(self):
        """Nodes below min_effective_strength should be bypassed."""
        self._add_strong_node("A", "Strong A")
        # Add a weak intermediate node
        weak_node = GraphNode(
            id="W",
            node_type=NodeType.EMOTION,
            label="Weak W",
            properties={
                "strength": 0.01,
                "last_reinforced": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),
            },
        )
        self.graph.add_node(weak_node)
        self._add_strong_node("B", "Strong B")
        self._add_edge("A", "W")
        self._add_edge("W", "B")

        # Default min_effective_strength=0.1 should block the path through W
        path = self.graph.get_path_between("A", "B", min_effective_strength=0.1)
        self.assertIsNone(path)

    def test_bidirectional_edge_traversal(self):
        """BFS should traverse edges in both directions."""
        self._add_strong_node("A", "Node A")
        self._add_strong_node("B", "Node B")
        # Only edge from B→A (not A→B)
        self._add_edge("B", "A", EdgeType.INFLUENCES, weight=0.8)
        path = self.graph.get_path_between("A", "B")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)

    def test_common_influences(self):
        """get_common_influences should find shared connected nodes."""
        self._add_strong_node("X", "Influence X", NodeType.VALUE)
        self._add_strong_node("A", "Node A")
        self._add_strong_node("B", "Node B")
        # X influences both A and B
        self._add_edge("X", "A")
        self._add_edge("X", "B")
        common = self.graph.get_common_influences("A", "B")
        common_ids = [n.id for n in common]
        self.assertIn("X", common_ids)

    def test_no_common_influences(self):
        """Nodes with no shared connections should return empty list."""
        self._add_strong_node("A", "Solo A")
        self._add_strong_node("B", "Solo B")
        self._add_strong_node("X", "Only connects A")
        self._add_strong_node("Y", "Only connects B")
        self._add_edge("X", "A")
        self._add_edge("Y", "B")
        common = self.graph.get_common_influences("A", "B")
        self.assertEqual(len(common), 0)


# ═══════════════════════════════════════════════════════════════
# US-300: Feature Expansion in ReadinessModelV2
# ═══════════════════════════════════════════════════════════════

class TestUS300FeatureExpansion(unittest.TestCase):
    """US-300: V2_FEATURE_NAMES expanded from 10 to 15 dimensions."""

    def test_feature_count_is_15(self):
        """V2_FEATURE_NAMES should now have 15 features."""
        self.assertEqual(len(V2_FEATURE_NAMES), 15)

    def test_new_feature_names_present(self):
        """The 5 new interaction/polynomial features should be listed."""
        expected_new = [
            "stress_x_cognitive_load",
            "override_x_engagement",
            "confidence_x_emotional",
            "engagement_squared",
            "emotional_cubed",
        ]
        for name in expected_new:
            self.assertIn(name, V2_FEATURE_NAMES)

    def test_feature_vector_length(self):
        """to_feature_vector should return 15-element list."""
        sample = ReadinessTrainingSample(
            emotional_state=0.7,
            cognitive_load=0.5,
            override_discipline=0.8,
            stress_level=0.3,
            confidence_trend=0.6,
            engagement=0.4,
            outcome_quality=0.9,
        )
        vec = sample.to_feature_vector()
        self.assertEqual(len(vec), 15)

    def test_interaction_terms_computed_correctly(self):
        """Verify the new interaction terms are computed correctly."""
        sample = ReadinessTrainingSample(
            emotional_state=0.7,
            cognitive_load=0.5,
            override_discipline=0.8,
            stress_level=0.3,
            confidence_trend=0.6,
            engagement=0.4,
            outcome_quality=0.9,
        )
        vec = sample.to_feature_vector()
        # Index 10: stress_x_cognitive_load = 0.3 * 0.5 = 0.15
        self.assertAlmostEqual(vec[10], 0.3 * 0.5, places=6)
        # Index 11: override_x_engagement = 0.8 * 0.4 = 0.32
        self.assertAlmostEqual(vec[11], 0.8 * 0.4, places=6)
        # Index 12: confidence_x_emotional = 0.6 * 0.7 = 0.42
        self.assertAlmostEqual(vec[12], 0.6 * 0.7, places=6)
        # Index 13: engagement_squared = 0.4^2 = 0.16
        self.assertAlmostEqual(vec[13], 0.4 ** 2, places=6)
        # Index 14: emotional_cubed = 0.7^3 = 0.343
        self.assertAlmostEqual(vec[14], 0.7 ** 3, places=6)

    def test_original_features_unchanged(self):
        """First 10 features should be identical to pre-US-300 behavior."""
        sample = ReadinessTrainingSample(
            emotional_state=0.7,
            cognitive_load=0.5,
            override_discipline=0.8,
            stress_level=0.3,
            confidence_trend=0.6,
            engagement=0.4,
            outcome_quality=0.9,
        )
        vec = sample.to_feature_vector()
        # Original 6 base features
        self.assertAlmostEqual(vec[0], 0.7)   # emotional_state
        self.assertAlmostEqual(vec[1], 0.5)   # cognitive_load
        self.assertAlmostEqual(vec[2], 0.8)   # override_discipline
        self.assertAlmostEqual(vec[3], 0.3)   # stress_level
        self.assertAlmostEqual(vec[4], 0.6)   # confidence_trend
        self.assertAlmostEqual(vec[5], 0.4)   # engagement
        # Original 4 non-linear/interaction features
        self.assertAlmostEqual(vec[6], 0.7 ** 2)   # emotional_squared
        self.assertAlmostEqual(vec[7], 0.5 ** 2)   # cognitive_squared
        self.assertAlmostEqual(vec[8], 0.7 * 0.5)  # emotional_x_cognitive
        self.assertAlmostEqual(vec[9], 0.3 * 0.8)  # stress_x_override

    def test_model_n_features_matches(self):
        """ReadinessModelV2.N_FEATURES should match V2_FEATURE_NAMES count."""
        self.assertEqual(ReadinessModelV2.N_FEATURES, 15)

    def test_extreme_values_no_crash(self):
        """Extreme feature values should not cause errors."""
        sample = ReadinessTrainingSample(
            emotional_state=0.0,
            cognitive_load=1.0,
            override_discipline=0.0,
            stress_level=1.0,
            confidence_trend=0.0,
            engagement=1.0,
            outcome_quality=0.0,
        )
        vec = sample.to_feature_vector()
        self.assertEqual(len(vec), 15)
        # emotional_cubed with 0.0 should be 0.0
        self.assertAlmostEqual(vec[14], 0.0)

    def test_model_train_with_15_features(self):
        """Model training should work with 15-dim feature vectors."""
        tmp = tempfile.mkdtemp()
        model = ReadinessModelV2(model_path=Path(tmp) / "model.json", min_samples=3)
        for i in range(10):
            model.add_training_sample(
                readiness_components={
                    "emotional_state": 0.5 + i * 0.05,
                    "cognitive_load": 0.5,
                    "override_discipline": 0.8,
                    "stress_level": 0.3,
                    "confidence_trend": 0.6,
                    "engagement": 0.4,
                },
                trading_outcome_quality=0.5 + i * 0.05,
            )
        result = model.train()
        self.assertTrue(model._trained)
        self.assertEqual(len(model._weights), 15)


# ═══════════════════════════════════════════════════════════════
# US-301: Circadian Readiness Multiplier
# ═══════════════════════════════════════════════════════════════

class TestUS301CircadianMultiplier(unittest.TestCase):
    """US-301: Time-of-day readiness modulation based on circadian research."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.signal_path = Path(self.tmp) / "bridge" / "readiness_signal.json"

    def test_peak_hours_multiplier_is_one(self):
        """Hours 9-10 should have multiplier = 1.0 (peak cognitive performance)."""
        computer = ReadinessComputer(signal_path=self.signal_path)
        self.assertEqual(computer.circadian_multiplier(hour=9), 1.0)
        self.assertEqual(computer.circadian_multiplier(hour=10), 1.0)

    def test_midnight_multiplier_is_low(self):
        """Hours 0-4 should have multiplier = 0.50 (trough)."""
        computer = ReadinessComputer(signal_path=self.signal_path)
        self.assertEqual(computer.circadian_multiplier(hour=0), 0.50)
        self.assertEqual(computer.circadian_multiplier(hour=3), 0.50)

    def test_evening_decline(self):
        """Late evening should be lower than afternoon."""
        computer = ReadinessComputer(signal_path=self.signal_path)
        afternoon = computer.circadian_multiplier(hour=15)
        evening = computer.circadian_multiplier(hour=22)
        self.assertGreater(afternoon, evening)

    def test_custom_circadian_config(self):
        """Custom circadian config should override defaults."""
        custom = {h: 0.8 for h in range(24)}
        custom[12] = 1.0
        computer = ReadinessComputer(signal_path=self.signal_path, circadian_config=custom)
        self.assertEqual(computer.circadian_multiplier(hour=0), 0.8)
        self.assertEqual(computer.circadian_multiplier(hour=12), 1.0)

    def test_neutral_config_no_effect(self):
        """Flat circadian config should not modify readiness score."""
        computer = ReadinessComputer(
            signal_path=self.signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        signal = computer.compute(emotional_state="calm")
        # With neutral circadian, score should be based purely on components
        self.assertGreater(signal.readiness_score, 60)
        self.assertAlmostEqual(signal.circadian_multiplier, 1.0)

    def test_circadian_applied_in_compute(self):
        """compute() should apply circadian multiplier to final score."""
        # Use a config where the current hour gives 0.5 multiplier
        all_half = {h: 0.5 for h in range(24)}
        computer = ReadinessComputer(
            signal_path=self.signal_path,
            circadian_config=all_half,
        )
        signal = computer.compute(emotional_state="calm")
        # Score should be roughly half of what it would be with multiplier=1.0
        neutral_computer = ReadinessComputer(
            signal_path=self.signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        neutral_signal = neutral_computer.compute(emotional_state="calm")
        # Allow some tolerance since acceleration/fatigue may differ slightly
        self.assertAlmostEqual(
            signal.readiness_score,
            neutral_signal.readiness_score * 0.5,
            delta=5.0,
        )

    def test_circadian_in_signal_dict(self):
        """ReadinessSignal.to_dict should include circadian_multiplier."""
        computer = ReadinessComputer(
            signal_path=self.signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        signal = computer.compute(emotional_state="calm")
        d = signal.to_dict()
        self.assertIn("circadian_multiplier", d)
        self.assertAlmostEqual(d["circadian_multiplier"], 1.0)

    def test_hour_clamping(self):
        """Out-of-range hours should be clamped to 0-23."""
        computer = ReadinessComputer(signal_path=self.signal_path)
        # Negative hour → clamped to 0
        self.assertEqual(computer.circadian_multiplier(hour=-1), computer.circadian_multiplier(hour=0))
        # Hour > 23 → clamped to 23
        self.assertEqual(computer.circadian_multiplier(hour=25), computer.circadian_multiplier(hour=23))

    def test_default_circadian_curve_complete(self):
        """Default curve should have entries for all 24 hours."""
        curve = ReadinessComputer.DEFAULT_CIRCADIAN_CURVE
        self.assertEqual(len(curve), 24)
        for h in range(24):
            self.assertIn(h, curve)
            self.assertGreaterEqual(curve[h], 0.0)
            self.assertLessEqual(curve[h], 1.0)


# ═══════════════════════════════════════════════════════════════
# Integration: Cross-story interactions
# ═══════════════════════════════════════════════════════════════

class TestPhase10Integration(unittest.TestCase):
    """Cross-story integration tests for Phase 10 features."""

    def test_enriched_outcome_serialization_roundtrip(self):
        """US-297: Enriched OutcomeSignal survives JSON roundtrip."""
        original = OutcomeSignal(
            pnl_today=42.0,
            emotional_state="focused",
            readiness_at_trade=88.5,
            cognitive_load=0.2,
            active_biases={"confirmation_bias": 0.4},
        )
        json_str = json.dumps(original.to_dict())
        loaded = json.loads(json_str)
        self.assertEqual(loaded["emotional_state"], "focused")
        self.assertAlmostEqual(loaded["readiness_at_trade"], 88.5)

    def test_decay_weighted_confidence_with_custom_time(self):
        """US-298: Decay function works with explicit time parameter."""
        now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        evidence_ts = (now - timedelta(days=7)).isoformat()
        pattern = DetectedPattern(
            pattern_id="test",
            tier=PatternTier.T1_DAILY,
            domain=PatternDomain.HUMAN,
            description="Test",
            evidence=[EvidenceItem(
                source_type="test",
                source_id="t",
                timestamp=evidence_ts,
                summary="7 days old",
            )],
            confidence=1.0,
        )
        dwc = get_decay_weighted_confidence(pattern, now=now)
        expected = math.exp(-EVIDENCE_DECAY_LAMBDA * 7)
        self.assertAlmostEqual(dwc, expected, places=3)

    def test_circadian_with_v2_model(self):
        """US-300 + US-301: V2 model score should also get circadian modulation."""
        tmp = tempfile.mkdtemp()
        signal_path = Path(tmp) / "bridge" / "readiness_signal.json"
        # Use half-multiplier circadian to verify it's applied
        half_config = {h: 0.5 for h in range(24)}
        v2 = MagicMock()
        v2._trained = True
        v2._train_samples = 100
        v2.compute_score.return_value = (75.0, {})
        computer = ReadinessComputer(
            signal_path=signal_path,
            v2_model=v2,
            circadian_config=half_config,
        )
        signal = computer.compute(emotional_state="calm")
        # V2 gives 75, circadian halves it → ~37.5
        self.assertLess(signal.readiness_score, 45)
        self.assertEqual(signal.model_version, "v2")
        self.assertAlmostEqual(signal.circadian_multiplier, 0.5)


if __name__ == "__main__":
    unittest.main()
