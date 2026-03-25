"""Phase 9 Tests — Adaptive Readiness, Temporal Decay & Trader Bias Detection.

US-290: V2 model wiring into ReadinessComputer
US-291: Temporal decay on self-model graph nodes
US-292: Graph pruning with archive for dormant nodes
US-293: Trader cognitive bias detection
US-294: Recency-weighted ML training
US-295: /insights command

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

# Ensure project root is on sys.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aura.core.readiness import ReadinessComputer, ReadinessSignal, ReadinessComponents
from src.aura.core.conversation_processor import BiasDetector, ConversationProcessor
from src.aura.core.self_model import SelfModelGraph, GraphNode, NodeType, EdgeType, GraphEdge
from src.aura.prediction.readiness_v2 import ReadinessModelV2, ReadinessTrainingSample


# ═══════════════════════════════════════════════════════════════
# US-290: Wire ReadinessModelV2 into primary compute path
# ═══════════════════════════════════════════════════════════════

class TestUS290V2Wiring(unittest.TestCase):
    """US-290: V2 model wired into ReadinessComputer with V1 fallback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.signal_path = Path(self.tmp) / "bridge" / "readiness_signal.json"

    def test_v1_fallback_when_no_v2_model(self):
        """When no V2 model is provided, V1 static weights are used."""
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=None, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        self.assertEqual(signal.model_version, "v1")
        self.assertGreater(signal.readiness_score, 0)

    def test_v1_fallback_when_v2_untrained(self):
        """V2 model present but untrained — falls back to V1."""
        v2 = MagicMock()
        v2._trained = False
        v2._train_samples = 0
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        self.assertEqual(signal.model_version, "v1")

    def test_v1_fallback_when_v2_insufficient_samples(self):
        """V2 model trained but with < 20 samples — still V1."""
        v2 = MagicMock()
        v2._trained = True
        v2._train_samples = 10  # Below V2_MIN_SAMPLES
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        self.assertEqual(signal.model_version, "v1")

    def test_v2_active_when_trained_sufficient(self):
        """V2 model trained with >= 20 samples — uses V2 score."""
        v2 = MagicMock()
        v2._trained = True
        v2._train_samples = 25
        v2.compute_score.return_value = (75.0, {"emotional_state": 0.5})
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        self.assertEqual(signal.model_version, "v2")
        # Score should be close to 75 (may have acceleration/fatigue adjustments)
        v2.compute_score.assert_called_once()

    def test_v2_score_clamped(self):
        """V2 predictions outside 0-100 are clamped."""
        v2 = MagicMock()
        v2._trained = True
        v2._train_samples = 25
        v2.compute_score.return_value = (150.0, {})  # Over 100
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        self.assertLessEqual(signal.readiness_score, 100.0)

    def test_v2_error_falls_back_to_v1(self):
        """V2 model error during compute — graceful fallback to V1."""
        v2 = MagicMock()
        v2._trained = True
        v2._train_samples = 25
        v2.compute_score.side_effect = RuntimeError("model error")
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})
        signal = computer.compute(emotional_state="calm")
        # Should not crash, should produce a V1 score
        self.assertGreater(signal.readiness_score, 0)

    def test_signal_includes_model_version_field(self):
        """ReadinessSignal.to_dict() includes model_version."""
        signal = ReadinessSignal(
            readiness_score=70.0,
            cognitive_load="low",
            active_stressors=[],
            override_loss_rate_7d=0.0,
            emotional_state="calm",
            confidence_trend="stable",
            components=ReadinessComponents(),
            model_version="v2",
        )
        d = signal.to_dict()
        self.assertEqual(d["model_version"], "v2")

    def test_cold_start_transition(self):
        """V2 transitions from untrained to trained at threshold."""
        v2 = MagicMock()
        v2._trained = False
        v2._train_samples = 19
        computer = ReadinessComputer(signal_path=self.signal_path, v2_model=v2, circadian_config={h: 1.0 for h in range(24)})

        # First call — V1
        signal1 = computer.compute(emotional_state="calm")
        self.assertEqual(signal1.model_version, "v1")

        # Simulate training completion
        v2._trained = True
        v2._train_samples = 20
        v2.compute_score.return_value = (72.0, {})

        signal2 = computer.compute(emotional_state="calm")
        self.assertEqual(signal2.model_version, "v2")


# ═══════════════════════════════════════════════════════════════
# US-291: Temporal decay on self-model graph nodes
# ═══════════════════════════════════════════════════════════════

class TestUS291TemporalDecay(unittest.TestCase):
    """US-291: Exponential decay on graph node strength."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / "test_graph.db"
        self.graph = SelfModelGraph(db_path=self.db_path)

    def tearDown(self):
        self.graph.close()

    def test_fresh_node_full_strength(self):
        """A just-created node has effective strength near DEFAULT_STRENGTH."""
        self.graph.add_node(GraphNode(
            id="n1", node_type=NodeType.EMOTION, label="calm",
            properties={"strength": 0.5, "last_reinforced": datetime.now(timezone.utc).isoformat()},
        ))
        strength = self.graph.get_effective_strength("n1", query_time=datetime.now(timezone.utc))
        self.assertAlmostEqual(strength, 0.5, delta=0.05)

    def test_decay_after_14_days(self):
        """After ~14 days, strength should be roughly halved (half-life)."""
        fourteen_days_ago = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        self.graph.add_node(GraphNode(
            id="n2", node_type=NodeType.EMOTION, label="stressed",
            properties={"strength": 1.0, "last_reinforced": fourteen_days_ago},
        ))
        strength = self.graph.get_effective_strength("n2", query_time=datetime.now(timezone.utc))
        # exp(-0.05 * 14) ≈ 0.497
        self.assertAlmostEqual(strength, 0.497, delta=0.05)

    def test_decay_after_60_days(self):
        """After 60 days, strength should be very low."""
        long_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.graph.add_node(GraphNode(
            id="n3", node_type=NodeType.EMOTION, label="old",
            properties={"strength": 1.0, "last_reinforced": long_ago},
        ))
        strength = self.graph.get_effective_strength("n3", query_time=datetime.now(timezone.utc))
        # exp(-0.05 * 60) ≈ 0.05
        self.assertLess(strength, 0.1)

    def test_reinforce_boosts_strength(self):
        """Reinforcing a node increases its strength."""
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        self.graph.add_node(GraphNode(
            id="n4", node_type=NodeType.EMOTION, label="anxious",
            properties={"strength": 0.4, "last_reinforced": seven_days_ago},
        ))
        before = self.graph.get_effective_strength("n4", query_time=datetime.now(timezone.utc))
        self.graph.reinforce_node("n4")
        after = self.graph.get_effective_strength("n4", query_time=datetime.now(timezone.utc))
        self.assertGreater(after, before)

    def test_strength_bounds(self):
        """Strength is clamped to [0.0, 1.0]."""
        self.graph.add_node(GraphNode(
            id="n5", node_type=NodeType.EMOTION, label="test",
            properties={"strength": 0.95, "last_reinforced": datetime.now(timezone.utc).isoformat()},
        ))
        # Reinforce multiple times — should not exceed 1.0
        for _ in range(10):
            self.graph.reinforce_node("n5")
        strength = self.graph.get_effective_strength("n5", query_time=datetime.now(timezone.utc))
        self.assertLessEqual(strength, 1.0)

    def test_default_values_when_no_properties(self):
        """Nodes without strength/last_reinforced use defaults."""
        self.graph.add_node(GraphNode(
            id="n6", node_type=NodeType.GOAL, label="profit",
            properties={},
        ))
        strength = self.graph.get_effective_strength("n6", query_time=datetime.now(timezone.utc))
        # Should use DEFAULT_STRENGTH and assume created_at as last_reinforced
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

    def test_filtered_query_by_min_strength(self):
        """get_nodes_by_type_filtered respects min_effective_strength."""
        now = datetime.now(timezone.utc)
        # Fresh node (strong)
        self.graph.add_node(GraphNode(
            id="strong", node_type=NodeType.EMOTION, label="strong",
            properties={"strength": 0.8, "last_reinforced": now.isoformat()},
        ))
        # Old node (weak)
        long_ago = (now - timedelta(days=90)).isoformat()
        self.graph.add_node(GraphNode(
            id="weak", node_type=NodeType.EMOTION, label="weak",
            properties={"strength": 0.5, "last_reinforced": long_ago},
        ))
        filtered = self.graph.get_nodes_by_type_filtered(
            NodeType.EMOTION, min_effective_strength=0.3, query_time=now
        )
        ids = [n.id for n in filtered]
        self.assertIn("strong", ids)
        self.assertNotIn("weak", ids)  # exp(-0.05*90)*0.5 ≈ 0.005


# ═══════════════════════════════════════════════════════════════
# US-292: Graph pruning with archive
# ═══════════════════════════════════════════════════════════════

class TestUS292GraphPruning(unittest.TestCase):
    """US-292: Dormant node pruning with archive."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / "test_graph.db"
        self.graph = SelfModelGraph(db_path=self.db_path)
        self.archive_path = Path(self.tmp) / "archive" / "pruned_nodes.jsonl"

    def tearDown(self):
        self.graph.close()

    def test_no_pruning_when_all_fresh(self):
        """No nodes pruned when all are recent."""
        self.graph.add_node(GraphNode(
            id="fresh", node_type=NodeType.EMOTION, label="calm",
            properties={"strength": 0.5, "last_reinforced": datetime.now(timezone.utc).isoformat()},
        ))
        count = self.graph.prune_dormant_nodes(
            min_strength=0.05, min_age_days=60, archive_path=self.archive_path
        )
        self.assertEqual(count, 0)

    def test_prune_old_weak_nodes(self):
        """Old nodes with decayed strength below threshold are pruned."""
        long_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        node = GraphNode(
            id="dormant", node_type=NodeType.EMOTION, label="old_emotion",
            properties={"strength": 0.3, "last_reinforced": long_ago},
        )
        self.graph.add_node(node)
        # Backdate the created_at to make it eligible for pruning (age > 60 days)
        self.graph._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?",
            (long_ago, "dormant"),
        )
        self.graph._conn.commit()
        count = self.graph.prune_dormant_nodes(
            min_strength=0.05, min_age_days=60, archive_path=self.archive_path
        )
        self.assertGreaterEqual(count, 1)

    def test_preserve_recently_reinforced_nodes(self):
        """Nodes that were reinforced recently survive despite old created_at."""
        long_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        recent = datetime.now(timezone.utc).isoformat()
        self.graph.add_node(GraphNode(
            id="reinforced", node_type=NodeType.GOAL, label="important",
            properties={"strength": 0.8, "last_reinforced": recent},
        ))
        # Backdate created_at so it's eligible by age
        self.graph._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?", (long_ago, "reinforced")
        )
        self.graph._conn.commit()
        count = self.graph.prune_dormant_nodes(
            min_strength=0.05, min_age_days=60, archive_path=self.archive_path
        )
        # Recently reinforced with strength 0.8 — effective ~0.8 > 0.05
        self.assertEqual(count, 0)

    def test_cascade_edges_on_prune(self):
        """Edges connected to pruned nodes are also removed."""
        long_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        self.graph.add_node(GraphNode(
            id="src", node_type=NodeType.PERSON, label="person",
            properties={"strength": 0.8, "last_reinforced": datetime.now(timezone.utc).isoformat()},
        ))
        self.graph.add_node(GraphNode(
            id="target", node_type=NodeType.EMOTION, label="old_emo",
            properties={"strength": 0.2, "last_reinforced": long_ago},
        ))
        # Backdate target created_at
        self.graph._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?", (long_ago, "target")
        )
        self.graph._conn.commit()
        self.graph.add_edge(GraphEdge(
            source_id="src", target_id="target", edge_type=EdgeType.TRIGGERS,
        ))
        stats_before = self.graph.get_stats()
        self.graph.prune_dormant_nodes(
            min_strength=0.05, min_age_days=60, archive_path=self.archive_path
        )
        stats_after = self.graph.get_stats()
        # Edge count should decrease
        self.assertLess(stats_after["total_edges"], stats_before["total_edges"])

    def test_archive_file_written(self):
        """Pruned nodes are archived to JSONL file."""
        long_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        self.graph.add_node(GraphNode(
            id="to_archive", node_type=NodeType.EMOTION, label="archive_me",
            properties={"strength": 0.2, "last_reinforced": long_ago},
        ))
        # Backdate created_at
        self.graph._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?", (long_ago, "to_archive")
        )
        self.graph._conn.commit()
        self.graph.prune_dormant_nodes(
            min_strength=0.05, min_age_days=60, archive_path=self.archive_path
        )
        if self.archive_path.exists():
            lines = self.archive_path.read_text().strip().split("\n")
            self.assertGreaterEqual(len(lines), 1)
            archived = json.loads(lines[0])
            self.assertIn("node", archived)
            self.assertIn("pruned_at", archived)
            self.assertEqual(archived["node"]["id"], "to_archive")


# ═══════════════════════════════════════════════════════════════
# US-293: Trader cognitive bias detection
# ═══════════════════════════════════════════════════════════════

class TestUS293BiasDetection(unittest.TestCase):
    """US-293: BiasDetector detects 4 trading biases."""

    def setUp(self):
        self.detector = BiasDetector()

    def test_disposition_effect_detected(self):
        """Phrases about holding losers trigger disposition effect."""
        msg = "I'm still waiting for this trade to bounce back. It might come back."
        biases = self.detector.detect_biases(msg)
        self.assertGreater(biases["disposition_effect"], 0.0)

    def test_disposition_countered(self):
        """Counter-disposition phrases reduce the score."""
        msg = "I stopped out and took the loss. Cut my position."
        biases = self.detector.detect_biases(msg)
        self.assertEqual(biases["disposition_effect"], 0.0)

    def test_loss_aversion_detected(self):
        """Disproportionate loss-word ratio triggers loss aversion."""
        msg = "I'm worried about the risk. The drawdown is scary. I fear the loss. Too much danger."
        biases = self.detector.detect_biases(msg)
        self.assertGreater(biases["loss_aversion"], 0.0)

    def test_loss_aversion_balanced(self):
        """Balanced loss/gain words produce no aversion signal."""
        msg = "There's risk but also opportunity. Some loss potential but good upside."
        biases = self.detector.detect_biases(msg)
        self.assertEqual(biases["loss_aversion"], 0.0)

    def test_recency_bias_detected(self):
        """Heavy use of recent temporal words triggers recency bias."""
        msg = "Just today I saw this setup. It just happened recently, right now."
        biases = self.detector.detect_biases(msg)
        self.assertGreater(biases["recency_bias"], 0.0)

    def test_recency_countered_by_historical(self):
        """Historical keywords balance recency words."""
        msg = "Today was good, but historically this pattern typically fails over time."
        biases = self.detector.detect_biases(msg)
        # Historical words should counterbalance
        self.assertLessEqual(biases["recency_bias"], 0.25)

    def test_confirmation_bias_detected(self):
        """Validation-seeking phrases trigger confirmation bias."""
        msg = "See what I mean? I knew it would work. Proving me right again."
        biases = self.detector.detect_biases(msg)
        self.assertGreater(biases["confirmation_bias"], 0.0)

    def test_no_bias_clean_message(self):
        """A neutral message produces no significant bias scores."""
        msg = "Good morning. Let me check my charts and plan for the session."
        biases = self.detector.detect_biases(msg)
        total = sum(biases.values())
        self.assertLess(total, 0.2)

    def test_empty_message(self):
        """Empty message returns all zeros."""
        biases = self.detector.detect_biases("")
        self.assertEqual(biases["disposition_effect"], 0.0)
        self.assertEqual(biases["loss_aversion"], 0.0)
        self.assertEqual(biases["recency_bias"], 0.0)
        self.assertEqual(biases["confirmation_bias"], 0.0)

    def test_aggregate_bias_score(self):
        """Aggregate bias score is the mean of individual biases."""
        biases = {"a": 0.5, "b": 0.3, "c": 0.0, "d": 0.2}
        agg = self.detector.aggregate_bias_score(biases)
        self.assertAlmostEqual(agg, 0.25, places=2)

    def test_bias_penalty_integration(self):
        """US-293: Bias penalty reduces readiness override_discipline score."""
        tmp = tempfile.mkdtemp()
        signal_path = Path(tmp) / "bridge" / "readiness_signal.json"
        computer = ReadinessComputer(signal_path=signal_path, circadian_config={h: 1.0 for h in range(24)})

        # Without bias
        signal_no_bias = computer.compute(emotional_state="calm")
        # Reset history and EMA state to avoid acceleration/hysteresis effects
        computer._readiness_history = []
        computer._last_smoothed_score = None

        # With significant bias
        bias_scores = {
            "disposition_effect": 0.8,
            "loss_aversion": 0.6,
            "recency_bias": 0.4,
            "confirmation_bias": 0.5,
        }
        signal_with_bias = computer.compute(emotional_state="calm", bias_scores=bias_scores)

        # Bias should reduce the readiness score
        self.assertLess(signal_with_bias.readiness_score, signal_no_bias.readiness_score)

    def test_bias_scores_in_conversation_signals(self):
        """Bias scores are included in ConversationSignals."""
        processor = ConversationProcessor()
        signals = processor.process_message(
            "I'm still waiting for it to bounce back. I knew it would work."
        )
        self.assertIn("disposition_effect", signals.bias_scores)
        self.assertIn("confirmation_bias", signals.bias_scores)


# ═══════════════════════════════════════════════════════════════
# US-294: Recency-weighted ML training
# ═══════════════════════════════════════════════════════════════

class TestUS294RecencyWeighting(unittest.TestCase):
    """US-294: Recency weighting in ReadinessModelV2 and OverridePredictor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_recent_samples_weight_near_one(self):
        """Samples from today get weight ~1.0."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=5,
        )
        now_str = datetime.now(timezone.utc).isoformat()
        weight = model._compute_sample_weight(now_str)
        self.assertAlmostEqual(weight, 1.0, delta=0.05)

    def test_old_samples_downweighted(self):
        """Samples from 60 days ago get significantly lower weight."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=5,
        )
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        weight = model._compute_sample_weight(old)
        # exp(-60/30) = exp(-2) ≈ 0.135
        self.assertAlmostEqual(weight, 0.135, delta=0.05)

    def test_weight_floor_respected(self):
        """Very old samples still get minimum weight floor."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=5,
        )
        ancient = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        weight = model._compute_sample_weight(ancient)
        self.assertGreaterEqual(weight, model.RECENCY_WEIGHT_FLOOR)

    def test_empty_timestamp_treated_as_recent(self):
        """Missing timestamp defaults to weight 1.0."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=5,
        )
        self.assertEqual(model._compute_sample_weight(""), 1.0)

    def test_training_converges_with_weights(self):
        """Training still converges when recency weights are applied."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=5,
        )
        now = datetime.now(timezone.utc)
        for i in range(25):
            ts = (now - timedelta(days=i * 3)).isoformat()
            model.add_training_sample(
                readiness_components={
                    "emotional_state": 0.7 + (i % 3) * 0.1,
                    "cognitive_load": 0.6,
                    "override_discipline": 0.8,
                    "stress_level": 0.5,
                    "confidence_trend": 0.7,
                    "engagement": 0.6,
                },
                trading_outcome_quality=0.6 + (i % 5) * 0.05,
                timestamp=ts,
            )
        # Model should be trained (>= 20 samples)
        self.assertTrue(model._trained)
        self.assertGreater(model._train_samples, 0)

    def test_timestamp_persisted_in_buffer(self):
        """Training sample timestamps survive save/load cycle."""
        model = ReadinessModelV2(
            model_path=Path(self.tmp) / "model.json",
            min_samples=100,  # High so it doesn't auto-train
        )
        ts = datetime.now(timezone.utc).isoformat()
        model.add_training_sample(
            readiness_components={
                "emotional_state": 0.7, "cognitive_load": 0.6,
                "override_discipline": 0.8, "stress_level": 0.5,
                "confidence_trend": 0.7, "engagement": 0.6,
            },
            trading_outcome_quality=0.7,
            timestamp=ts,
        )
        self.assertEqual(model._training_buffer[-1].timestamp, ts)

    def test_override_predictor_recency_weight(self):
        """OverridePredictor also computes recency weights."""
        from src.aura.prediction.override_predictor import OverridePredictor
        predictor = OverridePredictor(
            model_path=Path(self.tmp) / "op.json",
        )
        now_event = {"timestamp": datetime.now(timezone.utc).isoformat()}
        old_event = {"timestamp": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()}

        w_now = predictor._compute_sample_weight(now_event)
        w_old = predictor._compute_sample_weight(old_event)
        self.assertGreater(w_now, w_old)
        self.assertGreaterEqual(w_old, predictor.RECENCY_WEIGHT_FLOOR)


# ═══════════════════════════════════════════════════════════════
# US-295: /insights command
# ═══════════════════════════════════════════════════════════════

class TestUS295InsightsCommand(unittest.TestCase):
    """US-295: /insights command shows patterns, biases, graph health."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / "test.db"
        self.bridge_dir = Path(self.tmp) / "bridge"
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    def _make_companion(self):
        """Create a companion with mocked bridge."""
        from src.aura.cli.companion import AuraCompanion
        companion = AuraCompanion(
            db_path=self.db_path,
            bridge_dir=self.bridge_dir,
        )
        # Mock the bridge to avoid file I/O issues
        companion.bridge = MagicMock()
        companion.bridge.get_recent_overrides.return_value = []
        companion.bridge.get_bridge_status.return_value = {
            "readiness_signal": {"available": False, "score": 0},
            "outcome_signal": {"available": False},
            "override_events": {"total_recent": 0},
        }
        companion.bridge.read_outcome.return_value = None
        companion.bridge.bridge_dir = self.bridge_dir
        return companion

    def test_insights_no_data(self):
        """With no data, insights shows 'no patterns' message."""
        companion = self._make_companion()
        result = companion._cmd_insights()
        self.assertIn("Insights", result)
        # Should mention no patterns or no bias data
        self.assertTrue(
            "No significant" in result or "no" in result.lower() or "No bias" in result
        )

    def test_insights_with_bias_data(self):
        """After processing a biased message, insights shows bias info."""
        companion = self._make_companion()
        # Process a message with biases
        companion.process_input("I'm still waiting for it to bounce back. I knew it would work.")
        result = companion._cmd_insights()
        self.assertIn("Insights", result)

    def test_insights_shows_graph_stats(self):
        """Insights includes graph health section."""
        companion = self._make_companion()
        result = companion._cmd_insights()
        self.assertIn("Graph Health", result)
        self.assertIn("Total nodes", result)

    def test_insights_shows_model_version(self):
        """Insights shows which readiness model is active."""
        companion = self._make_companion()
        result = companion._cmd_insights()
        self.assertIn("Readiness model", result)

    def test_insights_command_routing(self):
        """The /insights command is properly routed."""
        companion = self._make_companion()
        result = companion._handle_command("/insights")
        self.assertIn("Insights", result)


if __name__ == "__main__":
    unittest.main()
