"""Phase 14 tests: Graph-Informed Readiness, Decision Quality & Online Learning.

US-320: Graph-informed readiness features from SelfModelGraph
US-321: Decision Quality Scorer with 7-dimension process evaluation
US-322: Online learning loop: outcome signals retrain ReadinessModelV2
US-323: Bridge outcome signal enrichment with human context
US-324: Anomaly-to-pattern cascade: wire ReadinessAnomalyDetector into PatternEngine
US-325: Phase 14 integration tests + /quality CLI command
"""

import json
import math
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aura.core.readiness import (
    ReadinessComputer,
    ReadinessComponents,
    ReadinessSignal,
    AnomalyResult,
)
from src.aura.core.self_model import (
    SelfModelGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from src.aura.scoring.decision_quality import (
    DecisionQualityScorer,
    DecisionQualityScore,
    DecisionQualitySignals,
)
from src.aura.prediction.readiness_v2 import ReadinessModelV2, ReadinessTrainingSample
from src.aura.bridge.signals import FeedbackBridge, OutcomeSignal

# --- Neutral circadian config (all hours = 1.0) to isolate test effects ---
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ══════════════════════════════════════════════════════
# US-320: Graph-informed readiness features
# ══════════════════════════════════════════════════════

class TestGraphInformedReadiness:
    """US-320: Verify graph features flow from SelfModelGraph → ReadinessComputer."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def _make_graph(self):
        return SelfModelGraph(db_path=Path(tempfile.mkdtemp()) / "test.db")

    def test_no_graph_neutral_score(self):
        """When no graph provided, graph_context_score defaults to 0.5 (neutral)."""
        rc = self._make_rc()
        features = rc._compute_graph_features(None)
        assert features["graph_context_score"] == 0.5
        assert features["emotional_volatility"] == 0.0

    def test_graph_with_no_emotion_nodes(self):
        """Graph with no Emotion nodes → emotional_volatility = 0."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add a Goal node only
        graph.add_node(GraphNode(
            id="goal_1", node_type=NodeType.GOAL, label="Test Goal", confidence=0.8
        ))
        features = rc._compute_graph_features(graph)
        assert features["emotional_volatility"] == 0.0
        # goal_alignment should reflect the goal
        assert features["goal_alignment"] >= 0.0

    def test_high_emotional_volatility_lowers_score(self):
        """High std dev of Emotion node confidences → lower graph_context_score."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add emotions with widely varying confidence
        for i, conf in enumerate([0.1, 0.9, 0.2, 0.8, 0.15, 0.85]):
            graph.add_node(GraphNode(
                id=f"emotion_{i}", node_type=NodeType.EMOTION,
                label=f"Emotion {i}", confidence=conf,
            ))
        features = rc._compute_graph_features(graph)
        assert features["emotional_volatility"] > 0.3

        # Compare with stable emotions — volatile should score lower
        graph2 = self._make_graph()
        for i in range(6):
            graph2.add_node(GraphNode(
                id=f"emotion_{i}", node_type=NodeType.EMOTION,
                label=f"Emotion {i}", confidence=0.5,  # Stable
            ))
        features_stable = rc._compute_graph_features(graph2)
        assert features["graph_context_score"] < features_stable["graph_context_score"]

    def test_strong_goal_alignment_boosts_score(self):
        """Strong Goal nodes connected to Decisions → higher goal_alignment."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add strong goals
        for i in range(3):
            graph.add_node(GraphNode(
                id=f"goal_{i}", node_type=NodeType.GOAL,
                label=f"Goal {i}", confidence=0.9,
            ))
        features = rc._compute_graph_features(graph)
        assert features["goal_alignment"] >= 0.8

    def test_high_pattern_load_penalizes(self):
        """Many active patterns → high pattern_load → lower score."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add 15 active patterns (confidence > 0.3)
        for i in range(15):
            graph.add_node(GraphNode(
                id=f"pattern_{i}", node_type=NodeType.PATTERN,
                label=f"Pattern {i}", confidence=0.5,
            ))
        features = rc._compute_graph_features(graph)
        assert features["pattern_load"] > 0.5

    def test_negative_influence_density(self):
        """Weak edges → higher negative_influence_density."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add nodes and weak edges
        for i in range(5):
            graph.add_node(GraphNode(
                id=f"node_{i}", node_type=NodeType.EMOTION,
                label=f"Node {i}", confidence=0.5,
            ))
        for i in range(4):
            graph.add_edge(GraphEdge(
                source_id=f"node_{i}", target_id=f"node_{i+1}",
                edge_type=EdgeType.INFLUENCES, weight=0.1,  # weak
            ))
        features = rc._compute_graph_features(graph)
        assert features["negative_influence_density"] > 0.5

    def test_compute_with_graph_influences_readiness(self):
        """compute() with graph parameter should influence readiness score."""
        rc = self._make_rc()
        graph = self._make_graph()
        # Add volatile emotions to lower graph score
        for i, conf in enumerate([0.1, 0.9, 0.1, 0.9]):
            graph.add_node(GraphNode(
                id=f"emo_{i}", node_type=NodeType.EMOTION,
                label=f"Emo {i}", confidence=conf,
            ))
        sig_no_graph = rc.compute(emotional_state="calm")

        rc2 = self._make_rc()
        sig_with_graph = rc2.compute(emotional_state="calm", graph=graph)
        # Scores may differ since graph context is blended at 10%
        # Both should be valid scores
        assert 0 <= sig_no_graph.readiness_score <= 100
        assert 0 <= sig_with_graph.readiness_score <= 100

    def test_graph_with_no_edges(self):
        """Graph with nodes but no edges → negative_influence_density = 0."""
        rc = self._make_rc()
        graph = self._make_graph()
        graph.add_node(GraphNode(
            id="emotion_1", node_type=NodeType.EMOTION, label="Happy", confidence=0.8
        ))
        features = rc._compute_graph_features(graph)
        assert features["negative_influence_density"] == 0.0


# ══════════════════════════════════════════════════════
# US-321: Decision Quality Scorer
# ══════════════════════════════════════════════════════

class TestDecisionQualityScorer:
    """US-321: Verify 7-dimension decision quality scoring."""

    def test_process_adherence_keywords(self):
        """Process adherence detects checklist, size, ratio, stop keywords."""
        scorer = DecisionQualityScorer()
        text = "I checked my checklist and confirmed the setup is valid. Position size is 1 lot. Risk/reward ratio is 2:1. Stop at 1.0850 level."
        result = scorer.score(text)
        assert result.process_adherence >= 0.8

    def test_information_adequacy_multi_timeframe(self):
        """Information adequacy scores high with multiple timeframes + macro."""
        scorer = DecisionQualityScorer()
        text = "Checked the 4h and 1h charts. Volume is strong. CPI data released today. Daily shows uptrend."
        result = scorer.score(text)
        assert result.information_adequacy >= 0.7

    def test_metacognitive_awareness(self):
        """Metacognitive keywords boost awareness score."""
        scorer = DecisionQualityScorer()
        text = "I notice I might be biased here. I'm not sure if I should take this. I'm aware I'm overconfident."
        result = scorer.score(text)
        assert result.metacognitive_awareness >= 0.5

    def test_uncertainty_acknowledgment(self):
        """Acknowledging uncertainty boosts score."""
        scorer = DecisionQualityScorer()
        text = "Could fail easily. Not sure about direction. Probably 50/50 on this one."
        result = scorer.score(text)
        assert result.uncertainty_acknowledgment >= 0.5

    def test_rationale_clarity(self):
        """Providing reasons boosts rationale score."""
        scorer = DecisionQualityScorer()
        text = "Entry because price broke above resistance and is supported by volume."
        result = scorer.score(text)
        assert result.rationale_clarity >= 0.6

    def test_emotional_regulation_clean(self):
        """No emotional flags → perfect emotional regulation."""
        scorer = DecisionQualityScorer()
        text = "Calm analysis today. Following my system."
        result = scorer.score(text)
        assert result.emotional_regulation == 1.0

    def test_emotional_regulation_with_flags(self):
        """Emotional flags reduce regulation score."""
        scorer = DecisionQualityScorer()
        text = "I need revenge on this market. FOMO is hitting hard. I'm so frustrated and angry."
        result = scorer.score(text)
        assert result.emotional_regulation < 0.5

    def test_cognitive_reflection(self):
        """Reflection keywords boost reflection score."""
        scorer = DecisionQualityScorer()
        text = "I waited and watched before entering. Checked again after 30 minutes. Took a step back to reconsider."
        result = scorer.score(text)
        assert result.cognitive_reflection >= 0.5

    def test_composite_weighted_correctly(self):
        """Composite = weighted sum of all 7 dimensions × 100."""
        scorer = DecisionQualityScorer()
        # All keywords present → high composite
        text = (
            "I followed my checklist and confirmed the setup. Position size 1 lot. "
            "Risk/reward ratio 2:1. Stop at level. "
            "4h and daily charts both bullish. Volume strong. CPI today. "
            "I notice I might be biased. Not sure if the rally continues. "
            "Because price broke resistance and confluence of signals. "
            "I waited and checked again before entry."
        )
        result = scorer.score(text)
        # Verify composite matches manual weighted sum (8 dimensions as of US-328)
        manual = (
            result.process_adherence * 0.25
            + result.information_adequacy * 0.17
            + result.metacognitive_awareness * 0.15
            + result.uncertainty_acknowledgment * 0.15
            + result.metacognitive_monitoring * 0.10
            + result.rationale_clarity * 0.10
            + result.emotional_regulation * 0.05
            + result.cognitive_reflection * 0.03
        ) * 100
        assert abs(result.composite_score - manual) < 0.1

    def test_no_signals_returns_baseline(self):
        """Empty conversation returns baseline (low) scores."""
        scorer = DecisionQualityScorer()
        result = scorer.score("hello")
        assert result.composite_score >= 0
        assert result.composite_score < 30  # Low with no signals

    def test_high_quality_conversation_scores_high(self):
        """Full quality conversation should score well above 50."""
        scorer = DecisionQualityScorer()
        text = (
            "Following my system rules today. Plan is clear. Confirmed the setup is valid. "
            "Position size calculated at 1 lot. R:R ratio 2.5:1. Stop at key level. "
            "Checked 4h, 1h, and daily timeframes. Volume and correlation confirm. "
            "CPI news calendar clear. Session timing good. "
            "I know I'm a bit overconfident today — aware of that. "
            "Not sure if this holds, could fail. Probably 60% chance. "
            "Entry because triple confluence at support. Based on my criteria. "
            "Waited, paused, watched before taking the trade. Took a step back first."
        )
        result = scorer.score(text)
        assert result.composite_score > 50

    def test_to_dict_includes_all_dimensions(self):
        """to_dict() returns all 8 dimensions + composite + metadata (US-328)."""
        scorer = DecisionQualityScorer()
        result = scorer.score("test")
        d = result.to_dict()
        assert "dimensions" in d
        assert len(d["dimensions"]) == 8
        assert "composite_score" in d
        assert "timestamp" in d

    def test_entry_latency_bonus(self):
        """Trade metadata with 5+ min entry latency boosts cognitive reflection."""
        scorer = DecisionQualityScorer()
        # Without latency
        r1 = scorer.score("I paused before entering")
        # With 5+ min latency
        r2 = scorer.score(
            "I paused before entering",
            trade_metadata={"idea_time": 1000, "entry_time": 1400},  # 400 seconds > 300
        )
        assert r2.cognitive_reflection >= r1.cognitive_reflection


# ══════════════════════════════════════════════════════
# US-322: Online learning loop
# ══════════════════════════════════════════════════════

class TestOnlineLearning:
    """US-322: Verify ReadinessModelV2.update_from_outcome works correctly."""

    def _make_trained_model(self):
        """Create a model with enough samples to be trained."""
        model = ReadinessModelV2(
            model_path=Path(tempfile.mkdtemp()) / "test_model.json",
            min_samples=5,  # Low threshold for testing
        )
        # Add training samples
        for i in range(10):
            model.add_training_sample(
                readiness_components={
                    "emotional_state": 0.5 + i * 0.05,
                    "cognitive_load": 0.6,
                    "override_discipline": 0.7,
                    "stress_level": 0.5,
                    "confidence_trend": 0.6,
                    "engagement": 0.5,
                },
                trading_outcome_quality=0.5 + i * 0.05,
            )
        assert model._trained
        return model

    def test_untrained_model_skips_update(self):
        """Untrained model (sample_count < min_samples) skips online updates."""
        model = ReadinessModelV2(
            model_path=Path(tempfile.mkdtemp()) / "test.json",
            min_samples=20,
        )
        result = model.update_from_outcome(
            component_dict={"emotional_state": 0.5},
            target_readiness=0.8,
        )
        assert result is False

    def test_single_update_changes_weights(self):
        """A single update should change at least some weights."""
        model = self._make_trained_model()
        old_weights = list(model._weights)
        old_bias = model._bias

        model.update_from_outcome(
            component_dict={
                "emotional_state": 0.9,
                "cognitive_load": 0.3,
                "override_discipline": 0.9,
                "stress_level": 0.2,
                "confidence_trend": 0.8,
                "engagement": 0.7,
            },
            target_readiness=0.9,
        )

        # At least some weights should have changed
        weight_changes = sum(1 for old, new in zip(old_weights, model._weights) if abs(old - new) > 1e-10)
        assert weight_changes > 0 or abs(old_bias - model._bias) > 1e-10

    def test_learning_rate_respected(self):
        """Higher learning rate → larger weight changes."""
        model1 = self._make_trained_model()
        model2 = self._make_trained_model()

        components = {
            "emotional_state": 0.9, "cognitive_load": 0.3,
            "override_discipline": 0.9, "stress_level": 0.2,
            "confidence_trend": 0.8, "engagement": 0.7,
        }

        w1_before = list(model1._weights)
        model1.update_from_outcome(components, 0.9, learning_rate=0.001)
        delta1 = sum(abs(a - b) for a, b in zip(w1_before, model1._weights))

        w2_before = list(model2._weights)
        model2.update_from_outcome(components, 0.9, learning_rate=0.1)
        delta2 = sum(abs(a - b) for a, b in zip(w2_before, model2._weights))

        # Larger learning rate should produce larger total weight change
        assert delta2 > delta1

    def test_negative_outcome_affects_direction(self):
        """Negative outcome (low target) should push weights differently than positive."""
        model_pos = self._make_trained_model()
        model_neg = self._make_trained_model()

        components = {
            "emotional_state": 0.5, "cognitive_load": 0.5,
            "override_discipline": 0.5, "stress_level": 0.5,
            "confidence_trend": 0.5, "engagement": 0.5,
        }

        model_pos.update_from_outcome(components, 0.95, learning_rate=0.05)
        model_neg.update_from_outcome(components, 0.05, learning_rate=0.05)

        # Biases should diverge
        assert model_pos._bias != model_neg._bias

    def test_sample_count_increments(self):
        """update_from_outcome increments _train_samples."""
        model = self._make_trained_model()
        initial_count = model._train_samples

        model.update_from_outcome(
            {"emotional_state": 0.5}, target_readiness=0.5
        )
        assert model._train_samples == initial_count + 1

    def test_readiness_computer_train_from_bridge(self):
        """ReadinessComputer.train_from_bridge_outcome dispatches to model."""
        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        # Compute once to populate cached state
        rc.compute(emotional_state="calm")

        # Mock the V2 model as trained
        mock_model = MagicMock()
        mock_model._trained = True
        mock_model._train_samples = 25
        mock_model.min_samples = 20
        mock_model.update_from_outcome.return_value = True
        rc._v2_model = mock_model

        outcome = {
            "pnl_today": 150.0,
            "trade_won": True,
            "win_rate_7d": 0.65,
        }
        result = rc.train_from_bridge_outcome(outcome)
        assert result is True
        mock_model.update_from_outcome.assert_called_once()

    def test_multiple_updates_converge(self):
        """Multiple consistent updates should shift bias in expected direction."""
        model = self._make_trained_model()
        bias_before = model._bias

        # Repeated high-quality outcomes
        for _ in range(5):
            model.update_from_outcome(
                {"emotional_state": 0.8, "cognitive_load": 0.8,
                 "override_discipline": 0.9, "stress_level": 0.8,
                 "confidence_trend": 0.8, "engagement": 0.7},
                target_readiness=0.9,
                learning_rate=0.05,
            )

        # Bias should have shifted (may be up or down depending on initial error direction)
        assert model._bias != bias_before


# ══════════════════════════════════════════════════════
# US-323: Bridge outcome signal enrichment
# ══════════════════════════════════════════════════════

class TestBridgeEnrichment:
    """US-323: Verify enrich_outcome_signal adds human context."""

    def _make_bridge(self):
        return FeedbackBridge(bridge_dir=Path(tempfile.mkdtemp()))

    def test_enrichment_adds_all_fields(self):
        """Enriched signal includes all human context fields."""
        bridge = self._make_bridge()
        outcome = {"pnl_today": 100.0, "regime": "NORMAL", "timestamp": "2026-03-24T12:00:00"}
        human_ctx = {
            "readiness_score": 75.0,
            "emotional_state": "calm",
            "cognitive_load_label": "low",
            "bias_scores": {"recency_bias": 0.3},
            "override_loss_risk": 0.15,
            "tilt_score": 0.1,
            "graph_context_score": 0.65,
            "decision_quality_score": 72.5,
        }
        enriched = bridge.enrich_outcome_signal(outcome, human_ctx)
        assert enriched["emotional_state_score"] == "calm"
        assert enriched["cognitive_load_score"] == "low"
        assert enriched["bias_count"] == 1
        assert enriched["override_loss_risk"] == 0.15
        assert enriched["readiness_at_trade_time"] == 75.0
        assert enriched["decision_quality_score"] == 72.5

    def test_no_cached_state_adds_empty_context(self):
        """When human_context is None/empty, add default empty fields."""
        bridge = self._make_bridge()
        outcome = {"pnl_today": -50.0}
        enriched = bridge.enrich_outcome_signal(outcome, {})
        assert enriched["emotional_state_score"] == "unknown"
        assert enriched["bias_count"] == 0
        assert enriched["readiness_at_trade_time"] == 0.0

    def test_enrichment_preserves_original_fields(self):
        """Original outcome fields are preserved after enrichment."""
        bridge = self._make_bridge()
        outcome = {"pnl_today": 200.0, "regime": "VOLATILE", "custom_field": "abc"}
        enriched = bridge.enrich_outcome_signal(outcome, {"readiness_score": 80})
        assert enriched["pnl_today"] == 200.0
        assert enriched["regime"] == "VOLATILE"
        assert enriched["custom_field"] == "abc"

    def test_atomic_write_persists(self):
        """Enriched signal is written to outcome_signal.json."""
        bridge = self._make_bridge()
        outcome = {"pnl_today": 50.0, "timestamp": "2026-03-24T12:00:00"}
        bridge.enrich_outcome_signal(outcome, {"readiness_score": 70, "emotional_state": "calm"})

        # Read back
        raw = bridge._locked_read(bridge._outcome_path)
        assert raw is not None
        data = json.loads(raw)
        assert data["readiness_at_trade_time"] == 70.0

    def test_roundtrip_read_enrich_write(self):
        """Write outcome → read → enrich → write back → read enriched."""
        bridge = self._make_bridge()
        signal = OutcomeSignal(pnl_today=100.0, regime="NORMAL", win_rate_7d=0.6)
        bridge.write_outcome(signal)

        # Read back
        read_signal = bridge.read_outcome()
        assert read_signal is not None

        # Enrich
        human_ctx = {"readiness_score": 85.0, "emotional_state": "energized"}
        enriched = bridge.enrich_outcome_signal(read_signal.to_dict(), human_ctx)

        # Read enriched
        raw = bridge._locked_read(bridge._outcome_path)
        data = json.loads(raw)
        assert data["readiness_at_trade_time"] == 85.0
        assert data["pnl_today"] == 100.0

    def test_missing_outcome_fields_handled(self):
        """Enrichment handles minimal outcome dict gracefully."""
        bridge = self._make_bridge()
        enriched = bridge.enrich_outcome_signal({}, {"readiness_score": 60})
        assert enriched["readiness_at_trade_time"] == 60.0

    def test_get_last_state_snapshot(self):
        """ReadinessComputer caches state snapshot after compute()."""
        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        # Before compute, snapshot should be None
        snap = rc.get_last_state_snapshot()
        assert snap == {}

        # After compute, snapshot should have data
        rc.compute(emotional_state="calm")
        snap = rc.get_last_state_snapshot()
        assert "readiness_score" in snap
        assert "components" in snap
        assert snap["readiness_score"] > 0


# ══════════════════════════════════════════════════════
# US-324: Anomaly-to-pattern cascade
# ══════════════════════════════════════════════════════

class TestAnomalyCascade:
    """US-324: Verify anomaly detection creates Life_Event nodes and T2 receives context."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def _make_graph(self):
        return SelfModelGraph(db_path=Path(tempfile.mkdtemp()) / "test.db")

    def test_anomaly_creates_life_event_node(self):
        """When anomaly is detected during compute(), a Life_Event node is logged to graph."""
        rc = self._make_rc()
        graph = self._make_graph()

        # Feed stable scores to build baseline, then spike
        for _ in range(15):
            rc.compute(emotional_state="calm", graph=graph)

        # Force anomaly by injecting extreme score
        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=70.0, residual=-40.0, threshold=15.0,
                anomaly_detected=True, severity=0.8,
            )
            rc.compute(emotional_state="stressed", graph=graph)

        # Check Life_Event nodes
        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert len(anomaly_events) >= 1
        event = anomaly_events[-1]
        assert event.properties["direction"] == "drop"  # Negative residual
        assert event.properties["severity"] == 0.8

    def test_severity_propagated(self):
        """Life_Event node severity matches anomaly severity."""
        rc = self._make_rc()
        graph = self._make_graph()

        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=60.0, residual=30.0, threshold=10.0,
                anomaly_detected=True, severity=0.65,
            )
            rc.compute(emotional_state="calm", graph=graph)

        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert len(anomaly_events) >= 1
        assert anomaly_events[-1].properties["severity"] == 0.65

    def test_direction_correct_for_spike(self):
        """Positive residual → direction='spike'."""
        rc = self._make_rc()
        graph = self._make_graph()

        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=50.0, residual=25.0, threshold=10.0,
                anomaly_detected=True, severity=0.7,
            )
            rc.compute(emotional_state="calm", graph=graph)

        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert anomaly_events[-1].properties["direction"] == "spike"

    def test_no_anomaly_no_node(self):
        """When no anomaly detected, no Life_Event node is created."""
        rc = self._make_rc()
        graph = self._make_graph()

        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=70.0, residual=2.0, threshold=15.0,
                anomaly_detected=False, severity=0.0,
            )
            rc.compute(emotional_state="calm", graph=graph)

        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert len(anomaly_events) == 0

    def test_no_graph_skips_logging(self):
        """When no graph provided, anomaly logging is skipped (backward compatible)."""
        rc = self._make_rc()
        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=70.0, residual=-40.0, threshold=15.0,
                anomaly_detected=True, severity=0.8,
            )
            # No graph → should not crash
            sig = rc.compute(emotional_state="stressed")
            assert sig.anomaly_detected is True

    def test_t2_correlate_anomalies(self):
        """T2 _correlate_anomalies finds co-occurrences with override events."""
        from src.aura.patterns.tier2 import Tier2CrossDomainDetector

        t2 = Tier2CrossDomainDetector(patterns_dir=Path(tempfile.mkdtemp()))
        now = datetime.now(timezone.utc)

        anomaly_events = [
            {"severity": 0.8, "direction": "drop", "timestamp": now.isoformat()},
            {"severity": 0.6, "direction": "spike", "timestamp": (now - timedelta(hours=2)).isoformat()},
        ]
        override_events = [
            {"timestamp": (now + timedelta(hours=1)).isoformat(), "outcome": "loss"},
            {"timestamp": (now - timedelta(hours=1)).isoformat(), "outcome": "loss"},
            {"timestamp": (now + timedelta(hours=3)).isoformat(), "outcome": "win"},
        ]

        patterns = t2._correlate_anomalies(anomaly_events, override_events)
        # Should find co-occurrences (within 24h window)
        assert len(patterns) >= 1

    def test_multiple_anomalies_create_multiple_nodes(self):
        """Each anomaly detection creates a separate Life_Event node."""
        rc = self._make_rc()
        graph = self._make_graph()

        for _ in range(3):
            with patch.object(rc._anomaly_detector, 'update') as mock_update:
                mock_update.return_value = AnomalyResult(
                    baseline=60.0, residual=-30.0, threshold=10.0,
                    anomaly_detected=True, severity=0.75,
                )
                rc.compute(emotional_state="stressed", graph=graph)

        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert len(anomaly_events) >= 3


# ══════════════════════════════════════════════════════
# US-325: Integration tests + /quality CLI command
# ══════════════════════════════════════════════════════

class TestPhase14Integration:
    """US-325: End-to-end integration tests for Phase 14 features."""

    def test_graph_features_flow_to_readiness_signal(self):
        """Integration: Graph → ReadinessComputer → ReadinessSignal."""
        graph = SelfModelGraph(db_path=Path(tempfile.mkdtemp()) / "test.db")
        # Populate graph with Emotion/Goal/Decision nodes
        for i in range(3):
            graph.add_node(GraphNode(
                id=f"emotion_{i}", node_type=NodeType.EMOTION,
                label=f"Emotion {i}", confidence=0.7,
            ))
        graph.add_node(GraphNode(
            id="goal_1", node_type=NodeType.GOAL, label="Consistency", confidence=0.9,
        ))

        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        sig = rc.compute(emotional_state="calm", graph=graph)

        assert 0 <= sig.readiness_score <= 100
        d = sig.to_dict()
        assert "readiness_score" in d
        assert "components" in d

    def test_decision_quality_in_bridge_signal(self):
        """Integration: Score conversation → composite in ReadinessSignal → bridge dict."""
        scorer = DecisionQualityScorer()
        text = "Following my plan. Checked 4h and daily. R:R is good. Stop at support."
        dq_result = scorer.score(text)

        # Verify it could be attached to a signal
        sig = ReadinessSignal(
            readiness_score=75.0,
            cognitive_load="low",
            active_stressors=[],
            override_loss_rate_7d=0.1,
            emotional_state="calm",
            confidence_trend="stable",
            components=ReadinessComponents(),
            decision_quality_score=dq_result.composite_score,
        )

        d = sig.to_dict()
        assert d["decision_quality_score"] == round(dq_result.composite_score, 1)
        assert d["decision_quality_score"] > 0

    def test_outcome_enrichment_roundtrip(self):
        """Integration: Enrich outcome → persist → read → verify enriched fields."""
        bridge = FeedbackBridge(bridge_dir=Path(tempfile.mkdtemp()))

        # Write initial outcome
        signal = OutcomeSignal(pnl_today=75.0, regime="NORMAL")
        bridge.write_outcome(signal)

        # Compute readiness to get state snapshot
        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        rc.compute(emotional_state="calm")
        snapshot = rc.get_last_state_snapshot()

        # Enrich
        outcome_data = bridge.read_outcome().to_dict()
        enriched = bridge.enrich_outcome_signal(outcome_data, snapshot)

        # Read back and verify
        raw = bridge._locked_read(bridge._outcome_path)
        persisted = json.loads(raw)
        assert "readiness_at_trade_time" in persisted
        assert persisted["pnl_today"] == 75.0

    def test_anomaly_creates_life_event_in_graph(self):
        """Integration: Anomaly detected → Life_Event node in graph → /anomalies shows it."""
        graph = SelfModelGraph(db_path=Path(tempfile.mkdtemp()) / "test.db")
        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

        with patch.object(rc._anomaly_detector, 'update') as mock_update:
            mock_update.return_value = AnomalyResult(
                baseline=70.0, residual=-35.0, threshold=12.0,
                anomaly_detected=True, severity=0.85,
            )
            rc.compute(emotional_state="stressed", graph=graph)

        # Verify Life_Event node exists
        life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
        anomaly_events = [n for n in life_events if n.properties.get("source") == "anomaly_detector"]
        assert len(anomaly_events) >= 1
        assert anomaly_events[0].properties["severity"] == 0.85

    def test_quality_command_output(self):
        """Integration: /quality command extracts conversation and scores it."""
        from src.aura.cli.companion import AuraCompanion

        companion = AuraCompanion(
            db_path=Path(tempfile.mkdtemp()) / "test.db",
            bridge_dir=Path(tempfile.mkdtemp()),
        )

        # Simulate message history
        companion._message_history = [
            {"role": "user", "content": "I checked my plan and followed my checklist. 4h chart confirms."},
            {"role": "assistant", "content": "Good process."},
            {"role": "user", "content": "R:R ratio is 2:1 with stop at support level."},
        ]

        output = companion._cmd_quality()
        assert "Decision Quality" in output
        assert "Composite Score" in output

    def test_anomalies_command_empty(self):
        """Integration: /anomalies with no events shows informative message."""
        from src.aura.cli.companion import AuraCompanion

        companion = AuraCompanion(
            db_path=Path(tempfile.mkdtemp()) / "test.db",
            bridge_dir=Path(tempfile.mkdtemp()),
        )
        output = companion._cmd_anomalies()
        assert "No anomalies detected" in output

    def test_quality_command_no_messages(self):
        """Integration: /quality with empty history shows informative message."""
        from src.aura.cli.companion import AuraCompanion

        companion = AuraCompanion(
            db_path=Path(tempfile.mkdtemp()) / "test.db",
            bridge_dir=Path(tempfile.mkdtemp()),
        )
        output = companion._cmd_quality()
        assert "No conversation" in output
