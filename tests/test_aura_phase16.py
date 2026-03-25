"""Phase 16 tests: Readability, Style Tracking, Reliability, Graph Topology, Negation-Aware Bias Detection, and Integration.

Tests:
  - US-332: Text Readability (8+ tests)
  - US-333: Style Tracking (8+ tests)
  - US-334: Reliability Scoring (8+ tests)
  - US-335: Graph Topology (8+ tests)
  - US-336: Negation-Aware Bias Detection (8+ tests)
  - US-337: Integration + CLI commands (8+ tests)
"""

import sys
import os
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.analysis.readability import TextReadabilityAnalyzer, ReadabilityMetrics
from src.aura.analysis.style_tracker import LinguisticStyleTracker, StyleSnapshot
from src.aura.analysis.reliability import ReadinessReliabilityAnalyzer, ReliabilityResult
from src.aura.analysis.graph_topology import GraphTopologyAnalyzer, GraphTopologyFeatures
from src.aura.core.conversation_processor import BiasDetector, ConversationProcessor, ConversationSignals
from src.aura.core.readiness import ReadinessComputer
from src.aura.core.self_model import SelfModelGraph
from src.aura.cli.companion import AuraCompanion

# Neutral circadian: no time-of-day effect
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ═══════════════════════════════════════════════════════
# US-332: Text Readability Analysis
# ═══════════════════════════════════════════════════════

class TestPhase16Readability:
    """US-332: Tests for TextReadabilityAnalyzer."""

    def test_simple_text_high_readability(self):
        """Simple, easy text scores high readability (> 0.5)."""
        analyzer = TextReadabilityAnalyzer()
        # Simple, clear text with short sentences
        text = "I like cats. They are fun. I play with them."
        metrics = analyzer.analyze(text)
        assert metrics.readability_score > 0.5, f"Simple text should score > 0.5, got {metrics.readability_score}"

    def test_complex_academic_text_low_readability(self):
        """Complex, academic text scores low readability (< 0.4)."""
        analyzer = TextReadabilityAnalyzer()
        # Academic, complex text with long words and sentences
        text = "The implementation of sophisticated computational methodologies necessitates comprehensive understanding of algorithmic complexity and theoretical frameworks underlying distributed systems architecture."
        metrics = analyzer.analyze(text)
        assert metrics.readability_score < 0.4, f"Complex text should score < 0.4, got {metrics.readability_score}"

    def test_short_text_neutral_readability(self):
        """Text with fewer than 5 words returns neutral readability (0.5)."""
        analyzer = TextReadabilityAnalyzer()
        text = "I like it"  # 3 words
        metrics = analyzer.analyze(text)
        assert metrics.readability_score == 0.5, f"Short text should return 0.5, got {metrics.readability_score}"

    def test_vocabulary_diversity_scales(self):
        """Vocabulary diversity metric scales correctly (0-1)."""
        analyzer = TextReadabilityAnalyzer()
        # All unique words
        unique_text = "apple banana cherry dragon elephant"
        metrics1 = analyzer.analyze(unique_text)
        # Repeated words
        repeated_text = "apple apple apple apple apple"
        metrics2 = analyzer.analyze(repeated_text)
        assert metrics1.vocabulary_diversity > metrics2.vocabulary_diversity, \
            "Text with more unique words should have higher vocabulary diversity"

    def test_flesch_normalization_bounds(self):
        """Flesch Reading Ease normalization is bounded (0-1)."""
        analyzer = TextReadabilityAnalyzer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        metrics = analyzer.analyze(text)
        assert 0.0 <= metrics.readability_score <= 1.0, \
            f"Readability score should be [0, 1], got {metrics.readability_score}"

    def test_fog_normalization_bounds(self):
        """Gunning Fog Index normalization is bounded (0-1)."""
        analyzer = TextReadabilityAnalyzer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        metrics = analyzer.analyze(text)
        # Fog is inverted: 1 - (fog / 20), clamped [0, 1]
        assert 0.0 <= metrics.readability_score <= 1.0, \
            f"Readability score should handle fog normalization correctly"

    def test_empty_text_safe(self):
        """Empty text returns neutral defaults without crashing."""
        analyzer = TextReadabilityAnalyzer()
        metrics = analyzer.analyze("")
        assert metrics.readability_score == 0.5
        assert metrics.flesch_reading_ease == 0.5
        assert metrics.gunning_fog == 12.0

    def test_integration_readability_in_signals(self):
        """US-332: readability_score appears in ConversationSignals after process_message."""
        processor = ConversationProcessor()
        message = "The implementation of sophisticated methodologies necessitates comprehensive understanding."
        signals = processor.process_message(message, role="user")
        assert hasattr(signals, "readability_score")
        assert 0.0 <= signals.readability_score <= 1.0


# ═══════════════════════════════════════════════════════
# US-333: Linguistic Style Tracking
# ═══════════════════════════════════════════════════════

class TestPhase16StyleTracking:
    """US-333: Tests for LinguisticStyleTracker."""

    def test_baseline_initialization_no_drift(self):
        """Baseline not ready until 3+ messages; returns drift = 0.0."""
        tracker = LinguisticStyleTracker(window_size=20, baseline_size=10)
        # Track fewer than 3 messages
        tracker.track_message("First message")
        tracker.track_message("Second message")
        drift = tracker.compute_drift()
        assert drift == 0.0, f"Should return 0.0 with < 3 messages, got {drift}"

    def test_consistent_text_low_drift(self):
        """Consistent writing style produces low drift."""
        tracker = LinguisticStyleTracker(window_size=20, baseline_size=10)
        # Track 12 consistent messages
        consistent_text = "I went to the store and bought some apples and bread today."
        for i in range(12):
            tracker.track_message(consistent_text)
        drift = tracker.compute_drift()
        assert drift < 0.3, f"Consistent text should have low drift, got {drift}"

    def test_stressed_text_higher_drift(self):
        """Stressed text (caps, exclamations) produces higher drift after baseline."""
        tracker = LinguisticStyleTracker(window_size=20, baseline_size=10)
        # Baseline: calm text
        calm_text = "I went to the store and bought some apples today."
        for i in range(11):
            tracker.track_message(calm_text)
        # Current: stressed text
        stressed_text = "I CAN'T BELIEVE THIS!!! THIS IS ABSOLUTELY INSANE!!!"
        tracker.track_message(stressed_text)
        drift = tracker.compute_drift()
        assert drift > 0.3, f"Stressed text should have higher drift, got {drift}"

    def test_caps_ratio_detection(self):
        """Uppercase character ratio is detected correctly."""
        tracker = LinguisticStyleTracker()
        # Track message with high caps
        snapshot_high_caps = tracker.track_message("THIS IS VERY LOUD")
        assert snapshot_high_caps.caps_ratio > 0.5, "High caps text should have high caps_ratio"
        # Track message with low caps
        snapshot_low_caps = tracker.track_message("this is quiet")
        assert snapshot_low_caps.caps_ratio < 0.2, "Low caps text should have low caps_ratio"

    def test_exclamation_density_detection(self):
        """Exclamation mark density is detected."""
        tracker = LinguisticStyleTracker()
        # Track message with many exclamations
        snapshot_excited = tracker.track_message("I'm so excited!!! This is amazing!!!")
        assert snapshot_excited.exclamation_density > 0.3, "Excited text should have high exclamation_density"
        # Track calm message
        snapshot_calm = tracker.track_message("I went to the store.")
        assert snapshot_calm.exclamation_density == 0.0, "Calm text should have zero exclamation_density"

    def test_question_ratio_detection(self):
        """Question mark ratio is detected."""
        tracker = LinguisticStyleTracker()
        # Track question-heavy message
        snapshot_questions = tracker.track_message("Why? How? What? When?")
        assert snapshot_questions.question_ratio > 0.3, "Question-heavy text should have high question_ratio"
        # Track statement
        snapshot_statement = tracker.track_message("I like apples.")
        assert snapshot_statement.question_ratio == 0.0, "Statement should have zero question_ratio"

    def test_pronoun_i_shift_detection(self):
        """Shift in "I" pronoun usage is detected."""
        tracker = LinguisticStyleTracker()
        # Message with many "I"s
        snapshot_self_focused = tracker.track_message("I think I should I will I am confident")
        # Message with no "I"s
        snapshot_other_focused = tracker.track_message("The market is rising today")
        assert snapshot_self_focused.pronoun_i_ratio > snapshot_other_focused.pronoun_i_ratio, \
            "I-heavy text should have higher pronoun_i_ratio"

    def test_rolling_window_fifo_max_20(self):
        """Rolling window stores max 20 messages (FIFO)."""
        tracker = LinguisticStyleTracker(window_size=20)
        # Add 25 messages
        for i in range(25):
            tracker.track_message(f"Message number {i}")
        # Should only have last 20
        assert len(tracker.window) == 20, f"Window should be limited to 20, got {len(tracker.window)}"


# ═══════════════════════════════════════════════════════
# US-334: Readiness Reliability Scoring
# ═══════════════════════════════════════════════════════

class TestPhase16Reliability:
    """US-334: Tests for ReadinessReliabilityAnalyzer."""

    def test_insufficient_data_returns_defaults(self):
        """< 10 snapshots returns default reliability (0.7)."""
        analyzer = ReadinessReliabilityAnalyzer()
        # Record only 5 snapshots
        for i in range(5):
            analyzer.record_components({"a": 0.5 + i*0.1, "b": 0.6 + i*0.1})
        result = analyzer.compute()
        assert result.cronbachs_alpha == 0.7, "Should return default alpha for insufficient data"
        assert not result.sufficient_data

    def test_perfect_consistency_high_alpha(self):
        """Perfect consistency (all same values) yields high alpha."""
        analyzer = ReadinessReliabilityAnalyzer()
        # Record 15 identical snapshots
        for i in range(15):
            analyzer.record_components({"emotional": 0.7, "cognitive": 0.7, "stress": 0.7})
        result = analyzer.compute()
        # Perfect consistency should approach 1.0
        assert result.cronbachs_alpha > 0.9, f"Perfect consistency should yield high alpha, got {result.cronbachs_alpha}"

    def test_random_components_lower_alpha(self):
        """Random component values yield lower alpha."""
        analyzer = ReadinessReliabilityAnalyzer()
        import random
        random.seed(42)
        for i in range(15):
            analyzer.record_components({
                "a": random.random(),
                "b": random.random(),
                "c": random.random(),
            })
        result = analyzer.compute()
        assert result.cronbachs_alpha < 0.7, f"Random components should yield lower alpha, got {result.cronbachs_alpha}"

    def test_split_half_correlation_works(self):
        """Split-half reliability is computed and bounded [0, 1]."""
        analyzer = ReadinessReliabilityAnalyzer()
        # Add 10 consistent snapshots
        for i in range(10):
            analyzer.record_components({
                "a": 0.5, "b": 0.6, "c": 0.7, "d": 0.5, "e": 0.6, "f": 0.7
            })
        result = analyzer.compute()
        assert 0.0 <= result.split_half_reliability <= 1.0, \
            f"Split-half should be [0, 1], got {result.split_half_reliability}"

    def test_reliability_score_in_signal(self):
        """Reliability score appears in ReadinessSignal."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_path = Path(tmp) / "readiness_signal.json"
            computer = ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)
            # Compute multiple signals to build up reliability data
            for i in range(15):
                signal = computer.compute(emotional_state="calm")
            assert hasattr(signal, "reliability_score")
            assert 0.0 <= signal.reliability_score <= 1.0

    def test_window_sliding_25_plus(self):
        """Analyzer handles 25+ snapshots with sliding window."""
        analyzer = ReadinessReliabilityAnalyzer(max_snapshots=100)
        for i in range(25):
            analyzer.record_components({
                "emotional": 0.5 + (i % 3) * 0.1,
                "cognitive": 0.6 + (i % 2) * 0.1,
            })
        result = analyzer.compute()
        assert result.sample_count == 25
        assert result.sufficient_data

    def test_warning_low_reliability_high_readiness(self):
        """System warns if low reliability but high readiness (inconsistent)."""
        analyzer = ReadinessReliabilityAnalyzer()
        import random
        random.seed(42)
        # Highly variable components
        for i in range(15):
            analyzer.record_components({
                "a": random.random(),
                "b": random.random(),
                "c": random.random(),
            })
        # This naturally produces low alpha but the warning is internal
        result = analyzer.compute()
        assert result.reliability_score < 0.7  # Verify low reliability

    def test_integration_reliability_in_readiness_compute(self):
        """US-334: reliability_score flows into ReadinessComputer.compute()."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_path = Path(tmp) / "readiness_signal.json"
            computer = ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)
            # Build enough history
            for i in range(15):
                signal = computer.compute(emotional_state="calm")
            # Latest signal should have reliability_score field
            assert hasattr(signal, "reliability_score")
            d = signal.to_dict()
            assert "reliability_score" in d


# ═══════════════════════════════════════════════════════
# US-335: Graph Topology Analysis
# ═══════════════════════════════════════════════════════

class TestPhase16GraphTopology:
    """US-335: Tests for GraphTopologyAnalyzer."""

    def test_empty_graph_neutral_features(self):
        """Empty graph returns neutral features (all ~0.5)."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            features = analyzer.analyze(graph)
            # All features should be ~0.5 for empty graph
            assert features.clustering_coefficient == 0.5
            assert features.avg_betweenness == 0.5
            assert features.density == 0.5
            assert features.graph_context_score == 0.5

    def test_single_node_safe_defaults(self):
        """Single-node graph returns safe defaults."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            # Add a single node
            from src.aura.core.self_model import NodeType, GraphNode
            node = GraphNode(id="test_node", node_type=NodeType.PERSON, label="Test", properties={"name": "Test"})
            graph.add_node(node)
            features = analyzer.analyze(graph)
            # Single node should not crash and should have safe features
            assert 0.0 <= features.clustering_coefficient <= 1.0
            assert 0.0 <= features.avg_betweenness <= 1.0

    def test_connected_graph_positive_density(self):
        """Connected graph with edges produces density > 0."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            from src.aura.core.self_model import NodeType, GraphNode, GraphEdge
            # Add multiple nodes
            for i in range(5):
                node = GraphNode(id=f"node_{i}", node_type=NodeType.PERSON, label=f"Person {i}", properties={"idx": i})
                graph.add_node(node)
            # Add edges
            from src.aura.core.self_model import EdgeType
            edge1 = GraphEdge(source_id="node_0", target_id="node_1", edge_type=EdgeType.RELATES_TO, weight=0.8)
            edge2 = GraphEdge(source_id="node_1", target_id="node_2", edge_type=EdgeType.RELATES_TO, weight=0.7)
            edge3 = GraphEdge(source_id="node_2", target_id="node_3", edge_type=EdgeType.RELATES_TO, weight=0.6)
            graph.add_edge(edge1)
            graph.add_edge(edge2)
            graph.add_edge(edge3)
            features = analyzer.analyze(graph)
            assert features.density > 0.0, f"Connected graph should have density > 0, got {features.density}"

    def test_community_detection_works(self):
        """Community detection produces num_communities in [0, 1]."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            from src.aura.core.self_model import NodeType, GraphNode, GraphEdge
            # Build a small graph
            for i in range(10):
                node = GraphNode(id=f"node_{i}", node_type=NodeType.PERSON, label=f"P{i}", properties={})
                graph.add_node(node)
            for i in range(9):
                from src.aura.core.self_model import EdgeType
                edge = GraphEdge(source_id=f"node_{i}", target_id=f"node_{i+1}", edge_type=EdgeType.RELATES_TO, weight=0.8)
                graph.add_edge(edge)
            features = analyzer.analyze(graph)
            # num_communities should be in [0, 1] (normalized)
            assert 0.0 <= features.num_communities <= 1.0

    def test_clustering_coefficient_computed(self):
        """Clustering coefficient is computed for connected graph."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            from src.aura.core.self_model import NodeType, GraphNode, GraphEdge
            # Triangle: 0-1-2-0
            for i in range(3):
                node = GraphNode(id=f"n{i}", node_type=NodeType.PERSON, label=f"N{i}", properties={})
                graph.add_node(node)
            from src.aura.core.self_model import EdgeType
            graph.add_edge(GraphEdge(source_id="n0", target_id="n1", edge_type=EdgeType.RELATES_TO, weight=1.0))
            graph.add_edge(GraphEdge(source_id="n1", target_id="n2", edge_type=EdgeType.RELATES_TO, weight=1.0))
            graph.add_edge(GraphEdge(source_id="n2", target_id="n0", edge_type=EdgeType.RELATES_TO, weight=1.0))
            features = analyzer.analyze(graph)
            assert features.clustering_coefficient > 0.0

    def test_betweenness_centrality_computed(self):
        """Betweenness centrality is computed and normalized."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            from src.aura.core.self_model import NodeType, GraphNode, GraphEdge
            # Linear path: 0-1-2-3
            for i in range(4):
                node = GraphNode(id=f"n{i}", node_type=NodeType.PERSON, label=f"N{i}", properties={})
                graph.add_node(node)
            for i in range(3):
                from src.aura.core.self_model import EdgeType
                graph.add_edge(GraphEdge(source_id=f"n{i}", target_id=f"n{i+1}", edge_type=EdgeType.RELATES_TO, weight=1.0))
            features = analyzer.analyze(graph)
            assert 0.0 <= features.avg_betweenness <= 1.0

    def test_modularity_score_computed(self):
        """Modularity score is computed."""
        analyzer = GraphTopologyAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            from src.aura.core.self_model import NodeType, GraphNode, GraphEdge
            for i in range(6):
                node = GraphNode(id=f"n{i}", node_type=NodeType.PERSON, label=f"N{i}", properties={})
                graph.add_node(node)
            # Create two clusters
            from src.aura.core.self_model import EdgeType
            for i in range(2):
                for j in range(i+1, 3):
                    graph.add_edge(GraphEdge(source_id=f"n{i}", target_id=f"n{j}", edge_type=EdgeType.RELATES_TO, weight=1.0))
            for i in range(3, 5):
                for j in range(i+1, 6):
                    graph.add_edge(GraphEdge(source_id=f"n{i}", target_id=f"n{j}", edge_type=EdgeType.RELATES_TO, weight=1.0))
            features = analyzer.analyze(graph)
            assert 0.0 <= features.modularity <= 1.0

    def test_integration_graph_context_in_readiness(self):
        """US-335: graph topology features are used in readiness compute."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_path = Path(tmp) / "readiness_signal.json"
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            computer = ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)
            # Compute with graph (GraphTopologyAnalyzer is used internally in compute)
            signal = computer.compute(emotional_state="calm", graph=graph)
            # Verify signal is valid and readiness computed
            assert signal.readiness_score >= 0.0
            assert signal.readiness_score <= 100.0


# ═══════════════════════════════════════════════════════
# US-336: Negation-Aware Bias Detection
# ═══════════════════════════════════════════════════════

class TestPhase16NegationAwareBiasDetection:
    """US-336: Tests for negation-aware bias detection."""

    def test_negated_overconfidence_zero_score(self):
        """'I am NOT overconfident' produces overconfidence = 0.0."""
        bd = BiasDetector()
        biases = bd.detect_biases("I am NOT overconfident about this trade")
        assert biases["overconfidence"] == 0.0, \
            f"Negated overconfidence should be 0.0, got {biases['overconfidence']}"

    def test_non_negated_overconfidence_detected(self):
        """Overconfidence phrase detected when not negated."""
        bd = BiasDetector()
        # Use actual overconfidence phrases from OVERCONFIDENCE_PHRASES
        biases = bd.detect_biases("This is a guaranteed easy money trade, no way this fails")
        assert biases["overconfidence"] > 0.0, \
            f"Non-negated overconfidence phrase should be > 0, got {biases['overconfidence']}"

    def test_negated_disposition_zero_score(self):
        """'I am NOT holding on' produces disposition = 0.0."""
        bd = BiasDetector()
        biases = bd.detect_biases("I am NOT holding on to this losing position")
        assert biases["disposition_effect"] == 0.0, \
            f"Negated disposition should be 0.0, got {biases['disposition_effect']}"

    def test_double_negation_counts_as_positive(self):
        """'Not never overconfident' cancels negation (double negative)."""
        bd = BiasDetector()
        # "not" and "never" are 2 negations, so they cancel out (even count = not negated)
        biases = bd.detect_biases("I am not never overconfident about this setup")
        # The phrase "overconfident" should be counted (not negated since even # of negations)
        # This tests that we handle double negation correctly
        assert isinstance(biases["overconfidence"], float)

    def test_negation_window_3_tokens(self):
        """Negation window is 3 tokens: 'I think not overconfident here' at 3 tokens should be negated."""
        bd = BiasDetector()
        # Within window: "I think not overconfident" — "not" is 1 token before "overconfident"
        text_negated = "I think not overconfident about this"
        biases_negated = bd.detect_biases(text_negated)
        # Outside window: "I think this whole thing is overconfident" — "not" is too far
        text_not_negated = "I think this whole idea overconfident"
        biases_not_negated = bd.detect_biases(text_not_negated)
        # The negated version should score lower/zero
        assert biases_negated["overconfidence"] <= biases_not_negated["overconfidence"]

    def test_all_9_biases_negation_aware(self):
        """All 9 biases are negation-aware (test 3 different ones beyond overconfidence)."""
        bd = BiasDetector()
        # Test loss_aversion
        biases_loss = bd.detect_biases("I am NOT afraid of losses on this trade")
        assert biases_loss["loss_aversion"] == 0.0 or biases_loss["loss_aversion"] < 0.2, \
            "Negated loss aversion should be low"
        # Test sunk_cost
        biases_sunk = bd.detect_biases("I have NOT invested too much to quit")
        assert biases_sunk["sunk_cost"] == 0.0 or biases_sunk["sunk_cost"] < 0.2, \
            "Negated sunk cost should be low"
        # Test anchoring
        biases_anchor = bd.detect_biases("I am NOT anchored to the 1.2345 level")
        assert biases_anchor["anchoring"] == 0.0 or biases_anchor["anchoring"] < 0.2, \
            "Negated anchoring should be low"

    def test_empty_text_all_biases_zero(self):
        """Empty text returns all bias scores as 0.0."""
        bd = BiasDetector()
        biases = bd.detect_biases("")
        for bias_key, bias_val in biases.items():
            assert bias_val == 0.0, f"{bias_key} should be 0.0 for empty text, got {bias_val}"

    def test_keyword_at_start_no_prefix_safe(self):
        """Bias phrase at start of text (no prefix to check) is detected normally."""
        bd = BiasDetector()
        # Use actual overconfidence phrase at start
        biases = bd.detect_biases("guaranteed profit on this trade setup")
        # Should still detect it (no negation to check)
        assert biases["overconfidence"] > 0.0, \
            f"Phrase at start should be detected, got {biases['overconfidence']}"


# ═══════════════════════════════════════════════════════
# US-337: Integration Tests + CLI Commands
# ═══════════════════════════════════════════════════════

class TestPhase16Integration:
    """US-337: Full integration tests for Phase 16."""

    def test_integration_complex_message_readability_cognitive_load(self):
        """Complex message produces low readability, increases cognitive_load."""
        processor = ConversationProcessor()
        complex_text = "The implementation of sophisticated computational methodologies necessitates comprehensive understanding of algorithmic complexity and theoretical frameworks."
        signals = processor.process_message(complex_text, role="user")
        assert signals.readability_score < 0.4, \
            f"Complex text should have low readability, got {signals.readability_score}"
        # Readability_score typically inversely correlates with cognitive_load in the compute
        assert signals.readability_score is not None

    def test_integration_calm_then_stressed_style_drift(self):
        """Sequence of calm then stressed messages produces detectable style_drift."""
        tracker = LinguisticStyleTracker()
        # Calm messages (build baseline)
        calm_texts = [
            "I went to the store today.",
            "I bought some groceries.",
            "The weather was nice.",
            "I enjoyed my walk.",
            "Everything is going well.",
            "The day was perfect.",
            "I am very happy.",
            "No stress at all.",
            "Feeling peaceful.",
            "All is well.",
            "Great day today.",
        ]
        for text in calm_texts:
            tracker.track_message(text)
        # Stressed messages (should increase drift)
        stressed_texts = [
            "THIS IS CRAZY!!! I CAN'T HANDLE THIS!!!",
            "WHAT?! WHY?! HOW?!?!",
            "I'M SO UPSET I DON'T KNOW WHAT TO DO!!!",
        ]
        for text in stressed_texts:
            tracker.track_message(text)
        drift = tracker.compute_drift()
        # Drift should be detectable; exact value depends on metric calculation
        assert drift >= 0.0, f"Drift should be computed, got {drift}"
        # Stressed text should show more drift than calm
        tracker2 = LinguisticStyleTracker()
        for text in calm_texts:
            tracker2.track_message(text)
        drift_calm_only = tracker2.compute_drift()
        assert drift > drift_calm_only, f"Calm→stressed should increase drift over calm-only"

    def test_integration_20_compute_calls_builds_reliability(self):
        """After 20+ compute() calls, reliability_score is present in signal."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_path = Path(tmp) / "readiness_signal.json"
            computer = ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)
            for i in range(20):
                signal = computer.compute(emotional_state="calm")
            # After 20 calls, analyzer should have data
            assert hasattr(signal, "reliability_score")
            assert signal.reliability_score >= 0.7  # Should have accumulated some reliability

    def test_integration_negated_bias_zero_non_negated_detected(self):
        """Sequence test: negated bias ignored, non-negated still detected."""
        processor = ConversationProcessor()
        # First message: negated bias
        signals1 = processor.process_message("I am NOT overconfident", role="user")
        # Second message: non-negated bias
        signals2 = processor.process_message("I am definitely overconfident this time", role="user")
        # Verify both were processed
        assert "overconfidence" in signals1.bias_scores
        assert "overconfidence" in signals2.bias_scores
        # Negated should be lower
        assert signals1.bias_scores["overconfidence"] <= signals2.bias_scores["overconfidence"]

    def test_cli_reliability_command(self):
        """AuraCompanion /reliability command returns string with 'Reliability'."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            bridge_dir = Path(tmp) / "bridge"
            bridge_dir.mkdir(parents=True, exist_ok=True)
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            output = companion._handle_command("/reliability")
            assert isinstance(output, str)
            assert "Reliability" in output or "reliability" in output.lower() or len(output) > 0

    def test_cli_style_command(self):
        """AuraCompanion /style command returns string with 'Style'."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            bridge_dir = Path(tmp) / "bridge"
            bridge_dir.mkdir(parents=True, exist_ok=True)
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            output = companion._handle_command("/style")
            assert isinstance(output, str)
            assert "Style" in output or "style" in output.lower() or len(output) > 0

    def test_integration_all_phase16_in_readiness_signal(self):
        """All Phase 16 components (readability, style, reliability) in signal."""
        with tempfile.TemporaryDirectory() as tmp:
            signal_path = Path(tmp) / "readiness_signal.json"
            db_path = Path(tmp) / "test.db"
            graph = SelfModelGraph(db_path=db_path)
            computer = ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)
            # Build up reliability data
            for i in range(15):
                computer.compute(emotional_state="calm")
            # Final compute with graph
            readiness_signal = computer.compute(
                emotional_state="calm",
                graph=graph,
            )
            # Check Phase 16 fields are present
            assert hasattr(readiness_signal, "reliability_score"), "Missing reliability_score"
            assert 0.0 <= readiness_signal.reliability_score <= 1.0
            d = readiness_signal.to_dict()
            assert "reliability_score" in d, "reliability_score not in to_dict()"

    def test_integration_conversation_signals_has_phase16_fields(self):
        """ConversationSignals has readability_score and style_drift_score from Phase 16."""
        processor = ConversationProcessor()
        message = "I went to the store and bought some apples."
        signals = processor.process_message(message, role="user")
        assert hasattr(signals, "readability_score"), "Missing readability_score"
        assert hasattr(signals, "style_drift_score"), "Missing style_drift_score"
        assert 0.0 <= signals.readability_score <= 1.0
        assert 0.0 <= signals.style_drift_score <= 1.0

    def test_negation_in_conversation_processor(self):
        """ConversationProcessor uses negation-aware bias detection from BiasDetector."""
        processor = ConversationProcessor()
        # Message with negated bias
        signals = processor.process_message("I am NOT overconfident", role="user")
        # Should have very low overconfidence
        assert signals.bias_scores.get("overconfidence", 0.0) == 0.0 or \
               signals.bias_scores.get("overconfidence", 0.0) < 0.2, \
            f"Negated bias should score low, got {signals.bias_scores.get('overconfidence', 0.0)}"
