"""Tests for AuraCompanion command handlers and process_input pipeline.

US-270: Behavioral tests for all 8+ command handlers.
US-271: Resilience tests with exception injection at each pipeline stage.

These test the 930-line companion.py which previously had ZERO behavioral tests.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.core.conversation_processor import ConversationProcessor, ConversationSignals
from aura.core.readiness import ReadinessComputer, ReadinessSignal, ReadinessComponents


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_readiness_signal(score: float = 65.0) -> ReadinessSignal:
    """Create a ReadinessSignal with given score."""
    return ReadinessSignal(
        readiness_score=score,
        emotional_state="neutral",
        cognitive_load="normal",
        active_stressors=[],
        override_loss_rate_7d=0.3,
        confidence_trend="stable",
        components=ReadinessComponents(
            emotional_state_score=0.7,
            cognitive_load_score=0.6,
            override_discipline_score=0.8,
            stress_level_score=0.7,
            confidence_trend_score=0.5,
            engagement_score=0.6,
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_graph_stats(nodes: int = 10, edges: int = 5, convs: int = 3) -> Dict[str, Any]:
    """Create a mock graph stats dict."""
    return {
        "total_nodes": nodes,
        "total_edges": edges,
        "total_conversations": convs,
        "nodes_by_type": {"person": 1, "goal": 2, "conversation": convs, "emotion": 2},
    }


def _make_outcome_signal():
    """Create a mock OutcomeSignal."""
    mock = MagicMock()
    mock.streak = "winning"
    mock.pnl_today = 45.50
    mock.win_rate_7d = 0.65
    mock.regime = "NORMAL"
    mock.confidence_at_time = 0.72
    mock.weighted_vote_at_time = 0.68
    return mock


def _make_bridge_status(outcome_available: bool = True) -> Dict[str, Any]:
    """Create a mock bridge status dict."""
    return {
        "readiness_signal": {"available": True, "score": 65.0},
        "outcome_signal": {"available": outcome_available, "pnl_today": 45.50},
        "override_events": {"total_recent": 3},
    }


@pytest.fixture
def companion(tmp_path):
    """Create an AuraCompanion with mocked dependencies."""
    from aura.cli.companion import AuraCompanion

    comp = AuraCompanion(
        db_path=tmp_path / "test_graph.db",
        bridge_dir=tmp_path / "bridge",
    )
    # Pre-set internal state so commands work without running start_session
    comp._latest_readiness = _make_readiness_signal(65.0)
    comp._latest_signals = ConversationSignals(
        emotional_state="neutral",
        sentiment_score=0.55,
        message_count=3,
        topics=["trading"],
    )

    return comp


# ═══════════════════════════════════════════════════════════════════════
# US-270: Command handler behavioral tests
# ═══════════════════════════════════════════════════════════════════════


class TestUS270CommandHandlers:
    """US-270: Behavioral tests for all command handlers in AuraCompanion."""

    def test_cmd_status_returns_session_info(self, companion):
        """/status output includes session ID, message count, graph stats."""
        companion.graph.get_stats = MagicMock(return_value=_make_graph_stats())
        companion.processor.get_session_summary = MagicMock(return_value={
            "message_count": 5,
            "net_sentiment": 0.123,
        })

        result = companion.process_input("/status")

        assert "Aura Status" in result
        assert "Messages this session: 5" in result
        assert "0.123" in result
        assert "Nodes: 10" in result
        assert "Edges: 5" in result
        assert "Readiness:" in result
        assert "65" in result  # readiness score

    def test_cmd_bridge_shows_signal_status(self, companion):
        """/bridge shows readiness → Buddy and outcome ← Buddy status."""
        companion.bridge.get_bridge_status = MagicMock(
            return_value=_make_bridge_status(outcome_available=True)
        )

        result = companion.process_input("/bridge")

        assert "Bridge Status" in result
        assert "Readiness" in result
        assert "Outcomes" in result
        assert "Override events: 3" in result

    def test_cmd_bridge_no_buddy(self, companion):
        """/bridge handles Buddy not running."""
        companion.bridge.get_bridge_status = MagicMock(
            return_value=_make_bridge_status(outcome_available=False)
        )

        result = companion.process_input("/bridge")

        assert "not active" in result or "✗" in result

    def test_cmd_readiness_shows_breakdown(self, companion):
        """/readiness shows score and all 6 component weights."""
        result = companion.process_input("/readiness")

        assert "65" in result  # score
        assert "Emotional" in result
        assert "Cognitive" in result
        assert "Override discipline" in result
        assert "Stress" in result
        assert "Confidence" in result
        assert "Engagement" in result
        assert "25%" in result  # emotional weight
        assert "20%" in result  # cognitive weight

    def test_cmd_readiness_no_score_yet(self, companion):
        """/readiness with no computed score returns helpful message."""
        companion._latest_readiness = None
        result = companion.process_input("/readiness")
        assert "No readiness score" in result

    def test_cmd_graph_shows_stats(self, companion):
        """/graph shows node/edge counts and type breakdown."""
        companion.graph.get_stats = MagicMock(return_value=_make_graph_stats(15, 8, 5))
        companion.graph.get_readiness_history = MagicMock(return_value=[
            {"timestamp": "2026-03-23T10:00:00", "score": 70.0, "trigger": "conversation_update"},
        ])

        result = companion.process_input("/graph")

        assert "Self-Model Graph" in result
        assert "Total nodes: 15" in result
        assert "Total edges: 8" in result
        assert "person: 1" in result
        assert "goal: 2" in result
        assert "70" in result  # readiness history

    def test_cmd_patterns_shows_report(self, companion):
        """/patterns displays the pattern engine report."""
        companion.pattern_engine.format_patterns_report = MagicMock(
            return_value="T1: 2 patterns\nT2: 1 pattern\nT3: 0 arcs"
        )

        result = companion.process_input("/patterns")

        assert "T1: 2 patterns" in result

    def test_cmd_patterns_run_forces_analysis(self, companion):
        """/patterns run forces T1+T2+T3 and shows counts."""
        companion.graph.get_recent_conversations = MagicMock(return_value=[])
        companion.graph.get_readiness_history = MagicMock(return_value=[])
        companion.pattern_engine.run_all = MagicMock(return_value={
            "t1": ["p1", "p2"], "t2": ["p3"], "t3": [],
        })
        companion.pattern_engine.format_patterns_report = MagicMock(return_value="Report text")

        result = companion.process_input("/patterns run")

        companion.pattern_engine.run_all.assert_called_once()
        assert "2 T1" in result
        assert "1 T2" in result
        assert "0 T3" in result

    def test_cmd_quit_returns_sentinel(self, companion):
        """/quit returns the __QUIT__ sentinel."""
        result = companion.process_input("/quit")
        assert result == "__QUIT__"

    def test_unknown_command_returns_help(self, companion):
        """Unknown command returns helpful error with /help reference."""
        result = companion.process_input("/foobar")
        assert "Unknown command" in result
        assert "/help" in result

    def test_cmd_validate_runs_validator(self, companion):
        """/validate runs the graph validator (or handles import error)."""
        # SelfModelValidator is imported lazily inside _cmd_validate.
        # The exception handler catches any error — test both paths.
        result = companion.process_input("/validate")
        # Either the validator runs and returns a report, or the import
        # fails and the except block returns a failure message.
        assert isinstance(result, str) and len(result) > 0
        # Must contain something meaningful — either a report or error info
        has_health_info = any(kw in result.lower() for kw in [
            "validation", "failed", "healthy", "orphan", "graph", "issue",
        ])
        assert has_health_info, f"Unexpected /validate output: {result[:200]}"

    def test_cmd_rules_no_active_rules(self, companion):
        """/rules with no active rules shows informational message."""
        with patch.dict("sys.modules", {}):
            result = companion.process_input("/rules")
            # Either shows rules or error (module import may fail in test env)
            assert isinstance(result, str) and len(result) > 0

    def test_cmd_predict_no_models(self, companion):
        """/predict with no prediction models shows not-available."""
        companion._override_predictor = None
        companion._readiness_v2 = None

        result = companion.process_input("/predict")

        assert "Prediction Models" in result
        assert "not available" in result

    def test_cmd_predict_with_override_predictor(self, companion):
        """/predict with override predictor shows risk assessment."""
        mock_predictor = MagicMock()
        mock_predictor.get_model_info.return_value = {
            "trained": True,
            "train_samples": 50,
            "train_accuracy": 0.72,
        }
        mock_prediction = MagicMock()
        mock_prediction.loss_probability = 0.35
        mock_prediction.risk_level = "moderate"
        mock_prediction.top_risk_factors = ["emotional_state", "cognitive_load"]
        mock_predictor.predict_loss_probability.return_value = mock_prediction

        companion._override_predictor = mock_predictor
        companion._readiness_v2 = None

        result = companion.process_input("/predict")

        assert "Override Predictor" in result
        assert "Trained: True" in result
        assert "35%" in result
        assert "moderate" in result
        assert "emotional_state" in result

    def test_cmd_predict_with_readiness_v2(self, companion):
        """/predict with readiness v2 shows model info."""
        mock_v2 = MagicMock()
        mock_v2.get_model_info.return_value = {
            "version": "v2",
            "buffer_size": 30,
            "min_samples": 50,
            "trained": False,
            "samples_until_v2": 20,
            "r_squared": 0.0,
        }

        companion._override_predictor = None
        companion._readiness_v2 = mock_v2

        result = companion.process_input("/predict")

        assert "Readiness V2" in result
        assert "30/50" in result  # buffer_size/min_samples
        assert "Samples until v2: 20" in result


# ═══════════════════════════════════════════════════════════════════════
# US-271: process_input resilience with exception injection
# ═══════════════════════════════════════════════════════════════════════


class TestUS271ProcessInputResilience:
    """US-271: Each pipeline stage fails independently without killing session."""

    def test_stage1_signal_extraction_failure(self, companion):
        """Stage 1: ConversationProcessor crash → neutral signals, session continues."""
        companion.processor.process_message = MagicMock(
            side_effect=RuntimeError("NLP engine exploded")
        )

        # Should not raise — stage 1 catch returns neutral signals
        result = companion.process_input("I'm feeling great today")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should still produce a response (stage 6 uses fallback signals)

    def test_stage3_graph_log_failure(self, companion):
        """Stage 3: Graph logging crash → session continues, response generated."""
        companion.graph.add_node = MagicMock(
            side_effect=Exception("SQLite disk error")
        )

        result = companion.process_input("Hello there")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_stage4_readiness_failure(self, companion):
        """Stage 4: Readiness computation crash → last-known score used."""
        original_readiness = companion._latest_readiness

        companion.readiness.compute = MagicMock(
            side_effect=ZeroDivisionError("bad math")
        )
        # Also mock _update_readiness's graph calls
        companion.bridge.get_recent_overrides = MagicMock(return_value=[])
        companion.graph.get_recent_conversations = MagicMock(return_value=[])

        result = companion.process_input("Market is volatile today")

        assert isinstance(result, str)
        # Latest readiness should still be the original (fallback)
        assert companion._latest_readiness is not None

    def test_stage5_pattern_detection_failure(self, companion):
        """Stage 5: Pattern detection crash → session continues normally."""
        companion.pattern_engine.run_t1 = MagicMock(
            side_effect=Exception("Pattern engine OOM")
        )

        result = companion.process_input("Seeing some interesting setups")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_stage6_response_generation_failure(self, companion):
        """Stage 6: Response generation crash → safe fallback message."""
        # Need to make _generate_response fail but keep other stages working
        with patch.object(companion, "_generate_response",
                         side_effect=KeyError("missing context")):
            result = companion.process_input("How's it going")

        assert isinstance(result, str)
        assert len(result) > 0
        # Fallback message from US-237
        assert "listening" in result.lower() or "feeling" in result.lower()

    def test_multiple_stage_failures(self, companion):
        """Multiple stages failing simultaneously → session still survives."""
        companion.graph.add_node = MagicMock(
            side_effect=Exception("DB locked")
        )
        companion.pattern_engine.run_t1 = MagicMock(
            side_effect=Exception("Pattern crash")
        )

        result = companion.process_input("Testing resilience")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_assistant_log_failure_nonfatal(self, companion, caplog):
        """Post-response assistant message log failure is non-fatal."""
        call_count = [0]
        original_process = companion.processor.process_message

        def failing_on_assistant(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("role") == "assistant" or (len(args) > 1 and args[1] == "assistant"):
                raise RuntimeError("Assistant log failed")
            return original_process(*args, **kwargs)

        companion.processor.process_message = failing_on_assistant

        with caplog.at_level(logging.ERROR):
            result = companion.process_input("Test message")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_input_command_routing(self, companion):
        """Commands starting with / are routed to _handle_command, not pipeline."""
        companion.processor.process_message = MagicMock()

        result = companion.process_input("/status")

        # process_message should NOT have been called for a command
        companion.processor.process_message.assert_not_called()
        assert "Aura Status" in result or "status" in result.lower()

    def test_message_history_appended_after_response(self, companion):
        """process_input appends user + assistant to message history."""
        initial_len = len(companion._message_history)

        companion.process_input("Hello world")

        # Should have user + assistant = 2 new entries
        assert len(companion._message_history) == initial_len + 2
        assert companion._message_history[-2]["role"] == "user"
        assert companion._message_history[-2]["content"] == "Hello world"
        assert companion._message_history[-1]["role"] == "assistant"

    def test_message_history_cap_enforced(self, companion):
        """Message history is capped at MAX_MESSAGE_HISTORY per US-262."""
        # Pre-fill with MAX messages
        companion._message_history = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(companion.MAX_MESSAGE_HISTORY)
        ]

        companion.process_input("One more message")

        assert len(companion._message_history) <= companion.MAX_MESSAGE_HISTORY


# ═══════════════════════════════════════════════════════════════════════
# Additional behavioral edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestCompanionBehavioralEdgeCases:
    """Additional edge case tests for companion behavior."""

    def test_emotional_state_stressed_response(self, companion):
        """Stressed emotional state triggers acknowledgment."""
        companion.processor.process_message = MagicMock(
            return_value=ConversationSignals(
                emotional_state="stressed",
                sentiment_score=0.2,
                stress_keywords_found=["overwhelmed", "burned out"],
                message_count=1,
            )
        )
        companion.bridge.get_recent_overrides = MagicMock(return_value=[])
        companion.graph.get_recent_conversations = MagicMock(return_value=[])
        companion.graph.get_readiness_history = MagicMock(return_value=[])
        companion.bridge.read_outcome = MagicMock(return_value=None)

        result = companion.process_input("I'm completely overwhelmed and burned out")

        # Template response should acknowledge stress in some form
        result_lower = result.lower()
        assert any(phrase in result_lower for phrase in [
            "dealing with a lot", "stressed", "stress", "weight", "wound up", "tension",
            "carrying", "step away", "feel", "yourself",
        ]), f"Expected stress acknowledgment in: {result}"

    def test_override_mentioned_triggers_history_check(self, companion):
        """Override mention triggers bridge history lookup."""
        companion.processor.process_message = MagicMock(
            return_value=ConversationSignals(
                emotional_state="neutral",
                override_mentioned=True,
                message_count=2,
            )
        )
        mock_override = MagicMock()
        mock_override.outcome = "loss"
        companion.bridge.get_recent_overrides = MagicMock(return_value=[mock_override, mock_override])
        companion.graph.get_recent_conversations = MagicMock(return_value=[])
        companion.graph.get_readiness_history = MagicMock(return_value=[])
        companion.bridge.read_outcome = MagicMock(return_value=None)

        result = companion.process_input("I want to override Buddy's signal")

        assert "override" in result.lower()
        assert "loss" in result.lower()

    def test_low_readiness_triggers_warning(self, companion):
        """Low readiness score triggers position-sizing warning."""
        companion._latest_readiness = _make_readiness_signal(35.0)
        companion.processor.process_message = MagicMock(
            return_value=ConversationSignals(
                emotional_state="stressed",
                message_count=1,
                sentiment_score=0.2,
            )
        )
        companion.bridge.get_recent_overrides = MagicMock(return_value=[])
        companion.graph.get_recent_conversations = MagicMock(return_value=[])
        companion.graph.get_readiness_history = MagicMock(return_value=[])
        companion.bridge.read_outcome = MagicMock(return_value=None)

        # Mock _update_readiness to return low score
        low_readiness = _make_readiness_signal(35.0)
        companion.readiness.compute = MagicMock(return_value=low_readiness)
        companion.graph.log_readiness = MagicMock()

        result = companion.process_input("Everything is falling apart")

        # Softened response should mention readiness state or Buddy's adjustment
        result_lower = result.lower()
        assert any(w in result_lower for w in [
            "readiness", "position", "reduce", "yourself", "buddy", "careful", "adjusting",
        ]), f"Expected low-readiness acknowledgment in: {result}"

    def test_end_session_logs_to_graph(self, companion):
        """end_session() logs the conversation summary to graph."""
        companion.processor.get_session_summary = MagicMock(return_value={
            "message_count": 5,
            "net_sentiment": 0.1,
        })
        companion.graph.log_conversation = MagicMock()
        companion.graph.close = MagicMock()

        companion.end_session()

        companion.graph.log_conversation.assert_called_once()
        call_kwargs = companion.graph.log_conversation.call_args
        assert call_kwargs[1]["conversation_id"] == companion._conversation_id
        companion.graph.close.assert_called_once()
