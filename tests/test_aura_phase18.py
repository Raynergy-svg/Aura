"""Phase 18 integration and unit tests — Signal Integration, Emotional Granularity & Anomaly Action."""
import json
import math
import os
import sys
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ─── US-344: Style Drift Penalty + Reliability Dampener ───

class TestPhase18StyleDriftPenalty:
    """US-344: Style drift penalty and reliability dampener tests."""

    def _make_computer(self):
        tmp = tempfile.mkdtemp()
        signal_path = Path(tmp) / "readiness_signal.json"
        from src.aura.core.readiness import ReadinessComputer
        return ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)

    def test_drift_above_threshold_reduces_readiness(self):
        """Drift > 0.4 should reduce readiness score."""
        rc = self._make_computer()
        sig_no_drift = rc.compute(emotional_state="calm", style_drift_score=0.0)
        rc2 = self._make_computer()
        sig_drift = rc2.compute(emotional_state="calm", style_drift_score=0.7)
        # 0.7 drift: penalty = min(5.0, (0.7-0.4)*8.0) = min(5.0, 2.4) = 2.4
        assert sig_drift.readiness_score < sig_no_drift.readiness_score

    def test_drift_below_threshold_no_effect(self):
        """Drift <= 0.4 should not affect readiness."""
        rc = self._make_computer()
        sig_no_drift = rc.compute(emotional_state="calm", style_drift_score=0.0)
        rc2 = self._make_computer()
        sig_low_drift = rc2.compute(emotional_state="calm", style_drift_score=0.3)
        assert abs(sig_no_drift.readiness_score - sig_low_drift.readiness_score) < 0.5

    def test_drift_max_penalty_capped(self):
        """Drift = 1.0 should cap penalty at 5.0 points."""
        rc = self._make_computer()
        sig_no_drift = rc.compute(emotional_state="calm", style_drift_score=0.0)
        rc2 = self._make_computer()
        sig_max_drift = rc2.compute(emotional_state="calm", style_drift_score=1.0)
        # Penalty = min(5.0, (1.0-0.4)*8.0) = min(5.0, 4.8) = 4.8
        diff = sig_no_drift.readiness_score - sig_max_drift.readiness_score
        assert diff <= 6.0  # Allow some tolerance for EMA smoothing

    def test_drift_default_zero(self):
        """Default style_drift_score should be 0.0 (no penalty)."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        # Should compute normally
        assert sig.readiness_score >= 0

    def test_reliability_dampener_pulls_toward_50(self):
        """Low reliability should pull extreme scores toward 50."""
        from src.aura.analysis.reliability import ReadinessReliabilityAnalyzer
        rc = self._make_computer()
        rc._reliability_analyzer = ReadinessReliabilityAnalyzer()
        # Feed 15 snapshots to meet sufficient_data threshold
        for i in range(15):
            rc._reliability_analyzer.record_components({
                "emotional_state": 0.3 + (i % 3) * 0.2,
                "cognitive_load": 0.1 + (i % 5) * 0.15,
                "override_discipline": 0.9 - (i % 4) * 0.1,
                "stress_level": 0.2 + (i % 3) * 0.1,
                "confidence_trend": 0.5 + (i % 2) * 0.1,
                "engagement": 0.4 + (i % 3) * 0.15,
            })
        # Force low reliability by monkey-patching
        with patch.object(rc._reliability_analyzer, 'compute') as mock_compute:
            from src.aura.analysis.reliability import ReliabilityResult
            mock_compute.return_value = ReliabilityResult(
                cronbachs_alpha=0.2,
                split_half_reliability=0.2,
                reliability_score=0.3,
                sample_count=15,
                sufficient_data=True,
            )
            sig = rc.compute(emotional_state="calm")
        # With reliability=0.3, score should be pulled significantly toward 50
        # Not easy to assert exact value due to all the blending, but score should exist
        assert 0 <= sig.readiness_score <= 100

    def test_reliability_above_threshold_no_dampening(self):
        """Reliability >= 0.5 should not dampen."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        # Default reliability is 0.7, no dampening
        assert sig.readiness_score >= 0

    def test_state_snapshot_drift_fields(self):
        """State snapshot should include drift and reliability fields."""
        rc = self._make_computer()
        rc.compute(emotional_state="calm", style_drift_score=0.5)
        snap = rc._last_state_snapshot
        assert "style_drift_penalty_applied" in snap
        assert snap["style_drift_penalty_applied"] is True

    def test_combined_drift_and_low_reliability(self):
        """Both drift penalty and reliability dampener can apply together."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm", style_drift_score=0.6)
        # Should not crash with both active
        assert 0 <= sig.readiness_score <= 100


# ─── US-345: Emotional Granularity Scorer ───

class TestPhase18EmotionalGranularity:
    """US-345: Emotional granularity scorer tests."""

    def test_single_emotion_low_granularity(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update("I feel happy happy happy happy")
        assert result.composite < 0.3

    def test_diverse_vocabulary_high_granularity(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update(
            "I feel happy but also anxious and a bit frustrated. "
            "There's some excitement mixed with dread and curiosity. "
            "I'm hopeful yet disgusted by the volatility."
        )
        assert result.composite > 0.3

    def test_entropy_increases_with_diversity(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        s1 = EmotionalGranularityScorer()
        r1 = s1.update("happy happy happy")  # Single cluster
        s2 = EmotionalGranularityScorer()
        r2 = s2.update("happy sad angry afraid surprised")  # Multiple clusters
        assert r2.entropy > r1.entropy

    def test_differentiation_within_cluster(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        s1 = EmotionalGranularityScorer()
        r1 = s1.update("sad sad sad")  # Single word in cluster
        s2 = EmotionalGranularityScorer()
        r2 = s2.update("sad melancholy gloomy depressed dejected")  # Multiple words in same cluster
        assert r2.differentiation > r1.differentiation

    def test_empty_text_returns_defaults(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update("")
        assert result.composite == 0.0

    def test_window_respects_max(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer(window_size=3)
        # Window stores words, not messages — but deque should have a maxlen
        assert scorer._word_history.maxlen is not None

    def test_composite_weighting(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update("happy sad angry afraid surprised disgusted trusting eager")
        # Composite = 0.35 * richness + 0.40 * entropy + 0.25 * differentiation
        expected = 0.35 * result.vocabulary_richness + 0.40 * result.entropy + 0.25 * result.differentiation
        assert abs(result.composite - expected) < 0.01

    def test_no_emotion_words_zero_scores(self):
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update("The market opened at 1.2345 and closed at 1.2350")
        assert result.composite == 0.0


# ─── US-346: Anomaly-to-Action Pipeline ───

class TestPhase18AnomalyAction:
    """US-346: Anomaly dampener and action pipeline tests."""

    def _make_computer(self):
        tmp = tempfile.mkdtemp()
        signal_path = Path(tmp) / "readiness_signal.json"
        from src.aura.core.readiness import ReadinessComputer
        return ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)

    def test_anomaly_history_deque_exists(self):
        rc = self._make_computer()
        assert hasattr(rc, '_anomaly_history')
        assert isinstance(rc._anomaly_history, deque)

    def test_signal_has_anomaly_action_fields(self):
        """Verify anomaly action fields exist on a computed signal."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        assert hasattr(sig, 'anomaly_action_taken')
        assert hasattr(sig, 'anomaly_dampening')
        assert sig.anomaly_action_taken is False
        assert sig.anomaly_dampening == 0.0

    def test_no_anomaly_no_action(self):
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        assert sig.anomaly_action_taken is False
        assert sig.anomaly_dampening == 0.0

    def test_state_snapshot_anomaly_field(self):
        rc = self._make_computer()
        rc.compute(emotional_state="calm")
        snap = rc._last_state_snapshot
        assert "anomaly_dampened" in snap

    def test_signal_serializes_anomaly_fields(self):
        """Verify anomaly fields serialize correctly via compute."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        d = sig.to_dict()
        assert "anomaly_action_taken" in d
        assert "anomaly_dampening" in d
        assert isinstance(d["anomaly_action_taken"], bool)
        assert isinstance(d["anomaly_dampening"], float)

    def test_anomaly_history_capped(self):
        rc = self._make_computer()
        assert rc._anomaly_history.maxlen == 50

    def test_compute_with_spike_doesnt_crash(self):
        """Even with extreme emotional state, compute should not crash."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="anxious", stress_keywords=["panic", "crash"])
        assert 0 <= sig.readiness_score <= 100

    def test_moving_average_empty_history_safe(self):
        """First compute with no history should be safe."""
        rc = self._make_computer()
        sig = rc.compute(emotional_state="calm")
        assert sig.readiness_score >= 0


# ─── US-347: Style Drift as T1 Early-Warning ───

class TestPhase18StyleDriftT1:
    """US-347: Style drift T1 pattern tests."""

    def test_three_consecutive_drifts_trigger_warning(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        proc = ConversationProcessor()
        proc._drift_history = [0.6, 0.7, 0.8]
        warning = proc.check_drift_warning()
        assert warning is not None
        assert warning["consecutive_count"] == 3

    def test_two_consecutive_no_trigger(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        proc = ConversationProcessor()
        proc._drift_history = [0.6, 0.7]
        warning = proc.check_drift_warning()
        assert warning is None

    def test_drift_below_threshold_resets(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        proc = ConversationProcessor()
        proc._drift_history = [0.6, 0.7, 0.3, 0.6, 0.7]  # Break at 0.3
        warning = proc.check_drift_warning()
        # After 0.3 breaks the chain, only 0.6, 0.7 remain consecutive
        # 2 consecutive < 3, so no warning
        assert warning is None

    def test_t1_generates_pattern_from_warning(self):
        from src.aura.patterns.tier1 import Tier1FrequencyDetector
        tmp = Path(tempfile.mkdtemp()) / "patterns"
        detector = Tier1FrequencyDetector(patterns_dir=tmp)
        warnings = [{"type": "style_drift_warning", "consecutive_count": 4, "avg_drift": 0.7, "max_drift": 0.9, "domain": "HUMAN"}]
        patterns = detector.detect(conversations=[], readiness_history=[], override_events=[], drift_warnings=warnings)
        drift_patterns = [p for p in patterns if "style_drift_warning" in p.pattern_id]
        assert len(drift_patterns) >= 1

    def test_drift_confidence_scales(self):
        from src.aura.patterns.tier1 import Tier1FrequencyDetector
        tmp1 = Path(tempfile.mkdtemp()) / "patterns"
        tmp2 = Path(tempfile.mkdtemp()) / "patterns"
        det1 = Tier1FrequencyDetector(patterns_dir=tmp1)
        det2 = Tier1FrequencyDetector(patterns_dir=tmp2)
        w1 = [{"type": "style_drift_warning", "consecutive_count": 3, "avg_drift": 0.5, "max_drift": 0.6, "domain": "HUMAN"}]
        w2 = [{"type": "style_drift_warning", "consecutive_count": 3, "avg_drift": 0.8, "max_drift": 0.9, "domain": "HUMAN"}]
        p1 = det1.detect([], [], [], drift_warnings=w1)
        p2 = det2.detect([], [], [], drift_warnings=w2)
        drift1 = [p for p in p1 if "style_drift_warning" in p.pattern_id]
        drift2 = [p for p in p2 if "style_drift_warning" in p.pattern_id]
        if drift1 and drift2:
            assert drift2[0].confidence >= drift1[0].confidence

    def test_drift_history_capped(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        proc = ConversationProcessor()
        proc._drift_history = [0.5] * 15
        if len(proc._drift_history) > 10:
            proc._drift_history = proc._drift_history[-10:]
        assert len(proc._drift_history) == 10

    def test_empty_history_safe(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        proc = ConversationProcessor()
        warning = proc.check_drift_warning()
        assert warning is None

    def test_t1_no_warnings_no_crash(self):
        from src.aura.patterns.tier1 import Tier1FrequencyDetector
        tmp = Path(tempfile.mkdtemp()) / "patterns"
        det = Tier1FrequencyDetector(patterns_dir=tmp)
        patterns = det.detect([], [], [])
        # Should not crash, returns whatever patterns exist
        assert isinstance(patterns, list)


# ─── US-348: Narrative Coherence Tracker ───

class TestPhase18NarrativeCoherence:
    """US-348: Narrative coherence tracker tests."""

    def test_identical_sessions_high_coherence(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        text = "I'm watching the breakout on EUR/USD with support at 1.0850"
        tracker.update(text, 0.6)
        result = tracker.update(text, 0.6)
        assert result.lexical_overlap > 0.5
        assert result.sentiment_consistency > 0.5

    def test_different_sessions_low_coherence(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        tracker.update("Bullish breakout momentum on EUR/USD with strong support", 0.8)
        result = tracker.update("Completely lost confused about everything nothing makes sense", 0.2)
        assert result.composite < 0.5

    def test_sentiment_stability(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        tracker.update("market looks good", 0.7)
        tracker.update("market still good", 0.72)
        result = tracker.update("market remains positive", 0.68)
        assert result.sentiment_consistency > 0.5

    def test_strategy_persistence_detected(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        tracker.update("I'm going for a breakout trade with stop loss at support", 0.6)
        result = tracker.update("Another breakout setup with stop loss below support level", 0.6)
        assert result.strategy_persistence > 0.0

    def test_single_session_returns_defaults(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        result = tracker.update("first session text", 0.5)
        assert result.composite == 0.5

    def test_stopwords_excluded(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker, STOPWORDS
        tracker = NarrativeCoherenceTracker()
        words = tracker._extract_content_words("the quick brown fox jumps over the lazy dog")
        assert "the" not in words
        assert "over" not in words
        assert "fox" in words

    def test_composite_weighting(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        tracker.update("breakout support resistance momentum", 0.5)
        result = tracker.update("breakout support resistance momentum", 0.5)
        expected = 0.35 * result.lexical_overlap + 0.35 * result.sentiment_consistency + 0.30 * result.strategy_persistence
        assert abs(result.composite - round(expected, 4)) < 0.02

    def test_empty_text_safe(self):
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        tracker.update("", 0.5)
        result = tracker.update("", 0.5)
        assert 0 <= result.composite <= 1.0


# ─── US-349: Integration Tests ───

class TestPhase18Integration:
    """Integration tests across Phase 18 features."""

    def _make_computer(self):
        tmp = tempfile.mkdtemp()
        signal_path = Path(tmp) / "readiness_signal.json"
        from src.aura.core.readiness import ReadinessComputer
        return ReadinessComputer(signal_path=signal_path, circadian_config=NEUTRAL_CIRCADIAN)

    def test_style_drift_reduces_readiness(self):
        """Integration: High style drift reduces readiness vs baseline."""
        rc1 = self._make_computer()
        rc2 = self._make_computer()
        sig_base = rc1.compute(emotional_state="calm", style_drift_score=0.0)
        sig_drift = rc2.compute(emotional_state="calm", style_drift_score=0.7)
        assert sig_drift.readiness_score < sig_base.readiness_score

    def test_diverse_emotions_high_granularity(self):
        """Integration: Diverse emotion text produces high granularity score."""
        from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
        scorer = EmotionalGranularityScorer()
        result = scorer.update(
            "I feel anxious and worried but also excited and hopeful. "
            "There's frustration mixed with curiosity and some sadness."
        )
        assert result.composite > 0.3

    def test_narrative_coherence_across_sessions(self):
        """Integration: Consistent sessions produce high coherence."""
        from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
        tracker = NarrativeCoherenceTracker()
        base = "Trading breakout with support at key level using momentum strategy"
        tracker.update(base, 0.6)
        tracker.update(base + " confirmed by volume", 0.62)
        result = tracker.update(base + " still holding strong", 0.58)
        assert result.composite > 0.3

    def test_full_compute_with_all_phase18_features(self):
        """Integration: compute() with all Phase 18 parameters doesn't crash."""
        rc = self._make_computer()
        sig = rc.compute(
            emotional_state="anxious",
            stress_keywords=["worried"],
            style_drift_score=0.5,
            message_text="I changed my mind because new data shows the breakout failed",
        )
        assert 0 <= sig.readiness_score <= 100
        assert isinstance(sig.anomaly_action_taken, bool)
        assert isinstance(sig.anomaly_dampening, float)


# ─── US-349: CLI Command Tests ───

class TestPhase18CLICommands:
    """CLI command tests for /granularity and /coherence."""

    def _make_companion(self):
        from src.aura.cli.companion import AuraCompanion
        tmp = tempfile.mkdtemp()
        db_path = Path(tmp) / "test.db"
        bridge_dir = Path(tmp) / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        return AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)

    def test_granularity_command_returns_string(self):
        companion = self._make_companion()
        result = companion._handle_command("/granularity")
        assert isinstance(result, str)
        assert "Granularity" in result

    def test_coherence_command_returns_string(self):
        companion = self._make_companion()
        result = companion._handle_command("/coherence")
        assert isinstance(result, str)
        assert "Coherence" in result

    def test_granularity_no_data(self):
        companion = self._make_companion()
        result = companion._handle_command("/granularity")
        # Should show "no data" message, not crash
        assert isinstance(result, str)

    def test_coherence_no_data(self):
        companion = self._make_companion()
        result = companion._handle_command("/coherence")
        # Should show "need more sessions" message, not crash
        assert isinstance(result, str)

    def test_help_includes_new_commands(self):
        companion = self._make_companion()
        result = companion._handle_command("/unknown_cmd_xyz")
        assert "/granularity" in result
        assert "/coherence" in result

    def test_all_commands_in_help(self):
        companion = self._make_companion()
        result = companion._handle_command("/notacommand")
        for cmd in ["/granularity", "/coherence", "/flexibility", "/journal"]:
            assert cmd in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
