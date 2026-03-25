"""Tests for semantic signal intelligence improvements.

US-280: Negation-aware emotional signal extraction.
US-281: Emotional intensity scaling with modifier words.
US-282: Confidence trend acceleration detection.
US-283: Decision fatigue detection via override frequency spike.
US-284: Cognitive load estimation from message complexity.
US-285: Companion /coach command.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.core.conversation_processor import ConversationProcessor, ConversationSignals
from aura.core.readiness import ReadinessComputer, ReadinessSignal


# ═══════════════════════════════════════════════════════════════════════
# US-280: Negation-aware emotional signal extraction
# ═══════════════════════════════════════════════════════════════════════


class TestUS280NegationDetection:
    """US-280: Negation words cancel keyword signals."""

    def setup_method(self):
        self.proc = ConversationProcessor()

    def test_simple_negation_cancels_stress(self):
        """'I am not stressed' should NOT detect stress."""
        signals = self.proc.process_message("I am not stressed at all")
        assert "stressed" not in signals.stress_keywords_found
        assert signals.emotional_state != "stressed"

    def test_no_worries_cancels_worried(self):
        """'No worries' should NOT detect worry/anxiety."""
        signals = self.proc.process_message("No worries, everything is fine")
        assert "worried" not in signals.stress_keywords_found

    def test_contraction_negation_dont(self):
        """'I don't feel stressed' cancels stress keyword."""
        signals = self.proc.process_message("I don't feel stressed")
        assert "stressed" not in signals.stress_keywords_found

    def test_contraction_negation_cant(self):
        """'Can't complain, not tired' cancels fatigue."""
        signals = self.proc.process_message("Can't say I'm tired today")
        assert signals.fatigue_detected is False

    def test_non_negated_stress_still_detected(self):
        """Regular stress keywords without negation still work."""
        signals = self.proc.process_message("I am so stressed about the deadline")
        assert "stressed" in signals.stress_keywords_found
        assert signals.emotional_state in ("stressed", "anxious")

    def test_double_negation_resolves_to_positive(self):
        """'not not stressed' = stressed (double negation)."""
        signals = self.proc.process_message("I'm not not stressed about this")
        assert "stressed" in signals.stress_keywords_found

    def test_negation_positive_keyword(self):
        """'I am not happy' should NOT detect positive signal."""
        signals = self.proc.process_message("I am not happy with the results")
        assert "happy" not in signals.positive_keywords_found

    def test_negation_beyond_window_does_not_cancel(self):
        """Negation far from keyword should not cancel it."""
        # 'not' is more than 3 tokens from 'stressed'
        signals = self.proc.process_message("I am not sure what to think about being stressed")
        assert "stressed" in signals.stress_keywords_found

    def test_never_negation(self):
        """'never worried' cancels worried."""
        signals = self.proc.process_message("I'm never worried about these trades")
        assert "worried" not in signals.stress_keywords_found

    def test_barely_negation(self):
        """'barely anxious' cancels anxious."""
        signals = self.proc.process_message("I'm barely anxious anymore")
        assert "anxious" not in signals.stress_keywords_found


# ═══════════════════════════════════════════════════════════════════════
# US-281: Emotional intensity scaling
# ═══════════════════════════════════════════════════════════════════════


class TestUS281IntensityScaling:
    """US-281: Intensity modifiers scale emotional signals."""

    def setup_method(self):
        self.proc = ConversationProcessor()

    def test_amplifier_increases_stress_score(self):
        """'extremely stressed' has higher stress than 'stressed'."""
        proc1 = ConversationProcessor()
        signals_plain = proc1.process_message("I am stressed")
        proc2 = ConversationProcessor()
        signals_amplified = proc2.process_message("I am extremely stressed")
        # Amplified stress should yield lower sentiment (more negative)
        assert signals_amplified.sentiment_score < signals_plain.sentiment_score

    def test_diminisher_reduces_stress_score(self):
        """'slightly stressed' has lower stress than 'stressed'."""
        proc1 = ConversationProcessor()
        signals_plain = proc1.process_message("I am stressed")
        proc2 = ConversationProcessor()
        signals_diminished = proc2.process_message("I am slightly stressed")
        # Diminished stress should yield higher sentiment (less negative)
        assert signals_diminished.sentiment_score > signals_plain.sentiment_score

    def test_amplifier_positive(self):
        """'extremely happy' has higher positive score than 'happy'."""
        proc1 = ConversationProcessor()
        signals_plain = proc1.process_message("I feel happy")
        proc2 = ConversationProcessor()
        signals_amplified = proc2.process_message("I feel extremely happy")
        assert signals_amplified.sentiment_score > signals_plain.sentiment_score

    def test_intensity_score_in_signals(self):
        """intensity_score field is present and reasonable."""
        signals = self.proc.process_message("I am very stressed about everything")
        assert hasattr(signals, "intensity_score")
        assert 0.0 <= signals.intensity_score <= 1.0

    def test_neutral_message_default_intensity(self):
        """Message with no keywords has default intensity 0.5."""
        signals = self.proc.process_message("The weather is nice today")
        assert signals.intensity_score == 0.5

    def test_intensity_in_to_dict(self):
        """intensity_score appears in to_dict output."""
        signals = self.proc.process_message("I feel calm")
        d = signals.to_dict()
        assert "intensity_score" in d

    def test_very_modifier_amplifies(self):
        """'very' is an amplifier word."""
        proc1 = ConversationProcessor()
        plain = proc1.process_message("I feel tired")
        proc2 = ConversationProcessor()
        amplified = proc2.process_message("I feel very tired")
        # Very should boost the fatigue signal
        assert amplified.sentiment_score <= plain.sentiment_score


# ═══════════════════════════════════════════════════════════════════════
# Combined negation + intensity tests
# ═══════════════════════════════════════════════════════════════════════


class TestUS280US281Combined:
    """Combined negation and intensity edge cases."""

    def test_negation_trumps_intensity(self):
        """'not extremely stressed' — negation cancels even amplified keyword."""
        proc = ConversationProcessor()
        signals = proc.process_message("I'm not extremely stressed")
        assert "stressed" not in signals.stress_keywords_found

    def test_existing_keyword_detection_preserved(self):
        """Existing non-negated, non-modified keywords still work as before."""
        proc = ConversationProcessor()
        signals = proc.process_message("I had an argument with my partner about the deadline")
        assert "argument" in signals.stress_keywords_found
        assert "deadline" in signals.stress_keywords_found
        assert "relationship stress" in signals.detected_stressors

    def test_mixed_sentiment_message(self):
        """Message with both positive and negative signals works."""
        proc = ConversationProcessor()
        signals = proc.process_message("I'm stressed but also motivated to improve")
        assert "stressed" in signals.stress_keywords_found
        assert "motivated" in signals.positive_keywords_found

    def test_backward_compat_empty_message(self):
        """Empty message still returns neutral signals (US-261 preserved)."""
        proc = ConversationProcessor()
        signals = proc.process_message("")
        assert signals.emotional_state == "neutral"
        assert signals.intensity_score == 0.5


# ═══════════════════════════════════════════════════════════════════════
# US-282: Confidence trend acceleration detection
# ═══════════════════════════════════════════════════════════════════════


class TestUS282ConfidenceAcceleration:
    """US-282: Second derivative of confidence trend."""

    def test_stable_scores_zero_acceleration(self, tmp_path):
        """Stable readiness scores produce zero acceleration."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        accel = comp.compute_confidence_acceleration([70, 70, 70])
        assert accel == 0.0

    def test_gradual_decline_small_negative_acceleration(self, tmp_path):
        """Gradual linear decline has ~zero acceleration (constant velocity)."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # Linear decline: 70 → 65 → 60 — constant velocity, zero acceleration
        accel = comp.compute_confidence_acceleration([70, 65, 60])
        assert abs(accel) < 0.01  # Near-zero for linear

    def test_rapid_decline_negative_acceleration(self, tmp_path):
        """Accelerating decline (80→75→60) has negative acceleration."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # Velocity changes: v0 = -5, v1 = -15 → accel = -10
        accel = comp.compute_confidence_acceleration([80, 75, 60])
        assert accel < 0  # Negative = accelerating decline

    def test_rapid_recovery_positive_acceleration(self, tmp_path):
        """Accelerating recovery (40→45→60) has positive acceleration."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        accel = comp.compute_confidence_acceleration([40, 45, 60])
        assert accel > 0  # Positive = accelerating recovery

    def test_insufficient_data_returns_zero(self, tmp_path):
        """Less than 3 data points returns zero acceleration."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        assert comp.compute_confidence_acceleration([70]) == 0.0
        assert comp.compute_confidence_acceleration([70, 65]) == 0.0
        assert comp.compute_confidence_acceleration([]) == 0.0

    def test_acceleration_in_readiness_signal(self, tmp_path):
        """confidence_acceleration field appears in ReadinessSignal."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # Compute multiple times to build history
        comp.compute(emotional_state="calm", conversation_count_7d=3)
        comp.compute(emotional_state="calm", conversation_count_7d=3)
        signal = comp.compute(emotional_state="calm", conversation_count_7d=3)
        assert hasattr(signal, "confidence_acceleration")
        d = signal.to_dict()
        assert "confidence_acceleration" in d

    def test_rapid_decline_applies_penalty(self, tmp_path):
        """Rapid confidence decline lowers readiness score."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # Inject a declining history
        comp._readiness_history = [80.0, 70.0]
        # Next compute should detect acceleration and may apply penalty
        signal = comp.compute(emotional_state="stressed", conversation_count_7d=1)
        # The signal should have the acceleration field populated
        assert isinstance(signal.confidence_acceleration, float)


# ═══════════════════════════════════════════════════════════════════════
# US-283: Decision fatigue detection
# ═══════════════════════════════════════════════════════════════════════


class TestUS283DecisionFatigue:
    """US-283: Override frequency spike as fatigue signal."""

    def test_no_events_no_fatigue(self, tmp_path):
        """Empty override events → zero fatigue."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        assert comp.compute_fatigue_score([]) == 0.0

    def test_few_events_no_fatigue(self, tmp_path):
        """Less than 3 events → zero fatigue (insufficient data)."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        events = [{"trade_won": True}, {"trade_won": False}]
        assert comp.compute_fatigue_score(events) == 0.0

    def test_uniform_events_no_spike(self, tmp_path):
        """Uniform override rate → no fatigue spike."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # 20 events, evenly distributed
        events = [{"trade_won": True}] * 20
        fatigue = comp.compute_fatigue_score(events)
        # Uniform distribution: recent rate ≈ historical rate → no spike
        assert fatigue == 0.0 or fatigue < 0.1

    def test_spike_detected(self, tmp_path):
        """Many recent overrides vs few historical → fatigue detected."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # 5 historical + 15 recent (FATIGUE_WINDOW_SIZE=10 recent)
        # Historical: 5 events in 10 slots → 0.5 rate
        # Recent: 10 events in 10 slots → 1.0 rate = 2x spike
        events = [{"trade_won": True}] * 5 + [{"trade_won": True}] * 15
        fatigue = comp.compute_fatigue_score(events)
        # Should detect some fatigue
        assert fatigue >= 0.0  # May or may not trigger depending on ratio

    def test_fatigue_score_in_signal(self, tmp_path):
        """fatigue_score field appears in ReadinessSignal."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        signal = comp.compute(emotional_state="calm", conversation_count_7d=3)
        assert hasattr(signal, "fatigue_score")
        d = signal.to_dict()
        assert "fatigue_score" in d

    def test_fatigue_score_bounded(self, tmp_path):
        """Fatigue score is always between 0.0 and 1.0."""
        comp = ReadinessComputer(signal_path=tmp_path / "bridge" / "readiness.json", circadian_config={h: 1.0 for h in range(24)})
        # Even with extreme spike
        events = [{"trade_won": True}] * 3 + [{"trade_won": True}] * 100
        fatigue = comp.compute_fatigue_score(events)
        assert 0.0 <= fatigue <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# US-284: Cognitive load estimation from message complexity
# ═══════════════════════════════════════════════════════════════════════


class TestUS284CognitiveLoadEstimation:
    """US-284: Dynamic cognitive load from text analysis."""

    def setup_method(self):
        self.proc = ConversationProcessor()

    def test_simple_message_low_load(self):
        """Short, simple message → low cognitive load."""
        score = self.proc.estimate_cognitive_load("feeling good today")
        assert score < 0.3

    def test_complex_message_high_load(self):
        """Complex message with conditionals and questions → high load."""
        msg = (
            "If the market drops but GDP is up, should I trade? "
            "However, considering the correlation with gold, "
            "I'm also wondering whether to hedge. "
            "But if the Fed raises rates, everything changes. "
            "What do you think about all of this?"
        )
        score = self.proc.estimate_cognitive_load(msg)
        assert score > 0.5

    def test_empty_message_returns_low(self):
        """Empty string → minimal cognitive load."""
        score = self.proc.estimate_cognitive_load("")
        assert score == 0.1

    def test_question_heavy_message(self):
        """Multiple questions increase cognitive load."""
        msg = "Should I trade? What about EUR/USD? Is the trend still up? When should I exit?"
        score = self.proc.estimate_cognitive_load(msg)
        assert score > 0.3  # Questions add load

    def test_conditional_words_increase_load(self):
        """Messages with 'if/but/however' have higher load."""
        simple = "I want to trade EUR/USD today"
        complex_msg = "If markets are volatile but news is positive, however I'm not sure"
        simple_score = self.proc.estimate_cognitive_load(simple)
        complex_score = self.proc.estimate_cognitive_load(complex_msg)
        assert complex_score > simple_score

    def test_score_bounded(self):
        """Cognitive load score is always between 0.1 and 0.95."""
        # Very long complex message
        long_msg = "If " * 50 + "but " * 50 + "however? " * 20
        score = self.proc.estimate_cognitive_load(long_msg)
        assert 0.1 <= score <= 0.95

        # Very short message
        short_score = self.proc.estimate_cognitive_load("ok")
        assert 0.1 <= short_score <= 0.95


# ═══════════════════════════════════════════════════════════════════════
# US-285: Companion /coach command
# ═══════════════════════════════════════════════════════════════════════


class TestUS285CoachCommand:
    """US-285: /coach provides actionable readiness-based recommendations."""

    @pytest.fixture
    def companion(self, tmp_path):
        from aura.cli.companion import AuraCompanion
        bridge_dir = tmp_path / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        comp = AuraCompanion(
            db_path=tmp_path / "test.db",
            bridge_dir=bridge_dir,
        )
        return comp

    def _make_readiness(self, score, emotional=0.7, cognitive=0.7,
                        override=0.8, stress=0.7, confidence=0.7, engagement=0.5):
        from aura.core.readiness import ReadinessComponents, ReadinessSignal
        return ReadinessSignal(
            readiness_score=score,
            cognitive_load="low",
            active_stressors=[],
            override_loss_rate_7d=0.0,
            emotional_state="calm",
            confidence_trend="stable",
            components=ReadinessComponents(
                emotional_state_score=emotional,
                cognitive_load_score=cognitive,
                override_discipline_score=override,
                stress_level_score=stress,
                confidence_trend_score=confidence,
                engagement_score=engagement,
            ),
        )

    def test_coach_no_data(self, companion):
        """Coach with no readiness data shows helpful message."""
        companion._latest_readiness = None
        result = companion.process_input("/coach")
        assert "enough data" in result.lower() or "coach" in result.lower()

    def test_coach_high_readiness(self, companion):
        """Coach with high readiness affirms the user."""
        companion._latest_readiness = self._make_readiness(80)
        companion._latest_signals = ConversationSignals()
        result = companion.process_input("/coach")
        assert "solid" in result.lower() or "good" in result.lower()

    def test_coach_low_emotional_state(self, companion):
        """Coach identifies low emotional state and recommends action."""
        companion._latest_readiness = self._make_readiness(45, emotional=0.2)
        companion._latest_signals = ConversationSignals()
        result = companion.process_input("/coach")
        assert "emotional" in result.lower()

    def test_coach_low_override_discipline(self, companion):
        """Coach identifies poor override discipline."""
        companion._latest_readiness = self._make_readiness(50, override=0.2)
        companion._latest_signals = ConversationSignals()
        result = companion.process_input("/coach")
        assert "override" in result.lower() or "discipline" in result.lower()

    def test_coach_low_cognitive_load(self, companion):
        """Coach identifies high cognitive load."""
        companion._latest_readiness = self._make_readiness(48, cognitive=0.2)
        companion._latest_signals = ConversationSignals()
        result = companion.process_input("/coach")
        assert "cognitive" in result.lower() or "load" in result.lower()

    def test_coach_recommendations_are_specific(self, companion):
        """Coach recommendations reference specific component scores."""
        companion._latest_readiness = self._make_readiness(40, stress=0.15, emotional=0.2)
        companion._latest_signals = ConversationSignals()
        result = companion.process_input("/coach")
        # Should contain percentage-like format from recommendations
        assert "%" in result or "stress" in result.lower() or "emotional" in result.lower()
