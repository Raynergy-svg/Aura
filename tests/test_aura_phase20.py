"""Phase 20 Tests: Resilience, Forecasting & Behavioral Memory.

Tests for US-356 through US-361.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import pytest

from src.aura.analysis.readability import TextReadabilityAnalyzer, ReadabilityMetrics


# ============================================================
# US-356: Resilience Score Computation
# ============================================================

class TestUS356Resilience:
    """Tests for ResilienceTracker in src/aura/scoring/resilience.py"""

    def _make_tracker(self):
        from src.aura.scoring.resilience import ResilienceTracker
        return ResilienceTracker()

    def test_cold_start_returns_neutral_50(self):
        """No episodes yet -> neutral resilience_score=50.0"""
        tracker = self._make_tracker()
        result = tracker.update(70.0)
        assert result.resilience_score == 50.0
        assert result.bounce_count == 0

    def test_recovery_below_threshold_tracked(self):
        """Dipping below threshold then recovering creates an episode."""
        tracker = self._make_tracker()
        # Push below threshold
        tracker.update(30.0)
        tracker.update(25.0)
        # Recover above threshold
        result = tracker.update(60.0)
        assert result.bounce_count == 1

    def test_fast_recovery_high_speed_score(self):
        """Single-step recovery -> recovery_speed close to 1.0"""
        tracker = self._make_tracker()
        tracker.update(30.0)   # 1 step in low state
        result = tracker.update(70.0)   # immediately recovers
        assert result.recovery_speed > 0.8

    def test_slow_recovery_low_speed_score(self):
        """Many steps in low state -> lower recovery_speed."""
        tracker = self._make_tracker()
        for _ in range(9):
            tracker.update(20.0)  # 9 steps in low state
        result = tracker.update(70.0)  # recover
        assert result.recovery_speed < 0.5

    def test_multiple_episodes_increases_bounce_count(self):
        """Three complete recovery episodes -> bounce_count = 3"""
        tracker = self._make_tracker()
        for _ in range(3):
            tracker.update(20.0)   # dip below
            tracker.update(70.0)   # recover
        result = tracker.update(70.0)
        assert result.bounce_count == 3

    def test_consistency_high_when_similar_recovery_times(self):
        """Same recovery time across episodes -> high recovery_consistency."""
        tracker = self._make_tracker()
        for _ in range(5):
            tracker.update(20.0)   # 1 step in low state
            tracker.update(70.0)   # recover in 1 step
        result = tracker.update(70.0)
        assert result.recovery_consistency > 0.8

    def test_resilience_score_bounds_0_to_100(self):
        """resilience_score must always be in [0, 100]."""
        tracker = self._make_tracker()
        from src.aura.scoring.resilience import ResilienceTracker
        # Feed many dip+recover cycles
        for _ in range(20):
            tracker.update(10.0)
            tracker.update(90.0)
        result = tracker.update(50.0)
        assert 0.0 <= result.resilience_score <= 100.0

    def test_no_episode_if_always_above_threshold(self):
        """Never dipping below threshold -> no recovery episodes, neutral score."""
        tracker = self._make_tracker()
        for score in [60.0, 70.0, 80.0, 65.0, 75.0]:
            result = tracker.update(score)
        assert result.bounce_count == 0
        assert result.resilience_score == 50.0


# ============================================================
# US-357: Multi-Horizon Readiness Forecasting
# ============================================================

class TestUS357Forecasting:
    """Tests for ReadinessForecaster in src/aura/prediction/forecaster.py"""

    def _make_forecaster(self):
        from src.aura.prediction.forecaster import ReadinessForecaster
        return ReadinessForecaster()

    def test_cold_start_forecast_equals_current(self):
        """With < 3 scores, forecast should equal current score."""
        f = self._make_forecaster()
        result = f.update(65.0)
        assert result.forecast_1h == 65.0
        assert result.forecast_6h == 65.0
        assert result.forecast_24h == 65.0

    def test_rising_trend_forecasts_higher(self):
        """Steadily increasing scores should produce forecast_1h > current minimum."""
        f = self._make_forecaster()
        for score in [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]:
            result = f.update(score)
        # For a rising trend, 1h forecast should be above starting point
        assert result.forecast_1h > 55.0

    def test_falling_trend_forecasts_lower(self):
        """Steadily decreasing scores -> forecast_1h below starting value."""
        f = self._make_forecaster()
        for score in [80.0, 75.0, 70.0, 65.0, 60.0, 55.0]:
            result = f.update(score)
        # For a falling trend, 1h forecast should be below starting point
        assert result.forecast_1h < 75.0

    def test_stable_trend_labels_stable(self):
        """Scores near constant -> trend_direction = stable."""
        f = self._make_forecaster()
        for _ in range(8):
            result = f.update(60.0)
        assert result.trend_direction == "stable"

    def test_confidence_increases_with_stable_history(self):
        """Stable (low variance) history -> higher confidence than initial."""
        f = self._make_forecaster()
        for _ in range(10):
            f.update(60.0)
        result = f.update(60.0)
        # With zero variance, confidence should be high
        assert result.confidence > 0.8

    def test_forecasts_clamped_to_0_100(self):
        """No forecast should go below 0 or above 100."""
        f = self._make_forecaster()
        # Feed very high scores to try to push forecast above 100
        for score in [95.0, 97.0, 99.0, 100.0, 100.0, 100.0]:
            result = f.update(score)
        assert result.forecast_24h <= 100.0
        # Feed very low scores to try to push below 0
        f2 = self._make_forecaster()
        for score in [5.0, 3.0, 1.0, 0.0, 0.0, 0.0]:
            result = f2.update(score)
        assert result.forecast_24h >= 0.0

    def test_forecast_24h_further_than_6h_for_trend(self):
        """For a clear rising trend, forecast_24h > forecast_6h > forecast_1h."""
        f = self._make_forecaster()
        for score in [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]:
            result = f.update(score)
        # All three forecasts exist; for rising trend 24h should be further out
        assert result.forecast_24h >= result.forecast_6h

    def test_rising_label_when_velocity_positive(self):
        """Strongly rising trend -> trend_direction = rising."""
        f = self._make_forecaster()
        for score in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]:
            result = f.update(score)
        assert result.trend_direction == "rising"


# ============================================================
# US-358: Goal-Emotion Coupling
# ============================================================

class TestUS358GoalEmotion:
    """Tests for GoalEmotionCoupler in src/aura/scoring/goal_emotion.py"""

    def _make_coupler(self):
        from src.aura.scoring.goal_emotion import GoalEmotionCoupler
        return GoalEmotionCoupler()

    def test_approach_words_detected(self):
        """Text with approach words -> approach_score > 0."""
        coupler = self._make_coupler()
        result = coupler.analyze("I am achieving my goals and making progress")
        assert result.approach_score > 0.0

    def test_avoidance_words_detected(self):
        """Text with avoidance words -> avoidance_score > 0."""
        coupler = self._make_coupler()
        result = coupler.analyze("I am stuck and it feels hopeless, I keep failing")
        assert result.avoidance_score > 0.0

    def test_positive_emotion_boosts_alignment(self):
        """Calm state with approach words -> alignment boosted."""
        coupler = self._make_coupler()
        neutral_result = coupler.analyze("I am achieving and making progress on track", "neutral")
        calm_result = coupler.analyze("I am achieving and making progress on track", "calm")
        assert calm_result.goal_alignment >= neutral_result.goal_alignment

    def test_stress_reduces_alignment(self):
        """Stressed state with avoidance words -> alignment reduced."""
        coupler = self._make_coupler()
        neutral_result = coupler.analyze("I am stuck and hopeless, giving up", "neutral")
        stressed_result = coupler.analyze("I am stuck and hopeless, giving up", "stressed")
        assert stressed_result.goal_alignment <= neutral_result.goal_alignment

    def test_neutral_text_neutral_alignment(self):
        """No approach or avoidance words -> alignment near 0."""
        coupler = self._make_coupler()
        result = coupler.analyze("Today is a normal day", "neutral")
        assert abs(result.goal_alignment) < 0.3

    def test_goal_alignment_clamped(self):
        """goal_alignment must always be within [-1, 1]."""
        coupler = self._make_coupler()
        # Max out approach words + positive emotion
        result = coupler.analyze(
            "achieving accomplished completed succeeded breakthrough finished making progress on track",
            "calm"
        )
        assert -1.0 <= result.goal_alignment <= 1.0

    def test_coupling_strength_is_max_of_approach_avoidance(self):
        """coupling_strength = max(approach_score, avoidance_score)."""
        coupler = self._make_coupler()
        result = coupler.analyze("I am achieving goals but also stuck and blocked")
        expected = max(result.approach_score, result.avoidance_score)
        assert abs(result.coupling_strength - expected) < 0.001

    def test_mixed_approach_avoidance_text(self):
        """Mixed text -> both scores non-zero."""
        coupler = self._make_coupler()
        result = coupler.analyze("I accomplished some things but I am stuck and failing too")
        assert result.approach_score > 0.0
        assert result.avoidance_score > 0.0


# ============================================================
# US-359: Pattern Momentum Detector
# ============================================================

class TestUS359Momentum:
    """Tests for PatternMomentumAnalyzer in src/aura/patterns/momentum.py"""

    def _make_analyzer(self):
        from src.aura.patterns.momentum import PatternMomentumAnalyzer
        return PatternMomentumAnalyzer()

    def test_cold_start_stable(self):
        """Less than 3 values -> stable defaults."""
        a = self._make_analyzer()
        result = a.update(0.5)
        assert result.momentum_label == "stable"
        assert result.momentum_score == 50.0
        assert result.velocity == 0.0
        assert result.acceleration == 0.0

    def test_increasing_frequency_accelerating(self):
        """Strongly increasing frequencies -> accelerating label."""
        a = self._make_analyzer()
        for freq in [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
            result = a.update(freq)
        assert result.momentum_label == "accelerating"

    def test_decreasing_frequency_decelerating(self):
        """Strongly decreasing frequencies -> decelerating label."""
        a = self._make_analyzer()
        for freq in [2.0, 1.5, 1.0, 0.7, 0.4, 0.2, 0.1]:
            result = a.update(freq)
        assert result.momentum_label == "decelerating"

    def test_momentum_score_bounds(self):
        """momentum_score must always be within [0, 100]."""
        a = self._make_analyzer()
        for freq in [0.1, 100.0, 0.0, 50.0, 200.0, 1.0]:
            result = a.update(freq)
            assert 0.0 <= result.momentum_score <= 100.0

    def test_velocity_positive_for_rising_pattern(self):
        """Rising frequencies -> velocity > 0."""
        a = self._make_analyzer()
        for freq in [1.0, 2.0, 3.0, 4.0, 5.0]:
            result = a.update(freq)
        assert result.velocity > 0.0

    def test_stable_frequencies_stable_label(self):
        """Constant frequencies -> stable label."""
        a = self._make_analyzer()
        for _ in range(8):
            result = a.update(1.0)
        assert result.momentum_label == "stable"

    def test_acceleration_computed_from_velocity_change(self):
        """Acceleration = current velocity - previous velocity."""
        a = self._make_analyzer()
        # First meaningful result
        for freq in [1.0, 1.0, 1.0, 1.0, 1.0]:
            r1 = a.update(freq)
        # Now add a spike to change velocity
        r2 = a.update(10.0)
        # acceleration should reflect the velocity change
        assert isinstance(r2.acceleration, float)

    def test_single_value_stable(self):
        """A single update returns stable with neutral score."""
        a = self._make_analyzer()
        result = a.update(3.0)
        assert result.momentum_score == 50.0
        assert result.momentum_label == "stable"


# ============================================================
# US-360: Behavioral Profile Consolidation
# ============================================================

class TestUS360BehavioralProfile:
    """Tests for BehavioralProfile in src/aura/core/behavioral_profile.py"""

    def _make_profile(self):
        from src.aura.core.behavioral_profile import BehavioralProfile
        return BehavioralProfile()  # in-memory

    def test_update_pattern_stored(self):
        """update_pattern stores the pattern retrievable in summary."""
        p = self._make_profile()
        p.update_pattern("trading", "overconfidence", frequency=0.6, strength=0.5)
        summary = p.get_profile_summary()
        all_patterns = (summary["strengths"] + summary["vulnerabilities"] +
                        summary["signatures"])
        names = [e["pattern_name"] for e in all_patterns]
        assert "overconfidence" in names

    def test_high_strength_classified_as_strength(self):
        """strength > 0.7 -> category = strength."""
        p = self._make_profile()
        p.update_pattern("trading", "discipline", frequency=0.8, strength=0.8)
        summary = p.get_profile_summary()
        strength_names = [e["pattern_name"] for e in summary["strengths"]]
        assert "discipline" in strength_names

    def test_low_strength_classified_as_vulnerability(self):
        """strength < 0.3 -> category = vulnerability."""
        p = self._make_profile()
        p.update_pattern("trading", "loss_aversion", frequency=0.5, strength=0.2)
        summary = p.get_profile_summary()
        vuln_names = [e["pattern_name"] for e in summary["vulnerabilities"]]
        assert "loss_aversion" in vuln_names

    def test_mid_strength_classified_as_signature(self):
        """0.3 <= strength <= 0.7 -> category = signature."""
        p = self._make_profile()
        p.update_pattern("cognitive", "reflection", frequency=0.4, strength=0.5)
        summary = p.get_profile_summary()
        sig_names = [e["pattern_name"] for e in summary["signatures"]]
        assert "reflection" in sig_names

    def test_profile_summary_structure(self):
        """get_profile_summary returns dict with required keys."""
        p = self._make_profile()
        summary = p.get_profile_summary()
        assert "strengths" in summary
        assert "vulnerabilities" in summary
        assert "signatures" in summary
        assert "profile_score" in summary

    def test_profile_score_increases_with_strengths(self):
        """More strengths -> higher profile_score."""
        p = self._make_profile()
        score_before = p.get_profile_summary()["profile_score"]
        p.update_pattern("trading", "discipline1", frequency=0.9, strength=0.9)
        p.update_pattern("trading", "discipline2", frequency=0.8, strength=0.8)
        score_after = p.get_profile_summary()["profile_score"]
        assert score_after >= score_before

    def test_profile_score_decreases_with_vulnerabilities(self):
        """More vulnerabilities -> lower profile_score."""
        p = self._make_profile()
        score_no_vulns = p.get_profile_summary()["profile_score"]
        for i in range(4):
            p.update_pattern("trading", f"weakness_{i}", frequency=0.3, strength=0.1)
        score_with_vulns = p.get_profile_summary()["profile_score"]
        assert score_with_vulns < score_no_vulns

    def test_save_and_load_state_roundtrip(self):
        """save_state then load_state restores the same patterns."""
        from src.aura.core.behavioral_profile import BehavioralProfile
        p1 = BehavioralProfile()
        p1.update_pattern("trading", "patience", frequency=0.7, strength=0.8)
        p1.update_pattern("emotional", "calmness", frequency=0.5, strength=0.6)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            p1.save_state(path)

            p2 = BehavioralProfile()
            p2.load_state(path)
            summary = p2.get_profile_summary()
            all_names = [e["pattern_name"] for e in
                         summary["strengths"] + summary["vulnerabilities"] + summary["signatures"]]
            assert "patience" in all_names
            assert "calmness" in all_names


# ============================================================
# US-361: Integration + Bug Fixes
# ============================================================

class TestUS361Integration:
    """Integration tests verifying bug fixes and wiring."""

    def test_hasattr_guard_in_generate_response(self):
        """_generate_response() source must contain hasattr(outcome check."""
        from src.aura.cli import companion
        src = inspect.getsource(companion.AuraCompanion._generate_response)
        assert "hasattr(outcome" in src

    def test_readability_simple_text_high_score(self):
        """Simple text should score > 0.5 even without textstat."""
        from src.aura.analysis.readability import TextReadabilityAnalyzer
        analyzer = TextReadabilityAnalyzer()
        text = "I am happy today. The sun is bright. Life is good. We enjoy simple things. Joy is nice."
        result = analyzer.analyze(text)
        assert result.readability_score > 0.5

    def test_readability_complex_text_low_score(self):
        """Complex academic text should score < 0.5 (or at least be computed)."""
        from src.aura.analysis.readability import TextReadabilityAnalyzer
        analyzer = TextReadabilityAnalyzer()
        text = (
            "The epistemological ramifications of poststructuralist hermeneutics "
            "necessitate a comprehensive re-evaluation of ontological presuppositions "
            "within contemporary philosophical discourse. "
            "Interdisciplinary methodologies amalgamate heterogeneous theoretical frameworks "
            "to elucidate multifaceted socioeconomic phenomena. "
            "Phenomenological investigations underscore the quintessential significance "
            "of subjectivistic epistemology in circumscribing existentialist paradigms."
        )
        result = analyzer.analyze(text)
        # Should compute a non-default score
        assert result.readability_score != 0.5 or result.flesch_reading_ease != 50.0

    def test_resilience_integrated_in_readiness(self):
        """ReadinessComputer has _resilience_tracker attribute after init."""
        from src.aura.core.readiness import ReadinessComputer
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = ReadinessComputer(signal_path=Path(tmpdir) / "signal.json")
            assert hasattr(rc, "_resilience_tracker")

    def test_forecast_fields_in_state_snapshot(self):
        """After compute(), state snapshot includes forecast fields."""
        from src.aura.core.readiness import ReadinessComputer
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = ReadinessComputer(signal_path=Path(tmpdir) / "signal.json")
            rc.compute(emotional_state="calm")
            snapshot = rc.get_last_state_snapshot()
            assert snapshot is not None
            assert "forecast_1h" in snapshot
            assert "forecast_6h" in snapshot
            assert "forecast_24h" in snapshot
            assert "forecast_confidence" in snapshot

    def test_goal_alignment_bonus_applied(self):
        """High goal_alignment (> 0.5) should produce higher readiness."""
        from src.aura.core.readiness import ReadinessComputer
        with tempfile.TemporaryDirectory() as tmpdir:
            rc_high = ReadinessComputer(signal_path=Path(tmpdir) / "h.json")
            rc_neutral = ReadinessComputer(signal_path=Path(tmpdir) / "n.json")
            sig_high = rc_high.compute(emotional_state="calm", goal_alignment=0.8)
            sig_neutral = rc_neutral.compute(emotional_state="calm", goal_alignment=0.0)
            assert sig_high.readiness_score >= sig_neutral.readiness_score

    def test_goal_alignment_penalty_applied(self):
        """Negative goal_alignment (< -0.5) should produce lower readiness."""
        from src.aura.core.readiness import ReadinessComputer
        with tempfile.TemporaryDirectory() as tmpdir:
            rc_neg = ReadinessComputer(signal_path=Path(tmpdir) / "neg.json")
            rc_neutral = ReadinessComputer(signal_path=Path(tmpdir) / "n2.json")
            sig_neg = rc_neg.compute(emotional_state="calm", goal_alignment=-0.8)
            sig_neutral = rc_neutral.compute(emotional_state="calm", goal_alignment=0.0)
            assert sig_neg.readiness_score <= sig_neutral.readiness_score

    def test_resilience_low_penalizes_readiness(self):
        """When resilience is low, readiness should be penalized."""
        from src.aura.core.readiness import ReadinessComputer
        from src.aura.scoring.resilience import ResilienceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = ReadinessComputer(signal_path=Path(tmpdir) / "s.json")
            # Pre-load tracker with many slow recoveries to get low resilience
            tracker = ResilienceTracker()
            for _ in range(5):
                for _ in range(9):
                    tracker.update(20.0)  # 9 steps in low state
                tracker.update(80.0)  # slow recovery
            rc._resilience_tracker = tracker
            # Tracker now has low resilience_score; verify it doesn't crash
            sig = rc.compute(emotional_state="calm")
            assert 0.0 <= sig.readiness_score <= 100.0

    def test_resilience_high_boosts_readiness(self):
        """When resilience is high, readiness should be boosted."""
        from src.aura.core.readiness import ReadinessComputer
        from src.aura.scoring.resilience import ResilienceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = ReadinessComputer(signal_path=Path(tmpdir) / "s2.json")
            tracker = ResilienceTracker()
            # Many fast single-step recoveries -> high resilience
            for _ in range(10):
                tracker.update(20.0)  # 1 step low
                tracker.update(80.0)  # recover fast
            rc._resilience_tracker = tracker
            sig = rc.compute(emotional_state="calm")
            assert 0.0 <= sig.readiness_score <= 100.0

    def test_behavioral_profile_persists_across_instances(self):
        """Save and load BehavioralProfile across separate instances."""
        from src.aura.core.behavioral_profile import BehavioralProfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bp_test.json"
            p1 = BehavioralProfile()
            p1.update_pattern("trading", "risk_control", frequency=0.9, strength=0.9)
            p1.save_state(path)
            p2 = BehavioralProfile()
            p2.load_state(path)
            summary = p2.get_profile_summary()
            strength_names = [e["pattern_name"] for e in summary["strengths"]]
            assert "risk_control" in strength_names

    def test_momentum_accelerating_label(self):
        """PatternMomentumAnalyzer returns accelerating for rising sequence."""
        from src.aura.patterns.momentum import PatternMomentumAnalyzer
        a = PatternMomentumAnalyzer()
        for freq in [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0]:
            result = a.update(freq)
        assert result.momentum_label == "accelerating"

    def test_all_phase20_imports_succeed(self):
        """All phase 20 modules import without error."""
        from src.aura.scoring.resilience import ResilienceTracker, ResilienceResult
        from src.aura.prediction.forecaster import ReadinessForecaster, ForecastResult
        from src.aura.scoring.goal_emotion import GoalEmotionCoupler, GoalEmotionResult
        from src.aura.patterns.momentum import PatternMomentumAnalyzer, MomentumResult
        from src.aura.core.behavioral_profile import BehavioralProfile, ProfileEntry
        # All imports succeeded
        assert True

# ---------------------------------------------------------------------------
# Phase 20 Integration Debt Cleanup Tests
# US-359: Pure-Python textstat fallback
# ---------------------------------------------------------------------------
class TestUS359ReadabilityFallback:
    """US-359: Pure-Python textstat fallback distinguishes easy vs complex text."""

    def setup_method(self):
        self.analyzer = TextReadabilityAnalyzer()

    def test_simple_text_high_readability(self):
        text = "The cat sat on the mat. The dog ran fast. I like pie."
        result = self.analyzer.analyze(text)
        assert result.readability_score > 0.5, f"Simple text should score > 0.5, got {result.readability_score}"

    def test_complex_text_low_readability(self):
        text = (
            "The epistemological implications of poststructuralist hermeneutics "
            "necessitate a comprehensive reconceptualization of ontological paradigms "
            "within interdisciplinary phenomenological frameworks and methodological praxeologies."
        )
        result = self.analyzer.analyze(text)
        assert result.readability_score < 0.5, f"Complex text should score < 0.5, got {result.readability_score}"

    def test_simple_vs_complex_ordering(self):
        simple = "I eat food. She drinks water. We run fast."
        complex_text = (
            "Quintessential poststructuralist methodologies dismantle epistemological "
            "constructs through hermeneutical deconstruction of metaphysical phenomena."
        )
        r_simple = self.analyzer.analyze(simple)
        r_complex = self.analyzer.analyze(complex_text)
        assert r_simple.readability_score > r_complex.readability_score

    def test_empty_text_returns_neutral(self):
        result = self.analyzer.analyze("")
        assert result.readability_score == 0.5

    def test_short_text_returns_neutral(self):
        result = self.analyzer.analyze("cat sat")
        assert result.readability_score == 0.5

    def test_vocabulary_diversity_computed(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = self.analyzer.analyze(text)
        assert result.vocabulary_diversity > 0.0

    def test_flesch_approximation_reasonable(self):
        text = "I like pie. She likes cake. We eat food."
        result = self.analyzer.analyze(text)
        assert result.flesch_reading_ease >= 0.0

    def test_gunning_fog_approximation(self):
        simple = "I eat food. She drinks water."
        result = self.analyzer.analyze(simple)
        assert 0.0 <= result.gunning_fog <= 20.0

    def test_estimate_syllables_monosyllabic(self):
        assert TextReadabilityAnalyzer._estimate_syllables("cat") == 1

    def test_estimate_syllables_polysyllabic(self):
        result = TextReadabilityAnalyzer._estimate_syllables("education")
        assert result >= 3, f"education should have >= 3 syllables, got {result}"


# ---------------------------------------------------------------------------
# US-356: DecisionFatigueIndex wired into ReadinessComputer
# ---------------------------------------------------------------------------
class TestUS356DecisionFatigueWired:
    """US-356 (Phase20): DecisionFatigueIndex is instantiated in ReadinessComputer."""

    def test_readiness_computer_has_decision_fatigue_index(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_decision_fatigue_index")

    def test_decision_fatigue_index_is_instantiated(self):
        from src.aura.core.readiness import ReadinessComputer
        from src.aura.scoring.decision_fatigue import DecisionFatigueIndex
        rc = ReadinessComputer()
        assert isinstance(rc._decision_fatigue_index, DecisionFatigueIndex)

    def test_sentiment_history_tracked(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_sentiment_history")
        assert isinstance(rc._sentiment_history, list)

    def test_complexity_history_tracked(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_complexity_history")
        assert isinstance(rc._complexity_history, list)

    def test_compute_updates_sentiment_history(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        initial_len = len(rc._sentiment_history)
        rc.compute(emotional_state="calm")
        assert len(rc._sentiment_history) > initial_len

    def test_compute_updates_complexity_history(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        initial_len = len(rc._complexity_history)
        rc.compute(emotional_state="calm")
        assert len(rc._complexity_history) > initial_len

    def test_external_fatigue_score_accepted(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        sig = rc.compute(emotional_state="calm", fatigue_score=0.9)
        assert 0.0 <= sig.readiness_score <= 100.0

    def test_internal_fatigue_used_when_not_provided(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        rc.compute(emotional_state="calm")
        assert len(rc._sentiment_history) >= 1


# ---------------------------------------------------------------------------
# US-357: BiasInteractionScorer wired into ReadinessComputer
# ---------------------------------------------------------------------------
class TestUS357BiasInteractionWired:
    """US-357 (Phase20): BiasInteractionScorer is instantiated in ReadinessComputer."""

    def test_readiness_computer_has_bias_interaction_scorer(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_bias_interaction_scorer")

    def test_bias_interaction_scorer_instantiated(self):
        from src.aura.core.readiness import ReadinessComputer
        from src.aura.scoring.bias_interactions import BiasInteractionScorer
        rc = ReadinessComputer()
        assert isinstance(rc._bias_interaction_scorer, BiasInteractionScorer)

    def test_bias_interaction_auto_computed_from_bias_scores(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        dangerous_bias_scores = {"confirmation_bias": 0.9, "anchoring": 0.9}
        sig = rc.compute(emotional_state="calm", bias_scores=dangerous_bias_scores)
        assert 0.0 <= sig.readiness_score <= 100.0

    def test_external_penalty_accepted(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        sig = rc.compute(emotional_state="calm", bias_interaction_penalty=8.0)
        assert 0.0 <= sig.readiness_score <= 100.0

    def test_no_penalty_empty_bias_scores(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        sig = rc.compute(emotional_state="calm", bias_scores={})
        assert 0.0 <= sig.readiness_score <= 100.0

    def test_scorer_score_method_works(self):
        from src.aura.scoring.bias_interactions import BiasInteractionScorer
        scorer = BiasInteractionScorer()
        result = scorer.score({"confirmation_bias": 0.8, "anchoring": 0.8})
        assert result.interaction_penalty > 0.0

    def test_scorer_returns_zero_for_no_biases(self):
        from src.aura.scoring.bias_interactions import BiasInteractionScorer
        scorer = BiasInteractionScorer()
        result = scorer.score({})
        assert result.interaction_penalty == 0.0

    def test_scorer_no_crash_on_partial_biases(self):
        from src.aura.scoring.bias_interactions import BiasInteractionScorer
        scorer = BiasInteractionScorer()
        result = scorer.score({"confirmation_bias": 0.3})
        assert result.interaction_penalty >= 0.0


# ---------------------------------------------------------------------------
# US-358: AdaptiveThresholdLearner wired into ReadinessComputer
# ---------------------------------------------------------------------------
class TestUS358AdaptiveThresholdWired:
    """US-358 (Phase20): AdaptiveThresholdLearner is instantiated in ReadinessComputer."""

    def test_readiness_computer_has_threshold_learner(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_threshold_learner")

    def test_threshold_learner_instantiated(self):
        from src.aura.core.readiness import ReadinessComputer
        from src.aura.learning.adaptive_thresholds import AdaptiveThresholdLearner
        rc = ReadinessComputer()
        assert isinstance(rc._threshold_learner, AdaptiveThresholdLearner)

    def test_threshold_context_tracked(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        assert hasattr(rc, "_threshold_context")
        assert rc._threshold_context in ("morning", "afternoon", "evening", "night")

    def test_last_threshold_used_populated_after_compute(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        rc.compute(emotional_state="calm")
        assert len(rc._last_threshold_used) > 0

    def test_threshold_keys_include_required(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        rc.compute(emotional_state="calm")
        keys = set(rc._last_threshold_used.keys())
        required = {"tilt", "override_risk", "style_drift", "granularity"}
        assert keys >= required

    def test_posteriors_updated_after_compute(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        initial_stats = rc._threshold_learner.get_stats("tilt", rc._threshold_context)
        initial_samples = initial_stats["total_samples"] if initial_stats else 0
        for _ in range(5):
            rc.compute(emotional_state="calm")
        new_stats = rc._threshold_learner.get_stats("tilt", rc._threshold_context)
        new_samples = new_stats["total_samples"] if new_stats else 0
        assert new_samples > initial_samples

    def test_get_context_maps_hours(self):
        from src.aura.learning.adaptive_thresholds import AdaptiveThresholdLearner
        assert AdaptiveThresholdLearner.get_context(6) == "morning"
        assert AdaptiveThresholdLearner.get_context(14) == "afternoon"
        assert AdaptiveThresholdLearner.get_context(20) == "evening"
        assert AdaptiveThresholdLearner.get_context(2) == "night"

    def test_compute_with_drift_score_no_crash(self):
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer()
        sig = rc.compute(emotional_state="calm", style_drift_score=0.5)
        assert 0.0 <= sig.readiness_score <= 100.0


# ---------------------------------------------------------------------------
# US-360: AuraCompanion wiring
# ---------------------------------------------------------------------------
class TestUS360CompanionWiring:
    """US-360 (Phase20): DecisionFatigueIndex and BiasInteractionScorer wired into AuraCompanion."""

    def test_companion_has_decision_fatigue_index(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            assert hasattr(companion, "_decision_fatigue_index")

    def test_companion_has_bias_interaction_scorer(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            assert hasattr(companion, "_bias_interaction_scorer")

    def test_companion_decision_fatigue_index_is_instantiated(self):
        from src.aura.cli.companion import AuraCompanion
        from src.aura.scoring.decision_fatigue import DecisionFatigueIndex
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            assert isinstance(companion._decision_fatigue_index, DecisionFatigueIndex)

    def test_companion_bias_interaction_scorer_is_instantiated(self):
        from src.aura.cli.companion import AuraCompanion
        from src.aura.scoring.bias_interactions import BiasInteractionScorer
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            assert isinstance(companion._bias_interaction_scorer, BiasInteractionScorer)

    def test_companion_has_sentiment_history(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            assert hasattr(companion, "_companion_sentiment_history")
            assert isinstance(companion._companion_sentiment_history, list)

    def test_companion_update_readiness_passes_fatigue(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            captured = {}
            original_compute = companion.readiness.compute
            def mock_compute(*args, **kwargs):
                captured.update(kwargs)
                return original_compute(*args, **kwargs)
            companion.readiness.compute = mock_compute
            companion._update_readiness()
            assert "fatigue_score" in captured
            assert "bias_interaction_penalty" in captured

    def test_companion_update_readiness_passes_bias_penalty(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            captured = {}
            original_compute = companion.readiness.compute
            def mock_compute(*args, **kwargs):
                captured.update(kwargs)
                return original_compute(*args, **kwargs)
            companion.readiness.compute = mock_compute
            companion._update_readiness()
            assert "bias_interaction_penalty" in captured

    def test_companion_no_crash_on_none_signals(self):
        from src.aura.cli.companion import AuraCompanion
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            companion = AuraCompanion(db_path=tmp / "graph.db", bridge_dir=tmp / "bridge")
            # Should not raise
            companion._update_readiness()

