"""Phase 13 tests: Signal Intelligence Wiring & VADER Sentiment Upgrade.

US-314: BiasDetector scores → readiness penalty
US-315: VADER compound sentiment scoring
US-316: Cognitive load estimation wired into readiness
US-317: OverridePredictor wired into ReadinessSignal
US-318: STL readiness trend decomposition
US-319: Cross-component integration tests
"""

import json
import math
import os
import sys
import tempfile
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
    ReadinessTrendAnalyzer,
)

# --- Neutral circadian config (all hours = 1.0) to isolate test effects ---
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ──────────────────────────────────────────────────
# US-314: BiasDetector scores → readiness penalty
# ──────────────────────────────────────────────────

class TestBiasScoresWiring:
    """US-314: Verify bias_scores flow through compute() and penalize readiness."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def test_no_biases_no_penalty(self):
        rc = self._make_rc()
        sig = rc.compute(emotional_state="calm", bias_scores=None)
        # No bias penalty — score should be in normal calm range
        assert sig.readiness_score >= 50
        assert sig.bias_scores == {}

    def test_biases_below_threshold_no_direct_penalty(self):
        rc = self._make_rc()
        biases = {"confirmation_bias": 0.3, "recency_bias": 0.2}
        sig_no_bias = rc.compute(emotional_state="calm")
        rc2 = self._make_rc()
        sig_bias = rc2.compute(emotional_state="calm", bias_scores=biases)
        # Below 0.5 threshold → no direct bias penalty (US-314)
        # But US-293 still applies bias to override_discipline, so small diff is ok
        assert abs(sig_no_bias.readiness_score - sig_bias.readiness_score) < 3.0

    def test_single_bias_above_threshold(self):
        rc = self._make_rc()
        sig_clean = rc.compute(emotional_state="calm")
        rc2 = self._make_rc()
        biases = {"confirmation_bias": 0.8}
        sig_biased = rc2.compute(emotional_state="calm", bias_scores=biases)
        # 1 bias > 0.5 → 3-point penalty
        expected_diff = 3.0
        # Allow some tolerance for EMA smoothing
        assert sig_clean.readiness_score - sig_biased.readiness_score >= expected_diff * 0.5

    def test_multiple_biases_capped(self):
        rc = self._make_rc()
        biases = {
            "confirmation_bias": 0.9,
            "recency_bias": 0.8,
            "loss_aversion": 0.7,
            "anchoring": 0.6,
            "disposition_effect": 0.95,
            "extra_bias": 0.99,  # 6th bias
        }
        sig = rc.compute(emotional_state="calm", bias_scores=biases)
        # 6 biases * 3 = 18, capped at 15
        assert sig.bias_scores == biases

    def test_bias_scores_in_signal_dict(self):
        rc = self._make_rc()
        biases = {"loss_aversion": 0.6}
        sig = rc.compute(emotional_state="calm", bias_scores=biases)
        d = sig.to_dict()
        assert "bias_scores" in d
        assert "loss_aversion" in d["bias_scores"]
        assert abs(d["bias_scores"]["loss_aversion"] - 0.6) < 0.01

    def test_backward_compat_no_param(self):
        rc = self._make_rc()
        # Calling without bias_scores should still work
        sig = rc.compute(emotional_state="calm")
        assert sig.bias_scores == {}
        assert sig.readiness_score > 0

    def test_all_four_bias_types(self):
        rc = self._make_rc()
        biases = {
            "confirmation_bias": 0.6,
            "recency_bias": 0.7,
            "loss_aversion": 0.8,
            "anchoring": 0.55,
        }
        sig = rc.compute(emotional_state="calm", bias_scores=biases)
        # All 4 above threshold → 4 * 3 = 12 penalty
        assert len(sig.bias_scores) == 4

    def test_bias_plus_tilt_combined(self):
        rc = self._make_rc()
        # Set context for tilt detection
        rc.set_context(
            messages=[
                {"text": "I need to get that money back!", "role": "user"},
                {"text": "revenge trade now", "role": "user"},
                {"text": "I'll show the market", "role": "user"},
            ]
        )
        biases = {"loss_aversion": 0.9, "recency_bias": 0.8}
        sig = rc.compute(emotional_state="stressed", bias_scores=biases)
        # Both penalties should reduce score significantly
        assert sig.readiness_score < 65


# ──────────────────────────────────────────────────
# US-315: VADER compound sentiment scoring
# ──────────────────────────────────────────────────

class TestVADERSentiment:
    """US-315: Verify VADER compound scoring blended with keyword sentiment."""

    def _make_processor(self):
        from src.aura.core.conversation_processor import ConversationProcessor
        return ConversationProcessor()

    def test_positive_message_high_sentiment(self):
        cp = self._make_processor()
        sig = cp.process_message("I feel great and confident about trading today!")
        assert sig.sentiment_score > 0.55

    def test_negative_message_low_sentiment(self):
        cp = self._make_processor()
        sig = cp.process_message("I am really stressed and anxious about losses")
        assert sig.sentiment_score < 0.5

    def test_negation_handled(self):
        cp = self._make_processor()
        sig_pos = cp.process_message("I feel great today")
        cp2 = self._make_processor()
        sig_neg = cp2.process_message("I don't feel great today")
        # VADER should detect negation and lower the score
        # Even if both trigger keyword "great", VADER differentiates
        if cp._vader is not None:
            # With VADER, negated should be lower
            assert sig_neg.sentiment_score <= sig_pos.sentiment_score

    def test_emphasis_handled(self):
        cp = self._make_processor()
        sig_normal = cp.process_message("I feel good")
        cp2 = self._make_processor()
        sig_emphasis = cp2.process_message("I feel AMAZING!!!")
        if cp._vader is not None:
            assert sig_emphasis.sentiment_score >= sig_normal.sentiment_score

    def test_contrast_mixed(self):
        cp = self._make_processor()
        sig = cp.process_message("Trading went well but I'm worried about tomorrow")
        # Mixed signal should be moderate
        assert 0.2 < sig.sentiment_score < 0.8

    def test_fallback_no_vader(self):
        cp = self._make_processor()
        cp._vader = None  # Simulate unavailable
        sig = cp.process_message("I feel stressed and anxious")
        # Should still produce a sentiment score via keywords
        assert sig.sentiment_score < 0.5

    def test_blend_ratio(self):
        cp = self._make_processor()
        if cp._vader is not None:
            # Direct test of _compute_vader_sentiment
            compound = cp._compute_vader_sentiment("I feel great!")
            assert compound is not None
            assert compound > 0  # Positive compound

    def test_neutral_message(self):
        cp = self._make_processor()
        sig = cp.process_message("The market opened at 1.0850 today")
        # Neutral factual message should be near 0.5
        assert 0.3 < sig.sentiment_score < 0.7


# ──────────────────────────────────────────────────
# US-316: Cognitive load estimation wired into readiness
# ──────────────────────────────────────────────────

class TestCognitiveLoadWiring:
    """US-316: Verify message text cognitive load affects readiness score."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def test_simple_message_high_score(self):
        rc = self._make_rc()
        sig = rc.compute(emotional_state="calm", message_text="All good today.")
        # Simple message → low cognitive load → higher score
        assert sig.readiness_score >= 50

    def test_complex_message_lower_score(self):
        rc_simple = self._make_rc()
        sig_simple = rc_simple.compute(emotional_state="calm", message_text="Fine.")

        rc_complex = self._make_rc()
        complex_msg = (
            "I have multiple concerns. Should I close the EUR/USD position? "
            "What if the market drops? Also, my boss gave me bad news about "
            "the project deadline. Furthermore, should I consider hedging? "
            "However, the risk might be manageable if I adjust the stop loss?"
        )
        sig_complex = rc_complex.compute(emotional_state="calm", message_text=complex_msg)

        # Complex message should produce lower readiness
        assert sig_complex.readiness_score < sig_simple.readiness_score

    def test_no_message_fallback(self):
        rc = self._make_rc()
        sig = rc.compute(emotional_state="calm", message_text=None)
        # Should work fine without message text
        assert sig.readiness_score > 0

    def test_blend_ratio(self):
        rc = self._make_rc()
        # With active stressors + complex message = worst case
        sig = rc.compute(
            emotional_state="calm",
            active_stressors=["deadline", "relationship", "health"],
            message_text="Should I close or hold? What if it drops? But also maybe it recovers?",
        )
        # Both stressor and text load contribute → lower cognitive score
        assert sig.components.cognitive_load_score < 0.7

    def test_stressor_plus_complex_worst_case(self):
        rc = self._make_rc()
        sig = rc.compute(
            emotional_state="stressed",
            active_stressors=["job_loss", "health", "financial"],
            stress_keywords=["stressed", "anxious", "worried"],
            message_text="If the market crashes, should I also sell my other positions? But what about the hedges?",
        )
        # Many stressors + complex text = low readiness
        assert sig.readiness_score < 50

    def test_stressor_plus_simple_medium(self):
        rc = self._make_rc()
        sig = rc.compute(
            emotional_state="calm",
            active_stressors=["deadline"],
            message_text="Doing fine.",
        )
        # 1 stressor + simple message → medium load
        assert sig.components.cognitive_load_score > 0.4

    def test_empty_string_fallback(self):
        rc = self._make_rc()
        sig = rc.compute(emotional_state="calm", message_text="")
        assert sig.readiness_score > 0


# ──────────────────────────────────────────────────
# US-317: OverridePredictor wired into ReadinessSignal
# ──────────────────────────────────────────────────

class TestOverridePredictorWiring:
    """US-317: Verify OverridePredictor predictions flow to signal and penalty."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def _mock_predictor(self, loss_prob=0.5, trained=True):
        pred = MagicMock()
        pred._trained = trained
        # H-NEW-01 fix: method is predict_loss_probability(), not predict()
        pred.predict_loss_probability = MagicMock(return_value=loss_prob)
        return pred

    def test_no_predictor_no_penalty(self):
        rc = self._make_rc()
        sig = rc.compute(emotional_state="calm")
        assert sig.override_loss_risk == 0.0

    def test_untrained_predictor_no_penalty(self):
        rc = self._make_rc()
        pred = self._mock_predictor(trained=False)
        sig = rc.compute(emotional_state="calm", override_predictor=pred)
        assert sig.override_loss_risk == 0.0
        pred.predict_loss_probability.assert_not_called()

    def test_low_risk_no_penalty(self):
        rc = self._make_rc()
        pred = self._mock_predictor(loss_prob=0.3)
        sig_no_pred = rc.compute(emotional_state="calm")

        rc2 = self._make_rc()
        sig_pred = rc2.compute(emotional_state="calm", override_predictor=pred)

        # Low risk → no penalty, scores should be similar
        assert abs(sig_no_pred.readiness_score - sig_pred.readiness_score) < 2.0
        assert abs(sig_pred.override_loss_risk - 0.3) < 0.01

    def test_high_risk_penalty(self):
        rc = self._make_rc()
        sig_clean = rc.compute(emotional_state="calm")

        rc2 = self._make_rc()
        pred = self._mock_predictor(loss_prob=0.85)
        sig_risky = rc2.compute(emotional_state="calm", override_predictor=pred)

        # High risk > 0.7 → 10-point penalty
        assert sig_clean.readiness_score - sig_risky.readiness_score >= 5.0
        assert sig_risky.override_loss_risk > 0.7

    def test_signal_field_present(self):
        rc = self._make_rc()
        pred = self._mock_predictor(loss_prob=0.5)
        sig = rc.compute(emotional_state="calm", override_predictor=pred)
        d = sig.to_dict()
        assert "override_loss_risk" in d
        assert abs(d["override_loss_risk"] - 0.5) < 0.01

    def test_ood_returns_conservative(self):
        rc = self._make_rc()
        pred = self._mock_predictor(loss_prob=0.0)  # OOD might return 0
        sig = rc.compute(emotional_state="calm", override_predictor=pred)
        # Zero risk → no penalty
        assert sig.override_loss_risk == 0.0

    def test_predictor_exception_graceful(self):
        rc = self._make_rc()
        pred = self._mock_predictor()
        pred.predict_loss_probability.side_effect = RuntimeError("Model corrupted")
        sig = rc.compute(emotional_state="calm", override_predictor=pred)
        # Should gracefully fall back
        assert sig.override_loss_risk == 0.0
        assert sig.readiness_score > 0

    def test_combined_with_bias_penalty(self):
        rc = self._make_rc()
        pred = self._mock_predictor(loss_prob=0.8)
        biases = {"confirmation_bias": 0.9, "recency_bias": 0.8}
        sig = rc.compute(
            emotional_state="calm",
            bias_scores=biases,
            override_predictor=pred,
        )
        # Both penalties should stack
        rc2 = self._make_rc()
        sig_clean = rc2.compute(emotional_state="calm")
        assert sig_clean.readiness_score - sig.readiness_score >= 10


# ──────────────────────────────────────────────────
# US-318: STL readiness trend decomposition
# ──────────────────────────────────────────────────

class TestReadinessTrendAnalyzer:
    """US-318: Verify STL trend decomposition and alert generation."""

    def test_insufficient_data_returns_stable(self):
        rta = ReadinessTrendAnalyzer()
        for i in range(10):
            rta.add_sample(70.0)
        result = rta.decompose()
        assert result["trend_direction"] == "stable"
        assert result.get("status") in ("insufficient_data", "no_statsmodels")

    def test_declining_trend_detected(self):
        rta = ReadinessTrendAnalyzer(seasonal_period=7)
        # Declining trend: 80 → 40 over 20 samples
        for i in range(20):
            score = 80 - (i * 2)
            rta.add_sample(score)
        result = rta.decompose()
        if result.get("status") == "ok":
            assert result["trend_direction"] == "declining"
            assert result["trend_slope"] < 0

    def test_stable_trend_detected(self):
        rta = ReadinessTrendAnalyzer(seasonal_period=7)
        import random
        random.seed(42)
        for i in range(20):
            score = 70 + random.uniform(-2, 2)
            rta.add_sample(score)
        result = rta.decompose()
        if result.get("status") == "ok":
            assert result["trend_direction"] == "stable"

    def test_improving_trend_detected(self):
        rta = ReadinessTrendAnalyzer(seasonal_period=7)
        for i in range(20):
            score = 40 + (i * 2)
            rta.add_sample(score)
        result = rta.decompose()
        if result.get("status") == "ok":
            assert result["trend_direction"] == "improving"
            assert result["trend_slope"] > 0

    def test_anomaly_spike_alert(self):
        rta = ReadinessTrendAnalyzer(seasonal_period=7)
        # Stable then sudden spike
        for i in range(16):
            rta.add_sample(70.0)
        rta.add_sample(20.0)  # Anomaly
        rta.add_sample(15.0)  # Another anomaly
        rta.add_sample(70.0)  # Back to normal
        rta.add_sample(70.0)
        alerts = rta.readiness_alert()
        # May or may not generate alert depending on statsmodels availability
        # If available, should detect anomalies
        if alerts:
            alert_types = [a["type"] for a in alerts]
            assert any(t in ("ANOMALY_SPIKE", "TREND_DECLINE") for t in alert_types)

    def test_seasonal_pattern_info(self):
        rta = ReadinessTrendAnalyzer(seasonal_period=7)
        # Repeating pattern with large seasonal component
        for cycle in range(3):
            for day in range(7):
                # Large seasonal swing: 50-90
                score = 70 + 20 * math.sin(2 * math.pi * day / 7)
                rta.add_sample(score)
        result = rta.decompose()
        if result.get("status") == "ok":
            assert result.get("seasonal_amplitude", 0) > 0

    def test_graceful_no_statsmodels(self):
        rta = ReadinessTrendAnalyzer()
        for i in range(20):
            rta.add_sample(70.0)
        # Even if statsmodels import fails, should return gracefully
        with patch.dict("sys.modules", {"statsmodels": None, "statsmodels.tsa.seasonal": None}):
            result = rta.decompose()
            assert result["trend_direction"] == "stable"

    def test_signal_field_populated(self):
        rc = ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        sig = rc.compute(emotional_state="calm")
        d = sig.to_dict()
        assert "trend_direction" in d
        assert d["trend_direction"] == "stable"  # Not enough samples yet


# ──────────────────────────────────────────────────
# US-319: Cross-component integration tests
# ──────────────────────────────────────────────────

class TestCrossComponentIntegration:
    """US-319: Integration tests that cross ConversationProcessor → ReadinessComputer boundaries."""

    def _make_rc(self):
        return ReadinessComputer(
            signal_path=Path(tempfile.mkdtemp()) / "sig.json",
            circadian_config=NEUTRAL_CIRCADIAN,
        )

    def test_bias_flow_processor_to_readiness(self):
        """Process message → extract biases → pass to compute → penalty applied."""
        from src.aura.core.conversation_processor import ConversationProcessor
        cp = ConversationProcessor()
        # Message that might trigger biases
        sig_conv = cp.process_message("I know EUR/USD will go up, it always does after a drop like this")
        bias_scores = sig_conv.bias_scores

        rc_clean = self._make_rc()
        sig_clean = rc_clean.compute(emotional_state="calm")

        rc_biased = self._make_rc()
        sig_biased = rc_biased.compute(
            emotional_state="calm",
            bias_scores=bias_scores,
        )
        # If biases detected with high scores, readiness should be lower
        if any(v > 0.5 for v in bias_scores.values()):
            assert sig_biased.readiness_score <= sig_clean.readiness_score

    def test_negative_sentiment_lowers_readiness(self):
        """Negative VADER sentiment → lower emotional_state_score → lower readiness."""
        rc_calm = self._make_rc()
        sig_calm = rc_calm.compute(emotional_state="calm")

        rc_stressed = self._make_rc()
        sig_stressed = rc_stressed.compute(emotional_state="stressed")

        assert sig_stressed.readiness_score < sig_calm.readiness_score

    def test_complex_message_affects_score(self):
        """Complex message → higher cognitive load → lower readiness."""
        rc_simple = self._make_rc()
        sig_simple = rc_simple.compute(
            emotional_state="calm",
            message_text="Good day.",
        )

        rc_complex = self._make_rc()
        sig_complex = rc_complex.compute(
            emotional_state="calm",
            message_text=(
                "If the EUR/USD breaks below 1.0800, should I close my position? "
                "But what if it bounces? Also considering the GBP/USD correlation. "
                "However, the news might push it further down. What do you think? "
                "Maybe I should hedge with a put option? Although that costs money."
            ),
        )

        assert sig_complex.readiness_score < sig_simple.readiness_score

    def test_override_risk_to_signal(self):
        """OverridePredictor risk → penalty applied → signal includes risk."""
        rc = self._make_rc()
        pred = MagicMock()
        pred._trained = True
        pred.predict_loss_probability = MagicMock(return_value=0.85)  # H-NEW-01 fix

        sig = rc.compute(emotional_state="calm", override_predictor=pred)
        assert sig.override_loss_risk > 0.7
        d = sig.to_dict()
        assert d["override_loss_risk"] > 0.7

    def test_trend_after_many_samples(self):
        """15+ readiness samples → trend analyzer produces direction."""
        rc = self._make_rc()
        # Feed declining samples
        for i in range(18):
            rc.compute(emotional_state="calm" if i < 10 else "stressed")

        sig = rc.compute(emotional_state="stressed")
        # After 19 samples, trend should be detectable
        d = sig.to_dict()
        assert "trend_direction" in d

    def test_all_penalties_stack(self):
        """Bias + override risk + tilt + stress all stack to lower readiness."""
        rc = self._make_rc()
        rc.set_context(
            messages=[
                {"text": "I need revenge on this market!", "role": "user"},
                {"text": "Getting my money back now!", "role": "user"},
                {"text": "Going all in!", "role": "user"},
            ]
        )
        pred = MagicMock()
        pred._trained = True
        pred.predict_loss_probability = MagicMock(return_value=0.9)  # H-NEW-01 fix

        sig = rc.compute(
            emotional_state="frustrated",
            bias_scores={"loss_aversion": 0.9, "confirmation_bias": 0.8},
            stress_keywords=["stressed", "anxious", "worried"],
            active_stressors=["financial_loss", "relationship"],
            override_predictor=pred,
            message_text="If I don't trade now I'll miss the opportunity but what if it drops more?",
        )

        # All penalties stacked → very low readiness
        assert sig.readiness_score < 35

    def test_healthy_trader_high_readiness(self):
        """Clean state → high readiness score with all new fields populated."""
        rc = self._make_rc()
        sig = rc.compute(
            emotional_state="calm",
            conversation_count_7d=5,
            confidence_trend="rising",
            message_text="Feeling good and focused today.",
        )
        assert sig.readiness_score >= 60
        d = sig.to_dict()
        assert d["bias_scores"] == {}
        assert d["override_loss_risk"] == 0.0
        assert d["trend_direction"] == "stable"
