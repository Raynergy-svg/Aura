"""Integration tests for critical untested Aura scenarios.

Tests:
  1. All readiness penalties stacking — verify floor at 0
  2. Cold start with empty everything — verify default compute succeeds
  3. ReadinessSignal JSON round-trip with ALL newer fields
  4. Scoring modules integration — recovery_score populated after history
  5. Readiness score monotonic with increasing stress
"""

import json
import math
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.core.readiness import (
    ReadinessComputer,
    ReadinessComponents,
    ReadinessSignal,
    _COMPONENT_WEIGHTS,
)

# Neutral circadian: no time-of-day effect
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


@pytest.fixture
def tmp_signal_path(tmp_path):
    """Provide a temp path for bridge signal files."""
    return tmp_path / "bridge" / "readiness_signal.json"


# ═══════════════════════════════════════════════════════════════════════
# 1. All readiness penalties fire simultaneously — score floors at 0
# ═══════════════════════════════════════════════════════════════════════

class TestAllPenaltiesStack:
    """Verify that when every penalty fires simultaneously the score
    remains in [0, 100] and never goes negative or NaN."""

    def test_readiness_all_penalties_stack_floor(self, tmp_signal_path):
        rc = ReadinessComputer(
            signal_path=tmp_signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        signal = rc.compute(
            emotional_state="overwhelmed",                   # 0.15
            active_stressors=[                               # 5+ → cognitive_load="high"
                "career", "health", "relationship", "money", "family",
            ],
            recent_override_events=[{"trade_won": False}] * 15,  # override_discipline → low
            confidence_trend="falling",                      # 0.4
            conversation_count_7d=0,                         # engagement → 0.2
            bias_scores={                                    # all 9 biases > 0.5
                "overconfidence": 0.9,
                "recency": 0.85,
                "anchoring": 0.8,
                "confirmation": 0.75,
                "loss_aversion": 0.7,
                "sunk_cost": 0.65,
                "herding": 0.6,
                "disposition": 0.55,
                "availability": 0.51,
            },
            stress_keywords=[
                "stressed", "exhausted", "angry", "losing money",
                "burnout", "frustrated", "anxious", "panicking",
            ],
            message_text=(
                "I can't believe I lost again. This is the fifth time this week and "
                "I'm so frustrated I can't think straight. My health is terrible, "
                "my relationship is falling apart, career is stalling, and my family "
                "keeps asking about the money I've lost. I feel overwhelmed and "
                "completely burned out. I should probably just throw more money at it "
                "to recover my losses."
            ),
        )

        # The score MUST be in [0, 100] — never negative, never NaN
        assert signal.readiness_score >= 0, (
            f"Score went negative: {signal.readiness_score}"
        )
        assert signal.readiness_score <= 100, (
            f"Score exceeded 100: {signal.readiness_score}"
        )
        assert not math.isnan(signal.readiness_score), "Score is NaN"
        assert not math.isinf(signal.readiness_score), "Score is Inf"

        # Under maximum stress, score should be very low
        assert signal.readiness_score < 30, (
            f"Expected score < 30 under max stress, got {signal.readiness_score}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Cold start — brand new user, first compute() with all defaults
# ═══════════════════════════════════════════════════════════════════════

class TestColdStart:
    """Verify that compute() succeeds with absolutely no prior state."""

    def test_readiness_cold_start_empty_everything(self, tmp_signal_path):
        rc = ReadinessComputer(
            signal_path=tmp_signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )
        # Call compute with zero arguments — all defaults
        signal = rc.compute()

        # Must not raise, and must produce a valid signal
        assert signal is not None
        assert isinstance(signal, ReadinessSignal)

        # Score should be in a reasonable "neutral" range
        assert 0 <= signal.readiness_score <= 100
        assert signal.readiness_score >= 50, (
            f"Cold start score too low: {signal.readiness_score} (expected >= 50)"
        )
        assert signal.readiness_score <= 90, (
            f"Cold start score too high: {signal.readiness_score} (expected <= 90)"
        )

        # Model version should be v1 (no trained V2 model)
        assert signal.model_version == "v1"

        # Components should have sensible defaults
        assert 0 <= signal.components.emotional_state_score <= 1
        assert 0 <= signal.components.cognitive_load_score <= 1
        assert 0 <= signal.components.override_discipline_score <= 1
        assert 0 <= signal.components.stress_level_score <= 1
        assert 0 <= signal.components.confidence_trend_score <= 1
        assert 0 <= signal.components.engagement_score <= 1


# ═══════════════════════════════════════════════════════════════════════
# 3. ReadinessSignal JSON round-trip with ALL newer fields populated
# ═══════════════════════════════════════════════════════════════════════

class TestSignalJsonRoundtripAllFields:
    """Verify that every field survives to_dict() -> json.dumps -> json.loads."""

    def test_signal_json_roundtrip_all_fields(self):
        components = ReadinessComponents(
            emotional_state_score=0.4,
            cognitive_load_score=0.3,
            override_discipline_score=0.2,
            stress_level_score=0.35,
            confidence_trend_score=0.6,
            engagement_score=0.7,
        )

        signal = ReadinessSignal(
            readiness_score=42.5,
            cognitive_load="high",
            active_stressors=["career", "health"],
            override_loss_rate_7d=0.75,
            emotional_state="stressed",
            confidence_trend="falling",
            components=components,
            timestamp="2026-03-24T12:00:00+00:00",
            conversation_count_7d=3,
            confidence_acceleration=-0.05,
            fatigue_score=0.6,
            model_version="v1",
            circadian_multiplier=0.85,
            raw_score=65.0,
            smoothed_score=63.5,
            tilt_score=0.4,
            decision_variability=0.8,
            anomaly_detected=True,
            anomaly_severity=0.7,
            bias_scores={"overconfidence": 0.8, "recency": 0.6},
            override_loss_risk=0.55,
            trend_direction="declining",
            decision_quality_score=72.0,
            recovery_score=0.75,
            regime_shift_detected=True,
            regime_shift_prob=0.82,
            reliability_score=0.65,
        )

        # Serialize
        d = signal.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        # Core fields
        assert loaded["readiness_score"] == 42.5
        assert loaded["cognitive_load"] == "high"
        assert loaded["active_stressors"] == ["career", "health"]
        assert loaded["override_loss_rate_7d"] == 0.75
        assert loaded["emotional_state"] == "stressed"
        assert loaded["confidence_trend"] == "falling"
        assert loaded["conversation_count_7d"] == 3
        assert loaded["model_version"] == "v1"

        # Component breakdown
        assert "components" in loaded
        assert isinstance(loaded["components"], dict)
        assert set(loaded["components"].keys()) == {
            "emotional_state", "cognitive_load", "override_discipline",
            "stress_level", "confidence_trend", "engagement",
        }

        # Newer fields (US-304 through US-334)
        assert loaded["recovery_score"] == 0.75
        assert loaded["regime_shift_detected"] is True
        assert loaded["regime_shift_prob"] == 0.82
        assert loaded["decision_quality_score"] == 72.0
        assert loaded["trend_direction"] == "declining"
        assert loaded["bias_scores"] == {"overconfidence": 0.8, "recency": 0.6}
        assert loaded["tilt_score"] == 0.4
        assert loaded["anomaly_detected"] is True
        assert loaded["anomaly_severity"] == 0.7
        assert loaded["raw_score"] == 65.0
        assert loaded["smoothed_score"] == 63.5
        assert loaded["override_loss_risk"] == 0.55
        assert loaded["circadian_multiplier"] == 0.85
        assert loaded["fatigue_score"] == 0.6
        assert loaded["decision_variability"] == 0.8
        assert loaded["reliability_score"] == 0.65

        # Verify confidence_acceleration round-trip (rounded to 4dp)
        assert loaded["confidence_acceleration"] == -0.05


# ═══════════════════════════════════════════════════════════════════════
# 4. Scoring modules integration — recovery_score after enough history
# ═══════════════════════════════════════════════════════════════════════

class TestScoringModulesIntegration:
    """Pump 20+ compute() calls to build up history and verify
    scoring modules produce non-default values."""

    def test_scoring_modules_integration(self, tmp_signal_path):
        rc = ReadinessComputer(
            signal_path=tmp_signal_path,
            circadian_config=NEUTRAL_CIRCADIAN,
        )

        # Pump 25 calls to build up history for recovery scoring
        # (needs >= 5 in _readiness_history for US-326 recovery scorer)
        signals = []
        for i in range(25):
            # Vary the emotional state to create some history variance
            states = ["calm", "energized", "neutral", "anxious", "calm"]
            state = states[i % len(states)]
            signal = rc.compute(
                emotional_state=state,
                conversation_count_7d=i % 7,
                active_stressors=["career"] if i % 3 == 0 else [],
            )
            signals.append(signal)

        last_signal = signals[-1]

        # After 25 calls, raw_score and smoothed_score should both be populated
        assert last_signal.raw_score is not None, "raw_score should be set"
        assert last_signal.smoothed_score is not None, "smoothed_score should be set"

        # recovery_score should have moved away from the 0.5 default
        # (with varied history, the recovery scorer should produce a real value)
        # We check across all signals after the 5th (when recovery scorer activates)
        recovery_scores = [s.recovery_score for s in signals[5:]]
        assert any(
            rs != 0.5 for rs in recovery_scores
        ), (
            f"recovery_score never moved from default 0.5 after 25 calls. "
            f"Values: {recovery_scores}"
        )

        # All signals should have valid fields
        for i, sig in enumerate(signals):
            assert 0 <= sig.readiness_score <= 100, (
                f"Signal {i}: score out of bounds = {sig.readiness_score}"
            )
            assert sig.model_version == "v1", (
                f"Signal {i}: model_version should be v1, got {sig.model_version}"
            )
            assert not math.isnan(sig.readiness_score), (
                f"Signal {i}: score is NaN"
            )


# ═══════════════════════════════════════════════════════════════════════
# 5. Readiness score monotonic with increasing stress
# ═══════════════════════════════════════════════════════════════════════

class TestReadinessMonotonicWithStress:
    """As stress increases (more stressors, worse emotional state),
    readiness should decrease — scores form a non-increasing sequence."""

    def test_readiness_score_monotonic_with_stress(self, tmp_signal_path):
        stressor_pool = ["career", "health", "relationship", "money", "family"]
        scores = []

        for num_stressors in range(6):  # 0, 1, 2, 3, 4, 5 stressors
            # Fresh computer each time to avoid EMA/hysteresis cross-contamination
            rc = ReadinessComputer(
                signal_path=tmp_signal_path,
                circadian_config=NEUTRAL_CIRCADIAN,
            )
            signal = rc.compute(
                emotional_state="neutral",
                active_stressors=stressor_pool[:num_stressors],
                confidence_trend="stable",
                conversation_count_7d=3,
            )
            scores.append(signal.readiness_score)

        # Verify non-increasing: each score <= the previous one
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 0.01, (
                f"Score increased with more stress: "
                f"stressors={i-1} -> score={scores[i-1]:.2f}, "
                f"stressors={i} -> score={scores[i]:.2f}. "
                f"Full sequence: {[f'{s:.2f}' for s in scores]}"
            )

        # Sanity: 0 stressors should score higher than 5 stressors
        assert scores[0] > scores[5], (
            f"0 stressors ({scores[0]:.2f}) should beat 5 stressors ({scores[5]:.2f})"
        )
