"""Unit tests for Aura ReadinessComputer — US-211.

Tests cover:
  1. Score bounds (always 0-100)
  2. Component weight sum (must equal 1.0)
  3. Emotional state mapping
  4. Override discipline scoring
  5. Cognitive load levels
  6. Confidence trend impact
  7. Engagement score tiers
  8. Signal serialization round-trip
"""

import json
import sys
import tempfile
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


@pytest.fixture
def tmp_signal_path(tmp_path):
    """Provide a temp path for bridge signal files."""
    return tmp_path / "bridge" / "readiness_signal.json"


# ── 1. Component weights sum to 1.0 ──────────────────────────────────────

def test_component_weights_sum():
    """Weights must sum to 1.0 — any drift silently breaks the score."""
    total = sum(_COMPONENT_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weight sum is {total}, expected 1.0"


# ── 2. Score is always in [0, 100] ───────────────────────────────────────

def test_score_bounds_default(tmp_signal_path):
    """Default (calm, no stressors) should produce a score in [60, 100]."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute()
    assert 0 <= signal.readiness_score <= 100


def test_score_bounds_worst_case(tmp_signal_path):
    """Max stress inputs should still yield score >= 0."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(
        emotional_state="overwhelmed",
        stress_keywords=["stressed", "exhausted", "angry", "losing money", "burnout", "frustrated"],
        active_stressors=["career", "health", "relationship", "money", "family"],
        recent_override_events=[{"trade_won": False}] * 10,
        conversation_count_7d=0,
        confidence_trend="falling",
    )
    assert 0 <= signal.readiness_score <= 100
    # Under severe stress the score should be low
    assert signal.readiness_score < 40


def test_score_bounds_best_case(tmp_signal_path):
    """Optimal state should produce a high score."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(
        emotional_state="calm",
        stress_keywords=[],
        active_stressors=[],
        recent_override_events=[{"trade_won": True}] * 5,
        conversation_count_7d=7,
        confidence_trend="rising",
    )
    assert signal.readiness_score >= 75


# ── 3. Emotional state mapping ───────────────────────────────────────────

@pytest.mark.parametrize("state,expected_range", [
    ("calm", (0.85, 0.95)),
    ("anxious", (0.35, 0.45)),
    ("stressed", (0.25, 0.35)),
    ("overwhelmed", (0.10, 0.20)),
    ("energized", (0.80, 0.90)),
    ("unknown_state", (0.45, 0.55)),  # Fallback default
])
def test_emotional_state_scores(tmp_signal_path, state, expected_range):
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(emotional_state=state)
    emo_score = signal.components.emotional_state_score
    lo, hi = expected_range
    assert lo <= emo_score <= hi, f"Emotional score for '{state}' = {emo_score}, expected [{lo}, {hi}]"


# ── 4. Override discipline ───────────────────────────────────────────────

def test_override_discipline_all_losses(tmp_signal_path):
    """100% losing overrides should tank override discipline."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(
        recent_override_events=[{"trade_won": False}] * 5,
    )
    assert signal.components.override_discipline_score <= 0.15
    assert signal.override_loss_rate_7d == 1.0


def test_override_discipline_no_overrides(tmp_signal_path):
    """No overrides → default discipline (0.8)."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(recent_override_events=[])
    assert signal.components.override_discipline_score == 0.8
    assert signal.override_loss_rate_7d == 0.0


# ── 5. Cognitive load levels ─────────────────────────────────────────────

def test_cognitive_load_low(tmp_signal_path):
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(active_stressors=[])
    assert signal.cognitive_load == "low"


def test_cognitive_load_medium(tmp_signal_path):
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(active_stressors=["career", "health"])
    assert signal.cognitive_load == "medium"


def test_cognitive_load_high(tmp_signal_path):
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(active_stressors=["career", "health", "relationship"])
    assert signal.cognitive_load == "high"


# ── 6. Confidence trend ──────────────────────────────────────────────────

def test_confidence_trend_impact(tmp_signal_path):
    """Rising trend should yield higher score than falling."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    rising = rc.compute(confidence_trend="rising")
    # Reset EMA state so second compute is independent (US-303 hysteresis fix)
    rc._last_smoothed_score = None
    rc._readiness_history = []
    falling = rc.compute(confidence_trend="falling")
    assert rising.readiness_score > falling.readiness_score


# ── 7. Engagement score tiers ────────────────────────────────────────────

def test_engagement_tiers(tmp_signal_path):
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    low = rc.compute(conversation_count_7d=0)
    mid = rc.compute(conversation_count_7d=3)
    high = rc.compute(conversation_count_7d=7)
    assert low.components.engagement_score < mid.components.engagement_score
    assert mid.components.engagement_score < high.components.engagement_score


# ── 8. Signal serialization round-trip ───────────────────────────────────

def test_signal_json_roundtrip(tmp_signal_path):
    """Signal → JSON → dict should preserve all fields."""
    rc = ReadinessComputer(signal_path=tmp_signal_path, circadian_config={h: 1.0 for h in range(24)})
    signal = rc.compute(
        emotional_state="anxious",
        active_stressors=["career"],
        confidence_trend="falling",
        conversation_count_7d=2,
    )
    data = json.loads(signal.to_json())
    assert "readiness_score" in data
    assert "components" in data
    assert isinstance(data["components"], dict)
    assert set(data["components"].keys()) == {
        "emotional_state", "cognitive_load", "override_discipline",
        "stress_level", "confidence_trend", "engagement",
    }
    assert data["emotional_state"] == "anxious"
    assert data["confidence_trend"] == "falling"


# ── 9. Regression: H-NEW-01 / L-01 — Override predictor method name ─────

def test_readiness_compute_override_predictor_method_name(tmp_signal_path):
    """Regression: predictor.predict_loss_probability() must be called, not .predict().

    Previously predictor.predict(ctx) raised AttributeError on every cycle, was
    silently swallowed by try/except, and left override_loss_risk always 0.0.
    This test forces the _trained=True path and confirms no AttributeError occurs.
    """
    from src.aura.prediction.override_predictor import OverridePredictor

    predictor = OverridePredictor()
    # Manually mark as trained so the compute() method enters the prediction branch
    predictor._trained = True

    rc = ReadinessComputer(
        signal_path=tmp_signal_path,
        circadian_config={h: 1.0 for h in range(24)},
    )
    rc._override_predictor = predictor

    # Should not raise AttributeError (would indicate .predict() was called instead
    # of .predict_loss_probability())
    result = rc.compute(emotional_state="calm")
    assert result is not None
    assert isinstance(result.readiness_score, float)
    # override_loss_risk is a valid float in [0.0, 1.0]
    assert 0.0 <= result.override_loss_risk <= 1.0
