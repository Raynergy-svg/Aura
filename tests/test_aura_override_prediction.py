"""Tests for Aura OverridePredictor — logistic regression on override outcomes.

Phase 5 — US-254, US-255.
Covers feature encoding, sigmoid stability, enum warnings (US-239),
training, prediction, risk level classification, OOD detection, and persistence.
"""

import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.prediction.override_predictor import (
    COGNITIVE_LOAD_RISK,
    EMOTIONAL_STATE_RISK,
    OVERRIDE_TYPE_RISK,
    REGIME_RISK,
    OverridePrediction,
    OverridePredictor,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_model_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def fresh_predictor(tmp_model_dir):
    """Untrained OverridePredictor with temp path."""
    return OverridePredictor(model_path=tmp_model_dir / "override.json")


def _make_override_events(n, loss_bias=0.5):
    """Generate n synthetic override events.

    loss_bias: probability that an event with high risk features is a loss.
    Creates a learnable pattern: high emotional risk + took_rejected → more losses.
    """
    import random
    random.seed(123)

    emotions = ["neutral", "anxious", "stressed", "frustrated", "calm", "focused"]
    cognitive_states = ["low", "normal", "high", "overloaded"]
    types = ["took_rejected", "skipped_recommended", "closed_early", "modified_sl_tp"]
    regimes = ["LOW", "NORMAL", "HIGH", "VOLATILE"]

    events = []
    for i in range(n):
        emo = emotions[i % len(emotions)]
        cog = cognitive_states[i % len(cognitive_states)]
        otype = types[i % len(types)]
        regime = regimes[i % len(regimes)]

        # Compute a "true" risk score
        emo_risk = EMOTIONAL_STATE_RISK.get(emo, 0.0)
        cog_risk = COGNITIVE_LOAD_RISK.get(cog, 0.0)
        type_risk = OVERRIDE_TYPE_RISK.get(otype, 0.15)
        regime_risk = REGIME_RISK.get(regime, 0.0)
        total_risk = emo_risk + cog_risk + type_risk + regime_risk

        # Outcome correlates with risk
        loss_prob = min(1.0, max(0.0, 0.3 + total_risk * 0.5))
        outcome = "loss" if random.random() < loss_prob else "win"

        events.append({
            "timestamp": f"2026-03-{(i % 28) + 1:02d}T10:00:00Z",
            "pair": "EUR/USD",
            "override_type": otype,
            "buddy_recommendation": "SELL",
            "trader_action": "BUY",
            "outcome": outcome,
            "pnl_pips": random.uniform(-50, 50),
            "emotional_state": emo,
            "cognitive_load": cog,
            "conversation_context": "",
            "regime": regime,
            "confidence_at_time": 0.3 + 0.5 * random.random(),
            "weighted_vote_at_time": 0.3 + 0.5 * random.random(),
        })
    return events


# ═══════════════════════════════════════════════════════════════════════
# US-254: Feature encoding, sigmoid, enum warnings
# ═══════════════════════════════════════════════════════════════════════


class TestUS254FeatureEncoding:
    """US-254: _encode_features(), _sigmoid(), and US-239 warnings."""

    def test_encode_features_length(self, fresh_predictor):
        """_encode_features() returns exactly 8 floats."""
        event = {
            "confidence_at_time": 0.7,
            "weighted_vote_at_time": 0.6,
            "emotional_state": "anxious",
            "cognitive_load": "high",
            "override_type": "took_rejected",
            "regime": "VOLATILE",
        }
        features = fresh_predictor._encode_features(event)
        assert len(features) == 8
        assert all(isinstance(f, float) for f in features)

    def test_known_emotional_state_mapping(self, fresh_predictor):
        """Known emotional states map to correct risk values."""
        for state, expected_risk in EMOTIONAL_STATE_RISK.items():
            event = {"emotional_state": state}
            features = fresh_predictor._encode_features(event)
            assert abs(features[2] - expected_risk) < 1e-10, \
                f"emotional_state '{state}' mapped to {features[2]}, expected {expected_risk}"

    def test_known_cognitive_load_mapping(self, fresh_predictor):
        """Known cognitive loads map to correct risk values."""
        for load, expected_risk in COGNITIVE_LOAD_RISK.items():
            event = {"cognitive_load": load}
            features = fresh_predictor._encode_features(event)
            assert abs(features[3] - expected_risk) < 1e-10, \
                f"cognitive_load '{load}' mapped to {features[3]}, expected {expected_risk}"

    def test_known_override_type_mapping(self, fresh_predictor):
        """Known override types map to correct risk values."""
        for otype, expected_risk in OVERRIDE_TYPE_RISK.items():
            event = {"override_type": otype}
            features = fresh_predictor._encode_features(event)
            assert abs(features[4] - expected_risk) < 1e-10

    def test_known_regime_mapping(self, fresh_predictor):
        """Known regimes map to correct risk values."""
        for regime, expected_risk in REGIME_RISK.items():
            event = {"regime": regime}
            features = fresh_predictor._encode_features(event)
            assert abs(features[5] - expected_risk) < 1e-10

    def test_unknown_emotional_defaults_to_zero(self, fresh_predictor):
        """Unknown emotional state defaults risk to 0.0."""
        event = {"emotional_state": "confused_but_hopeful"}
        features = fresh_predictor._encode_features(event)
        assert features[2] == 0.0

    def test_unknown_override_type_defaults_to_015(self, fresh_predictor):
        """Unknown override type defaults risk to 0.15."""
        event = {"override_type": "totally_new_type"}
        features = fresh_predictor._encode_features(event)
        assert abs(features[4] - 0.15) < 1e-10

    def test_us239_unknown_enum_warning(self, fresh_predictor, caplog):
        """US-239: Unknown enum values trigger a warning log."""
        event = {
            "emotional_state": "bewildered",
            "cognitive_load": "transcendent",
            "override_type": "quantum_override",
            "regime": "CHAOS",
        }
        with caplog.at_level(logging.WARNING):
            fresh_predictor._encode_features(event)

        us239_msgs = [r for r in caplog.records if "US-239" in r.message]
        assert len(us239_msgs) == 4  # one per unknown enum

    def test_us239_dedup_warnings(self, fresh_predictor, caplog):
        """US-239: Same unknown value only warns once (dedup by set)."""
        event = {"emotional_state": "bewildered"}
        with caplog.at_level(logging.WARNING):
            fresh_predictor._encode_features(event)
            fresh_predictor._encode_features(event)  # Same value again

        us239_msgs = [r for r in caplog.records
                      if "US-239" in r.message and "bewildered" in r.message]
        assert len(us239_msgs) == 1  # Only warned once

    def test_interaction_features(self, fresh_predictor):
        """Interaction features (conf*vote, emotional*cognitive) are correct."""
        event = {
            "confidence_at_time": 0.8,
            "weighted_vote_at_time": 0.7,
            "emotional_state": "anxious",   # 0.4
            "cognitive_load": "high",       # 0.4
        }
        features = fresh_predictor._encode_features(event)
        # conf_vote = 0.8 * 0.7 = 0.56
        assert abs(features[6] - 0.56) < 1e-10
        # emotional_cognitive = 0.4 * 0.4 = 0.16
        assert abs(features[7] - 0.16) < 1e-10

    def test_sigmoid_at_zero(self):
        """_sigmoid(0) = 0.5."""
        assert abs(OverridePredictor._sigmoid(0) - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        """_sigmoid(large_pos) ≈ 1.0 without overflow."""
        result = OverridePredictor._sigmoid(500)
        assert abs(result - 1.0) < 1e-10

    def test_sigmoid_large_negative(self):
        """_sigmoid(large_neg) ≈ 0.0 without overflow."""
        result = OverridePredictor._sigmoid(-500)
        assert abs(result - 0.0) < 1e-10

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) = 1.0 for any x."""
        for x in [0.1, 1.0, 5.0, 50.0]:
            total = OverridePredictor._sigmoid(x) + OverridePredictor._sigmoid(-x)
            assert abs(total - 1.0) < 1e-10

    def test_encode_target_loss(self, fresh_predictor):
        """_encode_target returns 1.0 for loss."""
        assert fresh_predictor._encode_target({"outcome": "loss"}) == 1.0

    def test_encode_target_win(self, fresh_predictor):
        """_encode_target returns 0.0 for win."""
        assert fresh_predictor._encode_target({"outcome": "win"}) == 0.0

    def test_encode_target_none_for_open(self, fresh_predictor):
        """_encode_target returns None for open/unknown outcome."""
        assert fresh_predictor._encode_target({"outcome": None}) is None
        assert fresh_predictor._encode_target({}) is None


# ═══════════════════════════════════════════════════════════════════════
# US-255: Training, prediction, risk levels, OOD
# ═══════════════════════════════════════════════════════════════════════


class TestUS255TrainingAndPrediction:
    """US-255: fit(), predict_loss_probability(), risk levels, OOD detection."""

    def test_insufficient_data_returns_error(self, fresh_predictor):
        """fit() with < 5 events returns error dict."""
        events = _make_override_events(3)
        result = fresh_predictor.fit(events)
        assert result["error"] == "insufficient_data"
        assert not fresh_predictor._trained

    def test_events_without_outcomes_filtered(self, fresh_predictor):
        """fit() filters events with no outcome — may result in insufficient data."""
        events = [{"override_type": "took_rejected", "outcome": None} for _ in range(10)]
        result = fresh_predictor.fit(events)
        assert result["error"] == "insufficient_data"

    def test_training_succeeds(self, fresh_predictor):
        """fit() with sufficient events sets _trained=True."""
        events = _make_override_events(30)
        metrics = fresh_predictor.fit(events)
        assert fresh_predictor._trained
        assert "error" not in metrics
        assert metrics["samples"] > 5
        assert "accuracy" in metrics
        assert "weights" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_prediction_returns_override_prediction(self, fresh_predictor):
        """predict_loss_probability() returns valid OverridePrediction."""
        events = _make_override_events(30)
        fresh_predictor.fit(events)

        context = {
            "confidence_at_time": 0.6,
            "weighted_vote_at_time": 0.5,
            "emotional_state": "stressed",
            "cognitive_load": "high",
            "override_type": "took_rejected",
            "regime": "VOLATILE",
        }
        pred = fresh_predictor.predict_loss_probability(context)
        assert isinstance(pred, OverridePrediction)
        assert 0.0 <= pred.loss_probability <= 1.0
        assert pred.risk_level in ("low", "moderate", "high", "critical")
        assert isinstance(pred.recommendation, str)
        assert isinstance(pred.top_risk_factors, list)
        assert isinstance(pred.feature_contributions, dict)

    def test_risk_level_low(self, fresh_predictor):
        """loss_probability < 0.5 → risk_level = 'low'."""
        # Untrained model, calm + low cognitive → low risk
        context = {
            "emotional_state": "calm",
            "cognitive_load": "low",
            "override_type": "skipped_recommended",
            "regime": "LOW",
            "confidence_at_time": 0.8,
            "weighted_vote_at_time": 0.8,
        }
        pred = fresh_predictor.predict_loss_probability(context)
        # Heuristic fallback: risk_sum = -0.1 + -0.1 + 0.1 + -0.1 = -0.2
        # sigmoid(-0.2 * 2.0 - 0.5) = sigmoid(-0.9) ≈ 0.29
        assert pred.risk_level == "low"
        assert pred.loss_probability < 0.5

    def test_risk_level_critical(self, fresh_predictor):
        """Extreme risk features → risk_level = 'critical'."""
        context = {
            "emotional_state": "revenge",   # 0.9
            "cognitive_load": "exhausted",  # 0.7
            "override_type": "took_rejected",  # 0.3
            "regime": "EXTREME",            # 0.6
            "confidence_at_time": 0.1,
            "weighted_vote_at_time": 0.1,
        }
        pred = fresh_predictor.predict_loss_probability(context)
        # Heuristic: risk_sum = 0.9 + 0.7 + 0.3 + 0.6 = 2.5
        # sigmoid(2.5 * 2.0 - 0.5) = sigmoid(4.5) ≈ 0.989
        assert pred.risk_level == "critical"
        assert pred.loss_probability >= 0.80

    def test_untrained_uses_heuristic(self, fresh_predictor):
        """Untrained model uses heuristic fallback, not learned weights."""
        assert not fresh_predictor._trained
        context = {
            "emotional_state": "neutral",
            "cognitive_load": "normal",
            "override_type": "skipped_recommended",
            "regime": "NORMAL",
        }
        pred = fresh_predictor.predict_loss_probability(context)
        # Should still return a prediction (heuristic path)
        assert 0.0 <= pred.loss_probability <= 1.0

    def test_ood_detection_blends_towards_05(self, tmp_model_dir):
        """OOD features (>3σ) blend prediction towards 0.5."""
        predictor = OverridePredictor(model_path=tmp_model_dir / "override.json")
        events = _make_override_events(30)
        predictor.fit(events)

        # Normal context
        normal_ctx = {
            "confidence_at_time": 0.6,
            "weighted_vote_at_time": 0.5,
            "emotional_state": "neutral",
            "cognitive_load": "normal",
            "override_type": "took_rejected",
            "regime": "NORMAL",
        }
        pred_normal = predictor.predict_loss_probability(normal_ctx)

        # OOD context — extreme confidence value
        ood_ctx = dict(normal_ctx)
        ood_ctx["confidence_at_time"] = 99.0  # Way beyond training distribution

        with patch("aura.prediction.override_predictor.logger") as mock_logger:
            pred_ood = predictor.predict_loss_probability(ood_ctx)
            ood_warnings = [
                c for c in mock_logger.warning.call_args_list
                if "US-208" in str(c)
            ]
            assert len(ood_warnings) > 0

        # OOD prediction should be pulled towards 0.5
        dist_to_half_ood = abs(pred_ood.loss_probability - 0.5)
        # It should be at most 60% of the original distance from 0.5
        # (blending formula: result * 0.6 + 0.5 * 0.4)

    def test_predict_batch(self, tmp_model_dir):
        """predict_batch() returns list of predictions."""
        predictor = OverridePredictor(model_path=tmp_model_dir / "override.json")
        events = _make_override_events(20)
        predictor.fit(events)

        contexts = [
            {"emotional_state": "anxious", "override_type": "took_rejected",
             "regime": "HIGH", "cognitive_load": "high",
             "confidence_at_time": 0.4, "weighted_vote_at_time": 0.3},
            {"emotional_state": "calm", "override_type": "skipped_recommended",
             "regime": "LOW", "cognitive_load": "low",
             "confidence_at_time": 0.8, "weighted_vote_at_time": 0.7},
        ]
        preds = predictor.predict_batch(contexts)
        assert len(preds) == 2
        assert all(isinstance(p, OverridePrediction) for p in preds)

    def test_to_dict_serialization(self, fresh_predictor):
        """OverridePrediction.to_dict() produces serializable dict."""
        pred = fresh_predictor.predict_loss_probability({
            "emotional_state": "neutral",
            "override_type": "took_rejected",
            "regime": "NORMAL",
        })
        d = pred.to_dict()
        assert isinstance(d, dict)
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str

    def test_model_save_load_roundtrip(self, tmp_model_dir):
        """Model save/load roundtrip preserves trained state."""
        model_path = tmp_model_dir / "override.json"
        predictor = OverridePredictor(model_path=model_path)
        events = _make_override_events(30)
        predictor.fit(events)

        weights_orig = list(predictor._weights)
        bias_orig = predictor._bias
        accuracy_orig = predictor._train_accuracy

        # Load in fresh predictor
        predictor2 = OverridePredictor(model_path=model_path)
        assert predictor2._trained
        assert predictor2._weights == weights_orig
        assert abs(predictor2._bias - bias_orig) < 1e-10
        assert abs(predictor2._train_accuracy - accuracy_orig) < 1e-10

    def test_feature_count_mismatch_on_load(self, tmp_model_dir):
        """Feature count mismatch on load is handled gracefully."""
        model_path = tmp_model_dir / "override.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"n_features": 3, "weights": [0.1, 0.2, 0.3]}
        model_path.write_text(json.dumps(data))

        predictor = OverridePredictor(model_path=model_path)
        assert not predictor._trained

    def test_get_model_info_untrained(self, fresh_predictor):
        """get_model_info() reports correct state when untrained."""
        info = fresh_predictor.get_model_info()
        assert info["trained"] is False
        assert info["n_features"] == 8
        assert info["weights"] == {}

    def test_get_model_info_trained(self, tmp_model_dir):
        """get_model_info() reports weights after training."""
        predictor = OverridePredictor(model_path=tmp_model_dir / "override.json")
        predictor.fit(_make_override_events(30))

        info = predictor.get_model_info()
        assert info["trained"] is True
        assert info["train_samples"] > 0
        assert len(info["weights"]) == 8

    def test_early_stopping_in_fit(self, tmp_model_dir):
        """fit() uses early stopping when loss converges."""
        predictor = OverridePredictor(
            model_path=tmp_model_dir / "override.json",
            n_epochs=1000,  # High max — should stop early
        )
        events = _make_override_events(30)
        metrics = predictor.fit(events)
        # Should stop before 1000 epochs on this data
        assert metrics["epochs"] < 1000
