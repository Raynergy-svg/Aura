"""Tests for Aura prediction models: ReadinessModelV2 and training pipeline.

Phase 5 — US-250 through US-253.
Covers feature encoding, V1 fallback, training with early stopping,
OOD detection, prediction bounds, and save/load roundtrip.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.prediction.readiness_v2 import (
    ReadinessModelV2,
    ReadinessTrainingSample,
    V2_FEATURE_NAMES,
    _V1_WEIGHTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temp directory for model persistence."""
    return tmp_path / "models"


@pytest.fixture
def fresh_model(tmp_model_dir):
    """Untrained ReadinessModelV2 with temp path (no disk load)."""
    model_path = tmp_model_dir / "readiness_v2.json"
    return ReadinessModelV2(model_path=model_path, min_samples=5)


@pytest.fixture
def sample_components():
    """Typical readiness component dict."""
    return {
        "emotional_state": 0.7,
        "cognitive_load": 0.6,
        "override_discipline": 0.8,
        "stress_level": 0.5,
        "confidence_trend": 0.65,
        "engagement": 0.4,
    }


def _make_training_samples(n, noise=0.05):
    """Generate n synthetic training samples with a known pattern.

    Pattern: high emotional + low stress → good outcome.
    This lets us verify that training learns something meaningful.
    """
    import random
    random.seed(42)
    samples = []
    for i in range(n):
        emotional = 0.3 + 0.5 * (i / n)
        cognitive = 0.4 + 0.3 * (i / n)
        override = 0.6 + 0.2 * (i / n)
        stress = 0.8 - 0.5 * (i / n)
        confidence = 0.5 + 0.3 * (i / n)
        engagement = 0.3 + 0.4 * (i / n)
        # outcome correlates with emotional * override - stress
        outcome = max(0.0, min(1.0,
            0.3 * emotional + 0.3 * override - 0.3 * stress + 0.1
            + random.uniform(-noise, noise)
        ))
        samples.append({
            "emotional_state": emotional,
            "cognitive_load": cognitive,
            "override_discipline": override,
            "stress_level": stress,
            "confidence_trend": confidence,
            "engagement": engagement,
            "outcome_quality": outcome,
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════
# US-250: Feature encoding and V1 fallback
# ═══════════════════════════════════════════════════════════════════════


class TestUS250FeatureEncoding:
    """US-250: ReadinessTrainingSample.to_feature_vector() and V1 fallback."""

    def test_feature_vector_length(self):
        """to_feature_vector() returns exactly 10 floats."""
        sample = ReadinessTrainingSample(
            emotional_state=0.7,
            cognitive_load=0.6,
            override_discipline=0.8,
            stress_level=0.5,
            confidence_trend=0.65,
            engagement=0.4,
            outcome_quality=0.75,
        )
        vec = sample.to_feature_vector()
        assert len(vec) == 15  # US-300: expanded from 10 to 15
        assert all(isinstance(v, float) for v in vec)

    def test_feature_vector_values(self):
        """Non-linear and interaction features are mathematically correct."""
        sample = ReadinessTrainingSample(
            emotional_state=0.8,
            cognitive_load=0.6,
            override_discipline=0.9,
            stress_level=0.4,
            confidence_trend=0.7,
            engagement=0.5,
            outcome_quality=0.0,  # irrelevant for encoding
        )
        vec = sample.to_feature_vector()

        # Base features (indices 0-5)
        assert vec[0] == 0.8   # emotional
        assert vec[1] == 0.6   # cognitive
        assert vec[2] == 0.9   # override
        assert vec[3] == 0.4   # stress
        assert vec[4] == 0.7   # confidence
        assert vec[5] == 0.5   # engagement

        # Non-linear (indices 6-7)
        assert abs(vec[6] - 0.8 ** 2) < 1e-10  # emotional²
        assert abs(vec[7] - 0.6 ** 2) < 1e-10  # cognitive²

        # Interactions (indices 8-9)
        assert abs(vec[8] - 0.8 * 0.6) < 1e-10   # emotional × cognitive
        assert abs(vec[9] - 0.4 * 0.9) < 1e-10   # stress × override

    def test_v1_fallback_when_untrained(self, fresh_model, sample_components):
        """compute_score() uses V1 weights when model is not trained."""
        assert not fresh_model._trained

        score, contributions = fresh_model.compute_score(sample_components)

        # Manually compute expected V1 score
        expected = (
            0.7 * 0.25  # emotional
            + 0.6 * 0.20  # cognitive
            + 0.8 * 0.25  # override
            + 0.5 * 0.15  # stress
            + 0.65 * 0.10  # confidence
            + 0.4 * 0.05  # engagement
        )
        assert abs(score - expected * 100) < 0.01

    def test_v1_contributions_keys(self, fresh_model, sample_components):
        """V1 fallback contributions contain all 6 base component keys."""
        _, contributions = fresh_model.compute_score(sample_components)
        for key in _V1_WEIGHTS:
            assert key in contributions

    def test_default_values_for_missing_keys(self, fresh_model):
        """compute_score() uses documented defaults for missing component keys."""
        score, _ = fresh_model.compute_score({})
        # Defaults: e=0.7, c=0.7, o=0.8, s=0.7, t=0.7, g=0.5
        expected = (
            0.7 * 0.25 + 0.7 * 0.20 + 0.8 * 0.25
            + 0.7 * 0.15 + 0.7 * 0.10 + 0.5 * 0.05
        )
        assert abs(score - expected * 100) < 0.01

    def test_v2_feature_names_count(self):
        """V2_FEATURE_NAMES has exactly 15 entries matching N_FEATURES (US-300)."""
        assert len(V2_FEATURE_NAMES) == ReadinessModelV2.N_FEATURES
        assert ReadinessModelV2.N_FEATURES == 15


# ═══════════════════════════════════════════════════════════════════════
# US-251: Training, early stopping, R² computation
# ═══════════════════════════════════════════════════════════════════════


class TestUS251Training:
    """US-251: train() mechanics, early stopping, metrics."""

    def test_insufficient_data_returns_error(self, fresh_model):
        """train() with < min_samples returns error dict."""
        # Add only 3 samples (min is 5)
        for s in _make_training_samples(3):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        result = fresh_model.train()
        assert result["error"] == "insufficient_data"
        assert result["samples"] == 3
        assert not fresh_model._trained

    def test_training_sets_trained_flag(self, fresh_model):
        """train() with sufficient data sets _trained=True and returns metrics."""
        for s in _make_training_samples(25):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        metrics = fresh_model.train()
        assert fresh_model._trained
        assert "error" not in metrics
        assert metrics["samples"] == 25
        assert metrics["using_v2"] is True
        assert "r_squared" in metrics
        assert "epochs_run" in metrics
        assert "weights" in metrics

    def test_r_squared_range(self, fresh_model):
        """R² is between 0 and 1 for well-behaved synthetic data."""
        for s in _make_training_samples(30, noise=0.01):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        metrics = fresh_model.train()
        assert 0.0 <= metrics["r_squared"] <= 1.0

    def test_early_stopping_convergence(self, fresh_model):
        """Early stopping fires before max_epochs on easy data."""
        # Very clean data with low noise → fast convergence
        for s in _make_training_samples(30, noise=0.001):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        metrics = fresh_model.train()
        # Should stop well before 500 epochs
        assert metrics["epochs_run"] < 500
        # Could be convergence or divergence, but should early-stop
        # (convergence most likely with clean data)

    def test_training_computes_feature_statistics(self, fresh_model):
        """Feature means and stds are computed after training."""
        for s in _make_training_samples(25):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        fresh_model.train()
        # Means should not all be zero anymore
        assert any(m != 0.0 for m in fresh_model._feature_means)
        # Stds should not all be 1.0 anymore (unless constant features)
        assert len(fresh_model._feature_stds) == fresh_model.N_FEATURES

    def test_trained_model_uses_v2_path(self, fresh_model, sample_components):
        """After training, compute_score() uses learned weights, not V1."""
        for s in _make_training_samples(25):
            fresh_model._training_buffer.append(ReadinessTrainingSample(**s))

        fresh_model.train()

        # Score should differ from V1 (learned weights ≠ static weights)
        score_v2, contribs_v2 = fresh_model.compute_score(sample_components)

        # V2 contributions have all 15 feature names (US-300)
        assert len(contribs_v2) == 15
        for name in V2_FEATURE_NAMES:
            assert name in contribs_v2

    def test_add_training_sample_auto_trains(self, tmp_model_dir):
        """add_training_sample() triggers train() when buffer reaches min_samples."""
        model = ReadinessModelV2(
            model_path=tmp_model_dir / "readiness_v2.json",
            min_samples=5,
        )
        samples = _make_training_samples(5)
        for s in samples:
            model.add_training_sample(
                {k: v for k, v in s.items() if k != "outcome_quality"},
                s["outcome_quality"],
            )

        # Should auto-train after 5th sample
        assert model._trained


# ═══════════════════════════════════════════════════════════════════════
# US-252: OOD detection and prediction bounds
# ═══════════════════════════════════════════════════════════════════════


class TestUS252OODAndBounds:
    """US-252: OOD detection blending and score clamping."""

    def _trained_model(self, tmp_model_dir):
        """Helper: create and train a model on centered data."""
        model = ReadinessModelV2(
            model_path=tmp_model_dir / "readiness_v2.json",
            min_samples=5,
        )
        for s in _make_training_samples(30, noise=0.02):
            model._training_buffer.append(ReadinessTrainingSample(**s))
        model.train()
        return model

    def test_normal_input_no_blending(self, tmp_model_dir):
        """Normal inputs (within 3σ) produce unblended prediction."""
        model = self._trained_model(tmp_model_dir)
        # Use components near training means → no OOD
        components = {
            "emotional_state": 0.55,
            "cognitive_load": 0.55,
            "override_discipline": 0.7,
            "stress_level": 0.55,
            "confidence_trend": 0.65,
            "engagement": 0.5,
        }
        with patch("aura.prediction.readiness_v2.logger") as mock_logger:
            score, _ = model.compute_score(components)
            # Should not log OOD warning
            for call in mock_logger.warning.call_args_list:
                assert "US-208" not in str(call)

    def test_ood_input_triggers_blending(self, tmp_model_dir):
        """OOD input (>3σ) triggers blending towards 0.5 and logs warning."""
        model = self._trained_model(tmp_model_dir)
        # Extreme value far from training distribution
        components = {
            "emotional_state": 99.0,  # Way beyond any training value
            "cognitive_load": 0.55,
            "override_discipline": 0.7,
            "stress_level": 0.55,
            "confidence_trend": 0.65,
            "engagement": 0.5,
        }
        with patch("aura.prediction.readiness_v2.logger") as mock_logger:
            score, _ = model.compute_score(components)
            # Should log OOD warning
            ood_warnings = [
                c for c in mock_logger.warning.call_args_list
                if "US-208" in str(c)
            ]
            assert len(ood_warnings) > 0

    def test_score_clamped_to_0_100(self, fresh_model):
        """compute_score() is always in [0, 100] even for extreme inputs."""
        # Very high values
        high = {k: 10.0 for k in _V1_WEIGHTS}
        score_high, _ = fresh_model.compute_score(high)
        assert 0.0 <= score_high <= 100.0

        # Very low / negative values
        low = {k: -10.0 for k in _V1_WEIGHTS}
        score_low, _ = fresh_model.compute_score(low)
        assert 0.0 <= score_low <= 100.0

    def test_score_clamped_to_0_100_trained(self, tmp_model_dir):
        """Clamping works for trained model too."""
        model = self._trained_model(tmp_model_dir)
        extreme = {k: 100.0 for k in _V1_WEIGHTS}
        score, _ = model.compute_score(extreme)
        assert 0.0 <= score <= 100.0


# ═══════════════════════════════════════════════════════════════════════
# US-253: Save/load roundtrip and buffer persistence
# ═══════════════════════════════════════════════════════════════════════


class TestUS253Persistence:
    """US-253: Model and buffer save/load roundtrip."""

    def test_model_save_load_roundtrip(self, tmp_model_dir):
        """Saved model loads with identical weights, bias, means, stds."""
        model_path = tmp_model_dir / "readiness_v2.json"
        model = ReadinessModelV2(model_path=model_path, min_samples=5)

        for s in _make_training_samples(25):
            model._training_buffer.append(ReadinessTrainingSample(**s))
        model.train()

        # Capture state
        weights_orig = list(model._weights)
        bias_orig = model._bias
        means_orig = list(model._feature_means)
        stds_orig = list(model._feature_stds)
        r2_orig = model._train_r_squared

        # Load into fresh model
        model2 = ReadinessModelV2(model_path=model_path, min_samples=5)
        assert model2._trained
        assert model2._weights == weights_orig
        assert abs(model2._bias - bias_orig) < 1e-10
        assert model2._feature_means == means_orig
        assert model2._feature_stds == stds_orig
        assert abs(model2._train_r_squared - r2_orig) < 1e-10

    def test_buffer_save_load_roundtrip(self, tmp_model_dir):
        """Training buffer survives save/load roundtrip."""
        model_path = tmp_model_dir / "readiness_v2.json"
        model = ReadinessModelV2(model_path=model_path, min_samples=100)

        # Add samples but don't trigger auto-train (min_samples=100)
        samples = _make_training_samples(10)
        for s in samples:
            model.add_training_sample(
                {k: v for k, v in s.items() if k != "outcome_quality"},
                s["outcome_quality"],
            )

        assert len(model._training_buffer) == 10

        # Load buffer in new model
        model2 = ReadinessModelV2(model_path=model_path, min_samples=100)
        assert len(model2._training_buffer) == 10
        # Verify first sample survived
        assert abs(model2._training_buffer[0].emotional_state
                    - samples[0]["emotional_state"]) < 1e-10

    def test_feature_count_mismatch_on_load(self, tmp_model_dir):
        """Feature count mismatch on load is handled gracefully."""
        model_path = tmp_model_dir / "readiness_v2.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a model with wrong feature count
        data = {
            "weights": [0.1, 0.2],
            "bias": 0.5,
            "feature_means": [0.0, 0.0],
            "feature_stds": [1.0, 1.0],
            "trained": True,
            "train_samples": 50,
            "train_r_squared": 0.9,
            "n_features": 2,  # Wrong! Should be 10
        }
        model_path.write_text(json.dumps(data))

        # Should load without crash, but remain untrained
        model = ReadinessModelV2(model_path=model_path, min_samples=5)
        assert not model._trained

    def test_missing_model_file_no_crash(self, tmp_model_dir):
        """Missing model file on load doesn't crash."""
        model_path = tmp_model_dir / "nonexistent.json"
        model = ReadinessModelV2(model_path=model_path, min_samples=5)
        assert not model._trained
        assert model._weights == [0.0] * 15  # US-300: expanded from 10 to 15

    def test_get_model_info_untrained(self, fresh_model):
        """get_model_info() returns correct structure when untrained."""
        info = fresh_model.get_model_info()
        assert info["version"] == "v1 (fallback)"
        assert info["trained"] is False
        assert info["samples_until_v2"] == 5  # min_samples=5, buffer=0

    def test_get_model_info_trained(self, tmp_model_dir):
        """get_model_info() returns v2 info after training."""
        model = ReadinessModelV2(
            model_path=tmp_model_dir / "readiness_v2.json",
            min_samples=5,
        )
        for s in _make_training_samples(25):
            model._training_buffer.append(ReadinessTrainingSample(**s))
        model.train()

        info = model.get_model_info()
        assert info["version"] == "v2"
        assert info["trained"] is True
        assert info["train_samples"] == 25
        assert info["samples_until_v2"] == 0

    def test_weight_comparison(self, tmp_model_dir):
        """get_weight_comparison() returns v1 vs v2 delta for base features."""
        model = ReadinessModelV2(
            model_path=tmp_model_dir / "readiness_v2.json",
            min_samples=5,
        )
        for s in _make_training_samples(25):
            model._training_buffer.append(ReadinessTrainingSample(**s))
        model.train()

        comparison = model.get_weight_comparison()
        for name in _V1_WEIGHTS:
            assert name in comparison
            assert "v1_weight" in comparison[name]
            assert "v2_weight" in comparison[name]
            assert "delta" in comparison[name]
