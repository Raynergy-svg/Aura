"""ML-Based Readiness Score v2 — learned component weights from outcomes.

PRD v2.2 §13 Phase 3: "Readiness score v2 (ML-based)"

Replaces the hardcoded component weights in ReadinessComputer with weights
learned from historical data: which readiness components actually predict
good/bad trading outcomes?

Approach:
1. Collect (readiness_components, trading_outcome) pairs from bridge data
2. Train a linear regression to predict outcome quality from readiness components
3. Use learned weights instead of the static {emotional: 0.25, cognitive: 0.20, ...}
4. Falls back to v1 weights when insufficient training data

The v2 model also adds:
- Non-linear features (squared terms for extreme states)
- Interaction terms (emotional × cognitive)
- Time-of-day feature (session timing matters for trading)

Zero-dependency — pure Python, no numpy/sklearn.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# V1 baseline weights (fall back to these)
_V1_WEIGHTS = {
    "emotional_state": 0.25,
    "cognitive_load": 0.20,
    "override_discipline": 0.25,
    "stress_level": 0.15,
    "confidence_trend": 0.10,
    "engagement": 0.05,
}

# V2 extended features (US-300: expanded from 10 to 15 dimensions)
V2_FEATURE_NAMES = [
    "emotional_state",
    "cognitive_load",
    "override_discipline",
    "stress_level",
    "confidence_trend",
    "engagement",
    # Non-linear features
    "emotional_squared",          # Captures extreme emotional states
    "cognitive_squared",          # Captures extreme cognitive load
    # Interaction features
    "emotional_x_cognitive",      # Compound impairment
    "stress_x_override",          # Stress + bad overrides = danger
    # US-300: Additional interaction/polynomial terms
    "stress_x_cognitive_load",    # Stress + overloaded = compounded impact
    "override_x_engagement",      # Low discipline + disengaged = double jeopardy
    "confidence_x_emotional",     # Confidence trend modulated by emotional state
    "engagement_squared",         # Diminishing returns of engagement
    "emotional_cubed",            # Saturation: small changes at extremes have outsized impact
]


@dataclass
class ReadinessTrainingSample:
    """A single (readiness_components, outcome) pair for training."""

    emotional_state: float
    cognitive_load: float
    override_discipline: float
    stress_level: float
    confidence_trend: float
    engagement: float
    outcome_quality: float  # 0-1, based on trading PnL / win rate that day
    # US-294: Timestamp for recency weighting
    timestamp: str = ""  # ISO-format UTC timestamp; empty = assumed recent

    def to_feature_vector(self) -> List[float]:
        """Expand into v2 feature vector (US-300: 15 dimensions)."""
        return [
            self.emotional_state,
            self.cognitive_load,
            self.override_discipline,
            self.stress_level,
            self.confidence_trend,
            self.engagement,
            # Non-linear
            self.emotional_state ** 2,
            self.cognitive_load ** 2,
            # Interactions (original)
            self.emotional_state * self.cognitive_load,
            self.stress_level * self.override_discipline,
            # US-300: Additional interaction/polynomial terms
            self.stress_level * self.cognitive_load,         # stress_x_cognitive_load
            self.override_discipline * self.engagement,      # override_x_engagement
            self.confidence_trend * self.emotional_state,    # confidence_x_emotional
            self.engagement ** 2,                            # engagement_squared
            self.emotional_state ** 3,                       # emotional_cubed
        ]


class ReadinessModelV2:
    """ML-based readiness scoring with learned component weights.

    Uses ridge regression (L2-regularized least squares) trained on
    historical readiness→outcome pairs. Falls back to v1 weights
    when insufficient training data.

    Args:
        model_path: Where to persist learned weights
        min_samples: Minimum samples required before using learned weights
        regularization: L2 penalty strength (prevents overfitting)
    """

    N_FEATURES = len(V2_FEATURE_NAMES)
    MAX_BUFFER_SIZE = 500  # US-262: Cap training buffer to prevent unbounded growth

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_samples: int = 20,
        regularization: float = 0.1,
    ):
        self.model_path = model_path or Path(".aura/models/readiness_v2.json")
        self.min_samples = min_samples
        self.regularization = regularization

        self._weights: List[float] = [0.0] * self.N_FEATURES
        self._bias: float = 0.5  # Start at neutral
        self._trained: bool = False
        self._train_samples: int = 0
        self._train_r_squared: float = 0.0
        self._feature_means: List[float] = [0.0] * self.N_FEATURES
        self._feature_stds: List[float] = [1.0] * self.N_FEATURES

        # Training data buffer (accumulates until min_samples reached)
        self._training_buffer: List[ReadinessTrainingSample] = []
        self._buffer_path = (
            self.model_path.parent / "readiness_v2_training_buffer.json"
            if self.model_path
            else Path(".aura/models/readiness_v2_training_buffer.json")
        )

        self._load_model()
        self._load_buffer()
        self._last_online_update: Optional[str] = None  # L-02: declare before first use

    # --- Scoring ---

    def compute_score(
        self, components: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute readiness score using learned weights.

        Args:
            components: Dict with keys matching ReadinessComponents.to_dict()
                {emotional_state, cognitive_load, override_discipline,
                 stress_level, confidence_trend, engagement}

        Returns:
            Tuple of (score_0_to_100, component_contributions)
        """
        # Build feature vector
        e = components.get("emotional_state", 0.7)
        c = components.get("cognitive_load", 0.7)
        o = components.get("override_discipline", 0.8)
        s = components.get("stress_level", 0.7)
        t = components.get("confidence_trend", 0.7)
        g = components.get("engagement", 0.5)

        features = [
            e, c, o, s, t, g,
            e ** 2, c ** 2,
            e * c, s * o,
            # US-300: Additional interaction/polynomial terms
            s * c,    # stress_x_cognitive_load
            o * g,    # override_x_engagement
            t * e,    # confidence_x_emotional
            g ** 2,   # engagement_squared
            e ** 3,   # emotional_cubed
        ]

        if self._trained:
            # Use learned weights
            score = self._predict(features)
        else:
            # Fall back to v1 weighted sum
            score = (
                e * _V1_WEIGHTS["emotional_state"]
                + c * _V1_WEIGHTS["cognitive_load"]
                + o * _V1_WEIGHTS["override_discipline"]
                + s * _V1_WEIGHTS["stress_level"]
                + t * _V1_WEIGHTS["confidence_trend"]
                + g * _V1_WEIGHTS["engagement"]
            )

        # Clamp to 0-100
        score_100 = max(0.0, min(100.0, score * 100))

        # Component contributions for interpretability
        contributions = self._compute_contributions(features)

        return score_100, contributions

    def _predict(self, features: List[float]) -> float:
        """Predict using learned weights.

        US-208: Includes OOD (out-of-distribution) detection — logs warning
        and caps prediction confidence when any feature is >3σ from training mean.
        """
        # Normalize
        normalized = []
        ood_features = []
        for i, val in enumerate(features):
            std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
            z = (val - self._feature_means[i]) / std
            normalized.append(z)
            if abs(z) > 3.0:
                fname = V2_FEATURE_NAMES[i] if i < len(V2_FEATURE_NAMES) else f"feature_{i}"
                ood_features.append((fname, val, z))

        if ood_features:
            logger.warning(
                "US-208: OOD input detected — %d feature(s) beyond 3σ: %s. "
                "Prediction may be unreliable (capping confidence).",
                len(ood_features),
                [(f, f"val={v:.3f}, z={z:.1f}σ") for f, v, z in ood_features],
            )

        # Linear prediction
        result = self._bias
        for w, x in zip(self._weights, normalized):
            result += w * x

        # US-208: Cap prediction towards 0.5 (uncertain) when OOD
        if ood_features:
            result = result * 0.6 + 0.5 * 0.4  # Blend towards 0.5

        return result

    def _compute_contributions(
        self, features: List[float]
    ) -> Dict[str, float]:
        """Compute per-feature contribution to the score."""
        contributions: Dict[str, float] = {}

        if self._trained:
            for i, (name, feat_val) in enumerate(
                zip(V2_FEATURE_NAMES, features)
            ):
                std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
                normalized = (feat_val - self._feature_means[i]) / std
                contributions[name] = self._weights[i] * normalized
        else:
            # V1 contributions (only base features)
            for name in list(_V1_WEIGHTS.keys()):
                idx = V2_FEATURE_NAMES.index(name)
                contributions[name] = features[idx] * _V1_WEIGHTS[name]

        return contributions

    # --- Training ---

    # US-294: Recency weighting constants
    RECENCY_HALFLIFE_DAYS = 30.0  # exp(-age_days/30) — 30-day half-life
    RECENCY_WEIGHT_FLOOR = 0.1    # Minimum sample weight (never fully ignored)

    def _compute_sample_weight(self, timestamp_str: str) -> float:
        """US-294: Compute recency weight for a training sample.

        Recent samples get weight ~1.0, older samples decay exponentially.
        Weight = max(0.1, exp(-age_days / 30))

        Args:
            timestamp_str: ISO-format UTC timestamp of the sample.

        Returns:
            Weight in [RECENCY_WEIGHT_FLOOR, 1.0]
        """
        if not timestamp_str:
            return 1.0  # No timestamp = assume recent
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            weight = math.exp(-age_days / self.RECENCY_HALFLIFE_DAYS)
            return max(self.RECENCY_WEIGHT_FLOOR, weight)
        except (ValueError, TypeError):
            return 1.0  # Parse failure = assume recent

    def add_training_sample(
        self,
        readiness_components: Dict[str, float],
        trading_outcome_quality: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a readiness→outcome pair to the training buffer.

        Call this after each trading day/session with the readiness components
        that were active and the resulting trading quality (0=terrible, 1=great).

        Args:
            readiness_components: Dict from ReadinessComponents.to_dict()
            trading_outcome_quality: 0-1 quality score derived from:
                - win rate (0-1)
                - PnL normalized to typical range
                - override discipline that day
            timestamp: Optional ISO-format UTC timestamp. Defaults to now.
        """
        sample = ReadinessTrainingSample(
            emotional_state=readiness_components.get("emotional_state", 0.7),
            cognitive_load=readiness_components.get("cognitive_load", 0.7),
            override_discipline=readiness_components.get("override_discipline", 0.8),
            stress_level=readiness_components.get("stress_level", 0.7),
            confidence_trend=readiness_components.get("confidence_trend", 0.7),
            engagement=readiness_components.get("engagement", 0.5),
            outcome_quality=trading_outcome_quality,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )
        self._training_buffer.append(sample)

        # US-262: Cap training buffer — drop oldest samples
        if len(self._training_buffer) > self.MAX_BUFFER_SIZE:
            dropped = len(self._training_buffer) - self.MAX_BUFFER_SIZE
            self._training_buffer = self._training_buffer[-self.MAX_BUFFER_SIZE:]
            logger.info("US-262: Training buffer capped — dropped %d oldest samples", dropped)

        self._save_buffer()

        # Auto-train when sufficient data accumulated
        if len(self._training_buffer) >= self.min_samples:
            self.train()

    def train(self) -> Dict[str, Any]:
        """Train the readiness model on accumulated samples.

        Uses closed-form ridge regression:
            w = (X^T X + λI)^(-1) X^T y

        For small feature sets (10 features), this is efficient without numpy.

        Returns:
            Training metrics dict.
        """
        if len(self._training_buffer) < self.min_samples:
            return {
                "error": "insufficient_data",
                "samples": len(self._training_buffer),
                "min_required": self.min_samples,
            }

        # Build feature matrix, target vector, and sample weights (US-294)
        X: List[List[float]] = []
        y: List[float] = []
        sample_weights: List[float] = []
        for sample in self._training_buffer:
            X.append(sample.to_feature_vector())
            y.append(sample.outcome_quality)
            sample_weights.append(self._compute_sample_weight(sample.timestamp))

        n = len(X)
        p = self.N_FEATURES

        # Compute feature statistics for normalization
        self._compute_means_stds(X)
        X_norm = self._normalize(X)

        # Gradient descent with early stopping (US-236)
        lr = 0.01
        max_epochs = 500
        patience = 10
        convergence_delta = 1e-5
        divergence_patience = 5

        self._weights = [0.0] * p
        self._bias = sum(y) / n  # Start at mean target

        prev_losses: List[float] = []
        converge_count = 0
        diverge_count = 0
        actual_epochs = 0
        early_stop_reason = None

        # US-294: Normalize sample weights so they sum to n (preserves learning rate behavior)
        weight_sum = sum(sample_weights)
        if weight_sum > 0:
            norm_weights = [w * n / weight_sum for w in sample_weights]
        else:
            norm_weights = [1.0] * n

        for epoch in range(max_epochs):
            total_loss = 0.0
            for i in range(n):
                # Forward
                pred = self._bias
                for j in range(p):
                    pred += self._weights[j] * X_norm[i][j]

                error = pred - y[i]
                # US-294: Weight the loss by recency
                weighted_error = error * norm_weights[i]
                total_loss += error ** 2 * norm_weights[i]

                # Update weights (recency-weighted gradient)
                for j in range(p):
                    gradient = weighted_error * X_norm[i][j] + self.regularization * self._weights[j]
                    self._weights[j] -= lr * gradient / n
                self._bias -= lr * weighted_error / n

            actual_epochs = epoch + 1
            avg_loss = total_loss / n

            # Early stopping checks (US-236)
            if prev_losses:
                loss_delta = abs(prev_losses[-1] - avg_loss)
                if loss_delta < convergence_delta:
                    converge_count += 1
                else:
                    converge_count = 0

                if avg_loss > prev_losses[-1]:
                    diverge_count += 1
                else:
                    diverge_count = 0

                if converge_count >= patience:
                    early_stop_reason = "convergence"
                    logger.info("US-236: Early stop at epoch %d — loss converged (delta < %.1e for %d epochs)",
                                actual_epochs, convergence_delta, patience)
                    break

                if diverge_count >= divergence_patience:
                    early_stop_reason = "divergence"
                    logger.warning("US-236: Early stop at epoch %d — loss diverging for %d epochs",
                                   actual_epochs, divergence_patience)
                    break

            prev_losses.append(avg_loss)

        if early_stop_reason is None:
            logger.info("US-236: Training completed full %d epochs", max_epochs)

        # Compute R² (coefficient of determination) — US-294: recency-weighted
        y_mean = sum(y[i] * norm_weights[i] for i in range(n)) / sum(norm_weights)
        ss_tot = sum(norm_weights[i] * (y[i] - y_mean) ** 2 for i in range(n))
        ss_res = 0.0
        for i in range(n):
            pred = self._bias
            for j in range(p):
                pred += self._weights[j] * X_norm[i][j]
            ss_res += norm_weights[i] * (y[i] - pred) ** 2

        self._train_r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        self._train_samples = n
        self._trained = True

        self._save_model()

        # Extract interpretable weight importances
        weight_importances = {
            name: round(abs(w), 4)
            for name, w in zip(V2_FEATURE_NAMES, self._weights)
        }
        sorted_importance = sorted(
            weight_importances.items(), key=lambda x: x[1], reverse=True
        )

        metrics = {
            "samples": n,
            "r_squared": round(self._train_r_squared, 4),
            "mse": round(ss_res / n, 6),
            "epochs_run": actual_epochs,
            "early_stop_reason": early_stop_reason,
            "top_features": sorted_importance[:5],
            "weights": {
                name: round(w, 4)
                for name, w in zip(V2_FEATURE_NAMES, self._weights)
            },
            "bias": round(self._bias, 4),
            "using_v2": True,
        }
        logger.info(
            "ReadinessV2 trained: %d samples, R²=%.3f, top=%s",
            n, self._train_r_squared,
            sorted_importance[0][0] if sorted_importance else "none",
        )
        return metrics

    def _compute_means_stds(self, X: List[List[float]]) -> None:
        """Compute feature means and stds."""
        n = len(X)
        p = self.N_FEATURES
        means = [0.0] * p
        for row in X:
            for j in range(p):
                means[j] += row[j]
        self._feature_means = [m / n for m in means]

        stds = [0.0] * p
        for row in X:
            for j in range(p):
                stds[j] += (row[j] - self._feature_means[j]) ** 2
        self._feature_stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

    def _normalize(self, X: List[List[float]]) -> List[List[float]]:
        """Z-score normalize feature matrix."""
        return [
            [
                (row[j] - self._feature_means[j]) /
                (self._feature_stds[j] if self._feature_stds[j] > 1e-8 else 1.0)
                for j in range(self.N_FEATURES)
            ]
            for row in X
        ]

    # --- Model Info ---

    # --- US-322: Online Learning ---

    def update_from_outcome(
        self,
        component_dict: Dict[str, float],
        target_readiness: float,
        learning_rate: float = 0.01,
    ) -> bool:
        """US-322/US-329: Single-sample SGD update from a trade outcome.

        Performs one gradient descent step using the new (components, target)
        pair. US-329: Uses AdaptiveLearningRateScheduler for warmup/decay/diversity.

        Args:
            component_dict: Readiness component values (emotional_state, etc.)
            target_readiness: Target readiness value (0-1 scale) derived from outcome
            learning_rate: Step size for weight update (used as initial_lr for scheduler)

        Returns:
            True if update was applied, False if skipped (model not yet batch-trained)
        """
        if not self._trained or self._train_samples < self.min_samples:
            logger.info(
                "US-322: Skipping online update — model not batch-trained yet "
                "(trained=%s, samples=%d, min=%d)",
                self._trained, self._train_samples, self.min_samples,
            )
            return False

        # Build feature vector from components
        e = component_dict.get("emotional_state", 0.7)
        c = component_dict.get("cognitive_load", 0.7)
        o = component_dict.get("override_discipline", 0.8)
        s = component_dict.get("stress_level", 0.7)
        t = component_dict.get("confidence_trend", 0.7)
        g = component_dict.get("engagement", 0.5)

        features = [
            e, c, o, s, t, g,
            e ** 2, c ** 2,
            e * c, s * o,
            s * c, o * g, t * e,
            g ** 2, e ** 3,
        ]

        # Normalize using stored means/stds
        normalized = []
        for i, val in enumerate(features):
            std = self._feature_stds[i] if self._feature_stds[i] > 1e-8 else 1.0
            normalized.append((val - self._feature_means[i]) / std)

        # Forward pass
        pred = self._bias
        for j in range(self.N_FEATURES):
            pred += self._weights[j] * normalized[j]

        # Compute error
        error = pred - target_readiness

        # Compute raw gradient vector
        raw_gradient = [
            error * normalized[j] + self.regularization * self._weights[j]
            for j in range(self.N_FEATURES)
        ]

        # US-329: Use adaptive learning rate if scheduler available
        actual_lr = learning_rate
        effective_gradient = raw_gradient
        try:
            if not hasattr(self, '_lr_scheduler') or self._lr_scheduler is None:
                from src.aura.prediction.lr_scheduler import AdaptiveLearningRateScheduler
                self._lr_scheduler = AdaptiveLearningRateScheduler(
                    initial_lr=learning_rate,
                    warmup_samples=50,
                )
            actual_lr = self._lr_scheduler.get_learning_rate(
                sample_count=self._train_samples,
                gradient_vector=raw_gradient,
            )
            effective_gradient = self._lr_scheduler.apply_momentum(raw_gradient)
            self._lr_scheduler.step(raw_gradient)
        except Exception as e:
            logger.debug("US-329: Adaptive LR unavailable, using fixed lr: %s", e)
            actual_lr = learning_rate
            effective_gradient = raw_gradient

        # SGD update: w -= lr * effective_gradient_j
        for j in range(self.N_FEATURES):
            self._weights[j] -= actual_lr * effective_gradient[j]
        self._bias -= actual_lr * error

        # Track update
        self._train_samples += 1
        self._last_online_update = datetime.now(timezone.utc).isoformat()

        logger.info(
            "US-322/US-329: Online SGD update — error=%.4f, lr=%.5f (adaptive), samples=%d",
            error, actual_lr, self._train_samples,
        )

        # Persist updated weights
        self._save_model()
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model status for display."""
        return {
            "version": "v2" if self._trained else "v1 (fallback)",
            "trained": self._trained,
            "train_samples": self._train_samples,
            "r_squared": round(self._train_r_squared, 3),
            "buffer_size": len(self._training_buffer),
            "min_samples": self.min_samples,
            "samples_until_v2": max(0, self.min_samples - len(self._training_buffer)),
            "weights": {
                name: round(w, 4)
                for name, w in zip(V2_FEATURE_NAMES, self._weights)
            } if self._trained else dict(_V1_WEIGHTS),
        }

    def get_weight_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare v1 vs v2 weights for interpretability."""
        comparison: Dict[str, Dict[str, float]] = {}
        for name in _V1_WEIGHTS:
            idx = V2_FEATURE_NAMES.index(name)
            comparison[name] = {
                "v1_weight": _V1_WEIGHTS[name],
                "v2_weight": round(self._weights[idx], 4) if self._trained else _V1_WEIGHTS[name],
                "delta": round(
                    self._weights[idx] - _V1_WEIGHTS[name], 4
                ) if self._trained else 0.0,
            }
        return comparison

    # --- Persistence ---

    def _save_model(self) -> None:
        """Persist model to JSON."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "weights": self._weights,
                "bias": self._bias,
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
                "trained": self._trained,
                "train_samples": self._train_samples,
                "train_r_squared": self._train_r_squared,
                "feature_names": V2_FEATURE_NAMES,
                "n_features": self.N_FEATURES,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            # US-273: Atomic locked write for crash safety
            from src.aura.persistence import atomic_write_json
            atomic_write_json(self.model_path, data)
        except Exception as e:
            logger.error("Failed to save readiness v2 model: %s", e)

    def _load_model(self) -> None:
        """Load model from JSON."""
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text())
            if data.get("n_features") != self.N_FEATURES:
                logger.warning(
                    "US-300: Readiness v2 feature count mismatch (saved=%s, current=%d) — "
                    "retrain required after feature expansion",
                    data.get("n_features"), self.N_FEATURES,
                )
                return
            self._weights = data["weights"]
            self._bias = data["bias"]
            self._feature_means = data["feature_means"]
            self._feature_stds = data["feature_stds"]
            self._trained = data.get("trained", False)
            self._train_samples = data.get("train_samples", 0)
            self._train_r_squared = data.get("train_r_squared", 0.0)
        except Exception as e:
            logger.warning("Failed to load readiness v2 model: %s", e)

    def _save_buffer(self) -> None:
        """Persist training buffer."""
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = [
                {
                    "emotional_state": s.emotional_state,
                    "cognitive_load": s.cognitive_load,
                    "override_discipline": s.override_discipline,
                    "stress_level": s.stress_level,
                    "confidence_trend": s.confidence_trend,
                    "engagement": s.engagement,
                    "outcome_quality": s.outcome_quality,
                    "timestamp": s.timestamp,  # US-294
                }
                for s in self._training_buffer
            ]
            from src.aura.persistence import atomic_write_json  # Fix C-02: was aura.persistence (wrong path, causes ImportError)
            atomic_write_json(self._buffer_path, data)
        except Exception as e:
            logger.warning("Failed to save training buffer: %s", e)

    def _load_buffer(self) -> None:
        """Load training buffer from disk."""
        if not self._buffer_path.exists():
            return
        try:
            data = json.loads(self._buffer_path.read_text())
            self._training_buffer = [
                ReadinessTrainingSample(**entry)
                for entry in data
                if isinstance(entry, dict)
            ]
        except Exception as e:
            logger.warning("Failed to load training buffer: %s", e)
