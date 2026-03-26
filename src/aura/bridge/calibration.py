"""US-346: Prediction Calibration Loop — Mutual Accuracy Scoring.

Inspired by MAE framework's Judge agent and the Human-AI Handshake's
validation attribute. Tracks how predictive each agent's signals are:

  Aura Calibration:  Was readiness_score predictive of trade outcome?
  Buddy Calibration: Were Buddy's recommendations correct on overrides?

Both scores stored in .aura/bridge/calibration_state.json.
When Aura's calibration drops below 0.5, the ReadinessSignal gains
a low_calibration flag so Buddy can discount it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CALIBRATION_FILENAME = "calibration_state.json"

# Thresholds
HIGH_READINESS_THRESHOLD = 70.0  # Readiness > this is considered "high"
LOW_CALIBRATION_THRESHOLD = 0.5  # Below this → low_calibration flag
MIN_SAMPLES_FOR_SCORING = 10     # Need at least this many samples


@dataclass
class CalibrationSample:
    """A single prediction-vs-outcome observation."""

    timestamp: str = ""
    prediction_value: float = 0.0    # The readiness score or recommendation confidence
    actual_outcome: str = ""         # "win" or "loss"
    score: float = 0.0               # +1 (correct prediction) or -1 (wrong)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prediction_value": round(self.prediction_value, 2),
            "actual_outcome": self.actual_outcome,
            "score": self.score,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSample":
        return cls(
            timestamp=data.get("timestamp", ""),
            prediction_value=data.get("prediction_value", 0.0),
            actual_outcome=data.get("actual_outcome", ""),
            score=data.get("score", 0.0),
            context=data.get("context", {}),
        )


class CalibrationTracker:
    """Tracks prediction accuracy for both Aura and Buddy.

    Uses a rolling window to compute calibration scores. Higher score
    means the agent's predictions are more predictive of actual outcomes.
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self.aura_predictions: List[CalibrationSample] = []
        self.buddy_predictions: List[CalibrationSample] = []

    def record_aura_prediction(
        self,
        readiness_score: float,
        trade_outcome: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record whether Aura's readiness score predicted the trade outcome.

        Scoring logic:
          High readiness (>70) + win  → +1 (correct)
          High readiness (>70) + loss → -1 (wrong)
          Low readiness  (≤70) + loss → +1 (correct — warned against trading)
          Low readiness  (≤70) + win  → -1 (wrong — unnecessarily cautious)
        """
        outcome = trade_outcome.lower().strip()
        if outcome not in ("win", "loss"):
            logger.warning("US-346: Unknown trade outcome '%s' — skipping", outcome)
            return

        high = readiness_score > HIGH_READINESS_THRESHOLD
        if (high and outcome == "win") or (not high and outcome == "loss"):
            score = 1.0
        else:
            score = -1.0

        sample = CalibrationSample(
            prediction_value=readiness_score,
            actual_outcome=outcome,
            score=score,
            context=context or {},
        )
        self.aura_predictions.append(sample)
        # Trim to window
        if len(self.aura_predictions) > self.window_size:
            self.aura_predictions = self.aura_predictions[-self.window_size:]

    def record_buddy_prediction(
        self,
        recommendation: str,
        trader_action: str,
        outcome: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record whether Buddy's recommendation was correct on an override.

        Scoring logic (only for overrides):
          Buddy said skip, trader took, lost  → +1 (Buddy was right)
          Buddy said skip, trader took, won   → -1 (Buddy was wrong)
          Buddy said take, trader skipped, won  → -1 (Buddy was wrong)
          Buddy said take, trader skipped, lost → +1 (Buddy was right)
        """
        outcome = outcome.lower().strip()
        if outcome not in ("win", "loss"):
            logger.warning("US-346: Unknown outcome '%s' — skipping", outcome)
            return

        # Determine if override was costly
        rec = recommendation.lower().strip()
        act = trader_action.lower().strip()

        # Only score when there's a disagreement (override)
        if rec == act:
            return  # No override, nothing to score

        # Buddy was right if the override resulted in a loss
        if outcome == "loss":
            score = 1.0   # Override hurt — Buddy was right
        else:
            score = -1.0  # Override helped — Buddy was wrong

        sample = CalibrationSample(
            prediction_value=0.0,  # Not a numeric prediction for Buddy
            actual_outcome=outcome,
            score=score,
            context=context or {"recommendation": rec, "action": act},
        )
        self.buddy_predictions.append(sample)
        if len(self.buddy_predictions) > self.window_size:
            self.buddy_predictions = self.buddy_predictions[-self.window_size:]

    def aura_calibration_score(self) -> float:
        """Compute Aura's prediction accuracy (0-1).

        Returns 0.5 (neutral) if insufficient samples.
        Score = (count of positive scores) / total.
        """
        if len(self.aura_predictions) < MIN_SAMPLES_FOR_SCORING:
            return 0.5
        positive = sum(1 for s in self.aura_predictions if s.score > 0)
        negative = sum(1 for s in self.aura_predictions if s.score < 0)
        # If all scores are zero (neutral), return 0.5 — no signal either way
        if positive == 0 and negative == 0:
            return 0.5
        return positive / len(self.aura_predictions)

    def buddy_calibration_score(self) -> float:
        """Compute Buddy's override accuracy (0-1).

        Returns 0.5 (neutral) if insufficient samples.
        """
        if len(self.buddy_predictions) < MIN_SAMPLES_FOR_SCORING:
            return 0.5
        positive = sum(1 for s in self.buddy_predictions if s.score > 0)
        negative = sum(1 for s in self.buddy_predictions if s.score < 0)
        if positive == 0 and negative == 0:
            return 0.5
        return positive / len(self.buddy_predictions)

    def is_low_calibration(self) -> bool:
        """Return True if Aura's calibration is below threshold with enough data."""
        if len(self.aura_predictions) < MIN_SAMPLES_FOR_SCORING:
            return False
        return self.aura_calibration_score() < LOW_CALIBRATION_THRESHOLD

    # --- Persistence ---

    def save_state(self, bridge_dir: Path) -> None:
        """Persist calibration state to bridge directory."""
        from src.aura.bridge.signals import FeedbackBridge

        state = {
            "aura_predictions": [s.to_dict() for s in self.aura_predictions],
            "buddy_predictions": [s.to_dict() for s in self.buddy_predictions],
            "aura_calibration_score": round(self.aura_calibration_score(), 4),
            "buddy_calibration_score": round(self.buddy_calibration_score(), 4),
            "window_size": self.window_size,
        }
        path = bridge_dir / _CALIBRATION_FILENAME
        FeedbackBridge._locked_write(path, json.dumps(state, indent=2, default=str))

    def load_state(self, bridge_dir: Path) -> bool:
        """Load calibration state from bridge directory. Returns True if loaded."""
        from src.aura.bridge.signals import FeedbackBridge

        path = bridge_dir / _CALIBRATION_FILENAME
        raw = FeedbackBridge._locked_read(path)
        if not raw:
            return False
        try:
            state = json.loads(raw)
            self.aura_predictions = [
                CalibrationSample.from_dict(s)
                for s in state.get("aura_predictions", [])
            ]
            self.buddy_predictions = [
                CalibrationSample.from_dict(s)
                for s in state.get("buddy_predictions", [])
            ]
            self.window_size = state.get("window_size", 20)
            return True
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("US-346: Failed to load calibration state: %s", e)
            return False
