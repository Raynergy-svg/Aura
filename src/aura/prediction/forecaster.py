"""Multi-Horizon Readiness Forecasting.

US-357: Simple exponential smoothing (SES) with linear regression velocity
to forecast readiness at 1h, 6h, and 24h horizons.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of multi-horizon readiness forecasting."""
    forecast_1h: float      # Readiness forecast in ~1 hour (0-100)
    forecast_6h: float      # Readiness forecast in ~6 hours (0-100)
    forecast_24h: float     # Readiness forecast in ~24 hours (0-100)
    confidence: float       # 0-1, higher when history is stable
    trend_direction: str    # "rising" | "falling" | "stable"


class ReadinessForecaster:
    """Forecasts future readiness using simple exponential smoothing (SES).

    SES with alpha=0.3: smoothed = alpha * score + (1-alpha) * prev_smoothed
    Velocity (trend): linear regression slope of last 5 scores, normalized by mean.
    Forecasts decay with distance: 1h uses velocity*1, 6h uses velocity*3, 24h uses velocity*5.

    Cold start (< 3 scores): forecast = current score, confidence = 0.3
    """

    ALPHA = 0.3  # SES smoothing factor
    VELOCITY_WINDOW = 5   # Scores to use for velocity computation
    CONFIDENCE_WINDOW = 10  # Scores to use for confidence computation

    def __init__(self):
        self._history: deque = deque(maxlen=50)
        self._smoothed: Optional[float] = None
        self._prev_velocity: float = 0.0

    def update(self, readiness_score: float) -> ForecastResult:
        """Update forecaster with new score and produce multi-horizon forecast.

        Args:
            readiness_score: Current readiness score (0-100)

        Returns:
            ForecastResult with 1h/6h/24h forecasts, confidence, and trend direction
        """
        self._history.append(readiness_score)

        # Cold start
        if len(self._history) < 3:
            return ForecastResult(
                forecast_1h=round(readiness_score, 1),
                forecast_6h=round(readiness_score, 1),
                forecast_24h=round(readiness_score, 1),
                confidence=0.3,
                trend_direction="stable",
            )

        # Update SES smoothed value
        if self._smoothed is None:
            self._smoothed = readiness_score
        else:
            self._smoothed = self.ALPHA * readiness_score + (1.0 - self.ALPHA) * self._smoothed

        # Compute velocity from last 5 scores
        velocity = self._compute_velocity()

        # Compute forecasts (clamped to 0-100)
        forecast_1h = max(0.0, min(100.0, self._smoothed + velocity * 1.0))
        forecast_6h = max(0.0, min(100.0, self._smoothed + velocity * 3.0))
        forecast_24h = max(0.0, min(100.0, self._smoothed + velocity * 5.0))

        # Compute confidence from std dev of last 10 scores
        confidence = self._compute_confidence()

        # Determine trend direction
        if velocity > 1.0:
            trend_direction = "rising"
        elif velocity < -1.0:
            trend_direction = "falling"
        else:
            trend_direction = "stable"

        self._prev_velocity = velocity

        logger.debug(
            "US-357: forecast 1h=%.1f 6h=%.1f 24h=%.1f conf=%.3f trend=%s velocity=%.3f",
            forecast_1h, forecast_6h, forecast_24h, confidence, trend_direction, velocity,
        )

        return ForecastResult(
            forecast_1h=round(forecast_1h, 1),
            forecast_6h=round(forecast_6h, 1),
            forecast_24h=round(forecast_24h, 1),
            confidence=round(confidence, 4),
            trend_direction=trend_direction,
        )

    def _compute_velocity(self) -> float:
        """Compute trend velocity using linear regression slope on last N scores.

        Returns:
            Velocity (slope in units per step, normalized by mean)
        """
        recent = list(self._history)[-self.VELOCITY_WINDOW:]
        n = len(recent)
        if n < 2:
            return 0.0

        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n

        # Linear regression slope
        numerator = sum((i - mean_x) * (recent[i] - mean_y) for i in range(n))
        denominator = sum((i - mean_x) ** 2 for i in range(n))

        if denominator == 0.0:
            return 0.0

        slope = numerator / denominator

        # Normalize by mean to get relative velocity
        if mean_y != 0.0:
            normalized_slope = slope / abs(mean_y) * 10.0  # Scale to meaningful range
        else:
            normalized_slope = slope

        return normalized_slope

    def _compute_confidence(self) -> float:
        """Compute forecast confidence from stability of recent history.

        confidence = 1.0 / (1.0 + std_dev / 20.0)
        """
        recent = list(self._history)[-self.CONFIDENCE_WINDOW:]
        n = len(recent)
        if n < 2:
            return 0.3

        mean = sum(recent) / n
        variance = sum((s - mean) ** 2 for s in recent) / n
        std_dev = math.sqrt(variance)

        confidence = 1.0 / (1.0 + std_dev / 20.0)
        return max(0.0, min(1.0, confidence))
