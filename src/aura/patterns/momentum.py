"""Pattern Momentum Detector.

US-359: Computes velocity and acceleration of pattern frequency changes
to detect momentum shifts in behavioral patterns.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MomentumResult:
    """Result of pattern momentum analysis."""
    velocity: float         # Rate of change (normalized)
    acceleration: float     # Change in velocity (delta velocity)
    momentum_score: float   # 0-100, 50 = neutral
    momentum_label: str     # "accelerating" | "decelerating" | "stable" | "reversing"


class PatternMomentumAnalyzer:
    """Analyzes momentum of behavioral pattern frequency changes.

    Maintains a rolling window of frequency values and computes:
    - velocity: slope of last 5 values (linear regression, normalized by mean)
    - acceleration: change in velocity between consecutive calls
    - momentum_score: 50 + clamp(velocity * 25, -50, 50)
    - momentum_label: based on velocity and acceleration thresholds

    Cold start (< 3 values): returns neutral defaults.
    """

    WINDOW_SIZE = 10
    VELOCITY_WINDOW = 5

    def __init__(self):
        self._history: deque = deque(maxlen=self.WINDOW_SIZE)
        self._prev_velocity: float = 0.0

    def update(self, pattern_frequency: float) -> MomentumResult:
        """Update with new pattern frequency value and compute momentum.

        Args:
            pattern_frequency: Current frequency of the pattern (0.0 to any positive float)

        Returns:
            MomentumResult with velocity, acceleration, momentum_score, momentum_label
        """
        self._history.append(pattern_frequency)

        # Cold start
        if len(self._history) < 3:
            return MomentumResult(
                velocity=0.0,
                acceleration=0.0,
                momentum_score=50.0,
                momentum_label="stable",
            )

        # Compute velocity using linear regression on last VELOCITY_WINDOW values
        velocity = self._compute_velocity()

        # Acceleration = change in velocity
        acceleration = velocity - self._prev_velocity

        # Momentum score: 50 = neutral, higher = more positive momentum
        momentum_score = 50.0 + max(-50.0, min(50.0, velocity * 25.0))
        momentum_score = max(0.0, min(100.0, momentum_score))

        # Determine label
        # Reversing: significant acceleration AND velocity changes sign
        if (abs(acceleration) > 0.3
                and ((self._prev_velocity > 0.0 and velocity < 0.0)
                     or (self._prev_velocity < 0.0 and velocity > 0.0))):
            momentum_label = "reversing"
        elif velocity > 0.5:
            momentum_label = "accelerating"
        elif velocity < -0.5:
            momentum_label = "decelerating"
        else:
            momentum_label = "stable"

        self._prev_velocity = velocity

        logger.debug(
            "US-359: momentum velocity=%.3f accel=%.3f score=%.1f label=%s",
            velocity, acceleration, momentum_score, momentum_label,
        )

        return MomentumResult(
            velocity=round(velocity, 4),
            acceleration=round(acceleration, 4),
            momentum_score=round(momentum_score, 1),
            momentum_label=momentum_label,
        )

    def _compute_velocity(self) -> float:
        """Compute velocity using linear regression slope on last N values.

        Returns:
            Normalized slope (slope / |mean| * 10 if mean != 0)
        """
        recent = list(self._history)[-self.VELOCITY_WINDOW:]
        n = len(recent)
        if n < 2:
            return 0.0

        mean_x = (n - 1) / 2.0
        mean_y = sum(recent) / n

        numerator = sum((i - mean_x) * (recent[i] - mean_y) for i in range(n))
        denominator = sum((i - mean_x) ** 2 for i in range(n))

        if denominator == 0.0:
            return 0.0

        slope = numerator / denominator

        # Normalize by mean to get relative velocity
        if mean_y != 0.0:
            normalized_slope = slope / abs(mean_y) * 10.0
        else:
            normalized_slope = slope

        return normalized_slope
