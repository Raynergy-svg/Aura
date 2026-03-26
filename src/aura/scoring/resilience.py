"""Resilience Score Computation.

US-356: Tracks recovery from low-readiness episodes to compute a resilience score.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResilienceResult:
    """Result of resilience computation."""
    recovery_speed: float       # 0-1, 1.0 = fastest recovery
    recovery_consistency: float # 0-1, 1.0 = most consistent recovery
    bounce_count: int           # Number of complete recovery episodes
    resilience_score: float     # 0-100, 50 = neutral (cold start)


class ResilienceTracker:
    """Tracks recovery from low-readiness episodes.

    Measures how quickly and consistently a trader recovers from low-readiness
    states (below threshold). Each dip below threshold followed by a recovery
    above threshold counts as one episode.

    resilience_score = (0.5 * recovery_speed + 0.3 * recovery_consistency + 0.2 * min(bounce_count/5, 1.0)) * 100
    Cold start (no episodes): resilience_score = 50.0
    """

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._score_history: deque = deque(maxlen=window_size)
        self._in_low_state: bool = False
        self._steps_in_low: int = 0
        self._recovery_steps: List[int] = []  # Steps to recover for each episode
        self._prev_result: Optional[ResilienceResult] = None

    def update(self, readiness_score: float, threshold: float = 40.0) -> ResilienceResult:
        """Update resilience tracking with new readiness score.

        Args:
            readiness_score: Current readiness score (0-100)
            threshold: Score below which the person is considered in a low state

        Returns:
            ResilienceResult with recovery metrics and overall resilience score
        """
        self._score_history.append(readiness_score)

        if readiness_score < threshold:
            # Entered or continuing a low state
            if not self._in_low_state:
                self._in_low_state = True
                self._steps_in_low = 1
            else:
                self._steps_in_low += 1
        else:
            # Above threshold
            if self._in_low_state:
                # Completed a recovery episode
                self._recovery_steps.append(self._steps_in_low)
                self._in_low_state = False
                self._steps_in_low = 0

        # Keep only last 10 recovery episodes
        if len(self._recovery_steps) > 10:
            self._recovery_steps = self._recovery_steps[-10:]

        return self._compute_result()

    def _compute_result(self) -> ResilienceResult:
        """Compute resilience metrics from recovery episode history."""
        if not self._recovery_steps:
            # Cold start — neutral score
            return ResilienceResult(
                recovery_speed=0.5,
                recovery_consistency=0.5,
                bounce_count=0,
                resilience_score=50.0,
            )

        bounce_count = len(self._recovery_steps)
        avg_steps = sum(self._recovery_steps) / len(self._recovery_steps)

        # recovery_speed: fewer steps = faster = higher score
        recovery_speed = 1.0 - min(avg_steps / 10.0, 1.0)

        # recovery_consistency: low std dev = consistent = higher score
        if len(self._recovery_steps) >= 2:
            mean_steps = avg_steps
            variance = sum((s - mean_steps) ** 2 for s in self._recovery_steps) / len(self._recovery_steps)
            std_dev = math.sqrt(variance)
            recovery_consistency = max(0.0, 1.0 - (std_dev / 5.0))
        else:
            recovery_consistency = 1.0  # Single episode: fully consistent

        # resilience_score
        resilience_score = (
            0.5 * recovery_speed
            + 0.3 * recovery_consistency
            + 0.2 * min(bounce_count / 5.0, 1.0)
        ) * 100.0

        resilience_score = max(0.0, min(100.0, resilience_score))

        logger.debug(
            "US-356: resilience speed=%.3f consistency=%.3f bounces=%d score=%.1f",
            recovery_speed, recovery_consistency, bounce_count, resilience_score,
        )

        return ResilienceResult(
            recovery_speed=round(recovery_speed, 4),
            recovery_consistency=round(recovery_consistency, 4),
            bounce_count=bounce_count,
            resilience_score=round(resilience_score, 1),
        )
