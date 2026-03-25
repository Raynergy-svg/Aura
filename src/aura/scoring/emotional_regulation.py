"""US-326: Emotional Regulation & Recovery Scorer.

Tracks how well the trader recovers from stress events, not just current state.
Three metrics:
  1. recovery_efficiency — how much of a readiness drop was recovered within window
  2. regulation_discipline — avoiding overrides during high-stress periods
  3. stress_absorption — maintaining readiness despite concurrent stressors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecoveryMetrics:
    """Recovery scoring output."""
    recovery_efficiency: float = 0.5   # 0-1, how well trader bounces back
    regulation_discipline: float = 1.0  # 0-1, avoiding overrides during stress
    stress_absorption: float = 0.5     # 0-1, resilience capacity
    composite_recovery_score: float = 0.5  # weighted composite


class EmotionalRegulationScorer:
    """Scores emotional regulation and recovery dynamics.

    Composite = 0.4 * recovery_efficiency + 0.35 * regulation_discipline + 0.25 * stress_absorption
    """

    WEIGHTS = {
        "recovery_efficiency": 0.4,
        "regulation_discipline": 0.35,
        "stress_absorption": 0.25,
    }
    MIN_HISTORY = 5  # Minimum readiness samples before scoring activates
    STRESS_THRESHOLD = 0.7  # Stress level above which overrides are penalized (0-1 scale, inverted: 0.7 stress_level_score = 0.3 stress)

    def recovery_efficiency(
        self,
        readiness_history: List[float],
        window_minutes: float = 60.0,
        sample_interval_minutes: float = 15.0,
    ) -> float:
        """Measure how much of a readiness drop the trader recovered.

        Uses the readiness_history list (most recent last). Finds the peak drop
        within the window and measures recovery from that trough.

        Args:
            readiness_history: List of readiness scores (0-100), newest last
            window_minutes: Recovery window in minutes
            sample_interval_minutes: Approximate time between samples

        Returns:
            0-1 float. 1.0 = full recovery, 0.0 = no recovery from drop
        """
        if len(readiness_history) < self.MIN_HISTORY:
            return 0.5  # Insufficient data, neutral

        # Determine how many samples fit in the window
        window_samples = max(3, int(window_minutes / max(sample_interval_minutes, 1)))
        recent = readiness_history[-window_samples:]

        if len(recent) < 3:
            return 0.5

        peak = max(recent)
        trough = min(recent)
        current = recent[-1]

        total_swing = peak - trough
        if total_swing < 2.0:
            return 1.0  # No significant drop, perfect recovery

        # How much of the drop has been recovered?
        # If trough was the lowest point and current is back near peak, recovery is high
        recovery_from_trough = current - trough
        efficiency = recovery_from_trough / total_swing

        return max(0.0, min(1.0, efficiency))

    def regulation_discipline(
        self,
        override_events: List[Dict[str, Any]],
        stress_levels: List[Dict[str, Any]],
    ) -> float:
        """Check if trader avoided overrides during high-stress periods.

        Args:
            override_events: List of override dicts with 'timestamp' field
            stress_levels: List of dicts with 'stress_level_score' (0-1, higher=less stressed)
                           and 'timestamp' fields

        Returns:
            0-1 float. 1.0 = no overrides during stress, 0.0 = all overrides during stress
        """
        if not override_events:
            return 1.0  # No overrides = perfect discipline

        if not stress_levels:
            return 0.5  # No stress data, neutral

        # Count overrides that happened during high-stress periods
        # (per-override stress lookup via _find_nearest_stress, not aggregate average)
        # High stress = stress_level_score < 0.4 (inverted scale)
        high_stress_overrides = 0
        for override in override_events:
            # Find nearest stress reading
            nearest_stress = self._find_nearest_stress(override, stress_levels)
            if nearest_stress is not None and nearest_stress < 0.4:
                high_stress_overrides += 1

        if len(override_events) == 0:
            return 1.0

        # Discipline = ratio of non-stress overrides
        stress_override_ratio = high_stress_overrides / len(override_events)
        discipline = 1.0 - stress_override_ratio

        return max(0.0, min(1.0, discipline))

    def stress_absorption(
        self,
        active_stressors_count: int,
        current_readiness: float,
    ) -> float:
        """Measure resilience: maintaining readiness despite stressors.

        Args:
            active_stressors_count: Number of concurrent life stressors
            current_readiness: Current readiness score (0-100)

        Returns:
            0-1 float. 1.0 = high readiness despite many stressors
        """
        if active_stressors_count == 0:
            return 0.5  # No stressors, neutral (not resilient, just unstressed)

        # Expected readiness drop per stressor: ~10 points
        expected_readiness = max(20.0, 70.0 - active_stressors_count * 10.0)

        if current_readiness >= expected_readiness:
            # Maintaining above expected — good absorption
            surplus = current_readiness - expected_readiness
            return min(1.0, 0.5 + surplus / 60.0)
        else:
            # Below expected — poor absorption
            deficit = expected_readiness - current_readiness
            return max(0.0, 0.5 - deficit / 60.0)

    def score(
        self,
        readiness_history: List[float],
        override_events: Optional[List[Dict[str, Any]]] = None,
        stress_levels: Optional[List[Dict[str, Any]]] = None,
        active_stressors_count: int = 0,
        current_readiness: float = 50.0,
    ) -> RecoveryMetrics:
        """Compute composite recovery score.

        Returns RecoveryMetrics with all 3 sub-scores plus composite.
        """
        override_events = override_events or []
        stress_levels = stress_levels or []

        re = self.recovery_efficiency(readiness_history)
        rd = self.regulation_discipline(override_events, stress_levels)
        sa = self.stress_absorption(active_stressors_count, current_readiness)

        composite = (
            self.WEIGHTS["recovery_efficiency"] * re
            + self.WEIGHTS["regulation_discipline"] * rd
            + self.WEIGHTS["stress_absorption"] * sa
        )

        return RecoveryMetrics(
            recovery_efficiency=round(re, 4),
            regulation_discipline=round(rd, 4),
            stress_absorption=round(sa, 4),
            composite_recovery_score=round(composite, 4),
        )

    def _find_nearest_stress(
        self, override: Dict[str, Any], stress_levels: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Find nearest stress_level_score to an override event."""
        override_ts = override.get("timestamp", 0)
        if not override_ts:
            # No timestamp, return average
            if stress_levels:
                return sum(s.get("stress_level_score", 0.7) for s in stress_levels) / len(stress_levels)
            return None

        best = None
        best_delta = float("inf")
        for s in stress_levels:
            ts = s.get("timestamp", 0)
            delta = abs(ts - override_ts)
            if delta < best_delta:
                best_delta = delta
                best = s.get("stress_level_score", 0.7)
        return best
