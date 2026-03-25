"""US-328: Metacognitive Monitoring Scorer.

Measures how well the trader knows what they know:
  1. calibration_score — alignment between stated confidence and actual outcomes
  2. resolution_score — ability to discriminate: high confidence on wins, low on losses
  3. effort_allocation — inverse correlation between confidence and message complexity

Feeds into DecisionQualityScorer as 8th dimension (metacognitive_monitoring, weight 0.10).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Single tracked decision for metacognitive analysis."""
    decision_id: str
    stated_confidence: float  # 0-1
    outcome_pnl: float       # positive = win, negative = loss
    message_complexity: float = 0.5  # 0-1, based on word count / unique words


@dataclass
class MetacognitiveScore:
    """Metacognitive monitoring score breakdown."""
    calibration: float = 0.5
    resolution: float = 0.5
    effort_allocation: float = 0.5
    composite: float = 0.5
    decision_count: int = 0


class MetacognitiveMonitoringScorer:
    """Scores trader's metacognitive monitoring ability.

    Composite = 0.4 * calibration + 0.35 * resolution + 0.25 * effort_allocation
    """

    WEIGHTS = {
        "calibration": 0.4,
        "resolution": 0.35,
        "effort_allocation": 0.25,
    }
    MIN_DECISIONS = 10  # Below this, returns 0.5 (neutral)

    def __init__(self):
        self._decisions: List[DecisionRecord] = []

    def track_decision(
        self,
        decision_id: str,
        stated_confidence: float,
        outcome_pnl: float,
        message_complexity: float = 0.5,
    ) -> None:
        """Record a decision for metacognitive tracking."""
        self._decisions.append(DecisionRecord(
            decision_id=decision_id,
            stated_confidence=max(0.0, min(1.0, stated_confidence)),
            outcome_pnl=outcome_pnl,
            message_complexity=max(0.0, min(1.0, message_complexity)),
        ))

    @property
    def decision_count(self) -> int:
        return len(self._decisions)

    def calibration_score(self) -> float:
        """Alignment between stated confidence and actual win rate.

        Perfect calibration = 0 error (80% confident → 80% win rate).
        Returns 0-1 (1.0 = perfect calibration).
        """
        if len(self._decisions) < self.MIN_DECISIONS:
            return 0.5

        # Mean stated confidence (0-1)
        mean_confidence = sum(d.stated_confidence for d in self._decisions) / len(self._decisions)

        # Actual win rate (0-1)
        wins = sum(1 for d in self._decisions if d.outcome_pnl > 0)
        win_rate = wins / len(self._decisions)

        # Calibration error: |confidence - win_rate|
        calibration_error = abs(mean_confidence - win_rate)

        # Convert to 0-1 score (lower error = higher score)
        # Max meaningful error is ~0.5 (e.g., 90% confident but only 40% win rate)
        return max(0.0, min(1.0, 1.0 - calibration_error * 2.0))

    def resolution_score(self) -> float:
        """Ability to discriminate: higher confidence on wins, lower on losses.

        Returns 0-1 (1.0 = perfect discrimination).
        """
        if len(self._decisions) < self.MIN_DECISIONS:
            return 0.5

        wins = [d for d in self._decisions if d.outcome_pnl > 0]
        losses = [d for d in self._decisions if d.outcome_pnl <= 0]

        if not wins or not losses:
            return 0.5  # Need both categories

        avg_win_conf = sum(d.stated_confidence for d in wins) / len(wins)
        avg_loss_conf = sum(d.stated_confidence for d in losses) / len(losses)

        # Gap: how much higher is confidence on wins vs losses?
        gap = avg_win_conf - avg_loss_conf  # Positive = good discrimination

        # Scale: gap of 0.3+ is excellent resolution
        resolution = 0.5 + gap / 0.6  # gap of 0.3 → 1.0, gap of -0.3 → 0.0
        return max(0.0, min(1.0, resolution))

    def effort_allocation_score(self) -> float:
        """Inverse correlation between confidence and deliberation effort.

        Good metacognition: uncertain → deliberate more (complex messages).
        Bad metacognition: uncertain but same effort, or overconfident with no deliberation.

        Returns 0-1 (1.0 = ideal effort allocation).
        """
        if len(self._decisions) < self.MIN_DECISIONS:
            return 0.5

        # We want NEGATIVE correlation: low confidence → high complexity
        n = len(self._decisions)
        confs = [d.stated_confidence for d in self._decisions]
        complexities = [d.message_complexity for d in self._decisions]

        mean_c = sum(confs) / n
        mean_x = sum(complexities) / n

        # Pearson correlation
        cov = sum((confs[i] - mean_c) * (complexities[i] - mean_x) for i in range(n)) / n
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in confs) / n)
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in complexities) / n)

        if std_c < 1e-8 or std_x < 1e-8:
            return 0.5  # No variance, can't assess

        correlation = cov / (std_c * std_x)

        # We want negative correlation (low confidence → high effort)
        # correlation of -1 is ideal, +1 is worst
        # Map [-1, 1] → [1.0, 0.0]
        return max(0.0, min(1.0, 0.5 - correlation * 0.5))

    def score(self) -> MetacognitiveScore:
        """Compute composite metacognitive monitoring score."""
        cal = self.calibration_score()
        res = self.resolution_score()
        eff = self.effort_allocation_score()

        composite = (
            self.WEIGHTS["calibration"] * cal
            + self.WEIGHTS["resolution"] * res
            + self.WEIGHTS["effort_allocation"] * eff
        )

        return MetacognitiveScore(
            calibration=round(cal, 4),
            resolution=round(res, 4),
            effort_allocation=round(eff, 4),
            composite=round(composite, 4),
            decision_count=len(self._decisions),
        )
