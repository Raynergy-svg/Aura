from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from collections import deque
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityResult:
    """Dataclass holding reliability analysis results."""
    cronbachs_alpha: float
    split_half_reliability: float
    reliability_score: float
    sample_count: int
    sufficient_data: bool


class ReadinessReliabilityAnalyzer:
    """
    Computes internal reliability metrics for the readiness scoring system.

    Uses Cronbach's alpha and split-half reliability to assess whether the
    6 readiness components are measuring a single underlying construct
    (internal consistency). Tracks component snapshots over time and produces
    a composite reliability score used to gate confidence in readiness signals.
    """

    def __init__(self, max_snapshots: int = 100):
        """
        Initialize the analyzer.

        Args:
            max_snapshots: Maximum number of component snapshots to retain (FIFO).
        """
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        logger.debug(f"ReadinessReliabilityAnalyzer initialized with max_snapshots={max_snapshots}")

    def record_components(self, components_dict: Dict[str, float]) -> None:
        """
        Record a snapshot of component values.

        Args:
            components_dict: Dictionary mapping component names to float values (typically 0-1).

        Returns:
            None
        """
        if not components_dict:
            logger.warning("record_components called with empty dict")
            return

        # Store as a list of values in consistent order
        snapshot = list(components_dict.values())
        self.snapshots.append(snapshot)
        logger.debug(f"Recorded component snapshot (total: {len(self.snapshots)})")

    def cronbachs_alpha(self, window: int = 20) -> float:
        """
        Compute Cronbach's alpha for internal consistency.

        Formula: α = (k / (k-1)) * (1 - sum(item_variances) / total_variance)

        Args:
            window: Number of recent snapshots to use (default 20).

        Returns:
            Cronbach's alpha value (0-1), or 0.7 if insufficient data.
        """
        if len(self.snapshots) < 10:
            logger.debug(f"Insufficient snapshots for Cronbach's alpha ({len(self.snapshots)} < 10), returning 0.7 default")
            return 0.7

        # Use last 'window' snapshots
        recent = list(self.snapshots)[-window:]

        # k = number of items (components)
        k = len(recent[0])  # Number of components per snapshot

        if k < 2:
            logger.warning("Cronbach's alpha requires at least 2 components")
            return 0.7

        # Compute variance for each component across snapshots
        item_variances = []
        for component_idx in range(k):
            values = [snapshot[component_idx] for snapshot in recent]
            var = self._compute_variance(values)
            item_variances.append(var)

        sum_item_variances = sum(item_variances)

        # Compute total variance (variance of sum of all components per snapshot)
        sums = [sum(snapshot) for snapshot in recent]
        total_variance = self._compute_variance(sums)

        # Handle edge case: perfect agreement
        if total_variance == 0:
            logger.debug("Total variance is 0, returning alpha=1.0 (perfect agreement)")
            return 1.0

        # Cronbach's alpha formula
        alpha = (k / (k - 1)) * (1 - sum_item_variances / total_variance)

        # Clamp to 0-1
        alpha = max(0.0, min(1.0, alpha))
        logger.debug(f"Cronbach's alpha computed: {alpha:.4f}")

        return alpha

    def split_half_reliability(self) -> float:
        """
        Compute split-half reliability with Spearman-Brown correction.

        Splits the 6 components into odd-indexed (0, 2, 4) and even-indexed (1, 3, 5)
        groups, computes mean scores for each half across snapshots, then applies
        Pearson correlation and Spearman-Brown correction.

        Returns:
            Corrected split-half reliability (0-1), or 0.7 if insufficient data.
        """
        if len(self.snapshots) < 10:
            logger.debug(f"Insufficient snapshots for split-half ({len(self.snapshots)} < 10), returning 0.7 default")
            return 0.7

        recent = list(self.snapshots)

        # Split into odd and even indices
        odd_scores = []  # indices 0, 2, 4
        even_scores = []  # indices 1, 3, 5

        for snapshot in recent:
            odd_mean = sum(snapshot[i] for i in range(0, len(snapshot), 2)) / (len(snapshot) // 2 + len(snapshot) % 2)
            even_mean = sum(snapshot[i] for i in range(1, len(snapshot), 2)) / (len(snapshot) // 2)
            odd_scores.append(odd_mean)
            even_scores.append(even_mean)

        # Compute Pearson correlation between the two half-score series
        r = self._pearson_correlation(odd_scores, even_scores)

        # Handle undefined correlation (zero variance)
        if r is None:
            logger.debug("Pearson correlation undefined (zero variance), returning split_half=0.5")
            return 0.5

        # Apply Spearman-Brown correction: r_sb = 2*r / (1 + r)
        if r == 1.0:
            # Perfect correlation: correction would be 2.0 / 2.0 = 1.0
            r_sb = 1.0
        elif r <= -1.0:
            # Perfect negative correlation: denominator would be zero, treat as zero reliability
            r_sb = 0.0
        else:
            r_sb = (2 * r) / (1 + r)

        # Clamp to 0-1
        r_sb = max(0.0, min(1.0, r_sb))
        logger.debug(f"Split-half reliability (with Spearman-Brown): {r_sb:.4f}")

        return r_sb

    @property
    def reliability_score(self) -> float:
        """
        Composite reliability score (weighted average).

        Returns:
            0.6 * cronbachs_alpha + 0.4 * split_half_reliability (0-1)
        """
        alpha = self.cronbachs_alpha()
        split_half = self.split_half_reliability()
        score = 0.6 * alpha + 0.4 * split_half
        return max(0.0, min(1.0, score))

    def compute(self) -> ReliabilityResult:
        """
        Compute all reliability metrics and return as dataclass.

        Returns:
            ReliabilityResult with alpha, split_half, composite score, and flags.
        """
        alpha = self.cronbachs_alpha()
        split_half = self.split_half_reliability()
        composite = self.reliability_score
        sample_count = len(self.snapshots)
        sufficient_data = sample_count >= 10

        result = ReliabilityResult(
            cronbachs_alpha=alpha,
            split_half_reliability=split_half,
            reliability_score=composite,
            sample_count=sample_count,
            sufficient_data=sufficient_data
        )

        logger.info(
            f"Reliability analysis complete: alpha={alpha:.4f}, split_half={split_half:.4f}, "
            f"composite={composite:.4f}, samples={sample_count}, sufficient={sufficient_data}"
        )

        return result

    # ────── Helper Methods ──────

    @staticmethod
    def _compute_variance(values: list[float]) -> float:
        """
        Compute sample variance.

        Args:
            values: List of numeric values.

        Returns:
            Sample variance (uses n-1 denominator, or 0 if only 1 value).
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        sum_sq_diff = sum((x - mean) ** 2 for x in values)
        variance = sum_sq_diff / (len(values) - 1)

        return variance

    @staticmethod
    def _pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
        """
        Compute Pearson correlation coefficient manually.

        Formula: r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2) * (n*sum_y2 - sum_y^2))

        Args:
            x_values: First series of values.
            y_values: Second series of values (must match length of x_values).

        Returns:
            Pearson r (-1 to +1), or None if correlation is undefined (zero variance).
        """
        if len(x_values) != len(y_values):
            logger.warning("Pearson correlation: mismatched series lengths")
            return None

        n = len(x_values)
        if n < 2:
            return None

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x ** 2 for x in x_values)
        sum_y2 = sum(y ** 2 for y in y_values)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

        # Handle zero variance case
        if denominator == 0:
            logger.debug("Pearson correlation denominator is zero (no variance in one or both series)")
            return None

        r = numerator / denominator

        # Clamp to -1 to +1 (due to floating-point precision)
        r = max(-1.0, min(1.0, r))

        return r
