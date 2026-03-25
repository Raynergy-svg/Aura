"""Multi-signal decision fatigue index.

Based on neurobiology research (eNeuro 2024, JNeurosci 2025) showing decision
fatigue manifests across 6 measurable dimensions, not just override frequency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecisionFatigueResult:
    """Result of decision fatigue computation."""
    frequency_signal: float = 0.0      # 0-1: decision frequency acceleration
    patience_signal: float = 0.0       # 0-1: holding period compression
    quality_signal: float = 0.0        # 0-1: win rate degradation
    reward_tolerance: float = 0.0      # 0-1: R:R ratio decline
    emotional_variability: float = 0.0 # 0-1: sentiment std_dev increase
    cognitive_decline: float = 0.0     # 0-1: linguistic complexity drop
    composite: float = 0.0            # 0-100: weighted composite


class DecisionFatigueIndex:
    """Computes multi-dimensional decision fatigue from trading and behavioral data."""

    def __init__(self, baseline_window: int = 20):
        """
        Args:
            baseline_window: Number of initial entries to establish baselines
        """
        self.baseline_window = baseline_window
        self._trade_frequencies: List[float] = []      # trades per hour
        self._holding_periods: List[float] = []         # hours held
        self._win_losses: List[bool] = []               # True=win, False=loss
        self._reward_ratios: List[float] = []           # R:R ratios
        self._sentiments: List[float] = []              # sentiment scores
        self._complexities: List[float] = []            # linguistic complexity scores
        self._baselines_computed = False
        self._baseline_freq = 0.5
        self._baseline_hold = 4.0
        self._baseline_rr = 1.5
        self._baseline_complexity = 0.5

    def _compute_baselines(self):
        """Compute baselines from first N entries."""
        if len(self._trade_frequencies) >= self.baseline_window:
            window = self._trade_frequencies[:self.baseline_window]
            self._baseline_freq = sum(window) / len(window) if window else 0.5
            self._baseline_freq = max(0.1, self._baseline_freq)  # floor

        if len(self._holding_periods) >= self.baseline_window:
            window = self._holding_periods[:self.baseline_window]
            self._baseline_hold = sum(window) / len(window) if window else 4.0
            self._baseline_hold = max(0.1, self._baseline_hold)

        if len(self._reward_ratios) >= self.baseline_window:
            window = self._reward_ratios[:self.baseline_window]
            self._baseline_rr = sum(window) / len(window) if window else 1.5
            self._baseline_rr = max(0.1, self._baseline_rr)

        if len(self._complexities) >= self.baseline_window:
            window = self._complexities[:self.baseline_window]
            self._baseline_complexity = sum(window) / len(window) if window else 0.5
            self._baseline_complexity = max(0.1, self._baseline_complexity)

        self._baselines_computed = True

    def update(self,
               trade_frequency: Optional[float] = None,
               holding_period: Optional[float] = None,
               win: Optional[bool] = None,
               reward_ratio: Optional[float] = None,
               sentiment: Optional[float] = None,
               complexity: Optional[float] = None) -> DecisionFatigueResult:
        """Update with new data and compute fatigue index.

        All parameters are optional — missing signals use neutral defaults.
        """
        # Append available data
        if trade_frequency is not None:
            self._trade_frequencies.append(trade_frequency)
        if holding_period is not None:
            self._holding_periods.append(holding_period)
        if win is not None:
            self._win_losses.append(win)
        if reward_ratio is not None:
            self._reward_ratios.append(reward_ratio)
        if sentiment is not None:
            self._sentiments.append(sentiment)
        if complexity is not None:
            self._complexities.append(complexity)

        # Recompute baselines if enough data
        total_entries = max(
            len(self._trade_frequencies),
            len(self._holding_periods),
            len(self._win_losses)
        )
        if not self._baselines_computed and total_entries >= self.baseline_window:
            self._compute_baselines()

        return self._compute()

    def _compute(self) -> DecisionFatigueResult:
        """Compute all 6 fatigue signals and composite."""
        # Recent window (last 10)
        recent_n = 10

        # 1. Frequency signal: trades/hour above baseline
        freq_signal = 0.0
        if self._trade_frequencies:
            recent_freq = self._trade_frequencies[-recent_n:]
            avg_freq = sum(recent_freq) / len(recent_freq)
            freq_signal = min(1.0, max(0.0, avg_freq / self._baseline_freq - 1.0))

        # 2. Patience signal: holding period compression
        patience_signal = 0.0
        if self._holding_periods:
            recent_hold = self._holding_periods[-recent_n:]
            avg_hold = sum(recent_hold) / len(recent_hold)
            patience_signal = min(1.0, max(0.0, 1.0 - avg_hold / self._baseline_hold))

        # 3. Quality signal: win rate degradation
        quality_signal = 0.0
        if self._win_losses:
            recent_wl = self._win_losses[-recent_n:]
            win_rate = sum(1 for w in recent_wl if w) / len(recent_wl)
            quality_signal = min(1.0, max(0.0, 1.0 - win_rate))

        # 4. Reward tolerance: R:R decline
        reward_signal = 0.0
        if self._reward_ratios:
            recent_rr = self._reward_ratios[-recent_n:]
            avg_rr = sum(recent_rr) / len(recent_rr)
            reward_signal = min(1.0, max(0.0, 1.0 - avg_rr / self._baseline_rr))

        # 5. Emotional variability: sentiment std_dev
        emotional_signal = 0.0
        if len(self._sentiments) >= 3:
            recent_sent = self._sentiments[-recent_n:]
            mean_sent = sum(recent_sent) / len(recent_sent)
            variance = sum((s - mean_sent) ** 2 for s in recent_sent) / len(recent_sent)
            std_dev = math.sqrt(variance)
            emotional_signal = min(1.0, std_dev / 0.5)  # normalize: 0.5 std = max

        # 6. Cognitive decline: complexity drop
        cognitive_signal = 0.0
        if self._complexities:
            recent_comp = self._complexities[-recent_n:]
            avg_comp = sum(recent_comp) / len(recent_comp)
            cognitive_signal = min(1.0, max(0.0, 1.0 - avg_comp / self._baseline_complexity))

        # Weighted composite (0-100)
        composite = (
            0.25 * freq_signal +
            0.20 * patience_signal +
            0.20 * quality_signal +
            0.15 * reward_signal +
            0.10 * emotional_signal +
            0.10 * cognitive_signal
        ) * 100.0

        composite = min(100.0, max(0.0, composite))

        if composite > 70:
            logger.warning(
                "US-352: High decision fatigue detected — composite=%.1f "
                "(freq=%.2f, patience=%.2f, quality=%.2f, reward=%.2f, emotional=%.2f, cognitive=%.2f)",
                composite, freq_signal, patience_signal, quality_signal,
                reward_signal, emotional_signal, cognitive_signal
            )

        return DecisionFatigueResult(
            frequency_signal=round(freq_signal, 3),
            patience_signal=round(patience_signal, 3),
            quality_signal=round(quality_signal, 3),
            reward_tolerance=round(reward_signal, 3),
            emotional_variability=round(emotional_signal, 3),
            cognitive_decline=round(cognitive_signal, 3),
            composite=round(composite, 1)
        )
