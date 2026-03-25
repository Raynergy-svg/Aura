"""Continuous valence-arousal affect tracking with emotional inertia detection.

US-351: Based on Russell's circumplex model and WELD 2025 workplace emotion dynamics research.
Tracks emotions on a continuous 2D (valence, arousal) plane and computes temporal
dynamics: inertia (persistence), volatility (variability), and stuck states.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AffectDynamicsResult:
    """Result of affect dynamics computation."""

    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (activated)
    inertia: float  # 0-1, persistence of emotional state (lag-1 autocorrelation)
    volatility: float  # 0-1, variability of emotional magnitude
    stuck_state: bool  # True if stuck in negative emotional state


# Urgency keywords for arousal detection
URGENCY_KEYWORDS = {
    "urgent",
    "asap",
    "immediately",
    "now",
    "hurry",
    "rush",
    "quick",
    "fast",
    "critical",
    "emergency",
}

# Stressed states for granular interaction
STRESSED_STATES = {
    "stressed",
    "anxious",
    "frustrated",
    "overwhelmed",
    "angry",
    "panicked",
    "fearful",
}


class AffectDynamicsTracker:
    """Continuous valence-arousal affect tracking with emotional dynamics.

    Maintains a sliding window of (valence, arousal) tuples and computes:
    - Inertia: lag-1 autocorrelation of valence (persistence)
    - Volatility: standard deviation of emotional magnitude sqrt(v² + a²)
    - Stuck state: detection of prolonged negative affect with high persistence

    Each call to update() processes text to extract:
    - Valence from VADER compound score or keyword fallback
    - Arousal from punctuation, caps, urgency, and stressed keywords
    """

    def __init__(self, window_size: int = 10):
        """Initialize tracker with empty history.

        Args:
            window_size: Number of (valence, arousal) tuples to retain (default 10)
        """
        self.window_size = window_size
        self._valence_history: deque = deque(maxlen=window_size)
        self._arousal_history: deque = deque(maxlen=window_size)
        self._consecutive_negative: int = 0

    def update(self, text: str, vader_compound: float = 0.0) -> AffectDynamicsResult:
        """Update affect tracking with new text and compute dynamics.

        Args:
            text: The message text to analyze
            vader_compound: VADER compound score (-1 to +1), or 0.0 if not available

        Returns:
            AffectDynamicsResult with valence, arousal, inertia, volatility, stuck_state
        """
        if not text or not text.strip():
            return AffectDynamicsResult(
                valence=0.0,
                arousal=0.0,
                inertia=0.0,
                volatility=0.0,
                stuck_state=False,
            )

        # --- VALENCE: Use vader_compound or keyword fallback ---
        if vader_compound != 0.0:
            valence = vader_compound  # Already in [-1, +1]
        else:
            # Simple keyword-based fallback
            valence = self._estimate_valence_from_text(text)

        # --- AROUSAL: Compute from text features ---
        arousal = self._compute_arousal(text)

        # --- INERTIA: Lag-1 autocorrelation of valence ---
        if len(self._valence_history) >= 2:
            inertia = self._compute_pearson_correlation(
                list(self._valence_history)[:-1], list(self._valence_history)[1:]
            )
            inertia = max(0.0, min(1.0, inertia))  # Clamp to [0, 1]
        else:
            inertia = 0.0

        # --- VOLATILITY: Std dev of emotional magnitude ---
        if len(self._valence_history) >= 2:
            magnitudes = []
            for v, a in zip(self._valence_history, self._arousal_history):
                mag = math.sqrt(v * v + a * a)
                magnitudes.append(mag)

            if len(magnitudes) >= 2:
                mean_mag = sum(magnitudes) / len(magnitudes)
                variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
                std_dev = math.sqrt(variance)
                # Normalize by dividing by sqrt(2) (max possible magnitude)
                volatility = std_dev / math.sqrt(2)
                volatility = max(0.0, min(1.0, volatility))  # Clamp to [0, 1]
            else:
                volatility = 0.0
        else:
            volatility = 0.0

        # --- STUCK STATE: Inertia > 0.8 AND mean valence < -0.3 AND consecutive_negative >= 5 ---
        mean_valence = (
            sum(self._valence_history) / len(self._valence_history)
            if self._valence_history
            else 0.0
        )

        # Track consecutive negative values
        if valence < -0.2:
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0

        stuck_state = (
            inertia > 0.8
            and mean_valence < -0.3
            and self._consecutive_negative >= 5
        )

        # Append to histories
        self._valence_history.append(valence)
        self._arousal_history.append(arousal)

        logger.debug(
            "US-351: affect_dynamics valence=%.3f arousal=%.3f inertia=%.3f volatility=%.3f stuck=%s",
            valence,
            arousal,
            inertia,
            volatility,
            stuck_state,
        )

        return AffectDynamicsResult(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            inertia=round(inertia, 4),
            volatility=round(volatility, 4),
            stuck_state=stuck_state,
        )

    def _estimate_valence_from_text(self, text: str) -> float:
        """Keyword-based valence estimation (fallback when VADER unavailable).

        Returns:
            Valence in [-1, +1] range
        """
        text_lower = text.lower()
        from src.aura.core.conversation_processor import (
            POSITIVE_KEYWORDS,
            STRESS_KEYWORDS,
        )

        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        stress_count = sum(1 for kw in STRESS_KEYWORDS if kw in text_lower)

        # Simple balance: positive vs negative
        net_score = positive_count - stress_count
        # Normalize to [-1, +1]
        # Assume up to ~10 keywords per message as reasonable range
        valence = net_score / 10.0
        return max(-1.0, min(1.0, valence))

    def _compute_arousal(self, text: str) -> float:
        """Compute arousal from text features (punctuation, caps, urgency).

        Returns:
            Arousal in [0, 1] range
        """
        if not text or not text.strip():
            return 0.0

        words = text.split()
        word_count = max(len(words), 1)

        # Exclamation density
        exclamation_count = text.count("!")
        exclamation_density = exclamation_count / word_count

        # Caps ratio
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)

        # Question density
        question_count = text.count("?")
        question_density = question_count / word_count

        # Urgency keywords
        urgency_count = 0
        for w in words:
            # Strip punctuation from word
            word_clean = w.strip(".,!?;:")
            if word_clean.lower() in URGENCY_KEYWORDS:
                urgency_count += 1
        urgency_density = urgency_count / word_count

        # Stressed state keywords (extra arousal boost)
        stressed_count = 0
        for w in words:
            word_clean = w.strip(".,!?;:")
            if word_clean.lower() in STRESSED_STATES:
                stressed_count += 1
        stressed_density = stressed_count / word_count

        # Weighted combination
        arousal = (
            exclamation_density * 0.4
            + caps_ratio * 0.3
            + question_density * 0.2
            + urgency_density * 0.1
            + stressed_density * 0.15
        )

        # Clamp to [0, 1]
        arousal = max(0.0, min(1.0, arousal))
        return arousal

    @staticmethod
    def _compute_pearson_correlation(x_list: List[float], y_list: List[float]) -> float:
        """Compute Pearson correlation coefficient manually (no scipy dependency).

        Args:
            x_list: First sequence
            y_list: Second sequence

        Returns:
            Pearson r in [-1, 1], or 0.0 if computation fails
        """
        if len(x_list) < 2 or len(y_list) < 2 or len(x_list) != len(y_list):
            return 0.0

        n = len(x_list)

        # Compute means
        mean_x = sum(x_list) / n
        mean_y = sum(y_list) / n

        # Compute covariance and standard deviations
        cov_xy = sum(
            (x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(n)
        ) / n
        var_x = sum((x - mean_x) ** 2 for x in x_list) / n
        var_y = sum((y - mean_y) ** 2 for y in y_list) / n

        std_x = math.sqrt(var_x)
        std_y = math.sqrt(var_y)

        # Avoid division by zero
        if std_x == 0.0 or std_y == 0.0:
            return 0.0

        # Pearson r = cov(x,y) / (std(x) * std(y))
        r = cov_xy / (std_x * std_y)
        return r
