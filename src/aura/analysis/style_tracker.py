"""Temporal Linguistic Style Tracking — measures writing style changes over time.

US-333: Linguistic style drift detection for Aura's cognitive load assessment.

Tracks five linguistic metrics per message:
    - avg_sentence_length: words per sentence
    - exclamation_density: exclamations per sentence
    - caps_ratio: uppercase characters / total characters (excl. spaces)
    - pronoun_i_ratio: count of standalone "I" / word count
    - question_ratio: question marks per sentence

Computes drift as Euclidean distance between current snapshot and rolling
10-message baseline, normalized to roughly 0-1 range and clamped.

Baseline starts after 3 messages (returns 0.0 before that).
Rolling window stores last 20 StyleSnapshot entries (FIFO).
Drift > 0.5 is logged as warning.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StyleSnapshot:
    """Single snapshot of linguistic style metrics for one message.

    Attributes:
        avg_sentence_length: Average words per sentence (total_words / sentence_count).
        exclamation_density: Exclamation marks per sentence (exclamation_count / sentence_count).
        caps_ratio: Uppercase characters / total non-space characters (0-1).
        pronoun_i_ratio: Count of standalone "I" / total words (0-1).
        question_ratio: Question marks per sentence (question_count / sentence_count).
    """
    avg_sentence_length: float = 0.0
    exclamation_density: float = 0.0
    caps_ratio: float = 0.0
    pronoun_i_ratio: float = 0.0
    question_ratio: float = 0.0

    def to_vector(self) -> tuple[float, float, float, float, float]:
        """Convert snapshot to a 5-element numeric vector for distance computation."""
        return (
            self.avg_sentence_length,
            self.exclamation_density,
            self.caps_ratio,
            self.pronoun_i_ratio,
            self.question_ratio,
        )


class LinguisticStyleTracker:
    """Tracks linguistic style changes over a rolling window of messages.

    Maintains a FIFO window of the last 20 StyleSnapshot entries and computes
    drift (distance from baseline) as messages are added. Baseline is computed
    from the oldest 10 snapshots; drift returns 0.0 until at least 3 messages
    have been tracked.
    """

    def __init__(self, window_size: int = 20, baseline_size: int = 10):
        """Initialize the style tracker.

        Args:
            window_size: Maximum number of snapshots to store (default 20).
            baseline_size: Number of oldest messages to use for baseline (default 10).
                Must be <= window_size. If fewer messages exist, baseline is not ready.
        """
        self.window_size = window_size
        self.baseline_size = baseline_size
        self.window: deque[StyleSnapshot] = deque(maxlen=window_size)
        logger.debug(
            f"LinguisticStyleTracker initialized: window_size={window_size}, "
            f"baseline_size={baseline_size}"
        )

    def track_message(self, message_text: str) -> StyleSnapshot:
        """Analyze a single message and add its snapshot to the rolling window.

        Args:
            message_text: The message text to analyze.

        Returns:
            StyleSnapshot with five computed metrics.
        """
        snapshot = self._compute_snapshot(message_text)
        self.window.append(snapshot)
        logger.debug(
            f"Tracked message snapshot: "
            f"avg_sent_len={snapshot.avg_sentence_length:.2f}, "
            f"excl_dens={snapshot.exclamation_density:.3f}, "
            f"caps={snapshot.caps_ratio:.3f}, "
            f"pronoun_i={snapshot.pronoun_i_ratio:.3f}, "
            f"question={snapshot.question_ratio:.3f} "
            f"(window size now {len(self.window)}/{self.window_size})"
        )
        return snapshot

    def compute_drift(self) -> float:
        """Compute Euclidean distance of current snapshot from baseline.

        Baseline is the mean of each metric over the oldest 10 messages.
        Returns 0.0 if fewer than (baseline_size + 1) messages exist
        (not enough history to compute a baseline and a current snapshot).

        Returns:
            Drift score (0-1 roughly). Clamped to [0, 1].
        """
        # Need at least baseline_size + 1 messages to compute drift
        # (baseline_size oldest for baseline, plus current)
        if len(self.window) < self.baseline_size + 1:
            logger.debug(
                f"Not enough history for drift: {len(self.window)} < "
                f"{self.baseline_size + 1}; returning 0.0"
            )
            return 0.0

        # Baseline: mean of oldest baseline_size snapshots
        baseline_snapshots = list(self.window)[: self.baseline_size]
        baseline_means = self._compute_means(baseline_snapshots)

        # Current: most recent snapshot
        current_snapshot = self.window[-1]
        current_vector = current_snapshot.to_vector()

        # Euclidean distance
        euclidean_dist = math.sqrt(
            sum((c - b) ** 2 for c, b in zip(current_vector, baseline_means))
        )

        # Normalize: divide by sqrt(5) to map to roughly 0-1 range
        normalized_drift = euclidean_dist / math.sqrt(5)

        # Clamp to [0, 1]
        drift = max(0.0, min(1.0, normalized_drift))

        if drift > 0.5:
            logger.warning(
                f"High linguistic style drift detected: {drift:.3f} > 0.5 "
                f"(baseline size: {len(baseline_snapshots)}, current snapshot variance)"
            )

        logger.debug(
            f"Drift computed: raw={euclidean_dist:.3f}, normalized={normalized_drift:.3f}, "
            f"clamped={drift:.3f}"
        )
        return drift

    @staticmethod
    def _compute_snapshot(text: str) -> StyleSnapshot:
        """Compute all five metrics for a single message.

        Args:
            text: The message text.

        Returns:
            StyleSnapshot with all metrics computed.
        """
        if not text or not text.strip():
            return StyleSnapshot()

        # Split into words
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return StyleSnapshot()

        # Count sentences (split on .!?)
        sentence_pattern = r'[.!?]'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        # Avoid division by zero
        if sentence_count == 0:
            sentence_count = 1

        # avg_sentence_length: total words / sentence count
        avg_sentence_length = word_count / sentence_count

        # exclamation_density: exclamation count / sentence count
        exclamation_count = text.count('!')
        exclamation_density = exclamation_count / max(sentence_count, 1)

        # caps_ratio: uppercase chars / non-space chars
        non_space_chars = ''.join(text.split())
        uppercase_count = sum(1 for c in non_space_chars if c.isupper())
        caps_ratio = (
            uppercase_count / len(non_space_chars)
            if non_space_chars else 0.0
        )

        # pronoun_i_ratio: count of standalone "I" / word count
        # Standalone "I" = surrounded by word boundaries
        i_pattern = r'\bI\b'
        i_count = len(re.findall(i_pattern, text, re.IGNORECASE))
        pronoun_i_ratio = i_count / max(word_count, 1)

        # question_ratio: question marks / sentence count
        question_count = text.count('?')
        question_ratio = question_count / max(sentence_count, 1)

        return StyleSnapshot(
            avg_sentence_length=avg_sentence_length,
            exclamation_density=exclamation_density,
            caps_ratio=caps_ratio,
            pronoun_i_ratio=pronoun_i_ratio,
            question_ratio=question_ratio,
        )

    @staticmethod
    def _compute_means(snapshots: list[StyleSnapshot]) -> tuple[float, float, float, float, float]:
        """Compute mean values for each metric across a list of snapshots.

        Args:
            snapshots: List of StyleSnapshot objects.

        Returns:
            Tuple of (avg_sent_len, excl_dens, caps, pronoun_i, question).
        """
        if not snapshots:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        vectors = [s.to_vector() for s in snapshots]
        n = len(vectors)

        means = tuple(
            sum(v[i] for v in vectors) / n
            for i in range(5)
        )
        return means
