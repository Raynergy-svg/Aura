"""US-342: Journal Reflection Quality Scorer — depth, causal density, pre-mortem presence.

Research (British Journal of Ed Tech 2025, Frontiers Psychology 2021) shows reflective
writing quality predicts future decision quality. Structured journaling with causal
analysis and pre-mortems improves performance by ~22.8%.

Depth levels:
  L1 = Summary only ("took trade, lost")
  L2 = Some reasoning ("thought it would breakout")
  L3 = Causal analysis ("misjudged volatility because...")
  L4 = Meta-reflection ("this pattern shows I overestimate edge in choppy markets")

reflection_quality = 0.50 * (depth_level / 4.0) + 0.30 * causal_density + 0.20 * premortem
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class JournalReflectionResult:
    """Result from journal reflection scoring."""
    depth_level: int          # 1-4
    causal_density: float     # 0-1
    premortem_present: bool
    reflection_quality: float  # 0-1


class JournalReflectionScorer:
    """Scores journal/conversation reflection quality.

    reflection_quality = 0.50 * (depth_level / 4.0) + 0.30 * causal_density + 0.20 * premortem
    """

    # Level 3: Causal analysis markers
    CAUSAL_MARKERS = [
        "because", "caused by", "led to", "as a result", "due to",
        "reason was", "the cause", "resulted in", "triggered by",
        "which meant", "therefore", "consequently", "since i",
    ]

    # Level 4: Meta-reflection markers
    META_MARKERS = [
        "pattern shows", "i notice", "this tells me", "lesson learned",
        "i tend to", "recurring theme", "i keep", "my habit of",
        "looking back", "bigger picture", "overall i see",
        "this reflects", "consistently i", "systematic issue",
    ]

    # Level 2: Basic reasoning markers
    REASONING_MARKERS = [
        "thought", "expected", "figured", "believed", "assumed",
        "my view was", "anticipated", "plan was", "idea was",
        "strategy was", "approach was", "target was",
    ]

    # Pre-mortem phrases
    PREMORTEM_PHRASES = [
        "if this fails", "risk is", "what could go wrong",
        "worst case", "downside is", "could lose", "danger is",
        "if it doesn't work", "potential failure", "my concern is",
        "the risk here", "if price goes against",
    ]

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text (simple split on .!?)."""
        sentences = re.split(r'[.!?]+', text)
        return max(1, sum(1 for s in sentences if s.strip()))

    def _count_causal_statements(self, text: str) -> int:
        """Count causal language markers in text."""
        lower = text.lower()
        return sum(1 for m in self.CAUSAL_MARKERS if m in lower)

    def _classify_depth(self, text: str) -> int:
        """Classify text into depth level 1-4."""
        lower = text.lower()

        # Check L4 first (meta-reflection)
        meta_count = sum(1 for m in self.META_MARKERS if m in lower)
        if meta_count >= 1:
            return 4

        # Check L3 (causal analysis)
        causal_count = self._count_causal_statements(text)
        if causal_count >= 1:
            return 3

        # Check L2 (basic reasoning)
        reasoning_count = sum(1 for m in self.REASONING_MARKERS if m in lower)
        if reasoning_count >= 1:
            return 2

        # L1 (summary only)
        return 1

    def _has_premortem(self, text: str) -> bool:
        """Check if text contains pre-mortem thinking."""
        lower = text.lower()
        return any(p in lower for p in self.PREMORTEM_PHRASES)

    def score(self, text: str) -> JournalReflectionResult:
        """Score journal reflection quality.

        Args:
            text: Journal/conversation text to analyze

        Returns:
            JournalReflectionResult with depth, causal density, premortem, and quality
        """
        if not text or not text.strip():
            return JournalReflectionResult(
                depth_level=1,
                causal_density=0.0,
                premortem_present=False,
                reflection_quality=0.0,
            )

        depth = self._classify_depth(text)
        total_sentences = self._count_sentences(text)
        causal_count = self._count_causal_statements(text)
        causal_density = min(causal_count / max(1, total_sentences), 1.0)
        premortem = self._has_premortem(text)

        reflection_quality = (
            0.50 * (depth / 4.0)
            + 0.30 * causal_density
            + 0.20 * (1.0 if premortem else 0.0)
        )

        return JournalReflectionResult(
            depth_level=depth,
            causal_density=round(causal_density, 4),
            premortem_present=premortem,
            reflection_quality=round(reflection_quality, 4),
        )
