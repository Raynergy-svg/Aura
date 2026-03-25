"""US-340: Cognitive Flexibility Scorer — measures belief updating and strategy adaptation.

Cognitive flexibility — the ability to update beliefs when evidence contradicts them —
is one of the strongest predictors of trading success (Dajani & Uddin 2015, Frontiers
Neuroscience 2024). Distinct from confirmation bias (which detects rigidity) by measuring
the POSITIVE signal: does the trader actively revise their view?

Three metrics:
  1. belief_update (0.40) — language indicating changed beliefs
  2. strategy_adaptation (0.35) — language indicating adjusted approach
  3. evidence_acknowledgment (0.25) — language acknowledging contradictory evidence
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# --- Negation awareness (reuses pattern from US-336) ---
NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "didn't", "isn't", "won't",
    "can't", "haven't", "shouldn't", "neither", "nor", "without",
    "doesn't", "wasn't", "weren't", "wouldn't", "couldn't",
})


def _is_negated(text: str, phrase: str) -> bool:
    """Check if a phrase is negated in the text (negation word within 3 tokens before)."""
    lower_text = text.lower()
    phrase_lower = phrase.lower()
    idx = lower_text.find(phrase_lower)
    if idx < 0:
        return False
    # Get up to 3 tokens before the phrase
    prefix = lower_text[:idx].strip()
    tokens = prefix.split()
    check_tokens = tokens[-3:] if len(tokens) >= 3 else tokens
    return any(t.strip(",.!?;:") in NEGATION_WORDS for t in check_tokens)


def _count_non_negated_phrases(text: str, phrases: List[str]) -> int:
    """Count phrases that appear in text without being negated."""
    lower_text = text.lower()
    count = 0
    for phrase in phrases:
        if phrase.lower() in lower_text and not _is_negated(text, phrase):
            count += 1
    return count


@dataclass
class CognitiveFlexibilityResult:
    """Result from cognitive flexibility scoring."""
    belief_update: float        # 0-1
    strategy_adaptation: float  # 0-1
    evidence_acknowledgment: float  # 0-1
    composite: float            # 0-1


class CognitiveFlexibilityScorer:
    """Scores cognitive flexibility from conversation text.

    Composite = 0.40 * belief_update + 0.35 * strategy_adaptation + 0.25 * evidence_acknowledgment
    """

    BELIEF_UPDATE_PHRASES = [
        "changed my mind", "i was wrong", "updated my thesis",
        "reconsidered my position", "revised my view", "new information changed",
        "i need to rethink", "that changes things", "i stand corrected",
        "my initial read was off", "adjusting my bias", "flipping my outlook",
    ]

    STRATEGY_ADAPTATION_PHRASES = [
        "adjusted my approach", "switching to", "modified my plan",
        "different strategy", "changing my entry", "adapting my method",
        "trying a new approach", "pivoting to", "revised my setup",
        "altered my risk", "changing my timeframe", "scaling differently",
    ]

    EVIDENCE_ACKNOWLEDGMENT_PHRASES = [
        "despite my expectation", "contrary to what i thought",
        "the data shows otherwise", "evidence suggests different",
        "market is telling me", "price action contradicts",
        "the chart shows the opposite", "against my initial view",
        "i have to respect the price", "numbers don't support",
        "invalidated my thesis", "proved me wrong",
    ]

    def score(self, text: str) -> CognitiveFlexibilityResult:
        """Score cognitive flexibility from conversation text.

        Args:
            text: Conversation/journal text to analyze

        Returns:
            CognitiveFlexibilityResult with 3 sub-scores and composite
        """
        if not text or not text.strip():
            return CognitiveFlexibilityResult(
                belief_update=0.0,
                strategy_adaptation=0.0,
                evidence_acknowledgment=0.0,
                composite=0.0,
            )

        belief_count = _count_non_negated_phrases(text, self.BELIEF_UPDATE_PHRASES)
        belief_score = min(belief_count / 2.0, 1.0)

        strategy_count = _count_non_negated_phrases(text, self.STRATEGY_ADAPTATION_PHRASES)
        strategy_score = min(strategy_count / 2.0, 1.0)

        evidence_count = _count_non_negated_phrases(text, self.EVIDENCE_ACKNOWLEDGMENT_PHRASES)
        evidence_score = min(evidence_count / 2.0, 1.0)

        composite = (
            0.40 * belief_score
            + 0.35 * strategy_score
            + 0.25 * evidence_score
        )

        return CognitiveFlexibilityResult(
            belief_update=belief_score,
            strategy_adaptation=strategy_score,
            evidence_acknowledgment=evidence_score,
            composite=composite,
        )
