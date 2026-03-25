"""US-348: Narrative Coherence Tracker — cross-session reasoning consistency."""
from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


@dataclass
class NarrativeCoherenceResult:
    """Result of narrative coherence analysis."""
    lexical_overlap: float  # 0-1, Jaccard similarity of content words
    sentiment_consistency: float  # 0-1, 1 - 2*std_dev(sentiments)
    strategy_persistence: float  # 0-1, overlap of strategy terms
    composite: float  # 0-1, weighted combination


# Common English stopwords to exclude from lexical analysis
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "up", "about", "also", "well",
}

# Trading strategy terms to track persistence
STRATEGY_TERMS = {
    "breakout", "pullback", "trend", "reversal", "momentum", "scalp",
    "swing", "position", "mean reversion", "support", "resistance",
    "fibonacci", "moving average", "rsi", "macd", "volume", "divergence",
    "consolidation", "accumulation", "distribution", "channel", "range",
    "stop loss", "take profit", "risk reward", "entry", "exit",
    "hedge", "correlation", "volatility", "atr", "bollinger",
}


class NarrativeCoherenceTracker:
    """
    Tracks reasoning consistency across sessions.

    Measures lexical overlap, sentiment consistency, and strategy
    persistence to assess how stable the trader's decision framework is.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._session_words: deque = deque(maxlen=window_size)  # List of word sets
        self._session_sentiments: deque = deque(maxlen=window_size)  # List of sentiment floats
        self._session_strategies: deque = deque(maxlen=window_size)  # List of strategy sets

    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract content words (excluding stopwords) from text."""
        if not text:
            return set()
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        return words - STOPWORDS

    def _extract_strategy_terms(self, text: str) -> Set[str]:
        """Extract trading strategy terms from text."""
        if not text:
            return set()
        text_lower = text.lower()
        found = set()
        for term in STRATEGY_TERMS:
            if term in text_lower:
                found.add(term)
        return found

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def update(self, text: str, sentiment: float = 0.5) -> NarrativeCoherenceResult:
        """
        Process a new session and compute coherence metrics.

        Args:
            text: The session's aggregated message text.
            sentiment: The session's overall sentiment score (0-1).

        Returns:
            NarrativeCoherenceResult with all metrics.
        """
        current_words = self._extract_content_words(text)
        current_strategies = self._extract_strategy_terms(text)

        # Compute metrics against previous sessions
        if len(self._session_words) == 0:
            # First session — no comparison possible
            self._session_words.append(current_words)
            self._session_sentiments.append(sentiment)
            self._session_strategies.append(current_strategies)
            return NarrativeCoherenceResult(
                lexical_overlap=0.5,  # Default neutral
                sentiment_consistency=0.5,
                strategy_persistence=0.5,
                composite=0.5,
            )

        # --- Lexical overlap: Jaccard similarity vs concatenated previous sessions ---
        prev_words = set()
        for ws in self._session_words:
            prev_words |= ws
        lexical_overlap = self._jaccard_similarity(current_words, prev_words)

        # --- Sentiment consistency: 1 - 2 * std_dev of recent sentiments ---
        all_sentiments = list(self._session_sentiments) + [sentiment]
        if len(all_sentiments) >= 2:
            mean_s = sum(all_sentiments) / len(all_sentiments)
            variance = sum((s - mean_s) ** 2 for s in all_sentiments) / (len(all_sentiments) - 1)
            std_dev = math.sqrt(variance)
            sentiment_consistency = max(0.0, 1.0 - 2.0 * std_dev)
        else:
            sentiment_consistency = 0.5

        # --- Strategy persistence: overlap of strategy terms ---
        prev_strategies = set()
        for ss in self._session_strategies:
            prev_strategies |= ss
        if current_strategies and prev_strategies:
            strategy_persistence = self._jaccard_similarity(current_strategies, prev_strategies)
        elif not current_strategies and not prev_strategies:
            strategy_persistence = 0.5  # Neither session mentioned strategies
        else:
            strategy_persistence = 0.0  # One mentions strategies, other doesn't

        # --- Composite ---
        composite = (
            0.35 * lexical_overlap
            + 0.35 * sentiment_consistency
            + 0.30 * strategy_persistence
        )
        composite = max(0.0, min(1.0, composite))

        # Store current session
        self._session_words.append(current_words)
        self._session_sentiments.append(sentiment)
        self._session_strategies.append(current_strategies)

        result = NarrativeCoherenceResult(
            lexical_overlap=round(lexical_overlap, 4),
            sentiment_consistency=round(sentiment_consistency, 4),
            strategy_persistence=round(strategy_persistence, 4),
            composite=round(composite, 4),
        )

        logger.debug(
            "US-348: Coherence — lexical=%.3f, sentiment=%.3f, strategy=%.3f, composite=%.3f",
            lexical_overlap, sentiment_consistency, strategy_persistence, composite,
        )

        return result
