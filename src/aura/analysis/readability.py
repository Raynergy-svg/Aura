"""Text Readability Metrics — measures text clarity and vocabulary diversity.

US-332: Text Readability Analysis for Aura's linguistic tracking.

Computes multiple readability indices:
    - Flesch Reading Ease (0-100, higher = easier)
    - Gunning Fog Index (years of education, higher = harder)
    - Vocabulary Diversity (unique_words / total_words)
    - Composite Readability Score (0-1 normalized)

Normalized composite uses weighted average:
    - Flesch norm (0.4): flesch_reading_ease / 100, clamped 0-1
    - Fog norm (0.3): max(0, min(1, 1 - gunning_fog / 20)), inverted
    - Vocab diversity (0.3): unique_words / total_words, direct 0-1

Short messages (< 5 words) and empty text return neutral defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available; readability analysis will return neutral defaults")

logger = logging.getLogger(__name__)


@dataclass
class ReadabilityMetrics:
    """Container for text readability analysis results.

    Attributes:
        flesch_reading_ease: Raw Flesch Reading Ease score (0-100).
            0-30: Very difficult, 30-50: Difficult, 50-60: Fairly difficult,
            60-70: Standard, 70-80: Fairly easy, 80-90: Easy, 90-100: Very easy.
        gunning_fog: Gunning Fog Index (raw, typically 6-16).
            Represents years of education needed to understand text.
        vocabulary_diversity: Unique words / total words (0-1).
            Measures lexical richness; higher = more varied vocabulary.
        readability_score: Normalized composite score (0-1).
            Weighted average of normalized flesch, fog, and vocabulary metrics.
    """
    flesch_reading_ease: float = 0.5
    gunning_fog: float = 12.0
    vocabulary_diversity: float = 0.5
    readability_score: float = 0.5


class TextReadabilityAnalyzer:
    """Analyzes text readability using multiple indices.

    Combines Flesch Reading Ease, Gunning Fog Index, and vocabulary diversity
    into a single normalized 0-1 composite score. Handles edge cases gracefully.
    """

    def __init__(self):
        """Initialize the analyzer. Warns if textstat is unavailable."""
        if not TEXTSTAT_AVAILABLE:
            logger.warning(
                "TextReadabilityAnalyzer initialized without textstat library; "
                "will use pure-Python fallback"
            )

    def analyze(self, text: str) -> ReadabilityMetrics:
        """Analyze text readability and return composite metrics.

        Args:
            text: Input text to analyze. Can be empty or very short.

        Returns:
            ReadabilityMetrics with flesch_reading_ease, gunning_fog,
            vocabulary_diversity, and normalized readability_score.
        """
        # Handle empty or None input
        if not text or not text.strip():
            logger.debug("analyze() called with empty text; returning neutral defaults")
            return ReadabilityMetrics()

        text = text.strip()
        words = text.split()
        word_count = len(words)

        # Short messages (< 5 words) return neutral readability
        if word_count < 5:
            logger.debug(
                f"Text too short ({word_count} words < 5); returning neutral readability"
            )
            return ReadabilityMetrics(
                flesch_reading_ease=50.0,
                gunning_fog=12.0,
                vocabulary_diversity=self._compute_vocabulary_diversity(words),
                readability_score=0.5,
            )

        # If textstat is not available, compute using pure-Python fallback
        if not TEXTSTAT_AVAILABLE:
            logger.debug("textstat unavailable; using pure-Python Flesch/Fog computation")
            vocab_div = self._compute_vocabulary_diversity(words)
            flesch_reading_ease, gunning_fog = self._compute_readability_pure_python(text, words)
            flesch_norm = max(0.0, min(1.0, flesch_reading_ease / 100.0))
            fog_norm = max(0.0, min(1.0, 1.0 - gunning_fog / 20.0))
            readability_score = (
                0.4 * flesch_norm + 0.3 * fog_norm + 0.3 * vocab_div
            )
            return ReadabilityMetrics(
                flesch_reading_ease=flesch_reading_ease,
                gunning_fog=gunning_fog,
                vocabulary_diversity=vocab_div,
                readability_score=readability_score,
            )

        # Compute readability indices using textstat
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            gunning_fog = textstat.gunning_fog(text)
        except Exception as e:
            logger.warning(f"Error computing textstat metrics: {e}; returning neutral defaults")
            vocab_div = self._compute_vocabulary_diversity(words)
            return ReadabilityMetrics(
                flesch_reading_ease=50.0,
                gunning_fog=12.0,
                vocabulary_diversity=vocab_div,
                readability_score=0.5,
            )

        # Compute vocabulary diversity
        vocabulary_diversity = self._compute_vocabulary_diversity(words)

        # Normalize and composite
        flesch_norm = max(0.0, min(1.0, flesch_reading_ease / 100.0))
        fog_norm = max(0.0, min(1.0, 1.0 - gunning_fog / 20.0))
        readability_score = (
            0.4 * flesch_norm + 0.3 * fog_norm + 0.3 * vocabulary_diversity
        )

        return ReadabilityMetrics(
            flesch_reading_ease=flesch_reading_ease,
            gunning_fog=gunning_fog,
            vocabulary_diversity=vocabulary_diversity,
            readability_score=readability_score,
        )

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word using vowel-group heuristic.

        Args:
            word: Single word (will be lowercased)

        Returns:
            Syllable count (minimum 1)
        """
        word = word.lower().strip(".,!?;:'\"-")
        if not word:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        # Silent 'e' at end: subtract 1 if word ends in 'e' and has > 1 syllable
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    # US-359: Alias for test compatibility
    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Estimate syllable count (alias for _count_syllables).

        US-359: Added as alias so both name variants work in tests.
        """
        word = word.lower().strip(".,;:!?\"'()-")
        if not word:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences by splitting on sentence-ending punctuation.

        Returns:
            Sentence count (minimum 1)
        """
        import re
        sentences = re.split(r"[.!?]+", text.strip())
        count = sum(1 for s in sentences if s.strip())
        return max(1, count)

    def _compute_readability_pure_python(self, text: str, words: list) -> tuple:
        """Compute Flesch Reading Ease and Gunning Fog without textstat.

        Args:
            text: Full text string
            words: Pre-split word tokens

        Returns:
            Tuple of (flesch_reading_ease, gunning_fog) both as float
        """
        word_count = max(len(words), 1)
        sentence_count = self._count_sentences(text)

        # Count syllables per word
        syllable_counts = [self._count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)

        # Flesch Reading Ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        asl = word_count / sentence_count  # average sentence length
        asw = total_syllables / word_count  # average syllables per word
        flesch = 206.835 - 1.015 * asl - 84.6 * asw
        flesch = max(0.0, min(100.0, flesch))

        # Gunning Fog Index
        # complex words = words with 3+ syllables
        complex_count = sum(1 for sc in syllable_counts if sc >= 3)
        percent_complex = complex_count / word_count
        fog = (asl + percent_complex * 100.0) * 0.4
        fog = max(0.0, fog)

        return round(flesch, 2), round(fog, 2)

    @staticmethod
    def _compute_vocabulary_diversity(words: list[str]) -> float:
        """Compute vocabulary diversity as unique_words / total_words.

        Args:
            words: List of word tokens (already split).

        Returns:
            Diversity ratio (0-1). Returns 0.5 for empty input.
        """
        if not words:
            return 0.5

        # Normalize to lowercase and strip punctuation for uniqueness check
        normalized_words = [word.lower() for word in words]
        unique_count = len(set(normalized_words))
        total_count = len(normalized_words)

        if total_count == 0:
            return 0.5

        diversity = unique_count / total_count
        return max(0.0, min(1.0, diversity))
