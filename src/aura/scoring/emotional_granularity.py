"""US-345: Emotional Granularity Scorer — emotion vocabulary diversity via Shannon entropy."""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmotionalGranularityResult:
    """Result of emotional granularity analysis."""
    vocabulary_richness: float  # 0-1, unique emotion words / total emotion words
    entropy: float  # 0-1, normalized Shannon entropy of emotion word distribution
    differentiation: float  # 0-1, within-cluster variety score
    composite: float  # 0-1, weighted combination


# Plutchik's wheel: 8 primary emotions with expanded vocabulary
EMOTION_CLUSTERS = {
    "joy": [
        "happy", "joyful", "elated", "ecstatic", "cheerful", "delighted", "glad",
        "pleased", "content", "satisfied", "euphoric", "thrilled",
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "gloomy", "melancholy", "sorrowful",
        "heartbroken", "disappointed", "despondent", "miserable", "dejected", "grief",
    ],
    "anger": [
        "angry", "furious", "enraged", "irritated", "annoyed", "frustrated",
        "resentful", "bitter", "hostile", "agitated", "livid", "outraged",
    ],
    "fear": [
        "afraid", "scared", "fearful", "terrified", "anxious", "nervous",
        "worried", "panicked", "dread", "apprehensive", "uneasy", "alarmed",
    ],
    "surprise": [
        "surprised", "astonished", "amazed", "shocked", "stunned", "startled",
        "bewildered", "dumbfounded",
    ],
    "disgust": [
        "disgusted", "revolted", "repulsed", "appalled", "sickened", "nauseated",
        "contempt", "loathing",
    ],
    "trust": [
        "trusting", "confident", "secure", "assured", "faithful", "reliable",
        "hopeful", "optimistic", "believing", "certain",
    ],
    "anticipation": [
        "eager", "excited", "anticipating", "expecting", "hopeful", "impatient",
        "vigilant", "curious", "interested", "enthusiastic",
    ],
}

# Flatten for fast lookup: word -> cluster name
_WORD_TO_CLUSTER = {}
for _cluster, _words in EMOTION_CLUSTERS.items():
    for _w in _words:
        _WORD_TO_CLUSTER[_w] = _cluster

# Total unique words across all clusters
_ALL_EMOTION_WORDS = set(_WORD_TO_CLUSTER.keys())
_NUM_CLUSTERS = len(EMOTION_CLUSTERS)


class EmotionalGranularityScorer:
    """
    Tracks emotion vocabulary diversity across recent messages.

    Uses Shannon entropy of emotion word frequency distribution,
    type-token ratio, and within-cluster differentiation to measure
    how precisely the user distinguishes between emotional states.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._word_history: deque = deque(maxlen=window_size * 50)  # Store individual words
        self._message_count = 0

    def _extract_emotion_words(self, text: str) -> List[str]:
        """Extract emotion words from text, matching against the lexicon."""
        if not text:
            return []
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if w in _ALL_EMOTION_WORDS]

    def update(self, text: str) -> EmotionalGranularityResult:
        """
        Process a new message and compute granularity metrics.

        Args:
            text: The message text to analyze.

        Returns:
            EmotionalGranularityResult with all metrics.
        """
        emotion_words = self._extract_emotion_words(text)
        self._word_history.extend(emotion_words)
        self._message_count += 1

        # Work with the full window of accumulated emotion words
        all_words = list(self._word_history)

        if not all_words:
            return EmotionalGranularityResult(
                vocabulary_richness=0.0,
                entropy=0.0,
                differentiation=0.0,
                composite=0.0,
            )

        # --- Vocabulary richness: type-token ratio ---
        unique_words = set(all_words)
        vocabulary_richness = len(unique_words) / max(1, len(all_words))

        # --- Shannon entropy of cluster distribution ---
        cluster_counts = Counter(_WORD_TO_CLUSTER.get(w, "unknown") for w in all_words)
        cluster_counts.pop("unknown", None)
        total = sum(cluster_counts.values())

        if total == 0 or len(cluster_counts) <= 1:
            entropy = 0.0
        else:
            raw_entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in cluster_counts.values()
                if c > 0
            )
            # Normalize by log2(num_clusters) so result is 0-1
            max_entropy = math.log2(_NUM_CLUSTERS)
            entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        # --- Differentiation: within-cluster variety ---
        activated_clusters = {}
        for w in all_words:
            cluster = _WORD_TO_CLUSTER.get(w)
            if cluster:
                if cluster not in activated_clusters:
                    activated_clusters[cluster] = set()
                activated_clusters[cluster].add(w)

        if activated_clusters:
            # For each activated cluster, what fraction of its vocabulary was used?
            cluster_varieties = []
            for cluster_name, used_words in activated_clusters.items():
                cluster_size = len(EMOTION_CLUSTERS[cluster_name])
                variety = len(used_words) / cluster_size
                cluster_varieties.append(variety)
            differentiation = sum(cluster_varieties) / len(cluster_varieties)
        else:
            differentiation = 0.0

        # --- Composite ---
        composite = (
            0.35 * vocabulary_richness
            + 0.40 * entropy
            + 0.25 * differentiation
        )
        composite = max(0.0, min(1.0, composite))

        result = EmotionalGranularityResult(
            vocabulary_richness=round(vocabulary_richness, 4),
            entropy=round(entropy, 4),
            differentiation=round(differentiation, 4),
            composite=round(composite, 4),
        )

        logger.debug(
            "US-345: Granularity — richness=%.3f, entropy=%.3f, diff=%.3f, composite=%.3f",
            vocabulary_richness, entropy, differentiation, composite,
        )

        return result
