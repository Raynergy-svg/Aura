"""Conversation processor — extracts emotional signals from user messages.

Phase 1: Keyword-based sentiment and stress detection.
US-280: Negation-aware emotional signal extraction.
US-281: Emotional intensity scaling with modifier words.
Phase 2: Will use Phi-4 14B via MLX for deep understanding.

This module is the "ear" of Aura — it listens to what the user says and
extracts structured signals that feed into the readiness computation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.aura.scoring.affect_dynamics import AffectDynamicsTracker

logger = logging.getLogger(__name__)


# --- Keyword Dictionaries ---
# These are Phase 1 heuristics. Phase 2 replaces with LLM inference.

STRESS_KEYWORDS: Set[str] = {
    # Fix H-02 (FOLLOWUP): Removed bare "stress" to prevent double-counting.
    # "stress" is a substring of "stressed" — the raw-text pre-check (kw in message_lower)
    # matched both when the message contained "stressed", inflating stress scores by ~50%.
    # "stressed" covers the intent; if a user says "under stress" it's covered by "pressure".
    "stressed", "overwhelmed", "exhausted", "tired", "anxious",
    "worried", "frustrated", "frustrating", "angry", "can't sleep", "insomnia", "burnout",
    "deadline", "pressure", "overworked", "losing money", "lost money",
    "argument", "fight", "conflict", "breakup", "divorce",
    # Desperation / revenge-trading / panic language
    "desperate", "desperation", "hopeless", "helpless", "terrified",
    "scared", "afraid", "panic", "panicking", "revenge",
    "can't take", "falling apart", "out of control", "need this to work",
    "another loss", "blown account", "blew my account", "reckless",
    "spiraling", "tilt", "tilted", "on tilt",
    # Overtrading / loss-related language
    "took a loss", "keep losing", "losing streak", "bad trade",
    "overtrad", "too many trades",
    # Frustration / regret / self-blame
    "i knew", "should have", "shouldn't have", "i can't", "can't keep",
    "held too long", "wrong call", "my fault", "so stupid", "idiot",
    "what was i thinking", "knew better", "lost pips", "lost 40",
    "going against me", "went against",
}

POSITIVE_KEYWORDS: Set[str] = {
    "great", "amazing", "wonderful", "happy", "excited", "energized",
    "productive", "focused", "calm", "relaxed", "confident", "winning",
    "breakthrough", "inspired", "motivated", "grateful", "optimistic",
}

# --- US-280: Negation words ---
# Negation within NEGATION_WINDOW tokens before a keyword cancels its signal.
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "neither", "nor", "hardly", "barely",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "can't", "cant", "cannot", "won't", "wont", "wouldn't", "wouldnt",
    "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent", "weren't", "werent",
    "hasn't", "hasnt", "haven't", "havent", "hadn't", "hadnt",
    "shouldn't", "shouldnt", "couldn't", "couldnt",
}
NEGATION_WINDOW = 3  # tokens before keyword to check for negation

# --- US-281: Intensity modifiers ---
# Amplifiers boost keyword signal; diminishers reduce it.
AMPLIFIER_WORDS: Set[str] = {
    "extremely", "very", "incredibly", "absolutely", "completely", "totally",
    "severely", "deeply", "intensely", "terribly", "really", "so", "utterly",
    "massively", "hugely", "thoroughly",
}
DIMINISHER_WORDS: Set[str] = {
    "slightly", "somewhat", "a bit", "a little", "mildly", "marginally",
    "fairly", "kind of", "sort of", "partly", "barely",
}
AMPLIFIER_MULTIPLIER = 1.5
DIMINISHER_MULTIPLIER = 0.5

FATIGUE_KEYWORDS: Set[str] = {
    "tired", "exhausted", "didn't sleep", "couldn't sleep", "insomnia",
    "drained", "low energy", "fatigued", "burnt out", "running on empty",
}

TRADING_OVERRIDE_KEYWORDS: Set[str] = {
    "i override", "i overrode", "overriding buddy", "override buddy",
    "ignored buddy", "took the trade anyway", "closed early",
    "moved my stop", "changed the sl", "changed the tp", "didn't listen",
    "went against", "manual trade", "gut feeling trade", "just felt like",
    "i'm overriding", "going to override", "gonna override",
}

STRESSOR_KEYWORDS: Dict[str, str] = {
    "career": "career decision",
    "job": "career change",
    "quit": "career change",
    "promotion": "career decision",
    "interview": "job search",
    "relationship": "relationship stress",
    "partner": "relationship stress",
    "health": "health concern",
    "sick": "health concern",
    "money": "financial stress",
    "debt": "financial stress",
    "moving": "relocation",
    "baby": "new parent",
    "pregnant": "expecting child",
    "parent": "family responsibility",
}


@dataclass
class ConversationSignals:
    """Extracted signals from a conversation exchange."""

    emotional_state: str = "neutral"  # calm, anxious, stressed, energized, fatigued, etc.
    stress_keywords_found: List[str] = field(default_factory=list)
    positive_keywords_found: List[str] = field(default_factory=list)
    detected_stressors: List[str] = field(default_factory=list)
    fatigue_detected: bool = False
    override_mentioned: bool = False
    sentiment_score: float = 0.5  # 0=very negative, 1=very positive
    topics: List[str] = field(default_factory=list)
    confidence_trend: str = "stable"  # rising, falling, stable
    message_count: int = 0
    # US-281: Intensity score — continuous 0.0 (minimal) to 1.0 (maximum)
    intensity_score: float = 0.5
    # US-293: Cognitive bias scores
    bias_scores: Dict[str, float] = field(default_factory=dict)
    # US-332: Text readability score (0-1, higher = more readable)
    readability_score: float = 0.5
    # US-333: Linguistic style drift score (0-1, higher = more drift from baseline)
    style_drift_score: float = 0.0
    # US-340: Cognitive flexibility score (0-1, higher = more flexible)
    cognitive_flexibility_score: float = 0.0
    # US-345: Emotional granularity score (0-1, higher = more precise emotional vocabulary)
    emotional_granularity_score: float = 0.0
    # US-348: Narrative coherence score (0-1, higher = more consistent reasoning across sessions)
    narrative_coherence_score: float = 0.5
    # US-351: Affect dynamics scores
    affect_valence: float = 0.0  # -1 (negative) to +1 (positive)
    affect_arousal: float = 0.0  # 0 (calm) to 1 (activated)
    affect_inertia: float = 0.0  # 0-1, persistence of emotional state
    affect_volatility: float = 0.0  # 0-1, variability of emotional magnitude

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotional_state": self.emotional_state,
            "stress_keywords": self.stress_keywords_found,
            "positive_keywords": self.positive_keywords_found,
            "detected_stressors": self.detected_stressors,
            "fatigue_detected": self.fatigue_detected,
            "override_mentioned": self.override_mentioned,
            "sentiment_score": round(self.sentiment_score, 3),
            "topics": self.topics,
            "confidence_trend": self.confidence_trend,
            "intensity_score": round(self.intensity_score, 3),
            "bias_scores": {k: round(v, 3) for k, v in self.bias_scores.items()},
            "readability_score": round(self.readability_score, 3),
            "style_drift_score": round(self.style_drift_score, 3),
            "cognitive_flexibility_score": round(self.cognitive_flexibility_score, 3),
            "emotional_granularity_score": round(self.emotional_granularity_score, 3),
            "narrative_coherence_score": round(self.narrative_coherence_score, 3),
            "affect_valence": round(self.affect_valence, 3),
            "affect_arousal": round(self.affect_arousal, 3),
            "affect_inertia": round(self.affect_inertia, 3),
            "affect_volatility": round(self.affect_volatility, 3),
        }


class BiasDetector:
    """US-293: Detects trading-specific cognitive biases from conversation text.

    Detects 9 biases:
    1. Disposition effect — holding losers, cutting winners early
    2. Loss aversion — disproportionate focus on downside
    3. Recency bias — overweighting recent events vs historical
    4. Confirmation bias — seeking validation for existing beliefs
    5. Sunk cost (US-327) — reluctance to abandon due to prior investment
    6. Anchoring (US-327) — fixation on specific prices or levels
    7. Overconfidence (US-327) — inflated self-assessment
    8. Hindsight (US-327) — retroactive inevitability
    9. Attribution error (US-327) — external blame / self-credit asymmetry

    US-336: All bias detection methods apply negation-awareness — phrases
    preceded by negation words within 3 tokens are excluded from scoring.
    """

    # US-336: Negation words for bias detection filtering
    NEGATION_WORDS = frozenset({
        "not", "no", "never", "don't", "didn't", "isn't", "won't",
        "can't", "haven't", "shouldn't", "neither", "nor", "without",
        "doesn't", "wasn't", "weren't", "wouldn't", "couldn't",
    })

    # Disposition effect phrases
    DISPOSITION_PHRASES = [
        "still waiting", "might come back", "bounce back", "hasn't hit my stop",
        "holding on", "just needs time", "average down", "bag holding",
        "it'll recover", "diamond hands",
    ]
    COUNTER_DISPOSITION_PHRASES = [
        "stopped out", "took profit", "cut loss", "closed position",
        "let it run", "stuck to plan", "followed the system", "took the loss",
    ]

    # Loss aversion keywords
    LOSS_WORDS = ["risk", "lose", "loss", "drawdown", "fear", "worried", "scared", "afraid", "downside", "danger"]
    GAIN_WORDS = ["profit", "gain", "win", "winning", "upside", "success", "opportunity", "reward"]

    # Recency bias keywords
    RECENT_WORDS = ["today", "just", "recent", "recently", "this week", "yesterday", "right now", "latest"]
    HISTORICAL_WORDS = ["historically", "typically", "always", "usually", "long term", "over time", "in the past"]

    # Confirmation bias phrases
    CONFIRMATION_PHRASES = [
        "proving me right", "i knew it", "told you so", "see what i mean",
        "exactly what i expected", "confirms my", "validates my", "that proves",
        "i was right", "just as i thought",
    ]
    DISMISSIVE_PHRASES = [
        "that's different", "special case", "doesn't apply", "exception",
        "but that's", "this time is different", "not the same",
    ]

    # US-327: Sunk cost bias
    SUNK_COST_PHRASES = [
        "already invested", "too much time", "can't give up now",
        "need to make it back", "put too much into", "come this far",
        "can't walk away", "too deep", "committed too much",
    ]

    # US-327: Anchoring bias
    ANCHORING_PHRASES = [
        "bought at", "entry was", "my cost basis", "waiting for",
        "need it to get back to", "original price", "break even",
    ]
    ANCHORING_PRICE_PATTERN = r'\b\d+\.\d{2,5}\b'  # Specific price mentions like 1.2345

    # US-327: Overconfidence bias
    OVERCONFIDENCE_PHRASES = [
        "can't lose", "easy money", "guaranteed", "i know exactly",
        "no way this fails", "sure thing", "100%", "slam dunk",
        "free money", "this is easy", "obvious trade",
    ]

    # US-327: Hindsight bias
    HINDSIGHT_PHRASES = [
        "knew it", "should have seen", "was obvious", "saw it coming",
        "told you so", "predicted this", "i called it", "knew this would happen",
    ]

    # US-327: Attribution error bias
    EXTERNAL_BLAME_PHRASES = [
        "rigged", "bad luck", "unfair", "manipulated", "market maker",
        "broker cheated", "stop hunted", "they always", "system is broken",
    ]
    SELF_CREDIT_PHRASES = [
        "i called it", "my analysis was right", "i'm the best",
        "nailed it", "genius trade", "my skill",
    ]

    def _is_negated(self, text: str, phrase: str) -> bool:
        """US-336: Check if a phrase match is negated within 3 tokens.

        Scans up to 3 tokens before the phrase's position in the text
        for any negation words. If found, the match is considered negated.

        Args:
            text: Full lowered message text
            phrase: The bias phrase to check (already lowered)

        Returns:
            True if the phrase is negated, False otherwise
        """
        pos = text.find(phrase)
        if pos < 0:
            return False

        # Get the text before the phrase, split into tokens
        prefix = text[:pos].strip()
        if not prefix:
            return False

        tokens_before = prefix.split()
        # Check last 3 tokens before the phrase
        check_tokens = tokens_before[-3:] if len(tokens_before) >= 3 else tokens_before
        for token in check_tokens:
            # Strip punctuation for matching
            cleaned = token.strip(".,!?;:'\"()[]")
            if cleaned in self.NEGATION_WORDS:
                return True
        return False

    def _count_non_negated_phrases(self, text: str, phrases: list) -> int:
        """US-336: Count phrase matches that are NOT negated.

        Args:
            text: Lowered message text
            phrases: List of bias phrases to check

        Returns:
            Count of non-negated phrase matches
        """
        count = 0
        for phrase in phrases:
            if phrase in text and not self._is_negated(text, phrase):
                count += 1
        return count

    def _count_non_negated_words(self, text: str, words: list) -> int:
        """US-336: Count word matches that are NOT negated.

        For single-word bias keywords, checks negation within 3 tokens before.

        Args:
            text: Lowered message text
            words: List of bias keywords to check

        Returns:
            Count of non-negated word matches
        """
        count = 0
        for word in words:
            if word in text and not self._is_negated(text, word):
                count += 1
        return count

    def detect_biases(self, message: str) -> Dict[str, float]:
        """Detect cognitive biases from message text.

        US-336: All bias detection now applies negation awareness —
        phrases preceded by negation words are excluded from scoring.

        Returns:
            Dict with bias scores (0.0-1.0 each):
            - disposition_effect
            - loss_aversion
            - recency_bias
            - confirmation_bias
            - sunk_cost (US-327)
            - anchoring (US-327)
            - overconfidence (US-327)
            - hindsight (US-327)
            - attribution_error (US-327)
        """
        if not message or not message.strip():
            return {
                "disposition_effect": 0.0,
                "loss_aversion": 0.0,
                "recency_bias": 0.0,
                "confirmation_bias": 0.0,
                "sunk_cost": 0.0,
                "anchoring": 0.0,
                "overconfidence": 0.0,
                "hindsight": 0.0,
                "attribution_error": 0.0,
            }

        text = message.lower()
        word_count = max(len(text.split()), 1)

        # 1. Disposition effect — US-336: negation-aware
        disposition_hits = self._count_non_negated_phrases(text, self.DISPOSITION_PHRASES)
        counter_hits = self._count_non_negated_phrases(text, self.COUNTER_DISPOSITION_PHRASES)
        disposition_raw = max(0, disposition_hits - counter_hits)
        disposition_score = min(1.0, disposition_raw / 3.0)

        # 2. Loss aversion — US-336: negation-aware
        loss_count = self._count_non_negated_words(text, self.LOSS_WORDS)
        gain_count = self._count_non_negated_words(text, self.GAIN_WORDS)
        if gain_count > 0 and loss_count > 2 * gain_count:
            loss_aversion_score = min(1.0, (loss_count - gain_count) / 5.0)
        elif gain_count == 0 and loss_count >= 2:
            loss_aversion_score = min(1.0, loss_count / 5.0)
        else:
            loss_aversion_score = 0.0

        # 3. Recency bias — US-336: negation-aware
        recent_count = self._count_non_negated_words(text, self.RECENT_WORDS)
        historical_count = self._count_non_negated_words(text, self.HISTORICAL_WORDS)
        recency_raw = max(0, recent_count - historical_count)
        recency_score = min(1.0, recency_raw / 4.0)

        # 4. Confirmation bias — US-336: negation-aware
        confirm_hits = self._count_non_negated_phrases(text, self.CONFIRMATION_PHRASES)
        dismiss_hits = self._count_non_negated_phrases(text, self.DISMISSIVE_PHRASES)
        confirmation_score = min(1.0, (confirm_hits + dismiss_hits) / 3.0)

        # 5. Sunk cost (US-327) — US-336: negation-aware
        sunk_cost_hits = self._count_non_negated_phrases(text, self.SUNK_COST_PHRASES)
        sunk_cost_score = min(1.0, sunk_cost_hits / 2.0)

        # 6. Anchoring (US-327) — US-336: negation-aware
        anchoring_hits = self._count_non_negated_phrases(text, self.ANCHORING_PHRASES)
        import re as _re
        price_mentions = len(_re.findall(self.ANCHORING_PRICE_PATTERN, text))
        anchoring_score = min(1.0, (anchoring_hits + min(price_mentions, 3) * 0.3) / 3.0)

        # 7. Overconfidence (US-327) — US-336: negation-aware
        overconfidence_hits = self._count_non_negated_phrases(text, self.OVERCONFIDENCE_PHRASES)
        overconfidence_score = min(1.0, overconfidence_hits / 2.0)

        # 8. Hindsight (US-327) — US-336: negation-aware
        hindsight_hits = self._count_non_negated_phrases(text, self.HINDSIGHT_PHRASES)
        hindsight_score = min(1.0, hindsight_hits / 2.0)

        # 9. Attribution error (US-327) — US-336: negation-aware
        external_hits = self._count_non_negated_phrases(text, self.EXTERNAL_BLAME_PHRASES)
        self_credit_hits = self._count_non_negated_phrases(text, self.SELF_CREDIT_PHRASES)
        # Both external blame AND excessive self-credit indicate attribution error
        attribution_raw = external_hits + max(0, self_credit_hits - 1)  # 1 self-credit is ok
        attribution_score = min(1.0, attribution_raw / 3.0)

        return {
            "disposition_effect": round(disposition_score, 3),
            "loss_aversion": round(loss_aversion_score, 3),
            "recency_bias": round(recency_score, 3),
            "confirmation_bias": round(confirmation_score, 3),
            "sunk_cost": round(sunk_cost_score, 3),
            "anchoring": round(anchoring_score, 3),
            "overconfidence": round(overconfidence_score, 3),
            "hindsight": round(hindsight_score, 3),
            "attribution_error": round(attribution_score, 3),
        }

    def aggregate_bias_score(self, biases: Dict[str, float]) -> float:
        """Compute aggregate bias score (0.0-1.0) from individual biases."""
        if not biases:
            return 0.0
        return sum(biases.values()) / max(len(biases), 1)


class TiltDetector:
    """US-304: Detects post-loss emotional spirals (tilt/revenge trading).

    Three indicators:
    1. Sentiment trajectory: 3+ consecutive messages with declining sentiment
    2. Revenge keywords: phrases signaling desire to "make it back"
    3. Override frequency spike: high override rate within 30min of a loss

    Tilt score (0-1) feeds as penalty into readiness.
    """

    REVENGE_KEYWORDS: Set[str] = {
        "make it back", "just one more", "recover", "double down",
        "revenge trade", "get it back", "need to win", "one more try",
        "can't end like this", "make up for", "undo the loss",
        "compensate", "recoup", "breakeven trade",
    }

    # Weights for combining indicators
    SENTIMENT_WEIGHT = 0.4
    REVENGE_WEIGHT = 0.35
    OVERRIDE_SPIKE_WEIGHT = 0.25

    def detect_tilt(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        recent_overrides: Optional[List[Dict[str, Any]]] = None,
        recent_outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Detect tilt from conversation patterns + override/outcome data.

        Args:
            messages: Recent messages with 'content' and optional 'sentiment' fields
            recent_overrides: Override events with timestamps
            recent_outcomes: Recent trade outcomes with 'trade_won' field

        Returns:
            Tilt score: 0.0 (no tilt) to 1.0 (severe tilt)
        """
        messages = messages or []
        recent_overrides = recent_overrides or []
        recent_outcomes = recent_outcomes or []

        if len(messages) < 2:
            return 0.0

        # 1. Sentiment trajectory — check for 3+ consecutive declines
        sentiment_indicator = self._check_sentiment_decline(messages)

        # 2. Revenge keyword detection
        revenge_indicator = self._check_revenge_keywords(messages)

        # 3. Override frequency spike after loss
        spike_indicator = self._check_override_spike(recent_overrides, recent_outcomes)

        # Combine with weights
        tilt = (
            sentiment_indicator * self.SENTIMENT_WEIGHT
            + revenge_indicator * self.REVENGE_WEIGHT
            + spike_indicator * self.OVERRIDE_SPIKE_WEIGHT
        )
        return max(0.0, min(1.0, tilt))

    def _check_sentiment_decline(self, messages: List[Dict[str, Any]]) -> float:
        """Check for consecutive declining sentiment in recent messages."""
        # Extract sentiment scores from messages (use 0.5 as default)
        sentiments = []
        for m in messages[-10:]:  # Last 10 messages
            if isinstance(m, dict):
                sentiments.append(m.get("sentiment", 0.5))
            else:
                sentiments.append(0.5)

        if len(sentiments) < 3:
            return 0.0

        # Count consecutive declines from the end
        consecutive_declines = 0
        for i in range(len(sentiments) - 1, 0, -1):
            if sentiments[i] < sentiments[i - 1]:
                consecutive_declines += 1
            else:
                break

        if consecutive_declines >= 3:
            return min(1.0, consecutive_declines / 5.0)  # 5+ declines = max
        return 0.0

    def _check_revenge_keywords(self, messages: List[Dict[str, Any]]) -> float:
        """Check for revenge trading keywords in recent messages."""
        total_hits = 0
        for m in messages[-5:]:  # Last 5 messages
            content = ""
            if isinstance(m, dict):
                content = m.get("content", "").lower()
            elif isinstance(m, str):
                content = m.lower()
            if not content:
                continue
            for kw in self.REVENGE_KEYWORDS:
                if kw in content:
                    total_hits += 1

        return min(1.0, total_hits / 3.0)  # 3+ hits = max

    def _check_override_spike(
        self,
        overrides: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
    ) -> float:
        """Check if override frequency spiked after a recent loss."""
        # Check if there was a recent loss
        has_recent_loss = False
        for o in outcomes[-5:]:
            if isinstance(o, dict) and not o.get("trade_won", True):
                has_recent_loss = True
                break

        if not has_recent_loss:
            return 0.0

        # Count overrides — if > 2 after a loss, it's a spike
        override_count = len(overrides[-5:]) if overrides else 0
        if override_count >= 3:
            return min(1.0, (override_count - 2) / 3.0)
        return 0.0


class ConversationProcessor:
    """Processes conversation messages and extracts emotional/cognitive signals.

    Phase 1: keyword-based analysis.
    Phase 2: Phi-4 14B inference via MLX for deep understanding.
    """

    def __init__(self):
        self._session_messages: List[Dict[str, str]] = []
        self._cumulative_stress: float = 0.0
        self._cumulative_positive: float = 0.0
        self._previous_sentiment: float = 0.5
        # US-347: Drift history tracking for T1 early-warning patterns
        self._drift_history: List[float] = []
        # US-293: Bias detection
        self._bias_detector = BiasDetector()
        # US-315: VADER sentiment analyzer (optional — graceful fallback)
        self._vader = None
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except (ImportError, LookupError):
            logger.warning("US-315: VADER not available — using keyword-only sentiment")
        # US-332: Text readability analyzer
        self._readability_analyzer = None
        try:
            from src.aura.analysis.readability import TextReadabilityAnalyzer
            self._readability_analyzer = TextReadabilityAnalyzer()
        except ImportError:
            logger.warning("US-332: TextReadabilityAnalyzer not available")
        # US-333: Linguistic style tracker
        self._style_tracker = None
        try:
            from src.aura.analysis.style_tracker import LinguisticStyleTracker
            self._style_tracker = LinguisticStyleTracker()
        except ImportError:
            logger.warning("US-333: LinguisticStyleTracker not available")
        # US-340: Cognitive flexibility scorer
        self._flexibility_scorer = None
        try:
            from src.aura.scoring.cognitive_flexibility import CognitiveFlexibilityScorer
            self._flexibility_scorer = CognitiveFlexibilityScorer()
        except ImportError:
            logger.warning("US-340: CognitiveFlexibilityScorer not available")
        # US-345: Emotional granularity scorer
        self._granularity_scorer = None
        try:
            from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer
            self._granularity_scorer = EmotionalGranularityScorer()
        except ImportError:
            logger.warning("US-345: EmotionalGranularityScorer not available")
        # US-348: Narrative coherence tracker
        self._narrative_coherence_tracker = None
        try:
            from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker
            self._narrative_coherence_tracker = NarrativeCoherenceTracker()
        except ImportError:
            logger.warning("US-348: NarrativeCoherenceTracker not available")
        # US-351: Affect dynamics tracker
        self._affect_tracker = AffectDynamicsTracker()

    # US-261: Input validation constants
    MAX_MESSAGE_LENGTH = 10000
    MAX_SESSION_MESSAGES = 500

    def _is_negated(self, tokens: List[str], keyword_start_idx: int) -> bool:
        """US-280: Check if keyword at index is negated by a preceding negation word.

        Scans up to NEGATION_WINDOW tokens before the keyword for negation words.
        Handles double negation: if two negation words precede, they cancel out.

        Args:
            tokens: Tokenized message (lowercased)
            keyword_start_idx: Index of the first token of the keyword

        Returns:
            True if keyword is negated (odd number of negation words in window)
        """
        window_start = max(0, keyword_start_idx - NEGATION_WINDOW)
        window = tokens[window_start:keyword_start_idx]
        negation_count = sum(1 for t in window if t in NEGATION_WORDS)
        return negation_count % 2 == 1  # Odd = negated, even = not negated

    def _get_intensity_modifier(self, tokens: List[str], keyword_start_idx: int) -> float:
        """US-281: Get intensity modifier for keyword from surrounding context.

        Scans up to NEGATION_WINDOW tokens before the keyword for
        amplifier or diminisher words.

        Returns:
            Multiplier: AMPLIFIER_MULTIPLIER, DIMINISHER_MULTIPLIER, or 1.0 (neutral)
        """
        window_start = max(0, keyword_start_idx - NEGATION_WINDOW)
        window = tokens[window_start:keyword_start_idx]

        for token in window:
            if token in AMPLIFIER_WORDS:
                return AMPLIFIER_MULTIPLIER
            if token in DIMINISHER_WORDS:
                return DIMINISHER_MULTIPLIER

        # Also check multi-word diminishers in the raw text window
        window_text = " ".join(window)
        for phrase in DIMINISHER_WORDS:
            if " " in phrase and phrase in window_text:
                return DIMINISHER_MULTIPLIER

        return 1.0

    def _find_keyword_in_tokens(self, tokens: List[str], keyword: str) -> int:
        """Find the starting token index of a keyword (handles multi-word keywords).

        Returns -1 if not found.
        """
        kw_tokens = keyword.split()
        kw_len = len(kw_tokens)
        for i in range(len(tokens) - kw_len + 1):
            if tokens[i:i + kw_len] == kw_tokens:
                return i
        return -1

    def _extract_keywords_with_negation(
        self, tokens: List[str], keywords: Set[str], message_lower: str
    ) -> Tuple[List[str], float]:
        """US-280+281: Extract keywords, filtering negated ones and computing intensity.

        Returns:
            Tuple of (found_keywords, total_intensity_multiplier)
        """
        found = []
        total_intensity = 0.0

        for kw in keywords:
            if kw not in message_lower:
                continue
            idx = self._find_keyword_in_tokens(tokens, kw)
            if idx == -1:
                # Multi-word keyword found in raw text but not cleanly tokenized
                # Fall back to simple check without negation (backward compat)
                found.append(kw)
                total_intensity += 1.0
                continue

            if self._is_negated(tokens, idx):
                logger.debug("US-280: Keyword '%s' negated — skipping", kw)
                continue

            modifier = self._get_intensity_modifier(tokens, idx)
            found.append(kw)
            total_intensity += modifier

        avg_intensity = total_intensity / len(found) if found else 1.0
        return found, avg_intensity

    def _compute_vader_sentiment(self, message: str) -> Optional[float]:
        """US-315: Compute VADER compound sentiment score.

        Returns compound score (-1 to +1) or None if VADER unavailable.
        VADER handles negation, emphasis, and contrast automatically.
        """
        if self._vader is None:
            return None
        try:
            scores = self._vader.polarity_scores(message)
            return scores.get("compound", 0.0)
        except Exception as e:
            logger.warning("US-315: VADER scoring failed: %s", e)
            return None

    def process_message(self, message: str, role: str = "user") -> ConversationSignals:
        """Process a single message and extract signals.

        Args:
            message: The message text
            role: "user" or "assistant"

        Returns:
            ConversationSignals with extracted emotional data
        """
        # US-261: Input validation — reject empty, truncate oversized
        if not message or not message.strip():
            logger.debug("US-261: Empty/whitespace message — returning neutral signals")
            return ConversationSignals(message_count=len(self._session_messages))

        if len(message) > self.MAX_MESSAGE_LENGTH:
            logger.warning("US-261: Message truncated from %d to %d chars", len(message), self.MAX_MESSAGE_LENGTH)
            message = message[:self.MAX_MESSAGE_LENGTH]

        self._session_messages.append({"role": role, "content": message, "timestamp": datetime.now(timezone.utc).isoformat()})

        # US-261: Cap session messages (sliding window)
        if len(self._session_messages) > self.MAX_SESSION_MESSAGES:
            self._session_messages = self._session_messages[-self.MAX_SESSION_MESSAGES:]

        if role != "user":
            # Only analyze user messages for emotional signals
            return ConversationSignals(message_count=len(self._session_messages))

        message_lower = message.lower()
        # US-280: Tokenize for negation + intensity analysis
        tokens = re.findall(r"[a-z']+", message_lower)

        # --- Keyword extraction with negation + intensity (US-280 + US-281) ---
        stress_found, stress_intensity = self._extract_keywords_with_negation(
            tokens, STRESS_KEYWORDS, message_lower
        )
        positive_found, positive_intensity = self._extract_keywords_with_negation(
            tokens, POSITIVE_KEYWORDS, message_lower
        )
        fatigue_found_kw, fatigue_intensity = self._extract_keywords_with_negation(
            tokens, FATIGUE_KEYWORDS, message_lower
        )
        fatigue_found = len(fatigue_found_kw) > 0
        # Override detection — action phrases only, not questions about overrides.
        # "I overrode Buddy" = override. "explain override" or "what is override" = not.
        _override_question_prefixes = ("explain", "what is", "what's", "tell me about", "how does", "how do", "why does", "why do", "define")
        _msg_is_question = any(message_lower.strip().startswith(p) for p in _override_question_prefixes) or message_lower.strip().endswith("?")
        if _msg_is_question:
            # Questions about overrides are NOT override actions
            override_found = False
        else:
            override_found = any(kw in message_lower for kw in TRADING_OVERRIDE_KEYWORDS)

        # Detect stressors — US-204: use word-boundary regex to prevent
        # false positives (e.g. "parent" matching inside "apartment")
        stressors = []
        for keyword, stressor in STRESSOR_KEYWORDS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower) and stressor not in stressors:
                stressors.append(stressor)

        # --- Sentiment computation (US-281: intensity-scaled) ---
        stress_score = len(stress_found) * 0.15 * stress_intensity
        # Override mentions compound into stress — overriding your system IS stress
        if override_found:
            stress_score += 0.15
        positive_score = len(positive_found) * 0.12 * positive_intensity
        fatigue_penalty = 0.15 * fatigue_intensity if fatigue_found else 0.0

        raw_sentiment = 0.5 + positive_score - stress_score - fatigue_penalty
        keyword_sentiment = max(0.0, min(1.0, raw_sentiment))

        # US-315: Blend VADER compound score with keyword sentiment
        vader_compound = self._compute_vader_sentiment(message)
        if vader_compound is not None:
            # Normalize compound (-1..+1) to (0..1) range
            vader_normalized = (vader_compound + 1.0) / 2.0
            # Blend: 60% VADER + 40% keyword (keyword as safety net)
            sentiment = 0.6 * vader_normalized + 0.4 * keyword_sentiment
            sentiment = max(0.0, min(1.0, sentiment))
            logger.debug("US-315: VADER=%.3f keyword=%.3f blended=%.3f", vader_normalized, keyword_sentiment, sentiment)
        else:
            sentiment = keyword_sentiment

        # US-281: Compute overall intensity score (0.0 to 1.0)
        all_keywords = len(stress_found) + len(positive_found) + len(fatigue_found_kw)
        if all_keywords > 0:
            weighted_intensity = (
                len(stress_found) * stress_intensity
                + len(positive_found) * positive_intensity
                + len(fatigue_found_kw) * fatigue_intensity
            ) / all_keywords
            # Normalize: 1.0 = neutral, scale to 0-1 range
            intensity_score = max(0.0, min(1.0, weighted_intensity / AMPLIFIER_MULTIPLIER))
        else:
            intensity_score = 0.5  # Neutral default when no keywords found

        # Update cumulative trackers
        self._cumulative_stress += stress_score
        self._cumulative_positive += positive_score

        # --- Determine emotional state ---
        if fatigue_found:
            emotional_state = "fatigued"
        elif stress_score >= 0.3:
            emotional_state = "stressed"
        elif stress_score >= 0.15:
            emotional_state = "anxious"
        elif positive_score > 0.3:
            emotional_state = "energized"
        elif positive_score > 0.15:
            emotional_state = "calm"
        else:
            emotional_state = "neutral"

        # --- Confidence trend ---
        if sentiment > self._previous_sentiment + 0.1:
            confidence_trend = "rising"
        elif sentiment < self._previous_sentiment - 0.1:
            confidence_trend = "falling"
        else:
            confidence_trend = "stable"
        self._previous_sentiment = sentiment

        # --- Build topics list ---
        topics = []
        if any(kw in message_lower for kw in ["trade", "trading", "buddy", "market", "forex", "fx"]):
            topics.append("trading")
        if any(kw in message_lower for kw in ["career", "job", "work", "promotion"]):
            topics.append("career")
        if any(kw in message_lower for kw in ["relationship", "partner", "family"]):
            topics.append("relationships")
        if any(kw in message_lower for kw in ["health", "sleep", "exercise", "sick"]):
            topics.append("health")
        if override_found:
            topics.append("trading_override")

        # --- US-293: Cognitive bias detection ---
        bias_scores = self._bias_detector.detect_biases(message)

        # US-332: Text readability scoring
        readability_score = 0.5  # neutral default
        if self._readability_analyzer is not None:
            try:
                readability_result = self._readability_analyzer.analyze(message)
                readability_score = readability_result.readability_score
                logger.debug("US-332: readability_score=%.3f (flesch=%.1f, fog=%.1f, vocab=%.3f)",
                             readability_score, readability_result.flesch_reading_ease,
                             readability_result.gunning_fog, readability_result.vocabulary_diversity)
            except Exception as e:
                logger.warning("US-332: Readability analysis failed: %s", e)

        # US-333: Linguistic style tracking + drift
        style_drift_score = 0.0
        if self._style_tracker is not None:
            try:
                self._style_tracker.track_message(message)
                style_drift_score = self._style_tracker.compute_drift()
                logger.debug("US-333: style_drift_score=%.3f", style_drift_score)
            except Exception as e:
                logger.warning("US-333: Style tracking failed: %s", e)

        # US-347: Track drift history for T1 early-warning patterns
        self._drift_history.append(style_drift_score)
        if len(self._drift_history) > 10:
            self._drift_history = self._drift_history[-10:]

        # US-340: Cognitive flexibility scoring
        cognitive_flexibility_score = 0.0
        if self._flexibility_scorer is not None:
            try:
                flex_result = self._flexibility_scorer.score(message)
                cognitive_flexibility_score = flex_result.composite
                if cognitive_flexibility_score > 0:
                    logger.debug("US-340: flexibility=%.3f (belief=%.2f, strategy=%.2f, evidence=%.2f)",
                                 flex_result.composite, flex_result.belief_update,
                                 flex_result.strategy_adaptation, flex_result.evidence_acknowledgment)
            except Exception as e:
                logger.warning("US-340: Cognitive flexibility scoring failed: %s", e)

        # US-345: Emotional granularity scoring
        emotional_granularity_score = 0.0
        if self._granularity_scorer is not None:
            try:
                granularity_result = self._granularity_scorer.update(message)
                emotional_granularity_score = granularity_result.composite
                if emotional_granularity_score > 0:
                    logger.debug("US-345: granularity=%.3f (richness=%.3f, entropy=%.3f, diff=%.3f)",
                                 granularity_result.composite, granularity_result.vocabulary_richness,
                                 granularity_result.entropy, granularity_result.differentiation)
            except Exception as e:
                logger.warning("US-345: Emotional granularity scoring failed: %s", e)

        # US-348: Narrative coherence tracking
        narrative_coherence_score = 0.5  # neutral default
        if self._narrative_coherence_tracker is not None:
            try:
                coherence_result = self._narrative_coherence_tracker.update(message, sentiment)
                narrative_coherence_score = coherence_result.composite
                logger.debug("US-348: narrative_coherence=%.3f (lexical=%.3f, sentiment=%.3f, strategy=%.3f)",
                             coherence_result.composite, coherence_result.lexical_overlap,
                             coherence_result.sentiment_consistency, coherence_result.strategy_persistence)
            except Exception as e:
                logger.warning("US-348: Narrative coherence tracking failed: %s", e)

        # US-351: Affect dynamics tracking (valence-arousal with inertia and volatility)
        affect_valence = 0.0
        affect_arousal = 0.0
        affect_inertia = 0.0
        affect_volatility = 0.0
        try:
            # Convert sentiment (0-1) to valence (-1 to +1) for affect tracker
            sentiment_to_valence = (sentiment * 2.0) - 1.0
            affect_result = self._affect_tracker.update(message, vader_compound=sentiment_to_valence)
            affect_valence = affect_result.valence
            affect_arousal = affect_result.arousal
            affect_inertia = affect_result.inertia
            affect_volatility = affect_result.volatility
            logger.debug("US-351: affect_dynamics valence=%.3f arousal=%.3f inertia=%.3f volatility=%.3f",
                         affect_valence, affect_arousal, affect_inertia, affect_volatility)
        except Exception as e:
            logger.warning("US-351: Affect dynamics tracking failed: %s", e)

        signals = ConversationSignals(
            emotional_state=emotional_state,
            stress_keywords_found=stress_found,
            positive_keywords_found=positive_found,
            detected_stressors=stressors,
            fatigue_detected=fatigue_found,
            override_mentioned=override_found,
            sentiment_score=sentiment,
            topics=topics,
            confidence_trend=confidence_trend,
            message_count=len(self._session_messages),
            intensity_score=intensity_score,
            bias_scores=bias_scores,
            readability_score=readability_score,
            style_drift_score=style_drift_score,
            cognitive_flexibility_score=cognitive_flexibility_score,
            emotional_granularity_score=emotional_granularity_score,
            narrative_coherence_score=narrative_coherence_score,
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            affect_inertia=affect_inertia,
            affect_volatility=affect_volatility,
        )

        logger.debug(
            "US-280/281/293/332/333/340/345/348/351: signals state=%s sentiment=%.2f intensity=%.2f readability=%.3f drift=%.3f flex=%.3f granularity=%.3f coherence=%.3f affect_val=%.3f arousal=%.3f inertia=%.3f volatility=%.3f",
            emotional_state, sentiment, intensity_score, readability_score, style_drift_score, cognitive_flexibility_score, emotional_granularity_score, narrative_coherence_score, affect_valence, affect_arousal, affect_inertia, affect_volatility,
        )

        return signals

    def estimate_cognitive_load(self, message: str) -> float:
        """US-284/US-332: Estimate cognitive load from message structural complexity.

        Analyzes message text for complexity indicators:
        - Sentence count (more sentences = higher load)
        - Question density (many questions = high cognitive demand)
        - Conditional words (if/but/however = complex reasoning)
        - Topic shift markers (also/another/besides = juggling multiple concerns)

        US-332: Blends with readability metrics at 30% weight when available.
        Low readability → high cognitive load (inverted).

        Returns:
            cognitive_load_score: 0.0 (minimal) to 1.0 (extreme complexity)
        """
        if not message or not message.strip():
            return 0.1

        message_lower = message.lower()
        words = message_lower.split()
        word_count = len(words)

        if word_count == 0:
            return 0.1

        # 1. Sentence count (rough — split on sentence-ending punctuation)
        sentences = re.split(r'[.!?]+', message.strip())
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        # 2. Question density
        question_marks = message.count('?')
        question_density = question_marks / max(sentence_count, 1)

        # 3. Conditional/complex reasoning words
        conditional_words = {"if", "but", "however", "although", "unless", "whereas",
                             "while", "despite", "yet", "though", "otherwise", "either",
                             "whether", "considering", "assuming", "given"}
        conditional_count = sum(1 for w in words if w in conditional_words)

        # 4. Topic shift markers
        shift_words = {"also", "another", "besides", "additionally", "plus",
                       "meanwhile", "separately", "furthermore", "moreover"}
        shift_count = sum(1 for w in words if w in shift_words)

        # 5. Message length factor (longer = more cognitive demand)
        length_factor = min(1.0, word_count / 100.0)  # Caps at 100 words

        # Combine text-structure indicators with weights
        text_structure_load = (
            0.20 * min(1.0, sentence_count / 6.0)      # 6+ sentences = max
            + 0.25 * min(1.0, question_density)          # Questions per sentence
            + 0.25 * min(1.0, conditional_count / 3.0)   # 3+ conditionals = max
            + 0.15 * min(1.0, shift_count / 2.0)         # 2+ topic shifts = max
            + 0.15 * length_factor                        # Message length
        )

        # US-332: Blend with readability score if available
        # Low readability (complex text) → high cognitive load
        raw_score = text_structure_load
        if self._readability_analyzer is not None:
            try:
                readability_result = self._readability_analyzer.analyze(message)
                readability_load = 1.0 - readability_result.readability_score  # Invert: low readability = high load
                raw_score = 0.70 * text_structure_load + 0.30 * readability_load
                logger.debug("US-332: cognitive_load blend — text_structure=%.3f readability_load=%.3f final=%.3f",
                             text_structure_load, readability_load, raw_score)
            except Exception as e:
                logger.warning("US-332: Readability blend failed in cognitive_load: %s", e)
                raw_score = text_structure_load

        # Clamp to 0.1-0.95 (never fully 0 or fully 1)
        return max(0.1, min(0.95, raw_score))

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation session."""
        return {
            "message_count": len(self._session_messages),
            "cumulative_stress": round(self._cumulative_stress, 3),
            "cumulative_positive": round(self._cumulative_positive, 3),
            "net_sentiment": round(self._cumulative_positive - self._cumulative_stress, 3),
        }

    def reset_session(self) -> None:
        """Reset session state for a new conversation."""
        self._session_messages = []
        self._cumulative_stress = 0.0
        self._cumulative_positive = 0.0
        self._previous_sentiment = 0.5

    def check_drift_warning(self) -> Optional[Dict[str, Any]]:
        """US-347: Check if style drift has been sustained above threshold for 3+ messages.

        Returns:
            Dict with drift warning data if sustained drift detected, else None.
            Warning includes: consecutive_count, avg_drift, max_drift, domain.
        """
        if len(self._drift_history) < 3:
            return None

        # Check last entries for consecutive drift > 0.5
        consecutive = 0
        for score in reversed(self._drift_history):
            if score > 0.5:
                consecutive += 1
            else:
                break

        if consecutive >= 3:
            recent = self._drift_history[-consecutive:]
            return {
                "type": "style_drift_warning",
                "consecutive_count": consecutive,
                "avg_drift": sum(recent) / len(recent),
                "max_drift": max(recent),
                "domain": "HUMAN",
            }
        return None
