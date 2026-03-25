"""US-321/US-328: Decision Quality Scorer — 8-dimension process evaluation.

Scores the QUALITY of trading decisions independent of outcomes.
A ready trader can make poor decisions (impulsive, no plan).
A stressed trader can follow excellent process (disciplined despite discomfort).

Dimensions (from behavioral economics research — Steenbarger, Douglas, metacognition):
  1. Process Adherence (0.25) — Did the trader follow their stated rules/plan?
  2. Information Adequacy (0.17) — Was sufficient information gathered?
  3. Metacognitive Awareness (0.15) — Was the trader self-aware?
  4. Uncertainty Acknowledgment (0.15) — Were unknowns properly acknowledged?
  5. Metacognitive Monitoring (0.10) — Calibration + resolution (US-328)
  6. Rationale Clarity (0.10) — Could the trader articulate why?
  7. Emotional Regulation (0.05) — Did emotions degrade the process?
  8. Cognitive Reflection (0.03) — Did the trader pause and reflect?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionQualitySignals:
    """Raw signals extracted from conversation + trade metadata."""

    process_adherence_indicators: Dict[str, bool] = field(default_factory=dict)
    info_sources_count: int = 0
    timeframe_check: bool = False
    macro_awareness: bool = False
    metacognitive_mentions: List[str] = field(default_factory=list)
    uncertainty_acknowledgments: List[str] = field(default_factory=list)
    rationale_statements: List[str] = field(default_factory=list)
    emotional_flags: Dict[str, bool] = field(default_factory=dict)
    reflection_indicators: Dict[str, bool] = field(default_factory=dict)
    entry_latency_seconds: Optional[float] = None


@dataclass
class DecisionQualityScore:
    """Per-decision quality score breakdown."""

    timestamp: str
    process_adherence: float  # 0-1
    information_adequacy: float
    metacognitive_awareness: float
    uncertainty_acknowledgment: float
    metacognitive_monitoring: float  # US-328
    rationale_clarity: float
    emotional_regulation: float
    cognitive_reflection: float
    composite_score: float  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for bridge signal."""
        return {
            "timestamp": self.timestamp,
            "dimensions": {
                "process_adherence": round(self.process_adherence, 3),
                "information_adequacy": round(self.information_adequacy, 3),
                "metacognitive_awareness": round(self.metacognitive_awareness, 3),
                "uncertainty_acknowledgment": round(self.uncertainty_acknowledgment, 3),
                "metacognitive_monitoring": round(self.metacognitive_monitoring, 3),
                "rationale_clarity": round(self.rationale_clarity, 3),
                "emotional_regulation": round(self.emotional_regulation, 3),
                "cognitive_reflection": round(self.cognitive_reflection, 3),
            },
            "composite_score": round(self.composite_score, 2),
            "metadata": self.metadata,
        }


class DecisionQualityScorer:
    """Scores trader decision-making process quality (independent of outcome).

    Composite = weighted sum of 8 dimensions, each scored 0-1:
      process_adherence (0.25) + information_adequacy (0.17) +
      metacognitive_awareness (0.15) + uncertainty_acknowledgment (0.15) +
      metacognitive_monitoring (0.10) + rationale_clarity (0.10) +
      emotional_regulation (0.05) + cognitive_reflection (0.03)
    """

    WEIGHTS = {
        "process_adherence": 0.25,
        "information_adequacy": 0.17,
        "metacognitive_awareness": 0.15,
        "uncertainty_acknowledgment": 0.15,
        "metacognitive_monitoring": 0.10,
        "rationale_clarity": 0.10,
        "emotional_regulation": 0.05,
        "cognitive_reflection": 0.03,
    }

    # --- Keyword sets for chat signal detection ---
    PROCESS_KEYWORDS = [
        "rule", "checklist", "supposed to", "before entering", "plan",
        "my system", "criteria", "confirmed", "setup valid",
    ]
    INFO_KEYWORDS = [
        "4h", "1h", "15m", "daily", "weekly", "correlation", "volume",
        "macro", "session", "news", "calendar", "confluence",
    ]
    METACOGNITIVE_KEYWORDS = [
        "not in zone", "biased", "uncertain", "aware", "overconfident",
        "not sure if i should", "questioning myself", "i know i'm",
        "i notice", "my state", "feeling off",
    ]
    UNCERTAINTY_KEYWORDS = [
        "could fail", "not sure", "probably", "percent", "likely",
        "no guarantee", "risk is", "might not work", "50/50",
    ]
    RATIONALE_KEYWORDS = [
        "because", "reason", "since", "broke", "rejected", "confluence",
        "setup is", "entry because", "based on", "supported by",
    ]
    EMOTION_KEYWORDS_MAP = {
        "revenge": ["revenge", "get it back", "make up for"],
        "fomo": ["fomo", "missing out", "everyone is", "don't want to miss"],
        "panic": ["panic", "anxious", "scared", "afraid"],
        "tilt": ["tilt", "annoyed", "frustrated", "angry", "pissed"],
    }
    REFLECTION_KEYWORDS = [
        "waited", "paused", "checked again", "reconsidered",
        "watched", "observed", "took a step back", "slept on it",
    ]

    def __init__(self, min_trades_for_trend: int = 3):
        """Initialize scorer.

        Args:
            min_trades_for_trend: Min trades before rolling average is meaningful
        """
        self.min_trades_for_trend = min_trades_for_trend
        self.trade_history: List[DecisionQualityScore] = []

    def extract_signals(
        self, conversation_text: str, trade_metadata: Optional[Dict[str, Any]] = None
    ) -> DecisionQualitySignals:
        """Extract decision quality signals from conversation + trade metadata.

        Args:
            conversation_text: User's conversation around this trade
            trade_metadata: Optional trade details (pair, timestamps, etc.)

        Returns:
            DecisionQualitySignals instance
        """
        trade_metadata = trade_metadata or {}
        signals = DecisionQualitySignals()
        lower_text = conversation_text.lower()

        # Process Adherence signals
        signals.process_adherence_indicators = {
            "checklist_mentioned": any(kw in lower_text for kw in self.PROCESS_KEYWORDS),
            "position_size_checked": "size" in lower_text or "lot" in lower_text,
            "ratio_mentioned": "ratio" in lower_text or "r:r" in lower_text or "r/r" in lower_text,
            "stop_level_set": ("stop" in lower_text or "sl" in lower_text) and (
                "at" in lower_text or "level" in lower_text or "pips" in lower_text
            ),
        }

        # Information Adequacy signals
        info_hits = sum(1 for kw in self.INFO_KEYWORDS if kw in lower_text)
        signals.info_sources_count = min(info_hits, 6)
        timeframe_mentions = sum(
            1 for tf in ["4h", "1h", "15m", "daily", "weekly", "1d", "5m"]
            if tf in lower_text
        )
        signals.timeframe_check = timeframe_mentions >= 2
        signals.macro_awareness = any(
            w in lower_text for w in ["jobs", "cpi", "fomc", "gdp", "calendar", "economic", "nfp"]
        )

        # Metacognitive Awareness
        signals.metacognitive_mentions = [
            kw for kw in self.METACOGNITIVE_KEYWORDS if kw in lower_text
        ]

        # Uncertainty Acknowledgment
        signals.uncertainty_acknowledgments = [
            kw for kw in self.UNCERTAINTY_KEYWORDS if kw in lower_text
        ]

        # Rationale Clarity
        signals.rationale_statements = [
            kw for kw in self.RATIONALE_KEYWORDS if kw in lower_text
        ]

        # Emotional Regulation
        flags = {}
        for emotion_name, keywords in self.EMOTION_KEYWORDS_MAP.items():
            flags[emotion_name] = any(kw in lower_text for kw in keywords)
        signals.emotional_flags = flags

        # Cognitive Reflection
        signals.reflection_indicators = {
            "waited": any(kw in lower_text for kw in ["waited", "paused"]),
            "checked_again": any(kw in lower_text for kw in ["checked again", "reconsidered"]),
            "observed_first": any(kw in lower_text for kw in ["watched", "observed", "took a step back"]),
        }

        # Entry latency from metadata
        idea_time = trade_metadata.get("idea_time")
        entry_time = trade_metadata.get("entry_time")
        if idea_time is not None and entry_time is not None:
            try:
                latency = float(entry_time) - float(idea_time)
                signals.entry_latency_seconds = max(0.0, latency)
            except (TypeError, ValueError):
                pass

        return signals

    def _score_process_adherence(self, signals: DecisionQualitySignals) -> float:
        """Dimension 1: Process Adherence (weight 0.25)."""
        ind = signals.process_adherence_indicators
        score = (
            (ind.get("checklist_mentioned", False) * 0.4)
            + (ind.get("position_size_checked", False) * 0.2)
            + (ind.get("ratio_mentioned", False) * 0.2)
            + (ind.get("stop_level_set", False) * 0.2)
        )
        return min(1.0, score)

    def _score_information_adequacy(self, signals: DecisionQualitySignals) -> float:
        """Dimension 2: Information Adequacy (weight 0.20)."""
        source_score = min(signals.info_sources_count / 3.0, 1.0) * 0.5
        tf_score = (1.0 if signals.timeframe_check else 0.0) * 0.3
        macro_score = (1.0 if signals.macro_awareness else 0.0) * 0.2
        return min(1.0, source_score + tf_score + macro_score)

    def _score_metacognitive_awareness(self, signals: DecisionQualitySignals) -> float:
        """Dimension 3: Metacognitive Awareness (weight 0.15)."""
        count = len(signals.metacognitive_mentions)
        return min(count / 2.0, 1.0)

    def _score_uncertainty_acknowledgment(self, signals: DecisionQualitySignals) -> float:
        """Dimension 4: Uncertainty Acknowledgment (weight 0.15)."""
        count = len(signals.uncertainty_acknowledgments)
        return min(count / 2.0, 1.0)

    def _score_rationale_clarity(self, signals: DecisionQualitySignals) -> float:
        """Dimension 5: Rationale Clarity (weight 0.10)."""
        count = len(signals.rationale_statements)
        if count >= 2:
            return 1.0
        elif count == 1:
            return 0.6
        return 0.0

    def _score_emotional_regulation(self, signals: DecisionQualitySignals) -> float:
        """Dimension 6: Emotional Regulation (weight 0.10).

        Inverse: more emotional flags = lower score.
        """
        flags = signals.emotional_flags
        emotion_count = sum(1 for v in flags.values() if v)
        penalty = emotion_count * 0.3
        return max(0.0, 1.0 - penalty)

    def _score_cognitive_reflection(self, signals: DecisionQualitySignals) -> float:
        """Dimension 7: Cognitive Reflection (weight 0.05)."""
        ind = signals.reflection_indicators
        indicator_count = sum(1 for v in ind.values() if v)
        indicator_score = min(indicator_count / 2.0, 1.0) * 0.7

        # Entry latency bonus
        latency_score = 0.0
        if signals.entry_latency_seconds is not None and signals.entry_latency_seconds >= 300:
            latency_score = 0.3

        return min(1.0, indicator_score + latency_score)

    def score(
        self,
        conversation_text: str,
        trade_metadata: Optional[Dict[str, Any]] = None,
        metacognitive_monitoring_score: float = 0.5,
    ) -> DecisionQualityScore:
        """Compute decision quality score from conversation + trade metadata.

        Args:
            conversation_text: User conversation around the trade
            trade_metadata: Optional trade details
            metacognitive_monitoring_score: US-328 metacognitive monitoring score (0-1)

        Returns:
            DecisionQualityScore with all 8 dimensions + composite
        """
        signals = self.extract_signals(conversation_text, trade_metadata)

        dim_scores = {
            "process_adherence": self._score_process_adherence(signals),
            "information_adequacy": self._score_information_adequacy(signals),
            "metacognitive_awareness": self._score_metacognitive_awareness(signals),
            "uncertainty_acknowledgment": self._score_uncertainty_acknowledgment(signals),
            "metacognitive_monitoring": metacognitive_monitoring_score,
            "rationale_clarity": self._score_rationale_clarity(signals),
            "emotional_regulation": self._score_emotional_regulation(signals),
            "cognitive_reflection": self._score_cognitive_reflection(signals),
        }

        composite = sum(dim_scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

        result = DecisionQualityScore(
            timestamp=datetime.now(timezone.utc).isoformat(),
            process_adherence=dim_scores["process_adherence"],
            information_adequacy=dim_scores["information_adequacy"],
            metacognitive_awareness=dim_scores["metacognitive_awareness"],
            uncertainty_acknowledgment=dim_scores["uncertainty_acknowledgment"],
            metacognitive_monitoring=dim_scores["metacognitive_monitoring"],
            rationale_clarity=dim_scores["rationale_clarity"],
            emotional_regulation=dim_scores["emotional_regulation"],
            cognitive_reflection=dim_scores["cognitive_reflection"],
            composite_score=composite * 100,
            metadata=trade_metadata or {},
        )

        self.trade_history.append(result)
        logger.info(
            "US-321: Decision quality scored — composite=%.1f (process=%.2f, info=%.2f, meta=%.2f)",
            result.composite_score,
            result.process_adherence,
            result.information_adequacy,
            result.metacognitive_awareness,
        )
        return result

    def get_rolling_average(self, window: int = 10) -> Optional[float]:
        """Rolling average composite score over last N trades."""
        if len(self.trade_history) < self.min_trades_for_trend:
            return None
        recent = self.trade_history[-window:]
        return sum(t.composite_score for t in recent) / len(recent)
