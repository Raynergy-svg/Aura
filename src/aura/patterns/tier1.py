"""Tier 1 Pattern Engine — Daily frequency-based patterns.

Lightweight pattern detection that runs after every conversation or trade cycle.
Detects:
  - Emotional frequency patterns (e.g., "stressed 4 of last 7 conversations")
  - Override frequency patterns (e.g., "3 overrides in 48 hours")
  - Readiness trend patterns (e.g., "readiness declining 5 consecutive sessions")
  - Stressor recurrence (e.g., "career_decision mentioned in 80% of conversations")

T1 patterns are cheap to compute — they're counting and averaging over
recent data windows. They feed into the T2 cross-domain engine as inputs.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.persistence import atomic_write_json  # Fix C-05: needed for atomic pattern writes

from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)

logger = logging.getLogger(__name__)

# --- Detection thresholds ---
STRESS_FREQUENCY_THRESHOLD = 0.6   # 60%+ conversations show stress
OVERRIDE_FREQUENCY_THRESHOLD = 3   # 3+ overrides in window
READINESS_DECLINE_STREAK = 3       # 3+ consecutive declines
STRESSOR_RECURRENCE_THRESHOLD = 0.5  # Same stressor in 50%+ of conversations

# US-298: Temporal decay constants (matching node decay from US-291)
EVIDENCE_DECAY_LAMBDA = 0.05  # ~14-day half-life
EVIDENCE_ARCHIVE_THRESHOLD = 0.1  # Auto-archive patterns below this


def get_decay_weighted_confidence(pattern: "DetectedPattern", now: Optional[datetime] = None) -> float:
    """Compute decay-weighted confidence for a pattern based on evidence freshness.

    US-298: Each evidence item decays with exp(-LAMBDA * days_old).
    Total confidence = base_confidence * (sum of decay weights / count).
    Patterns with decay-weighted confidence < ARCHIVE_THRESHOLD should be archived.

    Args:
        pattern: The detected pattern with evidence items
        now: Current time (default: UTC now)

    Returns:
        Decay-weighted confidence value (0.0 to pattern.confidence)
    """
    # M-02 (FOLLOWUP): Guard against empty evidence (division by zero at avg_weight line).
    # This guard was already present — confirmed correct. Pattern.confidence is returned
    # unchanged when no evidence exists, which is intentional (new pattern = full confidence).
    if not pattern.evidence:
        return pattern.confidence

    now = now or datetime.now(timezone.utc)
    total_weight = 0.0

    for evidence in pattern.evidence:
        try:
            ts = datetime.fromisoformat(evidence.timestamp.replace("Z", "+00:00"))
            days_old = max(0.0, (now - ts).total_seconds() / 86400.0)
        except (ValueError, TypeError, AttributeError):
            # Corrupted timestamps should NOT be treated as fresh — assign zero weight
            weight = 0.0
            total_weight += weight
            continue

        weight = math.exp(-EVIDENCE_DECAY_LAMBDA * days_old)
        total_weight += weight

    avg_weight = total_weight / len(pattern.evidence)
    return pattern.confidence * avg_weight


class Tier1FrequencyDetector:
    """Daily frequency-based pattern detection.

    Scans recent conversation history, readiness history, and override logs
    to detect recurring patterns based on simple frequency counts.

    Args:
        patterns_dir: Where to persist detected patterns
    """

    def __init__(self, patterns_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.patterns_dir = patterns_dir or Path(".aura/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = self.patterns_dir / "t1_patterns.json"
        self._active_patterns: Dict[str, DetectedPattern] = {}

        # US-272: Configurable thresholds — override module-level defaults
        cfg = config or {}
        self.stress_frequency_threshold = cfg.get("stress_frequency_threshold", STRESS_FREQUENCY_THRESHOLD)
        self.override_frequency_threshold = cfg.get("override_frequency_threshold", OVERRIDE_FREQUENCY_THRESHOLD)
        self.readiness_decline_streak = cfg.get("readiness_decline_streak", READINESS_DECLINE_STREAK)
        self.stressor_recurrence_threshold = cfg.get("stressor_recurrence_threshold", STRESSOR_RECURRENCE_THRESHOLD)

        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load persisted patterns from disk."""
        if not self._patterns_file.exists():
            return
        try:
            data = json.loads(self._patterns_file.read_text())
            for pid, pdata in data.items():
                evidence = [EvidenceItem(**e) for e in pdata.pop("evidence", [])]
                pdata["tier"] = PatternTier(pdata["tier"])
                pdata["domain"] = PatternDomain(pdata["domain"])
                pdata["status"] = PatternStatus(pdata["status"])
                self._active_patterns[pid] = DetectedPattern(
                    evidence=evidence, **pdata
                )
        except Exception as e:
            logger.warning(f"T1: Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Persist patterns to disk atomically.

        Fix C-05: Previously used write_text() (non-atomic). Now uses atomic_write_json()
        to prevent pattern data corruption on crash or power loss.
        """
        try:
            data = {
                pid: p.to_dict()
                for pid, p in self._active_patterns.items()
                if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
            }
            atomic_write_json(self._patterns_file, data)
        except Exception as e:
            logger.error(f"T1: Failed to save patterns: {e}")

    def detect(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
        drift_warnings: Optional[List[Dict]] = None,
    ) -> List[DetectedPattern]:
        """Run all T1 frequency detections.

        Args:
            conversations: Recent conversation records from self-model DB
            readiness_history: Recent readiness score entries
            override_events: Recent override events from bridge
            drift_warnings: Optional list of style drift warnings from ConversationProcessor (US-347)

        Returns:
            List of newly detected or updated patterns
        """
        new_patterns: List[DetectedPattern] = []

        new_patterns.extend(self._detect_emotional_frequency(conversations))
        new_patterns.extend(self._detect_stressor_recurrence(conversations))
        new_patterns.extend(self._detect_override_frequency(override_events))
        new_patterns.extend(self._detect_readiness_trends(readiness_history))

        # US-347: Style drift early-warning patterns
        if drift_warnings:
            for warning in drift_warnings:
                avg_drift = warning.get("avg_drift", 0.0)
                count = warning.get("consecutive_count", 0)
                confidence = min(1.0, avg_drift * 1.5)
                pattern_key = f"style_drift_warning_{count}"
                description = f"Linguistic style drift sustained ({count} messages)"

                result = self._upsert_pattern(
                    pattern_key=pattern_key,
                    domain=PatternDomain.HUMAN,
                    description=description,
                    evidence=EvidenceItem(
                        source_type="style_drift_detector",
                        source_id=f"drift_warning_{count}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        summary=f"Linguistic style drift sustained ({count} consecutive messages)",
                        data={
                            "avg_drift": avg_drift,
                            "max_drift": warning.get("max_drift", 0.0),
                            "consecutive_count": count,
                        },
                    ),
                    confidence=confidence,
                    suggested_rule=(
                        f"Style drift sustained at {avg_drift:.2%} intensity over {count} messages — "
                        f"human communication pattern shifting"
                    ),
                )
                new_patterns.extend(result)
                logger.info("US-347: Style drift T1 pattern — count=%d, avg=%.3f, confidence=%.3f",
                           count, avg_drift, confidence)

        # US-298: Auto-archive patterns whose evidence has decayed below threshold
        self._archive_decayed_patterns()

        self._save_patterns()

        if new_patterns:
            logger.info(f"T1: Detected {len(new_patterns)} patterns")

        return new_patterns

    def _detect_emotional_frequency(
        self, conversations: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect if negative emotional states dominate recent conversations."""
        if len(conversations) < 3:
            return []

        negative_states = {"stressed", "anxious", "fatigued", "frustrated", "overwhelmed"}
        recent = conversations[:7]  # Last 7 conversations
        negative_count = sum(
            1 for c in recent
            if c.get("emotional_state", "").lower() in negative_states
        )
        ratio = negative_count / len(recent)

        if ratio < self.stress_frequency_threshold:
            return []

        pattern_key = "emotional_frequency_negative"
        dominant_emotion = Counter(
            c.get("emotional_state", "neutral") for c in recent
            if c.get("emotional_state", "").lower() in negative_states
        ).most_common(1)

        dominant = dominant_emotion[0][0] if dominant_emotion else "stressed"
        description = (
            f"Negative emotional state ({dominant}) in {negative_count}/{len(recent)} "
            f"recent conversations ({ratio:.0%})"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.HUMAN,
            description=description,
            evidence=EvidenceItem(
                source_type="conversation_aggregate",
                source_id=f"last_{len(recent)}_conversations",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{negative_count}/{len(recent)} conversations negative ({dominant})",
                data={
                    "negative_count": negative_count,
                    "total": len(recent),
                    "ratio": round(ratio, 3),
                    "dominant_emotion": dominant,
                },
            ),
            confidence=min(0.9, 0.5 + ratio * 0.4),
            suggested_rule=(
                f"When emotional frequency of '{dominant}' exceeds {self.stress_frequency_threshold:.0%} "
                f"in recent conversations, reduce readiness score by 15%"
            ),
        )

    def _detect_stressor_recurrence(
        self, conversations: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect if specific stressors keep coming up across conversations."""
        if len(conversations) < 4:
            return []

        recent = conversations[:10]
        stressor_counts: Counter = Counter()

        for conv in recent:
            topics_raw = conv.get("topics", "[]")
            if isinstance(topics_raw, str):
                try:
                    topics = json.loads(topics_raw)
                except (json.JSONDecodeError, TypeError):
                    topics = []
            else:
                topics = topics_raw

            for topic in topics:
                stressor_counts[topic] += 1

        patterns = []
        for stressor, count in stressor_counts.items():
            ratio = count / len(recent)
            if ratio >= self.stressor_recurrence_threshold:
                pattern_key = f"stressor_recurrence_{stressor}"
                description = (
                    f"'{stressor}' appears in {count}/{len(recent)} recent "
                    f"conversations ({ratio:.0%}) — persistent stressor"
                )
                result = self._upsert_pattern(
                    pattern_key=pattern_key,
                    domain=PatternDomain.HUMAN,
                    description=description,
                    evidence=EvidenceItem(
                        source_type="stressor_frequency",
                        source_id=stressor,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        summary=f"'{stressor}' in {count}/{len(recent)} conversations",
                        data={"stressor": stressor, "count": count, "total": len(recent)},
                    ),
                    confidence=min(0.85, 0.5 + ratio * 0.35),
                )
                patterns.extend(result)

        return patterns

    def _detect_override_frequency(
        self, override_events: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect clustering of override events."""
        if len(override_events) < self.override_frequency_threshold:
            return []

        recent = override_events[-10:]  # Last 10 overrides
        losing = [e for e in recent if e.get("outcome") == "loss"]
        losing_rate = len(losing) / max(len(recent), 1)

        if len(recent) < self.override_frequency_threshold:
            return []

        pattern_key = "override_frequency_high"
        description = (
            f"{len(recent)} override events recently, "
            f"{len(losing)} losses ({losing_rate:.0%} loss rate)"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.TRADING,
            description=description,
            evidence=EvidenceItem(
                source_type="override_aggregate",
                source_id=f"last_{len(recent)}_overrides",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{len(recent)} overrides, {losing_rate:.0%} loss rate",
                data={
                    "total_overrides": len(recent),
                    "losing_overrides": len(losing),
                    "loss_rate": round(losing_rate, 3),
                },
            ),
            confidence=min(0.9, 0.5 + losing_rate * 0.4),
            suggested_rule=(
                f"Override loss rate at {losing_rate:.0%} — increase consensus "
                f"threshold when readiness < 60"
            ),
        )

    def _detect_readiness_trends(
        self, readiness_history: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect sustained readiness score decline."""
        if len(readiness_history) < self.readiness_decline_streak:
            return []

        # readiness_history is newest-first from the DB query
        recent = readiness_history[:10]
        scores = [r.get("score", 70) for r in recent]

        # Check for sustained decline (each score lower than previous)
        decline_streak = 0
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:  # newer < older = decline
                decline_streak += 1
            else:
                break

        if decline_streak < self.readiness_decline_streak:
            return []

        drop_amount = scores[-1] - scores[0]  # Positive = decline (newest is lower)
        pattern_key = "readiness_declining_streak"
        description = (
            f"Readiness score declining for {decline_streak} consecutive sessions "
            f"(from {scores[-1]:.0f} to {scores[0]:.0f}, drop of {abs(drop_amount):.0f} points)"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.HUMAN,
            description=description,
            evidence=EvidenceItem(
                source_type="readiness_trend",
                source_id=f"streak_{decline_streak}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{decline_streak}-session decline from {scores[-1]:.0f} to {scores[0]:.0f}",
                data={
                    "streak_length": decline_streak,
                    "start_score": scores[-1],
                    "end_score": scores[0],
                    "scores": [round(s, 1) for s in scores],
                },
            ),
            confidence=min(0.9, 0.5 + decline_streak * 0.1),
            suggested_rule=(
                f"Readiness declining {decline_streak} sessions — "
                f"trigger proactive check-in with user"
            ),
        )

    def _upsert_pattern(
        self,
        pattern_key: str,
        domain: PatternDomain,
        description: str,
        evidence: EvidenceItem,
        confidence: float = 0.5,
        suggested_rule: Optional[str] = None,
    ) -> List[DetectedPattern]:
        """Create or update a pattern. Returns list with the pattern if new/updated."""
        existing = self._active_patterns.get(pattern_key)

        if existing and existing.status not in (
            PatternStatus.ARCHIVED,
            PatternStatus.INVALIDATED,
        ):
            existing.add_evidence(evidence)
            existing.confidence = confidence
            existing.description = description
            if suggested_rule:
                existing.suggested_rule = suggested_rule
            logger.debug(
                f"T1: Updated pattern '{pattern_key}' "
                f"(obs={existing.observation_count}, conf={confidence:.2f})"
            )
            return [existing]

        pattern = DetectedPattern(
            pattern_id=pattern_key,
            tier=PatternTier.T1_DAILY,
            domain=domain,
            description=description,
            evidence=[evidence],
            observation_count=1,
            confidence=confidence,
            suggested_rule=suggested_rule,
        )
        self._active_patterns[pattern_key] = pattern
        logger.info(f"T1: New pattern detected: '{pattern_key}'")
        return [pattern]

    def _archive_decayed_patterns(self) -> int:
        """US-298: Archive patterns whose decay-weighted confidence drops below threshold.

        Returns:
            Count of patterns archived.
        """
        archived_count = 0
        now = datetime.now(timezone.utc)

        for pid, pattern in list(self._active_patterns.items()):
            if pattern.status in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED):
                continue

            dwc = get_decay_weighted_confidence(pattern, now)
            if dwc < EVIDENCE_ARCHIVE_THRESHOLD:
                pattern.status = PatternStatus.ARCHIVED
                archived_count += 1
                logger.info(
                    f"T1: Auto-archived pattern '{pid}' — "
                    f"decay-weighted confidence {dwc:.3f} < {EVIDENCE_ARCHIVE_THRESHOLD}"
                )

        return archived_count

    def get_active_patterns(self) -> List[DetectedPattern]:
        """Return all non-archived T1 patterns."""
        return [
            p for p in self._active_patterns.values()
            if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
        ]

    def get_promotable_patterns(self) -> List[DetectedPattern]:
        """Return patterns ready for rule promotion."""
        return [p for p in self._active_patterns.values() if p.is_promotable()]
