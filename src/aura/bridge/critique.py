"""US-347: Cross-Agent Critique — Structured Feedback Messages.

Inspired by DARWIN's cross-mutation and the self-challenging agent (SCA)
alternating critique pattern. Each agent issues structured observations
about the other's behavior with evidence. Critiques are not commands —
they inform learning without overriding autonomy.

Aura critiques Buddy:
  "You ignored my fatigue flag on 3/5 losing trades this week"

Buddy critiques Aura:
  "Your readiness was >80 on 4 consecutive losing days — recalibrate"

Stored in .aura/bridge/critique_log.jsonl.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CRITIQUE_LOG = "critique_log.jsonl"

# Minimum evidence strength to generate a critique
MIN_PATTERN_CONFIDENCE = 0.5
# Minimum losing trade count to trigger a critique
MIN_LOSING_TRADES = 2


@dataclass
class AgentCritique:
    """A structured observation from one agent about the other's behavior."""

    critique_id: str = ""
    critic: str = ""           # "aura" or "buddy"
    subject: str = ""          # "aura" or "buddy" (who is being critiqued)
    critique_type: str = ""
    # Types: prediction_accuracy, signal_ignored, weight_miscalibration,
    #        false_positive, false_negative, override_pattern
    observation: str = ""      # Natural language summary
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    confidence: float = 0.5
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.critique_id:
            self.critique_id = f"critique-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critique_id": self.critique_id,
            "critic": self.critic,
            "subject": self.subject,
            "critique_type": self.critique_type,
            "observation": self.observation,
            "evidence": self.evidence,
            "suggested_action": self.suggested_action,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCritique":
        return cls(
            critique_id=data.get("critique_id", ""),
            critic=data.get("critic", ""),
            subject=data.get("subject", ""),
            critique_type=data.get("critique_type", ""),
            observation=data.get("observation", ""),
            evidence=data.get("evidence", {}),
            suggested_action=data.get("suggested_action", ""),
            confidence=data.get("confidence", 0.5),
            timestamp=data.get("timestamp", ""),
        )


class CritiqueEngine:
    """Generates and manages cross-agent critiques."""

    def __init__(self, bridge_dir: Path) -> None:
        self.bridge_dir = bridge_dir
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = bridge_dir / _CRITIQUE_LOG
        if not self._log_path.exists():
            self._log_path.touch()

    def generate_aura_critique(
        self,
        outcome_history: List[Dict[str, Any]],
        override_history: List[Dict[str, Any]],
    ) -> Optional[AgentCritique]:
        """Analyze if Buddy ignored Aura's signals and lost.

        Looks for pattern: Aura's readiness was low (or fatigue/tilt flags set),
        Buddy still traded, and the trade lost.

        Args:
            outcome_history: List of outcome dicts with keys:
                pnl_today, win_rate_7d, timestamp, etc.
            override_history: List of override event dicts with keys:
                override_type, outcome, emotional_state, etc.
        """
        if not override_history:
            return None

        # Count overrides where Buddy's recommendation was ignored AND resulted in loss
        ignored_losses = []
        for event in override_history:
            outcome = event.get("outcome", "").lower()
            override_type = event.get("override_type", "")
            emotional_state = event.get("emotional_state", "").lower()

            # Trader took a rejected trade and lost
            if override_type == "took_rejected" and outcome == "loss":
                ignored_losses.append(event)

        if len(ignored_losses) < MIN_LOSING_TRADES:
            return None

        # Check if emotional signals were present during losses
        flagged_states = {"stressed", "fatigued", "anxious"}
        emotionally_flagged = sum(
            1 for e in ignored_losses
            if e.get("emotional_state", "").lower() in flagged_states
        )

        total_overrides = len([
            e for e in override_history if e.get("override_type") == "took_rejected"
        ])

        if total_overrides == 0:
            return None

        loss_rate = len(ignored_losses) / total_overrides
        confidence = min(1.0, loss_rate * 0.8 + (emotionally_flagged / max(1, len(ignored_losses))) * 0.2)

        if confidence < MIN_PATTERN_CONFIDENCE:
            return None

        return AgentCritique(
            critic="aura",
            subject="buddy",
            critique_type="signal_ignored",
            observation=(
                f"Trader overrode {len(ignored_losses)}/{total_overrides} rejected trades "
                f"resulting in losses. {emotionally_flagged} occurred during elevated "
                f"emotional states (stressed/fatigued/anxious)."
            ),
            evidence={
                "total_overrides": total_overrides,
                "losing_overrides": len(ignored_losses),
                "emotionally_flagged": emotionally_flagged,
                "loss_rate": round(loss_rate, 3),
            },
            suggested_action="Increase override gate threshold during elevated emotional states",
            confidence=confidence,
        )

    def generate_buddy_critique(
        self,
        readiness_history: List[Dict[str, Any]],
        outcome_history: List[Dict[str, Any]],
    ) -> Optional[AgentCritique]:
        """Analyze if Aura's readiness was misleading (high readiness + losses).

        Looks for pattern: readiness > 80 but trades lost.

        Args:
            readiness_history: List of readiness signal dicts (readiness_score, timestamp)
            outcome_history: List of outcome dicts (pnl_today, timestamp)
        """
        if not readiness_history or not outcome_history:
            return None

        # Match readiness signals to outcomes by proximity
        high_readiness_losses = 0
        high_readiness_total = 0

        for outcome in outcome_history:
            pnl = outcome.get("pnl_today", 0)
            # Find the closest readiness signal
            closest_readiness = None
            for rs in readiness_history:
                score = rs.get("readiness_score", 0)
                if score > 80:
                    closest_readiness = score
                    break  # Take the first one (most recent if sorted)

            if closest_readiness is not None and closest_readiness > 80:
                high_readiness_total += 1
                if pnl < 0:
                    high_readiness_losses += 1

        if high_readiness_total < MIN_LOSING_TRADES:
            return None
        if high_readiness_losses < MIN_LOSING_TRADES:
            return None

        loss_rate = high_readiness_losses / max(1, high_readiness_total)
        confidence = min(1.0, loss_rate)

        if confidence < MIN_PATTERN_CONFIDENCE:
            return None

        return AgentCritique(
            critic="buddy",
            subject="aura",
            critique_type="prediction_accuracy",
            observation=(
                f"Readiness score was >80 on {high_readiness_total} trading days, "
                f"but {high_readiness_losses} resulted in losses "
                f"({loss_rate:.0%} loss rate). "
                f"Consider recalibrating emotional_state weight."
            ),
            evidence={
                "high_readiness_days": high_readiness_total,
                "losing_days": high_readiness_losses,
                "loss_rate": round(loss_rate, 3),
            },
            suggested_action="Recalibrate emotional_state component weight in readiness computation",
            confidence=confidence,
        )

    def log_critique(self, critique: AgentCritique) -> None:
        """Append a critique to critique_log.jsonl."""
        from src.aura.bridge.signals import FeedbackBridge

        line = critique.to_json() + "\n"
        FeedbackBridge._locked_append(self._log_path, line)
        logger.info(
            "US-347: Logged critique '%s' from %s about %s (type=%s, confidence=%.2f)",
            critique.critique_id,
            critique.critic,
            critique.subject,
            critique.critique_type,
            critique.confidence,
        )

    def get_recent_critiques(
        self,
        critic: Optional[str] = None,
        limit: int = 10,
    ) -> List[AgentCritique]:
        """Read recent critiques from log, optionally filtered by critic."""
        from src.aura.bridge.signals import FeedbackBridge

        raw = FeedbackBridge._locked_read(self._log_path)
        if not raw:
            return []

        critiques: List[AgentCritique] = []
        for line in raw.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                c = AgentCritique.from_dict(data)
                if critic is None or c.critic == critic:
                    critiques.append(c)
            except (json.JSONDecodeError, ValueError):
                continue

        return critiques[-limit:]
