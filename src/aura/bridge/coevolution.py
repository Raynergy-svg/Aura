"""US-348: Co-Evolution Weight Drift — Automatic Calibration from Outcome Correlation.

Inspired by Agent0's bidirectional curriculum pressure and EvoAgentX's
prompt optimization. Uses calibration scores (US-346) and critique
evidence (US-347) to automatically adjust how much weight each system
gives the other's signals.

Key safety: max_drift_per_cycle caps weight changes at ±0.1 per update
to prevent runaway instability (lesson from DGM's unrestricted
self-modification).

Weight ranges:
  aura_outcome_weight:        0.5 – 1.5 (how much Aura weights Buddy's outcomes in T2)
  signal_weight_recommendation: 0.3 – 1.5 (how much Buddy should weight readiness)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Weight bounds
AURA_OUTCOME_WEIGHT_MIN = 0.5
AURA_OUTCOME_WEIGHT_MAX = 1.5
SIGNAL_WEIGHT_MIN = 0.3
SIGNAL_WEIGHT_MAX = 1.5

# Default drift cap
DEFAULT_MAX_DRIFT = 0.1

# Calibration thresholds for weight adjustment
HIGH_CALIBRATION = 0.7
LOW_CALIBRATION = 0.4

# Critique confidence threshold for weight nudge
CRITIQUE_CONFIDENCE_THRESHOLD = 0.8
CRITIQUE_NUDGE_AMOUNT = 0.05


@dataclass
class WeightChange:
    """Record of a weight adjustment for audit trail."""

    timestamp: str = ""
    parameter: str = ""       # "aura_outcome_weight" or "signal_weight_recommendation"
    old_value: float = 0.0
    new_value: float = 0.0
    trigger: str = ""         # "calibration" | "critique" | "manual"
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "parameter": self.parameter,
            "old_value": round(self.old_value, 4),
            "new_value": round(self.new_value, 4),
            "trigger": self.trigger,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightChange":
        return cls(
            timestamp=data.get("timestamp", ""),
            parameter=data.get("parameter", ""),
            old_value=data.get("old_value", 0.0),
            new_value=data.get("new_value", 0.0),
            trigger=data.get("trigger", ""),
            rationale=data.get("rationale", ""),
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply_drift(current: float, target: float, max_drift: float) -> float:
    """Move current toward target, capped by max_drift with damping.

    Applies 0.5 damping factor when delta is within max_drift to prevent
    oscillation between extremes (M-01 fix).
    """
    delta = target - current
    if abs(delta) > max_drift:
        delta = max_drift if delta > 0 else -max_drift
    else:
        delta *= 0.5  # Damping to prevent oscillation
    return current + delta


class CoEvolutionManager:
    """Manages bidirectional weight drift between Aura and Buddy.

    Better predictions earn more influence; poor predictions lose it.
    All weight changes are bounded to prevent runaway instability.
    """

    def __init__(self, max_drift_per_cycle: float = DEFAULT_MAX_DRIFT) -> None:
        self.aura_outcome_weight: float = 1.0
        self.signal_weight_recommendation: float = 1.0
        self.weight_history: List[WeightChange] = []
        self.max_drift_per_cycle: float = max_drift_per_cycle

    def update_from_calibration(
        self,
        aura_cal: float,
        buddy_cal: float,
    ) -> None:
        """Adjust weights based on calibration scores.

        Buddy calibration → aura_outcome_weight:
          High (>0.7): drift toward 1.5 (trust Buddy's outcomes more)
          Low  (<0.4): drift toward 0.5 (discount Buddy's outcomes)

        Aura calibration → signal_weight_recommendation:
          High (>0.7): drift toward 1.5 (recommend Buddy trust readiness more)
          Low  (<0.4): drift toward 0.3 (recommend Buddy discount readiness)
        """
        # --- Buddy calibration → aura_outcome_weight ---
        old_aow = self.aura_outcome_weight
        if buddy_cal > HIGH_CALIBRATION:
            target = AURA_OUTCOME_WEIGHT_MAX
        elif buddy_cal < LOW_CALIBRATION:
            target = AURA_OUTCOME_WEIGHT_MIN
        else:
            target = 1.0  # Neutral — drift toward default

        self.aura_outcome_weight = _clamp(
            _apply_drift(self.aura_outcome_weight, target, self.max_drift_per_cycle),
            AURA_OUTCOME_WEIGHT_MIN,
            AURA_OUTCOME_WEIGHT_MAX,
        )
        if self.aura_outcome_weight != old_aow:
            self._record_change(
                "aura_outcome_weight",
                old_aow,
                self.aura_outcome_weight,
                "calibration",
                f"buddy_cal={buddy_cal:.3f}, target={target:.2f}",
            )

        # --- Aura calibration → signal_weight_recommendation ---
        old_swr = self.signal_weight_recommendation
        if aura_cal > HIGH_CALIBRATION:
            target = SIGNAL_WEIGHT_MAX
        elif aura_cal < LOW_CALIBRATION:
            target = SIGNAL_WEIGHT_MIN
        else:
            target = 1.0

        self.signal_weight_recommendation = _clamp(
            _apply_drift(self.signal_weight_recommendation, target, self.max_drift_per_cycle),
            SIGNAL_WEIGHT_MIN,
            SIGNAL_WEIGHT_MAX,
        )
        if self.signal_weight_recommendation != old_swr:
            self._record_change(
                "signal_weight_recommendation",
                old_swr,
                self.signal_weight_recommendation,
                "calibration",
                f"aura_cal={aura_cal:.3f}, target={target:.2f}",
            )

    def update_from_critique(self, critique: Any) -> None:
        """Nudge weights based on a high-confidence critique.

        Only acts on weight_miscalibration critiques with confidence > 0.8.
        Nudge amount is fixed at 0.05 to prevent large jumps from single events.
        """
        if not hasattr(critique, "confidence") or not hasattr(critique, "critique_type"):
            return
        if critique.confidence < CRITIQUE_CONFIDENCE_THRESHOLD:
            return
        if critique.critique_type != "weight_miscalibration":
            return

        # Determine which weight to nudge based on subject
        if critique.subject == "aura":
            # Aura is miscalibrated — reduce signal_weight_recommendation
            old = self.signal_weight_recommendation
            self.signal_weight_recommendation = _clamp(
                self.signal_weight_recommendation - CRITIQUE_NUDGE_AMOUNT,
                SIGNAL_WEIGHT_MIN,
                SIGNAL_WEIGHT_MAX,
            )
            if self.signal_weight_recommendation != old:
                self._record_change(
                    "signal_weight_recommendation",
                    old,
                    self.signal_weight_recommendation,
                    "critique",
                    f"critique_id={critique.critique_id}",
                )
        elif critique.subject == "buddy":
            # Buddy is miscalibrated — reduce aura_outcome_weight
            old = self.aura_outcome_weight
            self.aura_outcome_weight = _clamp(
                self.aura_outcome_weight - CRITIQUE_NUDGE_AMOUNT,
                AURA_OUTCOME_WEIGHT_MIN,
                AURA_OUTCOME_WEIGHT_MAX,
            )
            if self.aura_outcome_weight != old:
                self._record_change(
                    "aura_outcome_weight",
                    old,
                    self.aura_outcome_weight,
                    "critique",
                    f"critique_id={critique.critique_id}",
                )

    def get_aura_outcome_weight(self) -> float:
        """Weight for T2 PatternEngine when processing outcome-derived correlations."""
        return self.aura_outcome_weight

    def get_signal_weight_recommendation(self) -> float:
        """Weight recommendation included in ReadinessSignal for Buddy."""
        return self.signal_weight_recommendation

    def _record_change(
        self, parameter: str, old: float, new: float, trigger: str, rationale: str
    ) -> None:
        change = WeightChange(
            parameter=parameter,
            old_value=old,
            new_value=new,
            trigger=trigger,
            rationale=rationale,
        )
        self.weight_history.append(change)
        logger.info(
            "US-348: Weight drift — %s: %.4f → %.4f (trigger=%s)",
            parameter, old, new, trigger,
        )

    # --- Persistence (stored alongside calibration data) ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aura_outcome_weight": round(self.aura_outcome_weight, 4),
            "signal_weight_recommendation": round(self.signal_weight_recommendation, 4),
            "max_drift_per_cycle": self.max_drift_per_cycle,
            "weight_history": [w.to_dict() for w in self.weight_history[-50:]],
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.aura_outcome_weight = data.get("aura_outcome_weight", 1.0)
        self.signal_weight_recommendation = data.get("signal_weight_recommendation", 1.0)
        self.max_drift_per_cycle = data.get("max_drift_per_cycle", DEFAULT_MAX_DRIFT)
        self.weight_history = [
            WeightChange.from_dict(w) for w in data.get("weight_history", [])
        ]
