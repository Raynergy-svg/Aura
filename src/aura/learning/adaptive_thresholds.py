"""Adaptive threshold learning via Thompson sampling contextual bandits.

Learns personalized thresholds for readiness signals (style_drift, reliability,
anomaly_severity, bias_penalty, override_risk, tilt) by tracking which threshold
values produce better outcomes per time-of-day context.

Based on contextual bandit literature (2025) and large-scale adaptive tutoring research.
"""

import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Time-of-day contexts
CONTEXTS = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
    "night": (0, 6),
}

# Default threshold configurations
DEFAULT_THRESHOLDS = {
    "style_drift": {"default": 0.4, "min": 0.2, "max": 0.8, "num_candidates": 5},
    "reliability": {"default": 0.5, "min": 0.3, "max": 0.7, "num_candidates": 5},
    "anomaly_severity": {"default": 0.5, "min": 0.3, "max": 0.8, "num_candidates": 5},
    "bias_penalty": {"default": 0.5, "min": 0.3, "max": 0.7, "num_candidates": 5},
    "override_risk": {"default": 0.7, "min": 0.5, "max": 0.9, "num_candidates": 5},
    "tilt": {"default": 0.6, "min": 0.3, "max": 0.8, "num_candidates": 5},
}


@dataclass
class ThresholdCandidate:
    """A candidate threshold value with Beta-Binomial posterior."""
    value: float
    alpha: float = 2.0   # success count + prior
    beta: float = 2.0    # failure count + prior

    @property
    def sample_count(self) -> int:
        """Total observations (excluding prior)."""
        return int(self.alpha + self.beta - 4)  # subtract initial priors

    def sample(self) -> float:
        """Thompson sample from Beta posterior."""
        return random.betavariate(max(0.01, self.alpha), max(0.01, self.beta))

    def update(self, outcome: float):
        """Update posterior with outcome (0-1)."""
        # Treat outcome as probability of success
        self.alpha += outcome
        self.beta += (1.0 - outcome)

    def to_dict(self) -> Dict[str, float]:
        return {"value": self.value, "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ThresholdCandidate":
        return cls(value=d["value"], alpha=d["alpha"], beta=d["beta"])


class AdaptiveThresholdLearner:
    """Thompson sampling contextual bandit for learning personalized thresholds."""

    MIN_SAMPLES = 20  # Minimum observations before overriding defaults

    def __init__(self, thresholds_config: Optional[Dict[str, Dict]] = None):
        """
        Args:
            thresholds_config: Maps threshold_name -> {default, min, max, num_candidates}.
                             Uses DEFAULT_THRESHOLDS if None.
        """
        self.config = thresholds_config or DEFAULT_THRESHOLDS

        # Build candidate sets: {threshold_name: {context: [ThresholdCandidate, ...]}}
        self._candidates: Dict[str, Dict[str, List[ThresholdCandidate]]] = {}

        for name, cfg in self.config.items():
            self._candidates[name] = {}
            num = cfg.get("num_candidates", 5)
            mn = cfg["min"]
            mx = cfg["max"]
            default = cfg["default"]

            # Generate evenly spaced candidates
            if num <= 1:
                values = [default]
            else:
                step = (mx - mn) / (num - 1)
                values = [round(mn + i * step, 3) for i in range(num)]

            for ctx in CONTEXTS:
                candidates = []
                for v in values:
                    # Stronger prior for candidate closest to default
                    dist = abs(v - default)
                    if num <= 1 or dist < step / 2:
                        alpha, beta = 5.0, 5.0  # strong prior on default
                    else:
                        alpha, beta = 2.0, 2.0  # weak prior on alternatives
                    candidates.append(ThresholdCandidate(value=v, alpha=alpha, beta=beta))
                self._candidates[name][ctx] = candidates

    @staticmethod
    def get_context(hour: int) -> str:
        """Map hour (0-23) to context string."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"

    def get_threshold(self, name: str, context: str) -> float:
        """Get threshold via Thompson sampling.

        Returns default if insufficient samples (< MIN_SAMPLES).
        """
        if name not in self._candidates:
            logger.warning("US-353: Unknown threshold '%s', returning 0.5", name)
            return 0.5

        if context not in self._candidates[name]:
            context = "morning"  # fallback

        candidates = self._candidates[name][context]

        # Guard against empty candidates list
        if not candidates:
            default = self.config[name]["default"]
            logger.warning("US-353: No candidates for %s/%s, returning default %.3f", name, context, default)
            return default

        # Check if we have enough samples
        total_samples = sum(c.sample_count for c in candidates)
        if total_samples < self.MIN_SAMPLES:
            # Return default
            default = self.config[name]["default"]
            return default

        # Thompson sample: pick candidate with highest sample
        best = max(candidates, key=lambda c: c.sample())
        return best.value

    def update(self, name: str, context: str, threshold_used: float, outcome: float):
        """Update posterior for the threshold that was used.

        Args:
            name: Threshold name (e.g., "style_drift")
            context: Time context (e.g., "morning")
            threshold_used: The threshold value that was applied
            outcome: 0-1 score (1 = good outcome, 0 = bad outcome)
        """
        if name not in self._candidates:
            return
        if context not in self._candidates[name]:
            return

        outcome = min(1.0, max(0.0, outcome))

        # Find closest candidate to threshold_used
        candidates = self._candidates[name][context]
        closest = min(candidates, key=lambda c: abs(c.value - threshold_used))
        closest.update(outcome)

        logger.debug(
            "US-353: Updated %s/%s threshold=%.3f outcome=%.2f "
            "(alpha=%.1f, beta=%.1f, samples=%d)",
            name, context, closest.value, outcome,
            closest.alpha, closest.beta, closest.sample_count
        )

    def save_state(self, path: Path):
        """Persist learner state to JSON."""
        state = {}
        for name, contexts in self._candidates.items():
            state[name] = {}
            for ctx, candidates in contexts.items():
                state[name][ctx] = [c.to_dict() for c in candidates]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(path)

        logger.info("US-353: Saved adaptive threshold state to %s", path)

    def load_state(self, path: Path) -> bool:
        """Load learner state from JSON. Returns True if loaded successfully."""
        path = Path(path)
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            for name, contexts in data.items():
                if name not in self._candidates:
                    continue
                for ctx, candidates_data in contexts.items():
                    if ctx not in self._candidates[name]:
                        continue
                    loaded = [ThresholdCandidate.from_dict(d) for d in candidates_data]
                    if len(loaded) == len(self._candidates[name][ctx]):
                        self._candidates[name][ctx] = loaded
            logger.info("US-353: Loaded adaptive threshold state from %s", path)
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("US-353: Failed to load threshold state: %s", e)
            return False

    def get_all_thresholds(self, context: str) -> Dict[str, float]:
        """Get all thresholds for a given context."""
        result = {}
        for name in self._candidates:
            result[name] = self.get_threshold(name, context)
        return result

    def get_stats(self, name: str, context: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific threshold in a context."""
        if name not in self._candidates or context not in self._candidates[name]:
            return None

        candidates = self._candidates[name][context]
        total_samples = sum(c.sample_count for c in candidates)

        return {
            "threshold_name": name,
            "context": context,
            "total_samples": total_samples,
            "using_learned": total_samples >= self.MIN_SAMPLES,
            "candidates": [
                {
                    "value": c.value,
                    "alpha": round(c.alpha, 1),
                    "beta": round(c.beta, 1),
                    "mean": round(c.alpha / (c.alpha + c.beta), 3),
                    "samples": c.sample_count
                }
                for c in candidates
            ]
        }
