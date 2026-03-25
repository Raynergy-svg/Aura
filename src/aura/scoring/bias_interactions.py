"""Bias interaction scoring — detects dangerous cognitive bias combinations.

Behavioral finance research shows certain bias pairs multiply risk non-linearly:
- Confirmation + Anchoring = locking onto wrong price level
- Sunk Cost + Loss Aversion = paralysis on losing positions
- Overconfidence + Hindsight = inability to learn from losses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiasInteractionResult:
    """Result of bias interaction analysis."""
    interaction_pairs: List[str] = field(default_factory=list)  # active pair descriptions
    interaction_penalty: float = 0.0   # additional penalty from interactions (0-10)
    total_pair_count: int = 0          # number of active dangerous pairs


# Dangerous bias pairs with risk multipliers
# Key: (bias_a, bias_b) — order doesn't matter
# Value: (multiplier, description)
DANGEROUS_PAIRS: Dict[Tuple[str, str], Tuple[float, str]] = {
    ("confirmation_bias", "anchoring"): (1.5, "Locking onto wrong price level — seeks confirming evidence for anchored price"),
    ("sunk_cost", "loss_aversion"): (1.4, "Paralysis on losing positions — can't cut losses due to invested cost"),
    ("overconfidence", "hindsight_bias"): (1.3, "Unable to learn from losses — hindsight rewrites history, overconfidence persists"),
    ("recency_bias", "anchoring"): (1.3, "Recent price anchoring — over-weights latest move as the anchor point"),
    ("confirmation_bias", "overconfidence"): (1.4, "Conviction without evidence — seeks only confirming data with excessive certainty"),
}


class BiasInteractionScorer:
    """Detects active dangerous bias pairs and computes interaction penalties."""

    def __init__(self, activation_threshold: float = 0.5):
        """
        Args:
            activation_threshold: Both biases in a pair must exceed this for interaction to fire.
        """
        self.activation_threshold = activation_threshold

    def score(self, bias_scores: Dict[str, float]) -> BiasInteractionResult:
        """Compute bias interaction penalty.

        Args:
            bias_scores: Dict mapping bias names to scores (0-1).
                         Expected keys: confirmation_bias, anchoring, sunk_cost,
                         loss_aversion, overconfidence, hindsight_bias, recency_bias,
                         attribution_error (not all required)

        Returns:
            BiasInteractionResult with active pairs and penalty
        """
        if not bias_scores:
            return BiasInteractionResult()

        active_pairs: List[str] = []
        total_penalty = 0.0

        for (bias_a, bias_b), (multiplier, description) in DANGEROUS_PAIRS.items():
            score_a = bias_scores.get(bias_a, 0.0)
            score_b = bias_scores.get(bias_b, 0.0)

            if score_a > self.activation_threshold and score_b > self.activation_threshold:
                # Interaction penalty = (score_a * score_b) * multiplier * 3.0
                pair_penalty = (score_a * score_b) * multiplier * 3.0
                total_penalty += pair_penalty

                pair_label = f"{bias_a}+{bias_b} ({score_a:.2f}×{score_b:.2f}, mult={multiplier})"
                active_pairs.append(pair_label)

                logger.info(
                    "US-354: Dangerous bias pair detected — %s+%s "
                    "(%.2f×%.2f, multiplier=%.1f, penalty=%.2f): %s",
                    bias_a, bias_b, score_a, score_b, multiplier, pair_penalty, description
                )

        # Cap interaction penalty at 10
        interaction_penalty = min(10.0, total_penalty)

        if active_pairs:
            logger.warning(
                "US-354: %d dangerous bias pair(s) active — interaction penalty=%.1f",
                len(active_pairs), interaction_penalty
            )

        return BiasInteractionResult(
            interaction_pairs=active_pairs,
            interaction_penalty=round(interaction_penalty, 2),
            total_pair_count=len(active_pairs)
        )
