"""Goal-Emotion Coupling Analysis.

US-358: Detects approach vs avoidance orientation in text and modulates
with emotional state to produce a goal alignment score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Set

logger = logging.getLogger(__name__)


# Words indicating goal approach behavior
APPROACH_WORDS: Set[str] = {
    "achieving",
    "making progress",
    "getting closer",
    "working toward",
    "accomplished",
    "completed",
    "finished",
    "succeeded",
    "on track",
    "going well",
    "breakthrough",
}

# Words indicating avoidance behavior
AVOIDANCE_WORDS: Set[str] = {
    "avoiding",
    "giving up",
    "cant",
    "can't",
    "failing",
    "stuck",
    "blocked",
    "impossible",
    "wont work",
    "won't work",
    "pointless",
    "hopeless",
    "quitting",
}

# Emotional states that positively modulate goal approach
POSITIVE_EMOTIONAL_STATES: Set[str] = {"calm", "energized"}

# Emotional states that negatively modulate goal avoidance
NEGATIVE_EMOTIONAL_STATES: Set[str] = {"stressed", "anxious", "overwhelmed"}


@dataclass
class GoalEmotionResult:
    """Result of goal-emotion coupling analysis."""
    approach_score: float       # 0-1, proportion of approach words detected
    avoidance_score: float      # 0-1, proportion of avoidance words detected
    coupling_strength: float    # 0-1, max(approach, avoidance)
    goal_alignment: float       # -1 to 1, positive = approach-oriented


class GoalEmotionCoupler:
    """Analyzes text for goal approach vs avoidance orientation.

    Detects approach and avoidance language patterns in text, then modulates
    the goal alignment score based on the user's current emotional state.

    goal_alignment = approach_score - avoidance_score
    Positive emotion + approach boosts alignment by +0.2
    Negative emotion + avoidance worsens alignment by -0.2
    """

    APPROACH_BOOST = 0.2   # Bonus when positive emotion + approach > 0.3
    AVOIDANCE_PENALTY = 0.2  # Penalty when stress + avoidance > 0.3

    def analyze(self, text: str, emotional_state: str = "neutral") -> GoalEmotionResult:
        """Analyze text for goal-emotion coupling.

        Args:
            text: User message to analyze
            emotional_state: Current emotional state label

        Returns:
            GoalEmotionResult with approach/avoidance scores and goal alignment
        """
        if not text or not text.strip():
            return GoalEmotionResult(
                approach_score=0.0,
                avoidance_score=0.0,
                coupling_strength=0.0,
                goal_alignment=0.0,
            )

        text_lower = text.lower()

        # Count approach words (including multi-word phrases)
        approach_count = sum(1 for w in APPROACH_WORDS if w in text_lower)
        avoidance_count = sum(1 for w in AVOIDANCE_WORDS if w in text_lower)

        # Normalize to 0-1 (3 matches = 1.0)
        approach_score = min(approach_count / 3.0, 1.0)
        avoidance_score = min(avoidance_count / 3.0, 1.0)

        # Coupling strength = max of both orientations
        coupling_strength = max(approach_score, avoidance_score)

        # Base alignment
        goal_alignment = approach_score - avoidance_score

        # Emotional state modulation
        state_lower = emotional_state.lower()
        if state_lower in POSITIVE_EMOTIONAL_STATES and approach_score > 0.3:
            goal_alignment += self.APPROACH_BOOST
        if state_lower in NEGATIVE_EMOTIONAL_STATES and avoidance_score > 0.3:
            goal_alignment -= self.AVOIDANCE_PENALTY

        # Clamp to [-1, 1]
        goal_alignment = max(-1.0, min(1.0, goal_alignment))

        logger.debug(
            "US-358: goal_emotion approach=%.3f avoidance=%.3f coupling=%.3f alignment=%.3f state=%s",
            approach_score, avoidance_score, coupling_strength, goal_alignment, emotional_state,
        )

        return GoalEmotionResult(
            approach_score=round(approach_score, 4),
            avoidance_score=round(avoidance_score, 4),
            coupling_strength=round(coupling_strength, 4),
            goal_alignment=round(goal_alignment, 4),
        )
