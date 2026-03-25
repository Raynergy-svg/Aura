"""US-329: Adaptive Learning Rate Scheduler for ReadinessModelV2 online SGD.

Replaces fixed lr=0.01 with:
  1. Warmup phase: linear ramp 0.001 → initial_lr over first warmup_samples
  2. Decay phase: lr = initial_lr / (1 + sample_count / warmup_samples)
  3. Gradient diversity: aligned gradients → full lr, diverse → reduced
  4. Momentum buffer (beta=0.9) for smoother convergence
  5. Floor at 0.0001 to prevent vanishing updates
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate with warmup, decay, gradient diversity, and momentum."""

    def __init__(
        self,
        initial_lr: float = 0.01,
        warmup_samples: int = 50,
        lr_floor: float = 0.0001,
        momentum_beta: float = 0.9,
        gradient_history_size: int = 10,
    ):
        self.initial_lr = initial_lr
        self.warmup_samples = max(1, warmup_samples)
        self.lr_floor = lr_floor
        self.momentum_beta = momentum_beta
        self.gradient_history_size = gradient_history_size

        self._gradient_history: List[List[float]] = []
        self._momentum_buffer: Optional[List[float]] = None
        self._samples_seen: int = 0

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    def get_learning_rate(
        self,
        sample_count: Optional[int] = None,
        gradient_vector: Optional[List[float]] = None,
    ) -> float:
        """Compute adaptive learning rate.

        Args:
            sample_count: Override for sample count (uses internal counter if None)
            gradient_vector: Current gradient for diversity check

        Returns:
            Learning rate (float)
        """
        t = sample_count if sample_count is not None else self._samples_seen

        # Phase 1: Warmup (linear ramp)
        if t < self.warmup_samples:
            warmup_progress = t / self.warmup_samples
            base_lr = self.initial_lr * (0.1 + 0.9 * warmup_progress)
        else:
            # Phase 2: Decay
            base_lr = self.initial_lr / (1.0 + t / self.warmup_samples)

        # Phase 3: Gradient diversity scaling
        diversity_factor = 1.0
        if gradient_vector is not None and self._gradient_history:
            diversity_factor = self._compute_diversity_factor(gradient_vector)

        lr = base_lr * diversity_factor
        return max(self.lr_floor, lr)

    def apply_momentum(self, gradient: List[float]) -> List[float]:
        """Apply momentum EMA to gradient.

        Args:
            gradient: Current raw gradient vector

        Returns:
            Momentum-smoothed gradient
        """
        if self._momentum_buffer is None:
            self._momentum_buffer = list(gradient)
        else:
            beta = self.momentum_beta
            self._momentum_buffer = [
                beta * m + (1.0 - beta) * g
                for m, g in zip(self._momentum_buffer, gradient)
            ]
        return list(self._momentum_buffer)

    def step(self, gradient_vector: Optional[List[float]] = None) -> None:
        """Record a training step (updates counters and history)."""
        self._samples_seen += 1
        if gradient_vector is not None:
            self._gradient_history.append(list(gradient_vector))
            if len(self._gradient_history) > self.gradient_history_size:
                self._gradient_history.pop(0)

    def _compute_diversity_factor(self, current_gradient: List[float]) -> float:
        """Compute gradient diversity factor via cosine similarity.

        High alignment (> 0.7) → full lr (exploit).
        Low alignment (< 0.3) → lr * 0.5 (explore).
        Between → linear interpolation.
        """
        if not self._gradient_history:
            return 1.0

        # Mean of recent gradients
        n_hist = len(self._gradient_history)
        dim = len(current_gradient)
        if dim == 0:
            return 1.0

        mean_grad = [0.0] * dim
        for g in self._gradient_history:
            for i in range(min(dim, len(g))):
                mean_grad[i] += g[i]
        mean_grad = [x / n_hist for x in mean_grad]

        # Cosine similarity
        similarity = self._cosine_similarity(current_gradient, mean_grad)

        # Map similarity to diversity factor
        if similarity > 0.7:
            return 1.0
        elif similarity < 0.3:
            return 0.5
        else:
            # Linear interpolation: 0.3 → 0.5, 0.7 → 1.0
            return 0.5 + (similarity - 0.3) / 0.4 * 0.5

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.5

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.5

        return dot / (norm_a * norm_b)
