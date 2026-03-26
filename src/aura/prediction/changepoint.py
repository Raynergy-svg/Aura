"""US-330: Bayesian Online Changepoint Detection for readiness regime shifts.

Detects sudden structural breaks in readiness time series using the
Adams & MacKay (2007) algorithm. Maintains posterior distribution over
run lengths — when P(changepoint) > threshold, a regime shift is flagged.

EWMA catches gradual drifts; BOCD catches sudden jumps:
  - Trader's baseline readiness dropped after a major loss streak
  - Emotional regulation degraded after a life event
  - Sudden improvement after a break/vacation

On detection: creates REGIME_SHIFT Life_Event node in graph.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ChangePointResult:
    """Result of a single changepoint detection update."""
    changepoint_prob: float = 0.0      # P(changepoint at this step)
    run_length: int = 0                # Most likely run length (steps since last changepoint)
    baseline_estimate: float = 50.0    # Current baseline estimate
    is_changepoint: bool = False       # Whether threshold was exceeded
    pre_baseline: float = 50.0         # Baseline before this update
    post_baseline: float = 50.0        # Baseline after this update


class BayesianChangePointDetector:
    """Online Bayesian changepoint detection for readiness time series.

    Uses per-run-length sufficient statistics with Gaussian likelihood.
    The changepoint hypothesis (r=0) uses a broad prior predictive, while
    existing run lengths use narrower posteriors based on accumulated data.
    This asymmetry is what makes changepoints detectable.
    """

    MIN_OBSERVATIONS = 20    # Minimum observations before detection activates
    DEFAULT_THRESHOLD = 0.10  # 10% posterior probability = changepoint

    def __init__(
        self,
        hazard_rate: float = 0.01,  # 1/100 = expect 1 shift per 100 observations
        threshold: float = 0.10,
        initial_mu: float = 50.0,
        initial_sigma: float = 15.0,
    ):
        """
        Args:
            hazard_rate: Prior P(changepoint) at any timestep
            threshold: Posterior probability above which changepoint is flagged
            initial_mu: Initial mean estimate for readiness scores
            initial_sigma: Initial std estimate for readiness scores
        """
        self.hazard_rate = max(1e-6, min(0.5, hazard_rate))
        self.threshold = threshold
        self.initial_mu = initial_mu
        self.initial_sigma = max(1.0, initial_sigma)

        # Run-length distribution (log-space for numerical stability)
        self._log_run_length_dist: List[float] = [0.0]  # P(r=0) = 1.0
        self._count: int = 0

        # Per-run-length sufficient statistics: (sum, sum_sq, count)
        # Index i corresponds to run length i
        self._rl_sums: List[float] = [0.0]
        self._rl_sum_sqs: List[float] = [0.0]
        self._rl_counts: List[int] = [0]

        # Global rolling stats
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._mu: float = initial_mu

        # Track baselines for regime shift reporting
        self._recent_scores: List[float] = []
        self._pre_baseline: float = initial_mu

    @property
    def observation_count(self) -> int:
        return self._count

    def _gaussian_log_pdf(self, x: float, mu: float, sigma: float) -> float:
        """Gaussian log-likelihood."""
        z = (x - mu) / sigma
        return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)

    def _predictive_log_likelihood(self, x: float, rl_idx: int) -> float:
        """Compute predictive log-likelihood for a specific run length.

        For r=0 (new segment): use broad prior (initial_mu, initial_sigma).
        For r>0: use accumulated sufficient stats for that run length.
        """
        if rl_idx == 0 or self._rl_counts[rl_idx] < 3:
            # Use prior predictive (broad)
            return self._gaussian_log_pdf(x, self.initial_mu, self.initial_sigma)

        # Posterior predictive from accumulated stats
        n = self._rl_counts[rl_idx]
        mu = self._rl_sums[rl_idx] / n
        variance = (self._rl_sum_sqs[rl_idx] / n) - mu * mu
        sigma = max(1.0, math.sqrt(max(0.0, variance)))
        return self._gaussian_log_pdf(x, mu, sigma)

    def update(self, readiness_score: float) -> ChangePointResult:
        """Update detector with new observation and check for changepoint.

        Args:
            readiness_score: Raw readiness score (0-100)

        Returns:
            ChangePointResult with detection status
        """
        self._count += 1
        self._recent_scores.append(readiness_score)
        if len(self._recent_scores) > 200:
            self._recent_scores = self._recent_scores[-100:]

        # Update global rolling statistics
        self._sum += readiness_score
        self._sum_sq += readiness_score ** 2
        self._mu = self._sum / self._count

        # Not enough observations yet
        if self._count < self.MIN_OBSERVATIONS:
            # Still update per-rl stats during warmup
            new_sums = [0.0]  # r=0 resets
            new_sum_sqs = [0.0]
            new_counts = [0]
            for i in range(len(self._rl_sums)):
                new_sums.append(self._rl_sums[i] + readiness_score)
                new_sum_sqs.append(self._rl_sum_sqs[i] + readiness_score ** 2)
                new_counts.append(self._rl_counts[i] + 1)
            self._rl_sums = new_sums
            self._rl_sum_sqs = new_sum_sqs
            self._rl_counts = new_counts

            return ChangePointResult(
                baseline_estimate=self._mu,
                pre_baseline=self._mu,
                post_baseline=self._mu,
            )

        # Compute per-run-length predictive log-likelihoods
        log_hazard = math.log(max(self.hazard_rate, 1e-10))
        log_1m_hazard = math.log(max(1.0 - self.hazard_rate, 1e-10))

        n_rl = len(self._log_run_length_dist)

        # Changepoint: sum of all existing P(r) * hazard * prior_predictive(x)
        log_prior_pred = self._gaussian_log_pdf(
            readiness_score, self.initial_mu, self.initial_sigma
        )
        log_cp_terms = [
            self._log_run_length_dist[i] + log_hazard + log_prior_pred
            for i in range(n_rl)
        ]
        log_cp_sum = self._log_sum_exp(log_cp_terms)

        # Growth: each existing r → r+1 with (1-hazard) * posterior_predictive(x|r)
        growth_log_probs = []
        for i in range(n_rl):
            ll = self._predictive_log_likelihood(readiness_score, i)
            growth_log_probs.append(
                self._log_run_length_dist[i] + log_1m_hazard + ll
            )

        # New distribution: [changepoint, growth_0, growth_1, ...]
        new_log_dist = [log_cp_sum] + growth_log_probs

        # Normalize (log-space) — guard against -inf norm which would produce Inf/NaN
        log_norm = self._log_sum_exp(new_log_dist)
        if math.isinf(log_norm) and log_norm < 0:
            # All probabilities collapsed — reset to uniform over current run lengths
            n = len(new_log_dist)
            self._log_run_length_dist = [math.log(1.0 / n) if n > 0 else 0.0] * n
        else:
            self._log_run_length_dist = [x - log_norm for x in new_log_dist]

        # Update per-run-length sufficient statistics
        # New r=0 starts fresh, existing ones accumulate
        new_sums = [0.0]  # r=0 resets
        new_sum_sqs = [0.0]
        new_counts = [0]
        for i in range(len(self._rl_sums)):
            new_sums.append(self._rl_sums[i] + readiness_score)
            new_sum_sqs.append(self._rl_sum_sqs[i] + readiness_score ** 2)
            new_counts.append(self._rl_counts[i] + 1)
        self._rl_sums = new_sums
        self._rl_sum_sqs = new_sum_sqs
        self._rl_counts = new_counts

        # Extract changepoint probability (P(r=0))
        changepoint_prob = math.exp(self._log_run_length_dist[0])

        # Most likely run length
        max_idx = 0
        max_val = self._log_run_length_dist[0]
        for i in range(1, len(self._log_run_length_dist)):
            if self._log_run_length_dist[i] > max_val:
                max_val = self._log_run_length_dist[i]
                max_idx = i

        # Determine if changepoint
        is_changepoint = changepoint_prob > self.threshold

        # Compute pre/post baselines for regime reporting
        pre_baseline = self._pre_baseline
        if is_changepoint:
            # Pre-baseline from scores before the changepoint
            split = max(1, len(self._recent_scores) - max(max_idx, 5))
            pre_scores = self._recent_scores[:split]
            post_scores = self._recent_scores[split:]
            if pre_scores:
                pre_baseline = sum(pre_scores) / len(pre_scores)
            post_baseline = sum(post_scores) / len(post_scores) if post_scores else readiness_score
            self._pre_baseline = post_baseline  # Update for next detection
            logger.info(
                "US-330: Changepoint detected — prob=%.3f, run_length=%d, pre=%.1f, post=%.1f",
                changepoint_prob, max_idx, pre_baseline, post_baseline,
            )
        else:
            post_baseline = self._mu

        # Trim run-length distribution to prevent unbounded growth
        max_rl = 300
        if len(self._log_run_length_dist) > max_rl:
            self._log_run_length_dist = self._log_run_length_dist[:max_rl]
            self._rl_sums = self._rl_sums[:max_rl]
            self._rl_sum_sqs = self._rl_sum_sqs[:max_rl]
            self._rl_counts = self._rl_counts[:max_rl]

        return ChangePointResult(
            changepoint_prob=changepoint_prob,
            run_length=max_idx,
            baseline_estimate=self._mu,
            is_changepoint=is_changepoint,
            pre_baseline=pre_baseline,
            post_baseline=post_baseline,
        )

    def reset(self) -> None:
        """Reset detector state after confirmed regime shift."""
        self._log_run_length_dist = [0.0]
        self._rl_sums = [0.0]
        self._rl_sum_sqs = [0.0]
        self._rl_counts = [0]
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self._recent_scores = []

    @staticmethod
    def _log_sum_exp(values: List[float]) -> float:
        """Numerically stable log-sum-exp."""
        if not values:
            return float('-inf')
        max_val = max(values)
        if max_val == float('-inf'):
            return float('-inf')
        return max_val + math.log(sum(math.exp(v - max_val) for v in values))
