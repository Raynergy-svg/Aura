"""Trader Readiness Score — Aura's primary output signal to Buddy.

Computes a 0-100 score reflecting the trader's cognitive and emotional
fitness for trading decisions. This score feeds into Buddy's agent team
as the TraderReadinessAgent (#13).

Components (PRD v2.2 §5):
    - Emotional state (detected from conversations)
    - Decision pattern quality (recent override history)
    - Cognitive load (concurrent life stressors)
    - Stress level (conversation sentiment analysis)
    - Confidence trend (rising/falling/stable)

The score modulates Buddy's behavior:
    - 80-100: Full trading capacity
    - 60-79:  Reduced position sizes (-20%)
    - 40-59:  Significantly reduced (-40%), wider SL buffers
    - 20-39:  Minimum positions only
    - 0-19:   Block new trades entirely
"""

from __future__ import annotations

import json
import logging
import dataclasses
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """US-313: Result of readiness anomaly detection."""
    baseline: float = 0.0          # EWMA baseline
    residual: float = 0.0          # score - baseline
    threshold: float = 0.0         # Current anomaly threshold
    anomaly_detected: bool = False
    severity: float = 0.0          # 0-1, |residual| / threshold


class ReadinessAnomalyDetector:
    """US-313: Detects anomalous readiness score deviations via EWMA residual analysis.

    Maintains an EWMA baseline of readiness scores and flags when the
    residual (raw - baseline) exceeds 2.5 standard deviations.
    """

    MIN_HISTORY = 10  # Minimum scores before activation
    THRESHOLD_MULTIPLIER = 2.5  # Standard deviations for anomaly

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: EWMA smoothing factor for baseline (0.1 = 90% inertia)
        """
        self.alpha = alpha
        self._baseline: Optional[float] = None
        self._variance_ewma: float = 0.0  # EWMA of squared residuals
        self._count: int = 0

    def update(self, score: float) -> AnomalyResult:
        """Update detector with new score and check for anomaly.

        Args:
            score: Raw readiness score (0-100)

        Returns:
            AnomalyResult with detection status
        """
        self._count += 1

        if self._baseline is None:
            self._baseline = score
            return AnomalyResult(baseline=score)

        # Update baseline
        new_baseline = self.alpha * score + (1.0 - self.alpha) * self._baseline

        # Compute residual
        residual = score - self._baseline

        # Update variance EWMA (using squared residual)
        self._variance_ewma = self.alpha * (residual ** 2) + (1.0 - self.alpha) * self._variance_ewma

        self._baseline = new_baseline

        # Not enough history yet
        if self._count < self.MIN_HISTORY:
            return AnomalyResult(baseline=new_baseline, residual=residual)

        # Compute threshold
        import math
        std = math.sqrt(max(self._variance_ewma, 1e-10))
        threshold = self.THRESHOLD_MULTIPLIER * std

        # Check for anomaly
        anomaly_detected = abs(residual) > threshold
        severity = min(1.0, abs(residual) / max(threshold, 1e-10)) if anomaly_detected else 0.0

        return AnomalyResult(
            baseline=new_baseline,
            residual=residual,
            threshold=threshold,
            anomaly_detected=anomaly_detected,
            severity=severity,
        )


@dataclass
class ReadinessComponents:
    """Individual components that make up the readiness score."""

    emotional_state_score: float = 0.7     # 0-1, higher = calmer/more positive
    cognitive_load_score: float = 0.7      # 0-1, higher = less loaded
    override_discipline_score: float = 0.8 # 0-1, higher = fewer bad overrides
    stress_level_score: float = 0.7        # 0-1, higher = less stressed
    confidence_trend_score: float = 0.7    # 0-1, higher = more stable/rising
    engagement_score: float = 0.5          # 0-1, higher = more recently engaged with Aura

    def to_dict(self) -> Dict[str, float]:
        return {
            "emotional_state": self.emotional_state_score,
            "cognitive_load": self.cognitive_load_score,
            "override_discipline": self.override_discipline_score,
            "stress_level": self.stress_level_score,
            "confidence_trend": self.confidence_trend_score,
            "engagement": self.engagement_score,
        }


@dataclass
class ReadinessSignal:
    """The complete readiness signal sent from Aura to Buddy.

    This is the JSON contract defined in PRD v2.2 §5:
    Human→Domain signal interface.
    """

    readiness_score: float              # 0-100
    cognitive_load: str                 # "low" | "medium" | "high"
    active_stressors: List[str]         # Current life stressors
    override_loss_rate_7d: float        # 0-1, rate of losing overrides last 7 days
    emotional_state: str                # "calm" | "anxious" | "stressed" | "energized" | "fatigued"
    confidence_trend: str               # "rising" | "falling" | "stable"
    components: ReadinessComponents     # Detailed breakdown
    timestamp: str = ""
    conversation_count_7d: int = 0      # How many conversations in last 7 days
    # US-282: Confidence acceleration (second derivative)
    confidence_acceleration: float = 0.0  # Negative = rapid decline, positive = rapid recovery
    # US-283: Decision fatigue flag
    fatigue_score: float = 0.0            # 0.0 = no fatigue, 1.0 = severe fatigue
    # US-290: Model version tracking
    model_version: str = "v1"             # "v1" (static weights) or "v2" (learned weights)
    # US-301: Circadian readiness multiplier
    circadian_multiplier: float = 1.0     # 0.5-1.0 time-of-day modifier
    # US-303: Raw vs smoothed scores for transparency
    raw_score: Optional[float] = None     # Pre-EMA score
    smoothed_score: Optional[float] = None  # Post-EMA score
    # US-304: Tilt/revenge trading detection
    tilt_score: float = 0.0               # 0.0-1.0 tilt severity
    # US-310: Decision cadence variability
    decision_variability: float = 1.0     # 0-1, 1.0 = optimal cadence
    # US-313: Anomaly detection
    anomaly_detected: bool = False
    anomaly_severity: float = 0.0
    # US-346: Anomaly-to-action pipeline
    anomaly_action_taken: bool = False
    anomaly_dampening: float = 0.0
    # US-314: Bias scores visibility for Buddy
    bias_scores: Dict[str, float] = field(default_factory=dict)
    # US-317: Override loss risk from predictor
    override_loss_risk: float = 0.0
    # US-318: Readiness trend direction from STL
    trend_direction: str = "stable"
    # US-321: Decision quality score (0-100, independent of readiness)
    decision_quality_score: float = 0.0
    # US-326: Recovery score from emotional regulation
    recovery_score: float = 0.5
    # US-330: Regime shift detection
    regime_shift_detected: bool = False
    regime_shift_prob: float = 0.0
    # US-334: Score reliability from Cronbach's alpha
    reliability_score: float = 0.7  # Default assumes decent reliability

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "readiness_score": round(self.readiness_score, 1),
            "cognitive_load": self.cognitive_load,
            "active_stressors": self.active_stressors,
            "override_loss_rate_7d": round(self.override_loss_rate_7d, 3),
            "emotional_state": self.emotional_state,
            "confidence_trend": self.confidence_trend,
            "components": self.components.to_dict(),
            "timestamp": self.timestamp,
            "conversation_count_7d": self.conversation_count_7d,
            "confidence_acceleration": round(self.confidence_acceleration, 4),
            "fatigue_score": round(self.fatigue_score, 3),
            "model_version": self.model_version,
            "circadian_multiplier": round(self.circadian_multiplier, 3),
            "tilt_score": round(self.tilt_score, 3),
            "decision_variability": round(self.decision_variability, 3),
            "anomaly_detected": self.anomaly_detected,
            "anomaly_severity": round(self.anomaly_severity, 3),
            "anomaly_action_taken": self.anomaly_action_taken,
            "anomaly_dampening": round(self.anomaly_dampening, 4),
            "bias_scores": {k: round(v, 3) for k, v in self.bias_scores.items()},
            "override_loss_risk": round(self.override_loss_risk, 3),
            "trend_direction": self.trend_direction,
            "decision_quality_score": round(self.decision_quality_score, 1),
            "recovery_score": round(self.recovery_score, 3),
            "regime_shift_detected": self.regime_shift_detected,
            "regime_shift_prob": round(self.regime_shift_prob, 4),
            "reliability_score": round(self.reliability_score, 3),
        }
        if self.raw_score is not None:
            result["raw_score"] = round(self.raw_score, 1)
        if self.smoothed_score is not None:
            result["smoothed_score"] = round(self.smoothed_score, 1)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# Component weights for the composite score (V1 static fallback)
_COMPONENT_WEIGHTS = {
    "emotional_state": 0.25,
    "cognitive_load": 0.20,
    "override_discipline": 0.25,
    "stress_level": 0.15,
    "confidence_trend": 0.10,
    "engagement": 0.05,
}

# US-302: Component names for adaptive weight tracking
_COMPONENT_NAMES = list(_COMPONENT_WEIGHTS.keys())


class AdaptiveWeightManager:
    """US-302: Bayesian adaptive component weights via Beta-Binomial conjugate priors.

    Each readiness component gets an (alpha, beta) pair. After each trade outcome,
    we update based on whether the component's signal predicted the result correctly.
    Exponential decay (half-life: 30 trades) ensures recent outcomes matter more.

    Falls back to static V1 weights until min_samples outcomes are recorded.
    """

    DECAY_HALF_LIFE = 30.0  # trades
    MIN_SAMPLES = 10        # require 10+ outcomes before using adaptive weights

    def __init__(self, persist_path: Optional[Path] = None):
        self._persist_path = persist_path
        # Initialize with uniform priors (alpha=1, beta=1) for all components
        self._priors: Dict[str, Dict[str, float]] = {
            name: {"alpha": 1.0, "beta": 1.0} for name in _COMPONENT_NAMES
        }
        self._sample_count = 0
        # Attempt to load persisted state
        if persist_path and persist_path.exists():
            self._load()

    def update(self, component_name: str, prediction_correct: bool, days_old: float = 0.0) -> None:
        """Update a component's prior based on trade outcome.

        Args:
            component_name: Which component to update
            prediction_correct: True if component signal matched outcome
            days_old: Age of this outcome in days (for decay weighting)
        """
        if component_name not in self._priors:
            logger.warning("US-302: Unknown component '%s' — skipping update", component_name)
            return

        decay = 0.5 ** (days_old / self.DECAY_HALF_LIFE) if days_old > 0 else 1.0
        prior = self._priors[component_name]
        if prediction_correct:
            prior["alpha"] += decay
        else:
            prior["beta"] += decay

        self._sample_count += 1

    def get_weights(self) -> Dict[str, float]:
        """Get normalized adaptive weights (alpha / (alpha + beta)) for all components.

        Returns weights normalized to sum=1.0.
        """
        raw = {}
        for name in _COMPONENT_NAMES:
            p = self._priors[name]
            raw[name] = p["alpha"] / (p["alpha"] + p["beta"])

        total = sum(raw.values())
        if total <= 0:
            return dict(_COMPONENT_WEIGHTS)  # Fallback to static
        return {k: v / total for k, v in raw.items()}

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def is_ready(self) -> bool:
        """Whether enough samples have been collected for adaptive weights."""
        return self._sample_count >= self.MIN_SAMPLES

    def bootstrap_from_history(self, outcomes: List[Dict[str, Any]]) -> int:
        """US-341: Replay past outcome signals to warm-start adaptive priors.

        Args:
            outcomes: List of dicts with 'success' (bool) and 'component_scores' (Dict[str, float])

        Returns:
            Number of valid outcomes processed
        """
        valid = 0
        skipped = 0
        for entry in outcomes:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            success = entry.get("success")
            comp_scores = entry.get("component_scores")
            if success is None or not isinstance(comp_scores, dict):
                skipped += 1
                continue
            # Update each component based on whether its signal was aligned with outcome
            for comp_name in _COMPONENT_NAMES:
                score = comp_scores.get(comp_name)
                if score is None:
                    continue
                # High score (> 0.5) predicted positive outcome
                predicted_positive = score > 0.5
                prediction_correct = predicted_positive == bool(success)
                self.update(comp_name, prediction_correct)
            valid += 1

        if skipped > 0:
            logger.info("US-341: Bootstrap skipped %d malformed entries", skipped)
        if valid > 0:
            logger.info("US-341: Bootstrap loaded %d valid outcomes (sample_count=%d)",
                        valid, self._sample_count)
        return valid

    def save(self) -> None:
        """Persist adaptive weights to disk (atomic write)."""
        if not self._persist_path:
            return
        data = {"priors": self._priors, "sample_count": self._sample_count}
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            import tempfile, os
            fd, tmp = tempfile.mkstemp(dir=str(self._persist_path.parent), suffix=".tmp")
            os.write(fd, json.dumps(data, indent=2).encode())
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp, str(self._persist_path))
            logger.debug("US-302: Adaptive weights saved (%d samples)", self._sample_count)
        except Exception as e:
            logger.warning("US-302: Failed to save adaptive weights: %s", e)

    def _load(self) -> None:
        """Load persisted adaptive weights."""
        try:
            data = json.loads(self._persist_path.read_text())
            self._priors = data.get("priors", self._priors)
            self._sample_count = data.get("sample_count", 0)
            logger.debug("US-302: Loaded adaptive weights (%d samples)", self._sample_count)
        except Exception as e:
            logger.warning("US-302: Failed to load adaptive weights: %s", e)


@dataclass
class CadenceMetrics:
    """US-310: Decision cadence analysis metrics."""
    rmssd: float = 0.0           # Root mean square successive differences
    sdnn: float = 0.0            # Standard deviation of intervals
    mean_interval: float = 0.0   # Mean inter-decision interval (seconds)
    variability_score: float = 1.0  # 0-1, 1.0 = optimal range


class DecisionCadenceAnalyzer:
    """US-310: Analyzes inter-decision intervals to detect cognitive fatigue.

    Uses HRV-inspired metrics (RMSSD, SDNN) on the time between trade
    evaluations/override events. Low variability = monotonic fatigued decisions.
    High variability = erratic stressed decisions.
    """

    MIN_TIMESTAMPS = 5  # Need at least 5 events

    def __init__(self, optimal_rmssd_low: float = 30.0, optimal_rmssd_high: float = 120.0):
        """
        Args:
            optimal_rmssd_low: Lower bound of optimal RMSSD range (seconds)
            optimal_rmssd_high: Upper bound of optimal RMSSD range (seconds)
        """
        self.optimal_rmssd_low = optimal_rmssd_low
        self.optimal_rmssd_high = optimal_rmssd_high

    def analyze(self, timestamps: List[float]) -> CadenceMetrics:
        """Analyze decision timestamps and compute cadence metrics.

        Args:
            timestamps: List of Unix timestamps (seconds) of decisions, sorted ascending

        Returns:
            CadenceMetrics with variability_score (1.0 = optimal, 0.0 = extreme)
        """
        if len(timestamps) < self.MIN_TIMESTAMPS:
            return CadenceMetrics(variability_score=1.0)  # Insufficient data, no penalty

        # Compute inter-decision intervals
        sorted_ts = sorted(timestamps)
        intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]

        if not intervals:
            return CadenceMetrics(variability_score=1.0)

        # SDNN: standard deviation of all intervals
        import math
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        sdnn = math.sqrt(variance)

        # RMSSD: root mean square of successive differences
        if len(intervals) >= 2:
            successive_diffs = [intervals[i+1] - intervals[i] for i in range(len(intervals) - 1)]
            rmssd = math.sqrt(sum(d**2 for d in successive_diffs) / len(successive_diffs))
        else:
            rmssd = sdnn  # Fallback

        # Compute variability score: 1.0 = in optimal range, 0.0 = extreme
        if rmssd < self.optimal_rmssd_low:
            # Too regular (fatigue/numbness)
            variability_score = max(0.0, rmssd / self.optimal_rmssd_low)
        elif rmssd > self.optimal_rmssd_high:
            # Too erratic (stress)
            variability_score = max(0.0, 1.0 - (rmssd - self.optimal_rmssd_high) / self.optimal_rmssd_high)
        else:
            # Optimal range
            variability_score = 1.0

        variability_score = max(0.0, min(1.0, variability_score))

        return CadenceMetrics(
            rmssd=rmssd,
            sdnn=sdnn,
            mean_interval=mean_interval,
            variability_score=variability_score,
        )


class ReadinessTrendAnalyzer:
    """US-318: STL-based readiness trend decomposition with alert generation.

    Decomposes readiness history into trend + seasonal + residual using
    statsmodels STL. Identifies: declining trends, anomaly spikes, and
    seasonal patterns invisible to point-wise scoring.
    """

    MIN_SAMPLES = 14  # STL needs at least 2 seasonal periods

    def __init__(self, seasonal_period: int = 7, window: int = 60):
        """Args:
            seasonal_period: Samples per seasonal cycle (default 7 = ~1 day if 4h samples)
            window: Max history length
        """
        self.seasonal_period = seasonal_period
        self.window = window
        self._history: List[float] = []

    def add_sample(self, score: float, timestamp: float = 0.0) -> None:
        """Add a readiness sample to history."""
        self._history.append(score)
        if len(self._history) > self.window * 2:
            self._history = self._history[-self.window:]

    def decompose(self) -> Dict[str, Any]:
        """Decompose readiness into trend + seasonal + residual.

        Returns dict with trend_direction, trend_slope, anomaly_count, etc.
        Returns {'status': 'insufficient_data'} if < MIN_SAMPLES.
        """
        if len(self._history) < self.MIN_SAMPLES:
            return {"status": "insufficient_data", "trend_direction": "stable"}

        try:
            from statsmodels.tsa.seasonal import STL
            import numpy as np
        except ImportError:
            logger.warning("US-318: statsmodels not available — STL decomposition disabled")
            return {"status": "no_statsmodels", "trend_direction": "stable"}

        scores = np.array(self._history, dtype=float)

        try:
            stl = STL(scores, seasonal=self.seasonal_period, period=self.seasonal_period)
            result = stl.fit()
        except Exception as e:
            logger.warning("US-318: STL decomposition failed: %s", e)
            return {"status": "decomposition_error", "trend_direction": "stable"}

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        # Trend direction from slope
        x = np.arange(len(trend))
        valid = ~np.isnan(trend)
        if valid.sum() < 2:
            return {"status": "insufficient_valid", "trend_direction": "stable"}

        slope = np.polyfit(x[valid], trend[valid], 1)[0]
        if slope < -0.5:
            trend_direction = "declining"
        elif slope > 0.5:
            trend_direction = "improving"
        else:
            trend_direction = "stable"

        # Anomaly detection in residuals
        valid_resid = residual[~np.isnan(residual)]
        if len(valid_resid) > 0:
            resid_std = np.std(valid_resid)
            anomalies = np.where(np.abs(valid_resid) > 1.5 * max(resid_std, 1e-10))[0]
            anomaly_count = len(anomalies)
        else:
            resid_std = 0.0
            anomaly_count = 0

        # Seasonal amplitude
        valid_seasonal = seasonal[~np.isnan(seasonal)]
        seasonal_amplitude = float(np.max(valid_seasonal) - np.min(valid_seasonal)) if len(valid_seasonal) > 0 else 0.0

        return {
            "status": "ok",
            "trend_direction": trend_direction,
            "trend_slope": float(slope),
            "anomaly_count": anomaly_count,
            "seasonal_amplitude": seasonal_amplitude,
            "sample_count": len(self._history),
        }

    def readiness_alert(self) -> List[Dict[str, str]]:
        """Generate alerts based on STL decomposition."""
        decomp = self.decompose()

        if decomp.get("status") != "ok":
            return []

        alerts = []
        if decomp["trend_direction"] == "declining" and decomp.get("anomaly_count", 0) > 2:
            alerts.append({
                "type": "TREND_DECLINE",
                "severity": "HIGH",
                "message": f"Readiness declining (slope={decomp['trend_slope']:.2f}) with {decomp['anomaly_count']} anomalies",
            })
        if decomp.get("anomaly_count", 0) > 0:
            alerts.append({
                "type": "ANOMALY_SPIKE",
                "severity": "MEDIUM",
                "message": f"{decomp['anomaly_count']} residual anomalies in recent readiness",
            })
        if decomp.get("seasonal_amplitude", 0) > 15:
            alerts.append({
                "type": "SEASONAL_PATTERN",
                "severity": "INFO",
                "message": f"Strong readiness cycle detected (amplitude={decomp['seasonal_amplitude']:.1f})",
            })

        return alerts


class ReadinessComputer:
    """Computes the trader readiness score from self-model data.

    This is the core algorithm that transforms Aura's understanding of the
    user into an actionable signal for Buddy.

    Args:
        signal_path: Where to write the readiness signal JSON for Buddy to read
    """

    # US-282: Confidence acceleration thresholds
    ACCEL_RAPID_DECLINE = -0.10   # Per-sample, rapid confidence drop
    ACCEL_RAPID_RECOVERY = 0.10   # Per-sample, rapid confidence recovery
    ACCEL_PENALTY = 0.10          # Readiness penalty for rapid decline
    ACCEL_BONUS = 0.05            # Readiness bonus for rapid recovery

    # US-283: Decision fatigue thresholds
    FATIGUE_WINDOW_SIZE = 10      # Number of recent override events to consider
    FATIGUE_SPIKE_MULTIPLIER = 2.0  # Override rate > 2x average = fatigue
    FATIGUE_PENALTY = 0.10        # Readiness penalty for fatigue

    # US-290: Minimum samples before V2 model is used
    V2_MIN_SAMPLES = 20
    # US-290: OOD blend factor — when V2 detects OOD, blend with V1
    V2_OOD_BLEND = 0.5  # 50% V1 + 50% V2 when OOD

    # US-303: EMA smoothing constants
    # M-01 (FOLLOWUP): Documented design choices to distinguish intentional behavior from bugs.
    EMA_ALPHA = 0.3               # Smoothing factor (lower = more inertia). Chosen so a single
                                  # conversation session doesn't whipsaw the score. 0.3 means
                                  # ~70% weight on history, preventing manipulation by single inputs.
    HYSTERESIS_THRESHOLD = 5.0    # Min delta (out of 100) to propagate a score change to Buddy.
                                  # Intentional: prevents Buddy from reacting to noise (< 5pt moves).
                                  # Trade-off: up to 3 consecutive sub-5pt changes can accumulate
                                  # unobserved — acceptable for daily-cadence readiness signals.
                                  # Raise this if signal instability becomes a problem.

    # US-304: Tilt/revenge trading penalty
    TILT_PENALTY_FACTOR = 0.4     # Max 40% readiness reduction at tilt_score=1.0

    # US-314/US-327: Direct bias penalty — each bias > 0.5 = 3 points, capped at 25 (9 biases max)
    BIAS_DIRECT_PENALTY_PER_BIAS = 3.0
    BIAS_DIRECT_PENALTY_CAP = 25.0
    BIAS_SCORE_THRESHOLD = 0.5

    # US-317: Override predictor risk penalty
    OVERRIDE_RISK_THRESHOLD = 0.7  # Risk > 70% triggers penalty
    OVERRIDE_RISK_PENALTY = 10.0   # 10-point readiness penalty

    # US-320: Graph context weight in composite
    GRAPH_CONTEXT_DEFAULT = 0.5  # Neutral score when no graph provided

    def __init__(self, signal_path: Optional[Path] = None, v2_model=None, circadian_config: Optional[Dict[int, float]] = None,
                 adaptive_weights: Optional[AdaptiveWeightManager] = None):
        self.signal_path = signal_path or Path(".aura/bridge/readiness_signal.json")
        self.signal_path.parent.mkdir(parents=True, exist_ok=True)
        self._override_history: List[Dict[str, Any]] = []
        # US-282: Store recent readiness scores for acceleration computation
        self._readiness_history: List[float] = []
        # US-326: Stress level history for emotional regulation scoring
        self._stress_history: List[Dict[str, Any]] = []
        # US-290: Optional V2 model for adaptive weighting
        self._v2_model = v2_model
        # US-301: Configurable circadian curve
        self._circadian_curve = circadian_config or self.DEFAULT_CIRCADIAN_CURVE
        # US-302: Bayesian adaptive component weights
        self._adaptive_weights = adaptive_weights
        # US-303: EMA smoothing state
        self._last_smoothed_score: Optional[float] = None
        # US-308: TiltDetector for revenge trading detection
        from src.aura.core.conversation_processor import TiltDetector
        self._tilt_detector = TiltDetector()
        self._recent_messages: List[Dict[str, Any]] = []
        self._recent_outcomes: List[Dict[str, Any]] = []
        # US-310: Decision cadence analysis
        self._cadence_analyzer = DecisionCadenceAnalyzer()
        self._decision_timestamps: List[float] = []
        # US-313: Anomaly detection via EWMA
        self._anomaly_detector = ReadinessAnomalyDetector()
        # US-346: Anomaly history for spike dampening
        self._anomaly_history: deque = deque(maxlen=50)
        # US-316: Lazy ConversationProcessor for cognitive load estimation
        self._conversation_processor = None
        # US-317: Optional override predictor
        self._override_predictor = None
        # US-318: STL trend analyzer
        self._trend_analyzer = ReadinessTrendAnalyzer()
        # US-320: Graph reference for graph-informed features
        self._graph = None
        # US-323: Last state snapshot for bridge enrichment
        self._last_state_snapshot: Optional[Dict[str, Any]] = None
        # US-326: Emotional regulation scorer
        self._recovery_scorer = None
        try:
            from src.aura.scoring.emotional_regulation import EmotionalRegulationScorer
            self._recovery_scorer = EmotionalRegulationScorer()
        except ImportError:
            logger.debug("US-326: EmotionalRegulationScorer not available")
        # US-330: Bayesian changepoint detector
        self._changepoint_detector = None
        self._last_changepoint_detected = False  # US-355: Track regime shift for anomaly dampening
        try:
            from src.aura.prediction.changepoint import BayesianChangePointDetector
            self._changepoint_detector = BayesianChangePointDetector()
        except ImportError:
            logger.debug("US-330: BayesianChangePointDetector not available")
        # US-334: Readiness reliability analyzer (Cronbach's alpha)
        self._reliability_analyzer = None
        try:
            from src.aura.analysis.reliability import ReadinessReliabilityAnalyzer
            self._reliability_analyzer = ReadinessReliabilityAnalyzer()
        except ImportError:
            logger.debug("US-334: ReadinessReliabilityAnalyzer not available")
        # US-335: Graph topology analyzer
        self._graph_topology_analyzer = None
        try:
            from src.aura.analysis.graph_topology import GraphTopologyAnalyzer
            self._graph_topology_analyzer = GraphTopologyAnalyzer()
        except ImportError:
            logger.debug("US-335: GraphTopologyAnalyzer not available")
        # US-321: Decision quality scorer
        self._decision_quality_scorer = None
        try:
            from src.aura.scoring.decision_quality import DecisionQualityScorer
            self._decision_quality_scorer = DecisionQualityScorer()
        except ImportError:
            logger.debug("US-321: DecisionQualityScorer not available")
        # US-328: Metacognitive monitoring scorer
        self._metacognitive_scorer = None
        try:
            from src.aura.scoring.metacognitive import MetacognitiveMonitoringScorer
            self._metacognitive_scorer = MetacognitiveMonitoringScorer()
        except ImportError:
            logger.debug("US-328: MetacognitiveMonitoringScorer not available")
        # US-340: Cognitive flexibility scorer
        self._flexibility_scorer = None
        try:
            from src.aura.scoring.cognitive_flexibility import CognitiveFlexibilityScorer
            self._flexibility_scorer = CognitiveFlexibilityScorer()
        except ImportError:
            logger.debug("US-340: CognitiveFlexibilityScorer not available")
        # US-342: Journal reflection scorer
        self._journal_reflection_scorer = None
        try:
            from src.aura.scoring.journal_reflection import JournalReflectionScorer
            self._journal_reflection_scorer = JournalReflectionScorer()
        except ImportError:
            logger.debug("US-342: JournalReflectionScorer not available")
        # US-341: Bootstrap adaptive weights from bridge outcome history
        if self._adaptive_weights and not self._adaptive_weights.is_ready():
            self._bootstrap_adaptive_weights()

    def _bootstrap_adaptive_weights(self) -> None:
        """US-341: Attempt to bootstrap adaptive weights from bridge outcome_signal.json history."""
        try:
            bridge_dir = self.signal_path.parent
            outcome_path = bridge_dir / "outcome_signal.json"
            if not outcome_path.exists():
                return
            raw = json.loads(outcome_path.read_text())
            # outcome_signal.json may be a single dict or a list of outcomes
            if isinstance(raw, dict):
                # Single outcome — wrap in list
                outcomes = [raw] if "success" in raw else raw.get("history", [])
            elif isinstance(raw, list):
                outcomes = raw
            else:
                return
            if outcomes:
                loaded = self._adaptive_weights.bootstrap_from_history(outcomes)
                if loaded > 0:
                    logger.info("US-341: Bootstrapped adaptive weights from %d bridge outcomes", loaded)
        except Exception as e:
            logger.debug("US-341: Could not bootstrap from bridge: %s", e)

    def _compute_graph_features(self, graph) -> Dict[str, float]:
        """US-320: Extract readiness-relevant features from the self-model graph.

        Returns dict with 4 features (each 0-1):
          emotional_volatility: std dev of recent Emotion node strengths
          goal_alignment: avg strength of Goal nodes linked to recent Decisions
          pattern_load: active pattern count / 20 (capped)
          negative_influence_density: ratio of weak edges to total recent edges

        Returns neutral features (all 0.5 mappings) if graph is unavailable.
        """
        import math  # math not imported at module level; keep local

        defaults = {
            "emotional_volatility": 0.0,
            "goal_alignment": 0.5,
            "pattern_load": 0.0,
            "negative_influence_density": 0.0,
            "graph_context_score": self.GRAPH_CONTEXT_DEFAULT,
        }

        if graph is None:
            return defaults

        try:
            # Import NodeType dynamically to avoid circular imports
            from src.aura.core.self_model import NodeType

            # 1. Emotional volatility — std dev of recent Emotion node confidences
            emotion_nodes = graph.get_nodes_by_type(NodeType.EMOTION)
            if len(emotion_nodes) >= 2:
                confidences = [n.confidence for n in emotion_nodes[:20]]  # Last 20 (already sorted desc)
                mean_c = sum(confidences) / len(confidences)
                variance = sum((c - mean_c) ** 2 for c in confidences) / len(confidences)
                emotional_volatility = min(1.0, math.sqrt(variance) * 2.0)  # Scale: 0.5 std → 1.0
            else:
                emotional_volatility = 0.0

            # 2. Goal alignment — avg confidence of Goal nodes
            goal_nodes = graph.get_nodes_by_type(NodeType.GOAL)
            if goal_nodes:
                goal_alignment = sum(n.confidence for n in goal_nodes) / len(goal_nodes)
                goal_alignment = min(1.0, max(0.0, goal_alignment))
            else:
                goal_alignment = 0.5  # No goals = neutral

            # 3. Pattern load — count active Pattern nodes / 20
            pattern_nodes = graph.get_nodes_by_type(NodeType.PATTERN)
            active_patterns = [n for n in pattern_nodes if n.confidence > 0.3]
            pattern_load = min(1.0, len(active_patterns) / 20.0)

            # 4. Negative influence density — count edges with low weight
            try:
                all_edges_result = graph._conn.execute(
                    "SELECT weight FROM edges ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
                if all_edges_result:
                    total_edges = len(all_edges_result)
                    weak_edges = sum(1 for row in all_edges_result if row[0] < 0.3)
                    negative_influence_density = weak_edges / total_edges
                else:
                    negative_influence_density = 0.0
            except Exception:
                negative_influence_density = 0.0

            # Composite graph context score
            graph_context_score = (
                0.3 * (1.0 - emotional_volatility)
                + 0.3 * goal_alignment
                + 0.2 * (1.0 - pattern_load)
                + 0.2 * (1.0 - negative_influence_density)
            )
            graph_context_score = max(0.0, min(1.0, graph_context_score))

            return {
                "emotional_volatility": emotional_volatility,
                "goal_alignment": goal_alignment,
                "pattern_load": pattern_load,
                "negative_influence_density": negative_influence_density,
                "graph_context_score": graph_context_score,
            }

        except Exception as e:
            logger.warning("US-320: Graph feature extraction failed: %s", e)
            return defaults

    def get_last_state_snapshot(self) -> Dict[str, Any]:
        """US-323: Return last computed readiness state for bridge enrichment."""
        return self._last_state_snapshot or {}

    def train_from_bridge_outcome(self, outcome_signal: Dict[str, Any]) -> bool:
        """US-322: Update ReadinessModelV2 from Buddy's outcome signal.

        Performs incremental SGD update with the trade outcome.

        Args:
            outcome_signal: Dict with trade_won, profit_pips, etc.

        Returns:
            True if online update was performed, False if skipped
        """
        if self._v2_model is None:
            logger.debug("US-322: No V2 model — skipping online update")
            return False

        # Need trained model with enough samples
        if not getattr(self._v2_model, '_trained', False):
            logger.debug("US-322: V2 model not yet trained — skipping online update")
            return False

        sample_count = getattr(self._v2_model, '_train_samples', 0)
        if sample_count < self.V2_MIN_SAMPLES:
            logger.debug("US-322: V2 model has only %d samples (need %d) — skipping online update",
                         sample_count, self.V2_MIN_SAMPLES)
            return False

        # Build training sample from last state + outcome
        snapshot = self.get_last_state_snapshot()
        if not snapshot:
            logger.debug("US-322: No state snapshot available — skipping online update")
            return False

        trade_won = outcome_signal.get("trade_won", False)
        profit = outcome_signal.get("profit_pips", 0.0)

        # Map outcome to target readiness: wins → high, losses → low
        if trade_won:
            target_readiness = min(100.0, 70.0 + abs(profit) * 0.5)
        else:
            target_readiness = max(0.0, 40.0 - abs(profit) * 0.5)

        try:
            if hasattr(self._v2_model, 'update_from_outcome'):
                self._v2_model.update_from_outcome(
                    snapshot.get("components", {}),
                    target_readiness,
                    learning_rate=0.01,
                )
                logger.info("US-322: Online update performed — target=%.1f, trade_won=%s",
                            target_readiness, trade_won)
                return True
        except Exception as e:
            logger.warning("US-322: Online update failed: %s", e)

        return False

    def set_context(self, messages: List[Dict[str, Any]] = None, outcomes: List[Dict[str, Any]] = None) -> None:
        """US-308: Set recent message and outcome context for tilt detection.

        Called by AuraCompanion before compute() so tilt detection has real data.
        """
        if messages is not None:
            self._recent_messages = messages[-20:]  # Keep last 20
        if outcomes is not None:
            self._recent_outcomes = outcomes[-10:]  # Keep last 10

    def circadian_multiplier(self, hour: Optional[int] = None) -> float:
        """US-301: Get circadian readiness multiplier for a given hour.

        Based on circadian rhythm research: 25% worse financial decision
        quality at 11pm vs 10am. Applied to final readiness score.

        Args:
            hour: Hour of day (0-23). Defaults to current hour.

        Returns:
            Multiplier between 0.5 and 1.0
        """
        if hour is None:
            hour = datetime.now(timezone.utc).hour
        hour = max(0, min(23, hour))
        return self._circadian_curve.get(hour, 0.75)

    def compute_confidence_acceleration(self, readiness_history: Optional[List[float]] = None) -> float:
        """US-282: Compute second derivative of confidence trend.

        Uses the last 3+ readiness scores to estimate acceleration.
        Negative = rapid decline, positive = rapid recovery, 0 = stable.

        Args:
            readiness_history: List of recent readiness scores (newest last).
                               Uses internal history if not provided.

        Returns:
            Acceleration value (typically -1.0 to +1.0)
        """
        history = readiness_history if readiness_history is not None else self._readiness_history
        if len(history) < 3:
            return 0.0

        # Use last 3 points to compute second derivative
        # First derivative (velocity): d1 = h[-1] - h[-2], d0 = h[-2] - h[-3]
        # Second derivative (acceleration): d1 - d0
        recent = history[-3:]
        v1 = (recent[2] - recent[1]) / 100.0  # Normalize to 0-1 scale
        v0 = (recent[1] - recent[0]) / 100.0
        acceleration = v1 - v0
        return acceleration

    def compute_fatigue_score(self, recent_override_events: Optional[List[Dict[str, Any]]] = None) -> float:
        """US-283: Compute decision fatigue score from override frequency spike.

        Compares recent override rate to historical average. A spike
        (>2x average) indicates decision fatigue.

        Args:
            recent_override_events: Override events with timestamps.

        Returns:
            Fatigue score 0.0 (no fatigue) to 1.0 (severe fatigue)
        """
        events = recent_override_events or []
        if len(events) < 3:
            return 0.0

        # Split events into recent window and historical
        window = min(self.FATIGUE_WINDOW_SIZE, len(events))
        recent = events[-window:]
        historical = events[:-window] if len(events) > window else events

        # Compute rates (events per "period")
        recent_count = len(recent)
        historical_count = len(historical) if historical else recent_count

        # If no historical baseline, use recent as baseline (no spike)
        if historical_count == 0:
            return 0.0

        # Normalize to comparable rates
        recent_rate = recent_count / max(window, 1)
        historical_rate = historical_count / max(len(events) - window, 1)

        if historical_rate <= 0:
            return 0.0

        spike_ratio = recent_rate / historical_rate

        if spike_ratio >= self.FATIGUE_SPIKE_MULTIPLIER:
            # Scale fatigue: 2x = 0.5, 3x = 0.75, 4x+ = 1.0
            fatigue = min(1.0, (spike_ratio - 1.0) / 3.0)
            logger.info("US-283: Decision fatigue detected — spike_ratio=%.2f, fatigue_score=%.2f",
                        spike_ratio, fatigue)
            return fatigue

        return 0.0

    # US-293: Bias penalty coefficient — max 30% reduction to override_discipline
    BIAS_PENALTY_FACTOR = 0.3

    # US-301: Default circadian curve — multiplier by hour (0-23)
    # Based on circadian research: peak at 9-11am, post-lunch dip, evening decline
    DEFAULT_CIRCADIAN_CURVE = {
        0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.55,
        6: 0.60, 7: 0.75, 8: 0.90, 9: 1.00, 10: 1.00, 11: 0.95,
        12: 0.90, 13: 0.85, 14: 0.90, 15: 0.95, 16: 0.90, 17: 0.85,
        18: 0.80, 19: 0.75, 20: 0.70, 21: 0.65, 22: 0.55, 23: 0.50,
    }

    def compute(
        self,
        emotional_state: str = "calm",
        stress_keywords: List[str] = None,
        active_stressors: List[str] = None,
        recent_override_events: Optional[List[Dict[str, Any]]] = None,
        conversation_count_7d: int = 0,
        confidence_trend: str = "stable",
        bias_scores: Optional[Dict[str, float]] = None,
        message_text: Optional[str] = None,
        override_predictor=None,
        style_drift_score: float = 0.0,
        granularity_score: float = 0.5,
        coherence_score: float = 0.5,
        affect_volatility: float = 0.0,
        affect_stuck: bool = False,
        fatigue_score: float = 0.0,
        bias_interaction_penalty: float = 0.0,
        graph=None,
    ) -> ReadinessSignal:
        """Compute the readiness score from current state.

        Args:
            emotional_state: Current emotional state label
            stress_keywords: Keywords from recent conversations indicating stress
            active_stressors: Known life stressors (career decision, relationship, etc.)
            recent_override_events: Override events from last 7 days with outcomes
            conversation_count_7d: Number of Aura conversations in last 7 days
            confidence_trend: "rising", "falling", or "stable"
            message_text: US-316 — Raw message text for cognitive load estimation
            override_predictor: US-317 — OverridePredictor instance for loss risk

        Returns:
            ReadinessSignal with composite score and all components
        """
        stress_keywords = stress_keywords or []
        active_stressors = active_stressors or []
        recent_override_events = recent_override_events or []

        # --- Compute individual components ---

        # Emotional state score
        emotional_scores = {
            "calm": 0.9,
            "energized": 0.85,
            "neutral": 0.7,
            "anxious": 0.4,
            "stressed": 0.3,
            "fatigued": 0.35,
            "frustrated": 0.25,
            "overwhelmed": 0.15,
        }
        emotional_score = emotional_scores.get(emotional_state.lower(), 0.5)

        # Cognitive load score — based on number of active stressors
        stressor_count = len(active_stressors)
        if stressor_count == 0:
            stressor_load_score = 0.9
            cognitive_load_label = "low"
        elif stressor_count <= 2:
            stressor_load_score = 0.65
            cognitive_load_label = "medium"
        else:
            stressor_load_score = 0.3
            cognitive_load_label = "high"

        # Stress keyword penalty
        stress_penalty = min(len(stress_keywords) * 0.05, 0.3)
        stressor_load_score = max(0.1, stressor_load_score - stress_penalty)

        # US-316: Blend with text-based cognitive load if message provided
        cognitive_load_score = stressor_load_score
        if message_text and message_text.strip():
            try:
                if self._conversation_processor is None:
                    from src.aura.core.conversation_processor import ConversationProcessor
                    self._conversation_processor = ConversationProcessor()
                text_load = self._conversation_processor.estimate_cognitive_load(message_text)
                # Invert: high text_load → low readiness component score
                text_load_score = max(0.1, 1.0 - text_load)
                # Blend 50/50 with stressor-based score
                cognitive_load_score = 0.5 * text_load_score + 0.5 * stressor_load_score
                if text_load > 0.6:
                    cognitive_load_label = "high"
                elif text_load > 0.3:
                    cognitive_load_label = "medium"
                logger.info("US-316: Cognitive load blend — text=%.2f stressor=%.2f final=%.2f",
                            text_load_score, stressor_load_score, cognitive_load_score)
            except Exception as e:
                logger.warning("US-316: Cognitive load estimation failed, using stressor-only: %s", e)
                cognitive_load_score = stressor_load_score

        # Override discipline score — US-233: validate each event before counting
        if recent_override_events:
            valid_events = []
            skipped = 0
            for e in recent_override_events:
                if not isinstance(e, dict):
                    skipped += 1
                    continue
                if "trade_won" not in e:
                    skipped += 1
                    continue
                valid_events.append(e)
            if skipped > 0:
                logger.warning(
                    "US-233: Skipped %d/%d malformed override events",
                    skipped, len(recent_override_events),
                )
            total_overrides = len(valid_events)
            losing_overrides = sum(
                1 for e in valid_events if not e.get("trade_won", True)
            )
            override_loss_rate = losing_overrides / max(total_overrides, 1)
            override_discipline = max(0.1, 1.0 - override_loss_rate)
        else:
            override_loss_rate = 0.0
            override_discipline = 0.8  # Default — no data, assume decent

        # US-293: Apply cognitive bias penalty to override discipline
        if bias_scores:
            aggregate_bias = sum(bias_scores.values()) / max(len(bias_scores), 1)
            if aggregate_bias > 0:
                penalty = 1.0 - (aggregate_bias * self.BIAS_PENALTY_FACTOR)
                override_discipline *= penalty
                override_discipline = max(0.1, override_discipline)
                logger.info("US-293: Bias penalty applied — aggregate=%.3f, discipline reduced to %.3f",
                            aggregate_bias, override_discipline)

        # Confidence trend score
        trend_scores = {"rising": 0.85, "stable": 0.7, "falling": 0.4}
        confidence_trend_score = trend_scores.get(confidence_trend, 0.5)

        # Engagement score — recent engagement with Aura is positive
        if conversation_count_7d >= 5:
            engagement_score = 0.9
        elif conversation_count_7d >= 3:
            engagement_score = 0.7
        elif conversation_count_7d >= 1:
            engagement_score = 0.5
        else:
            engagement_score = 0.2  # No recent conversations = lower readiness

        # Stress level (inverse of stress indicators)
        stress_indicator_count = len(stress_keywords) + stressor_count
        stress_level_score = max(0.1, 1.0 - (stress_indicator_count * 0.08))

        # --- Assemble components ---
        components = ReadinessComponents(
            emotional_state_score=emotional_score,
            cognitive_load_score=cognitive_load_score,
            override_discipline_score=override_discipline,
            stress_level_score=stress_level_score,
            confidence_trend_score=confidence_trend_score,
            engagement_score=engagement_score,
        )

        # --- US-320/US-335: Graph-informed features ---
        active_graph = graph or self._graph
        if self._graph_topology_analyzer is not None and active_graph is not None:
            try:
                topology_features = self._graph_topology_analyzer.analyze(active_graph)
                graph_context_score = topology_features.graph_context_score
                graph_features = {
                    "clustering_coefficient": topology_features.clustering_coefficient,
                    "avg_betweenness": topology_features.avg_betweenness,
                    "density": topology_features.density,
                    "num_communities": topology_features.num_communities,
                    "modularity": topology_features.modularity,
                    "largest_component_ratio": topology_features.largest_component_ratio,
                    "graph_context_score": graph_context_score,
                }
                logger.debug("US-335: Graph topology features — score=%.3f", graph_context_score)
            except Exception as e:
                logger.warning("US-335: Graph topology analysis failed, falling back to hand-crafted: %s", e)
                graph_features = self._compute_graph_features(active_graph)
                graph_context_score = graph_features["graph_context_score"]
        else:
            graph_features = self._compute_graph_features(active_graph)
            graph_context_score = graph_features["graph_context_score"]

        # --- Compute weighted composite score (0-100) ---
        # US-302: Use adaptive weights if available and ready, else V1 static
        component_scores = {
            "emotional_state": emotional_score,
            "cognitive_load": cognitive_load_score,
            "override_discipline": override_discipline,
            "stress_level": stress_level_score,
            "confidence_trend": confidence_trend_score,
            "engagement": engagement_score,
        }

        # --- US-334: Record components for reliability tracking ---
        reliability_score_val = 0.7  # Default
        if self._reliability_analyzer is not None:
            try:
                self._reliability_analyzer.record_components(component_scores)
                rel_result = self._reliability_analyzer.compute()
                reliability_score_val = rel_result.reliability_score
                if rel_result.sufficient_data and rel_result.cronbachs_alpha < 0.6:
                    logger.warning("US-334: Low internal consistency — alpha=%.3f, reliability=%.3f",
                                   rel_result.cronbachs_alpha, reliability_score_val)
            except Exception as e:
                logger.warning("US-334: Reliability computation failed: %s", e)

        if self._adaptive_weights and self._adaptive_weights.is_ready():
            weights = self._adaptive_weights.get_weights()
            logger.info("US-302: Using adaptive weights (samples=%d)", self._adaptive_weights.sample_count)
        else:
            weights = _COMPONENT_WEIGHTS

        # US-320: Blend graph context into composite (10% graph, 90% components)
        v1_raw = sum(component_scores[k] * weights[k] for k in _COMPONENT_NAMES)
        if graph is not None or self._graph is not None:
            v1_raw = 0.91 * v1_raw + 0.09 * graph_context_score
        v1_score = max(0.0, min(100.0, v1_raw * 100))

        # US-290: Try V2 model when available and trained
        model_version = "v1"
        readiness_score = v1_score
        if self._v2_model is not None:
            try:
                if (getattr(self._v2_model, '_trained', False)
                        and getattr(self._v2_model, '_train_samples', 0) >= self.V2_MIN_SAMPLES):
                    component_dict = components.to_dict()
                    v2_score, _contributions = self._v2_model.compute_score(component_dict)
                    v2_score = max(0.0, min(100.0, v2_score))
                    model_version = "v2"
                    readiness_score = v2_score
                    logger.info("US-290: V2 model active (samples=%d, v2=%.1f, v1=%.1f)",
                                self._v2_model._train_samples, v2_score, v1_score)
                else:
                    logger.debug("US-290: V1 fallback active (V2 not yet trained, samples=%d)",
                                 getattr(self._v2_model, '_train_samples', 0))
            except Exception as e:
                logger.warning("US-290: V2 model error, falling back to V1: %s", e)
                readiness_score = v1_score

        # --- US-282: Confidence acceleration adjustment ---
        self._readiness_history.append(readiness_score)
        # Keep history bounded
        if len(self._readiness_history) > 50:
            self._readiness_history = self._readiness_history[-50:]

        confidence_acceleration = self.compute_confidence_acceleration()
        if confidence_acceleration < self.ACCEL_RAPID_DECLINE:
            readiness_score = max(0.0, readiness_score - self.ACCEL_PENALTY * 100)
            logger.info("US-282: Rapid confidence decline — penalty applied (accel=%.4f)", confidence_acceleration)
        elif confidence_acceleration > self.ACCEL_RAPID_RECOVERY:
            readiness_score = min(100.0, readiness_score + self.ACCEL_BONUS * 100)
            logger.info("US-282: Rapid confidence recovery — bonus applied (accel=%.4f)", confidence_acceleration)

        # --- US-283: Decision fatigue adjustment ---
        # Use provided fatigue_score if given (US-355), else compute from events
        if fatigue_score == 0.0:  # Default value, not provided
            fatigue_score = self.compute_fatigue_score(recent_override_events)
        # Convert computed fatigue (0-1 scale) to 0-100 if needed
        if fatigue_score <= 1.0:
            fatigue_score_normalized = fatigue_score * 100.0  # Scale up for US-355 processing
        else:
            fatigue_score_normalized = fatigue_score  # Already on 0-100 scale
        if fatigue_score <= 1.0 and fatigue_score > 0:  # Computed value (0-1 scale)
            readiness_score = max(0.0, readiness_score - fatigue_score * self.FATIGUE_PENALTY * 100)

        # --- US-301: Circadian readiness modulation ---
        circ_mult = self.circadian_multiplier()
        readiness_score = max(0.0, min(100.0, readiness_score * circ_mult))

        # --- US-304/US-308: Tilt/revenge trading penalty ---
        tilt_score = 0.0
        try:
            tilt_score = self._tilt_detector.detect_tilt(
                messages=self._recent_messages,
                recent_overrides=recent_override_events,
                recent_outcomes=self._recent_outcomes,
            )
        except Exception as e:
            logger.warning("US-304/US-308: Tilt detection error: %s", e)
            tilt_score = 0.0
        if tilt_score > 0:
            readiness_score = max(0.0, readiness_score * (1.0 - self.TILT_PENALTY_FACTOR * tilt_score))
            logger.info("US-304/US-308: Tilt penalty applied — tilt_score=%.3f", tilt_score)

        # --- US-314: Direct cognitive bias penalty ---
        bias_penalty_applied = 0.0
        if bias_scores:
            high_biases = sum(1 for v in bias_scores.values() if v > self.BIAS_SCORE_THRESHOLD)
            bias_penalty_applied = min(
                high_biases * self.BIAS_DIRECT_PENALTY_PER_BIAS,
                self.BIAS_DIRECT_PENALTY_CAP,
            )
            if bias_penalty_applied > 0:
                readiness_score = max(0.0, readiness_score - bias_penalty_applied)
                logger.info("US-314: Direct bias penalty — %d biases > %.1f, penalty=%.1f",
                            high_biases, self.BIAS_SCORE_THRESHOLD, bias_penalty_applied)

        # --- US-317: Override predictor loss risk penalty ---
        override_loss_risk = 0.0
        predictor = override_predictor or self._override_predictor
        if predictor is not None:
            try:
                if getattr(predictor, '_trained', False):
                    # Build context from available data
                    ctx = {}
                    if bias_scores:
                        ctx['bias_scores'] = bias_scores
                    ctx['emotional_state'] = emotional_state
                    ctx['confidence_trend'] = confidence_trend
                    override_loss_risk = predictor.predict_loss_probability(ctx)
                    override_loss_risk = max(0.0, min(1.0, override_loss_risk))
                    if override_loss_risk > self.OVERRIDE_RISK_THRESHOLD:
                        readiness_score = max(0.0, readiness_score - self.OVERRIDE_RISK_PENALTY)
                        logger.info("US-317: Override risk penalty — risk=%.2f > threshold=%.2f, penalty=%.1f",
                                    override_loss_risk, self.OVERRIDE_RISK_THRESHOLD, self.OVERRIDE_RISK_PENALTY)
            except Exception as e:
                logger.warning("US-317: Override predictor error: %s", e)
                override_loss_risk = 0.0

        # --- US-310: Decision cadence variability penalty ---
        import time
        self._decision_timestamps.append(time.time())
        if len(self._decision_timestamps) > 50:
            self._decision_timestamps = self._decision_timestamps[-50:]
        cadence = self._cadence_analyzer.analyze(self._decision_timestamps)
        decision_variability = cadence.variability_score
        readiness_score = max(0.0, readiness_score * (0.8 + 0.2 * decision_variability))

        # --- US-313: Readiness anomaly detection ---
        anomaly_result = self._anomaly_detector.update(readiness_score)
        anomaly_detected = anomaly_result.anomaly_detected
        anomaly_severity = anomaly_result.severity
        if anomaly_detected:
            logger.info("US-313: Readiness anomaly detected — residual=%.1f, threshold=%.1f, severity=%.2f",
                        anomaly_result.residual, anomaly_result.threshold, anomaly_severity)
            # US-324: Log Life_Event node to graph on anomaly detection
            active_graph = graph or self._graph
            if active_graph is not None:
                try:
                    from src.aura.core.self_model import GraphNode, NodeType
                    import uuid as _uuid
                    direction = "drop" if anomaly_result.residual < 0 else "spike"
                    life_event_node = GraphNode(
                        id=f"life_event_anomaly_{_uuid.uuid4().hex[:8]}",
                        node_type=NodeType.LIFE_EVENT,
                        label=f"Readiness anomaly ({direction})",
                        properties={
                            "severity": round(anomaly_severity, 3),
                            "direction": direction,
                            "readiness_at_time": round(readiness_score, 1),
                            "residual": round(anomaly_result.residual, 2),
                            "threshold": round(anomaly_result.threshold, 2),
                            "source": "anomaly_detector",
                        },
                        confidence=anomaly_severity,
                    )
                    active_graph.add_node(life_event_node)
                    logger.info("US-324: Life_Event node logged — direction=%s, severity=%.2f",
                                direction, anomaly_severity)
                except Exception as e:
                    logger.warning("US-324: Failed to log Life_Event node: %s", e)

        # --- US-346: Anomaly-to-action pipeline ---
        anomaly_action_taken = False
        anomaly_dampening_val = 0.0
        regime_dampening_reduced = False  # US-355: Track if dampening was reduced
        if anomaly_detected and anomaly_severity > 0.5:
            # Compute moving average from recent history
            recent_scores = list(self._readiness_history)[-10:] if len(self._readiness_history) >= 2 else []
            if recent_scores:
                moving_avg = sum(recent_scores) / len(recent_scores)
                # US-355: Anomaly-regime coordination — reduce dampening if regime shift active
                if self._last_changepoint_detected:
                    dampening_factor = anomaly_severity * 0.1  # regime shift: minimal dampening
                    regime_dampening_reduced = True
                    logger.info("US-355: Regime shift active — reduced anomaly dampening from %.1f%% to %.1f%%",
                                anomaly_severity * 40, anomaly_severity * 10)
                else:
                    dampening_factor = anomaly_severity * 0.4  # normal dampening
                pre_dampen = readiness_score
                readiness_score = readiness_score * (1.0 - dampening_factor) + moving_avg * dampening_factor
                readiness_score = max(0.0, min(100.0, readiness_score))
                anomaly_action_taken = True
                anomaly_dampening_val = dampening_factor
                logger.info("US-346: Anomaly dampened — severity=%.2f, dampening=%.0f%%, pre=%.1f, post=%.1f",
                           anomaly_severity, dampening_factor * 100, pre_dampen, readiness_score)

        # US-346: Record anomaly event in history
        if anomaly_detected:
            import time as _anomaly_t
            self._anomaly_history.append({
                "timestamp": _anomaly_t.time(),
                "direction": "drop" if anomaly_result.residual < 0 else "spike",
                "severity": round(anomaly_severity, 3),
                "readiness_before": round(readiness_score + (anomaly_dampening_val * readiness_score if anomaly_action_taken else 0), 1),
                "readiness_after": round(readiness_score, 1),
                "dampened": anomaly_action_taken,
            })

        # --- US-326: Emotional regulation & recovery scoring ---
        # GAP-003: Accumulate stress history with real timestamps
        import time as _stress_t
        self._stress_history.append({"stress_level_score": stress_level_score, "timestamp": _stress_t.time()})
        if len(self._stress_history) > 50:
            self._stress_history = self._stress_history[-50:]

        recovery_score = 0.5  # Default neutral
        if self._recovery_scorer is not None and len(self._readiness_history) >= 5:
            try:
                recovery_metrics = self._recovery_scorer.score(
                    readiness_history=self._readiness_history,
                    override_events=recent_override_events,
                    stress_levels=self._stress_history,
                    active_stressors_count=len(active_stressors),
                    current_readiness=readiness_score,
                )
                recovery_score = recovery_metrics.composite_recovery_score
                # Blend recovery into readiness at 6% weight (scale remaining 94%) — US-338 reduced from 8%
                readiness_score = 0.94 * readiness_score + 0.06 * (recovery_score * 100)
                readiness_score = max(0.0, min(100.0, readiness_score))
                logger.info("US-326: Recovery score=%.3f (efficiency=%.3f, discipline=%.3f, absorption=%.3f)",
                            recovery_score, recovery_metrics.recovery_efficiency,
                            recovery_metrics.regulation_discipline, recovery_metrics.stress_absorption)
            except Exception as e:
                logger.warning("US-326: Recovery scoring failed: %s", e)
                recovery_score = 0.5

        # --- US-330: Bayesian changepoint detection ---
        regime_shift_detected = False
        regime_shift_prob = 0.0
        self._last_changepoint_detected = False  # US-355: Reset for this cycle
        if self._changepoint_detector is not None:
            try:
                cp_result = self._changepoint_detector.update(readiness_score)
                regime_shift_detected = cp_result.is_changepoint
                regime_shift_prob = cp_result.changepoint_prob
                if regime_shift_detected:
                    self._last_changepoint_detected = True  # US-355: Flag for anomaly dampening coordination
                    logger.info("US-330: Regime shift detected — prob=%.3f, pre=%.1f, post=%.1f",
                                cp_result.changepoint_prob, cp_result.pre_baseline, cp_result.post_baseline)
                    # Create REGIME_SHIFT Life_Event node in graph
                    active_graph = graph or self._graph
                    if active_graph is not None:
                        try:
                            from src.aura.core.self_model import GraphNode, NodeType
                            import uuid as _uuid2
                            regime_node = GraphNode(
                                id=f"life_event_regime_{_uuid2.uuid4().hex[:8]}",
                                node_type=NodeType.LIFE_EVENT,
                                label="Readiness regime shift",
                                properties={
                                    "severity": round(cp_result.changepoint_prob, 3),
                                    "direction": "shift",
                                    "pre_baseline": round(cp_result.pre_baseline, 1),
                                    "post_baseline": round(cp_result.post_baseline, 1),
                                    "run_length": cp_result.run_length,
                                    "readiness_at_time": round(readiness_score, 1),
                                    "source": "changepoint_detector",
                                },
                                confidence=cp_result.changepoint_prob,
                            )
                            active_graph.add_node(regime_node)
                            logger.info("US-330: REGIME_SHIFT Life_Event logged")
                        except Exception as e:
                            logger.warning("US-330: Failed to log REGIME_SHIFT node: %s", e)
            except Exception as e:
                logger.warning("US-330: Changepoint detection failed: %s", e)

        # --- US-318: STL trend analysis ---
        import time as _time_mod
        self._trend_analyzer.add_sample(readiness_score, _time_mod.time())
        trend_decomp = self._trend_analyzer.decompose()
        trend_direction = trend_decomp.get("trend_direction", "stable")

        # --- US-321/US-328: Decision quality scoring ---
        decision_quality_score_val = 0.0
        if self._decision_quality_scorer is not None:
            try:
                # US-328: Compute metacognitive monitoring score first
                metacog_score = 0.5
                if self._metacognitive_scorer is not None:
                    try:
                        metacog_result = self._metacognitive_scorer.score()
                        metacog_score = metacog_result.composite
                    except Exception as e:
                        logger.warning("US-328: Metacognitive scoring failed: %s", e)

                # US-342: Compute journal reflection quality
                reflection_quality = 0.0
                if self._journal_reflection_scorer is not None:
                    try:
                        conversation_text_for_reflection = message_text or ""
                        refl_result = self._journal_reflection_scorer.score(conversation_text_for_reflection)
                        reflection_quality = refl_result.reflection_quality
                    except Exception as e:
                        logger.warning("US-342: Journal reflection scoring failed: %s", e)

                # Use message_text if available, otherwise empty string (yields neutral scores)
                conversation_text = message_text or ""
                dq_result = self._decision_quality_scorer.score(
                    conversation_text=conversation_text,
                    metacognitive_monitoring_score=metacog_score,
                    reflection_quality=reflection_quality,
                )
                decision_quality_score_val = dq_result.composite_score
            except Exception as e:
                logger.warning("US-321: Decision quality scoring failed: %s", e)

        # --- US-338: Blend decision quality into readiness (7% weight) ---
        if decision_quality_score_val > 0:
            readiness_score = 0.93 * readiness_score + 0.07 * decision_quality_score_val
            readiness_score = max(0.0, min(100.0, readiness_score))
            logger.info("US-338: Decision quality blended — dq=%.1f, readiness=%.1f",
                        decision_quality_score_val, readiness_score)
            # Warn on high readiness with poor decision process
            if decision_quality_score_val < 30 and readiness_score > 70:
                logger.warning("US-338: High readiness (%.1f) with poor decision process (%.1f) — verify before trading",
                               readiness_score, decision_quality_score_val)

        # --- US-340: Cognitive flexibility bonus ---
        flexibility_score = 0.0
        if self._flexibility_scorer is not None and message_text:
            try:
                flex_result = self._flexibility_scorer.score(message_text)
                flexibility_score = flex_result.composite
                if flexibility_score > 0.3:
                    flexibility_bonus = min(5.0, flexibility_score * 7.0)
                    readiness_score = min(100.0, readiness_score + flexibility_bonus)
                    logger.info("US-340: Flexibility bonus +%.1f (composite=%.3f)", flexibility_bonus, flexibility_score)
                # Warn on rigid thinking with high confirmation bias
                conf_bias = (bias_scores or {}).get("confirmation", 0.0)
                if flexibility_score < 0.1 and conf_bias > 0.5:
                    logger.warning("US-340: Rigid thinking detected — flexibility=%.2f, confirmation_bias=%.2f",
                                   flexibility_score, conf_bias)
            except Exception as e:
                logger.warning("US-340: Flexibility scoring failed: %s", e)

        # --- US-344: Style drift penalty ---
        style_drift_penalty_applied = False
        if style_drift_score > 0.4:
            drift_penalty = min(5.0, (style_drift_score - 0.4) * 8.0)
            readiness_score = max(0.0, readiness_score - drift_penalty)
            style_drift_penalty_applied = True
            logger.info("US-344: Style drift penalty -%.1f (drift=%.3f)", drift_penalty, style_drift_score)
            if style_drift_score > 0.6:
                logger.warning("US-344: High linguistic drift (%.3f) — verify emotional state", style_drift_score)

        # --- US-350: Emotional granularity bonus/penalty ---
        granularity_bonus_applied = 0.0
        if granularity_score > 0.6:
            granularity_bonus = min(3.0, (granularity_score - 0.6) * 7.5)
            readiness_score = min(100.0, readiness_score + granularity_bonus)
            granularity_bonus_applied = granularity_bonus
            logger.info("US-350: Granularity bonus +%.1f (granularity=%.3f)", granularity_bonus, granularity_score)
        elif granularity_score < 0.3 and (active_stressors or emotional_state.lower() in ["stressed", "anxious", "frustrated", "overwhelmed", "angry"]):
            granularity_penalty = min(3.0, (0.3 - granularity_score) * 10.0)
            readiness_score = max(0.0, readiness_score - granularity_penalty)
            granularity_bonus_applied = -granularity_penalty
            logger.warning("US-350: Granularity penalty -%.1f (granularity=%.3f) under stress", granularity_penalty, granularity_score)

        readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-350: Narrative coherence bonus/penalty ---
        coherence_bonus_applied = 0.0
        if coherence_score > 0.6:
            coherence_bonus = min(2.0, (coherence_score - 0.6) * 5.0)
            readiness_score = min(100.0, readiness_score + coherence_bonus)
            coherence_bonus_applied = coherence_bonus
            logger.info("US-350: Coherence bonus +%.1f (coherence=%.3f)", coherence_bonus, coherence_score)
        elif coherence_score < 0.3:
            coherence_penalty = min(2.0, (0.3 - coherence_score) * 6.67)
            readiness_score = max(0.0, readiness_score - coherence_penalty)
            coherence_bonus_applied = -coherence_penalty
            logger.warning("US-350: Coherence penalty -%.1f (coherence=%.3f)", coherence_penalty, coherence_score)

        readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-355: Affect dynamics penalties ---
        affect_penalty = 0.0
        if affect_stuck:
            affect_penalty = 5.0
            readiness_score -= affect_penalty
            logger.info("US-355: Stuck negative affect state — penalty=%.1f", affect_penalty)
        elif affect_volatility > 0.6:
            affect_penalty = min(3.0, (affect_volatility - 0.6) * 7.5)
            readiness_score -= affect_penalty
            logger.info("US-355: High affect volatility (%.2f) — penalty=%.1f", affect_volatility, affect_penalty)
        readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-355: Decision fatigue penalty ---
        fatigue_penalty = 0.0
        if fatigue_score_normalized > 70.0:
            fatigue_penalty = min(6.0, (fatigue_score_normalized - 70.0) * 0.2)
            readiness_score -= fatigue_penalty
            logger.info("US-355: Decision fatigue active (%.1f) — penalty=%.1f", fatigue_score_normalized, fatigue_penalty)
        readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-355: Bias interaction penalty (on top of existing) ---
        bias_interaction_penalty_applied = 0.0
        if bias_interaction_penalty > 0:
            readiness_score -= bias_interaction_penalty
            bias_interaction_penalty_applied = bias_interaction_penalty
            logger.info("US-355: Bias interaction penalty=%.1f", bias_interaction_penalty)
            readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-344: Reliability dampener ---
        reliability_dampened = False
        if reliability_score_val < 0.5 and self._reliability_analyzer is not None and len(self._reliability_analyzer.snapshots) >= 10:
            readiness_score = 50.0 + (readiness_score - 50.0) * reliability_score_val
            readiness_score = max(0.0, min(100.0, readiness_score))
            reliability_dampened = True
            logger.info("US-344: Reliability dampener applied (reliability=%.3f)", reliability_score_val)
            if reliability_score_val < 0.4:
                logger.warning("US-344: Low readiness reliability (%.3f) — score may be unreliable", reliability_score_val)

        # --- GAP-009: Apply temporal maintenance to graph nodes ---
        active_graph = graph or self._graph
        if active_graph is not None and hasattr(active_graph, 'prune_dormant_nodes'):
            try:
                active_graph.prune_dormant_nodes()
            except Exception as e:
                logger.debug("GAP-009: Graph temporal maintenance skipped: %s", e)

        # --- US-303: EMA smoothing with hysteresis ---
        raw_score_val = readiness_score
        if self._last_smoothed_score is not None:
            # Use higher alpha (faster response) when emotional state signals
            # genuine deterioration, so stressed/anxious/overwhelmed states
            # aren't masked by the smoothing inertia.
            negative_states = {"stressed", "anxious", "frustrated", "overwhelmed", "fatigued"}
            if emotional_state.lower() in negative_states and readiness_score < self._last_smoothed_score:
                ema_alpha = min(0.7, self.EMA_ALPHA * 2.0)  # 2x faster response on deterioration
            else:
                ema_alpha = self.EMA_ALPHA
            smoothed = ema_alpha * readiness_score + (1.0 - ema_alpha) * self._last_smoothed_score
            # Hysteresis: only update if delta exceeds threshold
            # Use a lower threshold (2.0) for drops under negative emotional states
            # to prevent the smoothing from masking real deterioration.
            if emotional_state.lower() in negative_states and readiness_score < self._last_smoothed_score:
                effective_threshold = min(self.HYSTERESIS_THRESHOLD, 2.0)
            else:
                effective_threshold = self.HYSTERESIS_THRESHOLD
            if abs(smoothed - self._last_smoothed_score) >= effective_threshold:
                self._last_smoothed_score = smoothed
                readiness_score = smoothed
            else:
                readiness_score = self._last_smoothed_score
        else:
            # First computation — no smoothing, set baseline
            self._last_smoothed_score = readiness_score

        readiness_score = max(0.0, min(100.0, readiness_score))

        # --- US-323/US-326/US-330/US-334: Cache state snapshot for bridge enrichment ---
        self._last_state_snapshot = {
            "readiness_score": readiness_score,
            "components": components.to_dict(),
            "bias_scores": bias_scores or {},
            "override_loss_risk": override_loss_risk,
            "tilt_score": tilt_score,
            "emotional_state": emotional_state,
            "cognitive_load_label": cognitive_load_label,
            "graph_context_score": graph_context_score,
            "recovery_score": recovery_score,
            "regime_shift_detected": regime_shift_detected,
            "regime_shift_prob": regime_shift_prob,
            "reliability_score": reliability_score_val,
            "decision_quality_blended": decision_quality_score_val > 0,
            "style_drift_penalty_applied": style_drift_penalty_applied,
            "reliability_dampened": reliability_dampened,
            "anomaly_dampened": anomaly_action_taken,
            "granularity_bonus_applied": granularity_bonus_applied,
            "coherence_bonus_applied": coherence_bonus_applied,
            # US-355: Phase 19 state snapshot fields
            "affect_penalty_applied": affect_penalty,
            "fatigue_penalty_applied": fatigue_penalty,
            "bias_interaction_penalty_applied": bias_interaction_penalty_applied,
            "regime_dampening_reduced": regime_dampening_reduced,
        }

        # --- Build signal ---
        signal = ReadinessSignal(
            readiness_score=readiness_score,
            cognitive_load=cognitive_load_label,
            active_stressors=active_stressors,
            override_loss_rate_7d=override_loss_rate,
            emotional_state=emotional_state,
            confidence_trend=confidence_trend,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
            conversation_count_7d=conversation_count_7d,
            confidence_acceleration=confidence_acceleration,
            fatigue_score=fatigue_score,
            model_version=model_version,
            circadian_multiplier=circ_mult,
            raw_score=raw_score_val,
            smoothed_score=self._last_smoothed_score,
            tilt_score=tilt_score,
            decision_variability=decision_variability,
            anomaly_detected=anomaly_detected,
            anomaly_severity=anomaly_severity,
            anomaly_action_taken=anomaly_action_taken,
            anomaly_dampening=anomaly_dampening_val,
            bias_scores=bias_scores or {},
            override_loss_risk=override_loss_risk,
            trend_direction=trend_direction,
            recovery_score=recovery_score,
            regime_shift_detected=regime_shift_detected,
            decision_quality_score=decision_quality_score_val,
            regime_shift_prob=regime_shift_prob,
            reliability_score=reliability_score_val,
        )

        # --- Write signal to bridge file for Buddy to read ---
        self._write_signal(signal)

        logger.info(
            f"Readiness score: {readiness_score:.0f}/100 "
            f"(emotional={emotional_score:.2f}, cognitive={cognitive_load_score:.2f}, "
            f"override={override_discipline:.2f}, stress={stress_level_score:.2f})"
        )

        return signal

    def _write_signal(self, signal: ReadinessSignal) -> None:
        """Write the readiness signal to the bridge file.

        US-202: Uses FeedbackBridge._locked_write for concurrent access safety.
        """
        try:
            from src.aura.bridge.signals import FeedbackBridge
            FeedbackBridge._locked_write(self.signal_path, signal.to_json())
        except ImportError:
            # Fallback to direct write if bridge not available
            try:
                self.signal_path.write_text(signal.to_json())
            except (OSError, IOError) as e:
                logger.error("US-260: Failed to write readiness signal (direct): %s", e)
        except (OSError, IOError, ValueError) as e:
            logger.error("US-260: Failed to write readiness signal (locked): %s", e)

    def train_from_outcome(self, outcome_signal, readiness_signal: Optional[ReadinessSignal] = None) -> bool:
        """US-309: Update adaptive weights from trade outcome.

        Determines which readiness components predicted the outcome correctly,
        then updates the Bayesian priors for each component.

        Args:
            outcome_signal: OutcomeSignal with trade result
            readiness_signal: The ReadinessSignal that was active during the trade

        Returns:
            True if training was performed, False if skipped
        """
        if not self._adaptive_weights:
            logger.debug("US-309: No adaptive weights manager — skipping training")
            return False

        if readiness_signal is None:
            readiness_signal = self.read_latest_signal()
        if readiness_signal is None:
            logger.debug("US-309: No readiness signal available — skipping training")
            return False

        # Determine trade outcome
        profit = 0.0
        if isinstance(outcome_signal, dict):
            trade_won = outcome_signal.get("trade_won", False)
            profit = outcome_signal.get("profit_pips", 0.0)
        else:
            trade_won = getattr(outcome_signal, "trade_won", False)
            profit = getattr(outcome_signal, "profit_pips", 0.0)

        # Compute days_old from outcome timestamp
        days_old = 0.0
        outcome_ts = None
        if isinstance(outcome_signal, dict):
            outcome_ts = outcome_signal.get("timestamp")
        else:
            outcome_ts = getattr(outcome_signal, "timestamp", None)
        if outcome_ts:
            try:
                from datetime import datetime, timezone
                if isinstance(outcome_ts, str):
                    ot = datetime.fromisoformat(outcome_ts.replace("Z", "+00:00"))
                else:
                    ot = outcome_ts
                days_old = max(0.0, (datetime.now(timezone.utc) - ot).total_seconds() / 86400)
            except Exception:
                days_old = 0.0

        # For each component, determine if it predicted correctly
        # "Correct" = component score > 0.5 and trade won, OR component score <= 0.5 and trade lost
        components = readiness_signal.components.to_dict()
        for comp_name, comp_score in components.items():
            predicted_good = comp_score > 0.5
            prediction_correct = (predicted_good and trade_won) or (not predicted_good and not trade_won)
            self._adaptive_weights.update(comp_name, prediction_correct, days_old)

        # Persist updated weights
        self._adaptive_weights.save()
        logger.info("US-309: Trained adaptive weights from outcome (won=%s, profit=%.1f, days_old=%.1f)",
                    trade_won, profit, days_old)
        return True

    def read_latest_signal(self) -> Optional[ReadinessSignal]:
        """Read the latest readiness signal from disk.

        US-202: Uses FeedbackBridge._locked_read for concurrent access safety.
        """
        try:
            from src.aura.bridge.signals import FeedbackBridge
            raw = FeedbackBridge._locked_read(self.signal_path)
        except ImportError:
            if not self.signal_path.exists():
                return None
            raw = self.signal_path.read_text()
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            components = ReadinessComponents(**data.pop("components", {}))
            # Fix H-03: Filter data to only known ReadinessSignal fields before **kwargs expansion.
            # Previously any unexpected key in the JSON caused TypeError: __init__() got an
            # unexpected keyword argument. Using dataclasses.fields() ensures forward-compatible
            # deserialization — unknown keys are logged and ignored rather than crashing.
            known_fields = {f.name for f in dataclasses.fields(ReadinessSignal)} - {"components"}
            unknown_keys = set(data.keys()) - known_fields
            if unknown_keys:
                logger.warning(
                    "H-03 fix: Ignoring unknown ReadinessSignal keys from disk: %s", unknown_keys
                )
            filtered_data = {k: v for k, v in data.items() if k in known_fields}
            return ReadinessSignal(components=components, **filtered_data)
        except Exception as e:
            logger.warning(f"Failed to read readiness signal: {e}")
            return None
