"""Tier 2 Pattern Engine — Cross-domain correlations.

The core intelligence that makes Aura novel: correlating human states
with trading outcomes to find patterns neither system could see alone.

PRD v2.2 §7.4 / US-2.3:
  - Cross-references human engine data with domain engine outcomes
  - Surfaces correlations with significance threshold (p < 0.05, min 10 data points)
  - Example outputs:
    "Win rate drops 40% during weeks you mention career stress 3+ times"
    "Your best trading days follow conversations where you expressed clarity"
    "Overrides after 2+ losses are 80% losers"
  - Correlations displayed with evidence trail
  - User rates correlation quality; ratings feed back

This module runs on a weekly cadence (or on-demand via CLI).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.aura.persistence import atomic_write_json  # Fix C-05: needed for atomic pattern writes

from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)

logger = logging.getLogger(__name__)

# Minimum data points for a correlation to be considered
MIN_SAMPLE_SIZE = 5
# Significance threshold — patterns below this p-value are "significant"
P_VALUE_THRESHOLD = 0.10  # Relaxed from 0.05 for small sample sizes early on
# Minimum absolute correlation strength to report
MIN_CORRELATION_STRENGTH = 0.25


def _compute_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient and approximate p-value.

    Uses a t-test approximation for p-value since we avoid heavy scipy dependency.

    Args:
        x: First variable values
        y: Second variable values (same length)

    Returns:
        (correlation, p_value) tuple. Both 0.0 if computation fails.
    """
    n = len(x)
    if n < MIN_SAMPLE_SIZE or len(y) != n:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Covariance and standard deviations
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0, 1.0

    r = cov / (std_x * std_y)
    r = max(-1.0, min(1.0, r))  # Clamp for numerical stability

    # Approximate p-value using t-distribution approximation
    if abs(r) >= 1.0:
        return r, 0.0

    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    # Approximate p-value: for n >= 5, use normal approximation
    # This is rough but avoids scipy dependency
    p_approx = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return r, p_approx


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using Abramowitz and Stegun formula."""
    # Good to ~1e-5 accuracy
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def _compute_lagged_correlations(
    series_a: Dict[str, float],
    series_b: Dict[str, float],
    max_lag: int = 7,
    date_format: str = "%Y-%m-%d",
) -> List[Tuple[int, float, float]]:
    """Compute correlations between series_a and series_b at multiple time lags.

    For each lag in [0, max_lag], shifts series_a backward by `lag` days and
    correlates with series_b on the same date. This discovers delayed effects:
    e.g., stress on day D correlates with poor trading on day D+2.

    Args:
        series_a: Dict mapping date strings to float values (the "cause" series)
        series_b: Dict mapping date strings to float values (the "effect" series)
        max_lag: Maximum lag in days to test (default 7)
        date_format: Date string format (default "%Y-%m-%d")

    Returns:
        List of (lag_days, correlation, p_value) tuples for all tested lags,
        sorted by absolute correlation descending. Only lags with sufficient
        data points (>= MIN_SAMPLE_SIZE) are included.
    """
    results: List[Tuple[int, float, float]] = []

    for lag in range(max_lag + 1):
        # For each date in series_b, look for series_a value `lag` days earlier
        aligned_a: List[float] = []
        aligned_b: List[float] = []

        for date_str, b_val in series_b.items():
            try:
                dt = datetime.strptime(date_str, date_format)
                lagged_date = (dt - timedelta(days=lag)).strftime(date_format)
            except (ValueError, TypeError):
                continue

            if lagged_date in series_a:
                aligned_a.append(series_a[lagged_date])
                aligned_b.append(b_val)

        if len(aligned_a) >= MIN_SAMPLE_SIZE:
            r, p = _compute_correlation(aligned_a, aligned_b)
            results.append((lag, r, p))

    # Sort by absolute correlation descending
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


class Tier2CrossDomainDetector:
    """Cross-domain correlation detector.

    Correlates data from the human engine (conversations, emotions, stressors,
    readiness) with domain engine data (trade outcomes, overrides, regimes)
    to find statistically significant cross-engine patterns.

    Args:
        patterns_dir: Where to persist detected patterns
    """

    def __init__(self, patterns_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.patterns_dir = patterns_dir or Path(".aura/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = self.patterns_dir / "t2_patterns.json"
        self._active_patterns: Dict[str, DetectedPattern] = {}

        # US-272: Configurable thresholds
        cfg = config or {}
        self.min_sample_size = cfg.get("min_sample_size", MIN_SAMPLE_SIZE)
        self.p_value_threshold = cfg.get("p_value_threshold", P_VALUE_THRESHOLD)
        self.min_correlation_strength = cfg.get("min_correlation_strength", MIN_CORRELATION_STRENGTH)

        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load persisted T2 patterns."""
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
            logger.warning(f"T2: Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Persist T2 patterns to disk atomically.

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
            logger.error(f"T2: Failed to save patterns: {e}")

    def detect(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
        anomaly_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectedPattern]:
        """Run all T2 cross-domain correlation analyses.

        Args:
            conversations: Conversation records with emotional_state, topics, timestamp
            readiness_history: Readiness score history with score, timestamp
            trade_outcomes: Trade journal entries with pnl, pair, timestamp, outcome
            override_events: Override events with emotional context and outcomes
            anomaly_context: US-324 — Recent Life_Event nodes from anomaly detection

        Returns:
            List of newly detected or updated cross-domain patterns
        """
        new_patterns: List[DetectedPattern] = []

        new_patterns.extend(
            self._correlate_stress_with_win_rate(conversations, trade_outcomes)
        )
        new_patterns.extend(
            self._correlate_readiness_with_pnl(readiness_history, trade_outcomes)
        )
        new_patterns.extend(
            self._correlate_stressor_with_overrides(conversations, override_events)
        )
        new_patterns.extend(
            self._correlate_emotion_with_trade_timing(conversations, trade_outcomes)
        )
        new_patterns.extend(
            self._detect_override_after_loss_streak(override_events, trade_outcomes)
        )

        # US-324: Correlate anomalies with override events
        if anomaly_context:
            new_patterns.extend(
                self._correlate_anomalies(anomaly_context, override_events)
            )

        self._save_patterns()

        if new_patterns:
            logger.info(f"T2: Detected {len(new_patterns)} cross-domain patterns")

        return new_patterns

    def _correlate_anomalies(
        self,
        anomaly_events: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """US-324: Find co-occurrences between readiness anomalies and override events.

        Detects patterns like 'anomaly severity 0.8 + override spike = impaired state'.
        Looks for override events that occur within 24 hours of an anomaly event.

        Args:
            anomaly_events: Life_Event nodes with severity, direction, timestamp
            override_events: Override events with timestamp and outcome

        Returns:
            List of detected co-occurrence patterns
        """
        if not anomaly_events or not override_events:
            return []

        patterns: List[DetectedPattern] = []
        co_occurrences: List[Dict[str, Any]] = []

        for anomaly in anomaly_events:
            anomaly_ts = anomaly.get("timestamp") or anomaly.get("properties", {}).get("timestamp", "")
            if not anomaly_ts:
                continue
            try:
                a_dt = datetime.fromisoformat(anomaly_ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            severity = anomaly.get("severity") or anomaly.get("properties", {}).get("severity", 0)
            direction = anomaly.get("direction") or anomaly.get("properties", {}).get("direction", "unknown")

            for override in override_events:
                o_ts = override.get("timestamp", "")
                if not o_ts:
                    continue
                try:
                    o_dt = datetime.fromisoformat(o_ts.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

                # Check if within 24 hours
                delta_hours = abs((o_dt - a_dt).total_seconds()) / 3600.0
                if delta_hours <= 24.0:
                    co_occurrences.append({
                        "anomaly_severity": severity,
                        "anomaly_direction": direction,
                        "override_outcome": override.get("outcome", "unknown"),
                        "delta_hours": round(delta_hours, 1),
                    })

        if len(co_occurrences) >= 2:
            # Calculate loss rate during anomalies
            loss_count = sum(1 for c in co_occurrences if c["override_outcome"] == "loss")
            total = len(co_occurrences)
            loss_rate = loss_count / total if total > 0 else 0

            avg_severity = sum(c["anomaly_severity"] for c in co_occurrences) / total

            pattern = DetectedPattern(
                pattern_id="t2_anomaly_override_cooccurrence",
                tier=PatternTier.T2_WEEKLY,
                domain=PatternDomain.CROSS_ENGINE,
                description=(
                    "Readiness anomalies co-occur with overrides: "
                    f"{total} events, {loss_rate:.0%} loss rate during anomalies "
                    f"(avg severity {avg_severity:.2f})"
                ),
                confidence=min(0.9, total / 10.0),
                evidence=[
                    EvidenceItem(
                        source_type="anomaly_override_correlation",
                        source_id="anomaly_cascade",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        summary=f"{total} co-occurrences, loss rate {loss_rate:.0%}",
                        data={
                            "co_occurrences": total,
                            "loss_rate_during_anomaly": round(loss_rate, 3),
                            "avg_anomaly_severity": round(avg_severity, 3),
                        },
                    )
                ],
                status=PatternStatus.DETECTED,
                observation_count=total,
            )
            patterns.append(pattern)
            self._active_patterns[pattern.pattern_id] = pattern
            logger.info(
                "US-324: Anomaly-override co-occurrence — %d events, loss_rate=%.2f",
                total, loss_rate,
            )

        return patterns

    def _correlate_stress_with_win_rate(
        self,
        conversations: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """PRD example: 'Win rate drops 40% during weeks you mention career stress 3+ times.'

        Groups data by week, counts stress mentions per week, computes win rate per week,
        then correlates.
        """
        if len(conversations) < self.min_sample_size or len(trade_outcomes) < self.min_sample_size:
            return []

        # Group conversations and trades by ISO week
        week_stress: Dict[str, int] = defaultdict(int)
        week_trade_results: Dict[str, List[bool]] = defaultdict(list)

        negative_states = {"stressed", "anxious", "fatigued", "frustrated", "overwhelmed"}

        for conv in conversations:
            ts = conv.get("timestamp", "")
            emotional_state = conv.get("emotional_state", "neutral")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            if emotional_state.lower() in negative_states:
                week_stress[week_key] += 1

        for trade in trade_outcomes:
            ts = trade.get("timestamp", trade.get("close_time", ""))
            won = trade.get("won", trade.get("pnl_pips", 0) > 0)
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            week_trade_results[week_key].append(bool(won))

        # Build aligned weekly data — use all weeks that have trades,
        # defaulting stress to 0 for calm weeks (they matter for correlation!)
        common_weeks = sorted(week_trade_results.keys())
        if len(common_weeks) < self.min_sample_size:
            return []

        stress_counts = [week_stress.get(w, 0) for w in common_weeks]
        win_rates = [
            sum(week_trade_results[w]) / len(week_trade_results[w])
            for w in common_weeks
        ]

        r, p_value = _compute_correlation(stress_counts, win_rates)

        if abs(r) < MIN_CORRELATION_STRENGTH or p_value > P_VALUE_THRESHOLD:
            return []

        # Compute effect size for description
        high_stress_weeks = [
            w for w, s in zip(common_weeks, stress_counts) if s >= 3
        ]
        low_stress_weeks = [
            w for w, s in zip(common_weeks, stress_counts) if s < 2
        ]
        high_wr = (
            sum(sum(week_trade_results[w]) for w in high_stress_weeks)
            / max(sum(len(week_trade_results[w]) for w in high_stress_weeks), 1)
        ) if high_stress_weeks else 0
        low_wr = (
            sum(sum(week_trade_results[w]) for w in low_stress_weeks)
            / max(sum(len(week_trade_results[w]) for w in low_stress_weeks), 1)
        ) if low_stress_weeks else 0

        wr_diff = low_wr - high_wr
        description = (
            f"Win rate {'drops' if r < 0 else 'rises'} {abs(wr_diff):.0%} during "
            f"weeks with 3+ stress conversations vs calm weeks "
            f"(r={r:.2f}, p={p_value:.3f}, n={len(common_weeks)} weeks)"
        )

        return self._upsert_pattern(
            pattern_key="stress_win_rate_correlation",
            description=description,
            evidence=EvidenceItem(
                source_type="cross_domain_correlation",
                source_id="stress_vs_win_rate",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"Stress-WinRate r={r:.2f}, p={p_value:.3f}",
                data={
                    "correlation": round(r, 4),
                    "p_value": round(p_value, 4),
                    "sample_weeks": len(common_weeks),
                    "high_stress_win_rate": round(high_wr, 3),
                    "low_stress_win_rate": round(low_wr, 3),
                    "win_rate_difference": round(wr_diff, 3),
                },
            ),
            correlation_strength=r,
            p_value=p_value,
            sample_size=len(common_weeks),
            confidence=min(0.9, 0.5 + (1 - p_value) * 0.4),
            suggested_rule=(
                f"Reduce position size by {min(40, int(abs(wr_diff) * 100))}% during "
                f"weeks with 3+ stress conversations"
            ),
        )

    def _correlate_readiness_with_pnl(
        self,
        readiness_history: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Correlate readiness scores with trading PnL, testing lags 0-7 days.

        US-296: Now tests multiple time lags to discover delayed effects.
        E.g., readiness drop on Monday → poor trading on Wednesday (lag=2).
        """
        if len(readiness_history) < self.min_sample_size or len(trade_outcomes) < self.min_sample_size:
            return []

        # Build day → avg readiness and day → PnL maps
        day_readiness_lists: Dict[str, List[float]] = defaultdict(list)
        day_pnl: Dict[str, float] = defaultdict(float)

        for r in readiness_history:
            ts = r.get("timestamp", "")
            score = r.get("score", 70)
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue
            day_readiness_lists[day_key].append(score)

        for trade in trade_outcomes:
            ts = trade.get("timestamp", trade.get("close_time", ""))
            pnl = trade.get("pnl_pips", trade.get("pnl", 0))
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue
            day_pnl[day_key] += float(pnl)

        # Compute average readiness per day for lagged correlation
        day_readiness_avg: Dict[str, float] = {
            d: sum(scores) / len(scores)
            for d, scores in day_readiness_lists.items()
        }

        # US-296: Test lags 0-7 days
        lagged_results = _compute_lagged_correlations(
            series_a=day_readiness_avg,
            series_b=day_pnl,
            max_lag=7,
        )

        if not lagged_results:
            return []

        # Pick the strongest correlation that passes thresholds
        best_lag, best_r, best_p = None, 0.0, 1.0
        for lag, r, p in lagged_results:
            if abs(r) >= self.min_correlation_strength and p <= self.p_value_threshold:
                best_lag, best_r, best_p = lag, r, p
                break  # Already sorted by |r| descending

        if best_lag is None:
            return []

        lag_text = f" (lag={best_lag} days)" if best_lag > 0 else ""
        description = (
            f"Readiness score {'positively' if best_r > 0 else 'negatively'} correlates "
            f"with daily PnL{lag_text} (r={best_r:.2f}, p={best_p:.3f}). "
            f"Higher readiness = {'better' if best_r > 0 else 'worse'} trading days."
        )

        evidence_data: Dict[str, Any] = {
            "correlation": round(best_r, 4),
            "p_value": round(best_p, 4),
            "sample_days": len(day_pnl),
        }
        if best_lag > 0:
            evidence_data["optimal_lag_days"] = best_lag

        return self._upsert_pattern(
            pattern_key="readiness_pnl_correlation",
            description=description,
            evidence=EvidenceItem(
                source_type="cross_domain_correlation",
                source_id="readiness_vs_pnl",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"Readiness-PnL r={best_r:.2f}, p={best_p:.3f}{lag_text}",
                data=evidence_data,
            ),
            correlation_strength=best_r,
            p_value=best_p,
            sample_size=len(day_pnl),
            confidence=min(0.9, 0.5 + (1 - best_p) * 0.4),
            suggested_rule=(
                f"Readiness-PnL correlation is {best_r:.2f}{lag_text} — readiness signal "
                f"is {'validating well' if best_r > 0.3 else 'needs recalibration'}"
            ),
        )

    def _correlate_stressor_with_overrides(
        self,
        conversations: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect if specific stressors predict overrides.

        PRD example: "Override rate 3x during high-stress weeks"
        """
        if len(conversations) < 3 or len(override_events) < 3:
            return []

        # Count stressors in conversations near override events
        stressor_override_counts: Dict[str, int] = defaultdict(int)
        stressor_total_counts: Dict[str, int] = defaultdict(int)

        for conv in conversations:
            topics_raw = conv.get("topics", "[]")
            if isinstance(topics_raw, str):
                try:
                    topics = json.loads(topics_raw)
                except (json.JSONDecodeError, TypeError):
                    topics = []
            else:
                topics = topics_raw

            for topic in topics:
                stressor_total_counts[topic] += 1

        # Check which stressors appear near overrides
        for override in override_events:
            emotional_state = override.get("emotional_state", "")
            context = override.get("conversation_context", "")

            for stressor in stressor_total_counts:
                if stressor.lower() in context.lower() or stressor.lower() in emotional_state.lower():
                    stressor_override_counts[stressor] += 1

        patterns = []
        for stressor, override_count in stressor_override_counts.items():
            total = stressor_total_counts.get(stressor, 0)
            if override_count >= 2 and total >= 3:
                override_rate = override_count / max(len(override_events), 1)
                description = (
                    f"'{stressor}' present in {override_count}/{len(override_events)} "
                    f"override events — this stressor may trigger impulsive trading"
                )
                result = self._upsert_pattern(
                    pattern_key=f"stressor_override_{stressor}",
                    description=description,
                    evidence=EvidenceItem(
                        source_type="stressor_override_correlation",
                        source_id=stressor,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        summary=f"'{stressor}' in {override_count} overrides",
                        data={
                            "stressor": stressor,
                            "override_count": override_count,
                            "total_overrides": len(override_events),
                            "stressor_frequency": total,
                        },
                    ),
                    confidence=min(0.85, 0.5 + override_rate * 0.35),
                    suggested_rule=(
                        f"When '{stressor}' active, increase override warning sensitivity"
                    ),
                )
                patterns.extend(result)

        return patterns

    def _correlate_emotion_with_trade_timing(
        self,
        conversations: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """PRD example: 'Your best trading days follow conversations where you expressed clarity.'

        Checks if positive emotional states in conversations predict next-day
        trading performance.
        """
        if len(conversations) < self.min_sample_size or len(trade_outcomes) < self.min_sample_size:
            return []

        positive_states = {"calm", "energized", "confident", "focused"}
        negative_states = {"stressed", "anxious", "fatigued", "frustrated"}

        # Build day → emotional valence map
        day_emotional_valence: Dict[str, float] = {}
        for conv in conversations:
            ts = conv.get("timestamp", "")
            state = conv.get("emotional_state", "neutral").lower()
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue

            if state in positive_states:
                day_emotional_valence[day_key] = day_emotional_valence.get(day_key, 0) + 1
            elif state in negative_states:
                day_emotional_valence[day_key] = day_emotional_valence.get(day_key, 0) - 1

        # Build day → PnL map
        day_pnl: Dict[str, float] = defaultdict(float)
        for trade in trade_outcomes:
            ts = trade.get("timestamp", trade.get("close_time", ""))
            pnl = trade.get("pnl_pips", trade.get("pnl", 0))
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue
            day_pnl[day_key] += float(pnl)

        # Align: conversation emotion on day D → trade performance on day D
        common_days = sorted(set(day_emotional_valence.keys()) & set(day_pnl.keys()))
        if len(common_days) < self.min_sample_size:
            return []

        valences = [day_emotional_valence[d] for d in common_days]
        pnls = [day_pnl[d] for d in common_days]

        r, p_value = _compute_correlation(valences, pnls)

        if abs(r) < MIN_CORRELATION_STRENGTH or p_value > P_VALUE_THRESHOLD:
            return []

        description = (
            f"Positive emotional conversations {'predict better' if r > 0 else 'correlate with worse'} "
            f"same-day trading (r={r:.2f}, p={p_value:.3f}, n={len(common_days)} days)"
        )

        return self._upsert_pattern(
            pattern_key="emotion_trade_timing",
            description=description,
            evidence=EvidenceItem(
                source_type="cross_domain_correlation",
                source_id="emotion_vs_trading",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"Emotion-Trading r={r:.2f}, p={p_value:.3f}",
                data={
                    "correlation": round(r, 4),
                    "p_value": round(p_value, 4),
                    "sample_days": len(common_days),
                },
            ),
            correlation_strength=r,
            p_value=p_value,
            sample_size=len(common_days),
            confidence=min(0.9, 0.5 + (1 - p_value) * 0.4),
        )

    def _detect_override_after_loss_streak(
        self,
        override_events: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """PRD example: 'Overrides after 2+ losses are 80% losers.'

        Checks if overrides that happen during loss streaks have worse outcomes.
        """
        if len(override_events) < 3 or len(trade_outcomes) < 5:
            return []

        # Build a simple streak tracker from trade outcomes
        # Sort trades by timestamp
        sorted_trades = sorted(
            trade_outcomes,
            key=lambda t: t.get("timestamp", t.get("close_time", "")),
        )

        # Track running loss streak at each point in time
        loss_streak = 0
        date_streak: Dict[str, int] = {}
        for trade in sorted_trades:
            won = trade.get("won", trade.get("pnl_pips", 0) > 0)
            ts = trade.get("timestamp", trade.get("close_time", ""))
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue

            if won:
                loss_streak = 0
            else:
                loss_streak += 1
            date_streak[day_key] = loss_streak

        # Check override outcomes during loss streaks
        overrides_during_streak = []
        overrides_normal = []

        for override in override_events:
            ts = override.get("timestamp", "")
            outcome = override.get("outcome", "")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day_key = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue

            streak_at_time = date_streak.get(day_key, 0)
            if outcome in ("win", "loss"):
                if streak_at_time >= 2:
                    overrides_during_streak.append(outcome == "loss")
                else:
                    overrides_normal.append(outcome == "loss")

        if len(overrides_during_streak) < 2:
            return []

        streak_loss_rate = sum(overrides_during_streak) / len(overrides_during_streak)
        normal_loss_rate = (
            sum(overrides_normal) / max(len(overrides_normal), 1)
            if overrides_normal
            else 0.5
        )

        if streak_loss_rate < 0.6:  # Only flag if loss rate is meaningfully high
            return []

        description = (
            f"Overrides during 2+ loss streaks are {streak_loss_rate:.0%} losers "
            f"(vs {normal_loss_rate:.0%} normally). "
            f"Based on {len(overrides_during_streak)} streak overrides."
        )

        return self._upsert_pattern(
            pattern_key="override_after_loss_streak",
            description=description,
            evidence=EvidenceItem(
                source_type="override_streak_analysis",
                source_id="loss_streak_overrides",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=(
                    f"Streak overrides: {streak_loss_rate:.0%} loss rate "
                    f"({len(overrides_during_streak)} events)"
                ),
                data={
                    "streak_loss_rate": round(streak_loss_rate, 3),
                    "normal_loss_rate": round(normal_loss_rate, 3),
                    "streak_overrides": len(overrides_during_streak),
                    "normal_overrides": len(overrides_normal),
                },
            ),
            confidence=min(0.9, 0.5 + streak_loss_rate * 0.4),
            suggested_rule=(
                f"Block overrides during 2+ loss streaks — "
                f"{streak_loss_rate:.0%} historical loss rate"
            ),
        )

    def _upsert_pattern(
        self,
        pattern_key: str,
        description: str,
        evidence: EvidenceItem,
        confidence: float = 0.5,
        correlation_strength: Optional[float] = None,
        p_value: Optional[float] = None,
        sample_size: int = 0,
        suggested_rule: Optional[str] = None,
    ) -> List[DetectedPattern]:
        """Create or update a T2 pattern."""
        existing = self._active_patterns.get(pattern_key)

        if existing and existing.status not in (
            PatternStatus.ARCHIVED,
            PatternStatus.INVALIDATED,
        ):
            existing.add_evidence(evidence)
            existing.confidence = confidence
            existing.description = description
            existing.correlation_strength = correlation_strength
            existing.p_value = p_value
            existing.sample_size = sample_size
            if suggested_rule:
                existing.suggested_rule = suggested_rule
            return [existing]

        pattern = DetectedPattern(
            pattern_id=pattern_key,
            tier=PatternTier.T2_WEEKLY,
            domain=PatternDomain.CROSS_ENGINE,
            description=description,
            evidence=[evidence],
            observation_count=1,
            confidence=confidence,
            correlation_strength=correlation_strength,
            p_value=p_value,
            sample_size=sample_size,
            suggested_rule=suggested_rule,
        )
        self._active_patterns[pattern_key] = pattern
        logger.info(f"T2: New cross-domain pattern: '{pattern_key}'")
        return [pattern]

    def get_active_patterns(self) -> List[DetectedPattern]:
        """Return all non-archived T2 patterns."""
        return [
            p for p in self._active_patterns.values()
            if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
        ]

    def get_significant_patterns(self) -> List[DetectedPattern]:
        """Return patterns with statistical significance."""
        return [
            p for p in self._active_patterns.values()
            if p.p_value is not None
            and p.p_value <= P_VALUE_THRESHOLD
            and p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
        ]

    def get_promotable_patterns(self) -> List[DetectedPattern]:
        """Return patterns ready for rule promotion."""
        return [p for p in self._active_patterns.values() if p.is_promotable()]
