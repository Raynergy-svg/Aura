"""Pattern Engine Orchestrator — coordinates T1, T2, and T3 detectors.

This is the central hub that:
1. Gathers data from self-model graph, bridge signals, and trade journal
2. Feeds data into each tier's detector
3. Collects detected patterns
4. Manages pattern lifecycle (detect → validate → promote → archive)
5. Optionally enriches patterns via cloud synthesis (LLM fallback)
6. Provides the CLI command interface (/patterns in Aura companion)

PRD v2.2 §7.1:
  The pattern engine is the intelligence layer that transforms raw data
  from both engines into actionable insights. Each tier runs at different
  cadences with different computational costs.

  T1 (daily):   Frequency-based, lightweight
  T2 (weekly):  Cross-domain correlations
  T3 (monthly): Narrative arcs, prediction models
  Cloud:        Optional LLM synthesis for deeper explanations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.patterns.base import (
    DetectedPattern,
    PatternDomain,
    PatternStatus,
    PatternTier,
)
from src.aura.patterns.tier1 import Tier1FrequencyDetector
from src.aura.patterns.tier2 import Tier2CrossDomainDetector
from src.aura.patterns.tier3 import Tier3NarrativeArcDetector
from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

logger = logging.getLogger(__name__)


class TierReloadResult:
    """US-312: Result of a tier pattern reload."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class PatternEngine:
    """Orchestrates pattern detection across all tiers.

    Coordinates data gathering, tier execution, pattern lifecycle management,
    and rule promotion.

    Args:
        patterns_dir: Base directory for pattern persistence
        bridge_dir: Directory for feedback bridge signals
        trade_journal_path: Path to Buddy's trade journal (for T2 correlations)
    """

    def __init__(
        self,
        patterns_dir: Optional[Path] = None,
        bridge_dir: Optional[Path] = None,
        trade_journal_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.patterns_dir = patterns_dir or Path(".aura/patterns")
        self.bridge_dir = bridge_dir or Path(".aura/bridge")
        self.trade_journal_path = trade_journal_path or Path(
            "trained_data/trade_journal_rl.json"
        )

        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # US-272: Load config from .aura/config.json, with overrides
        # L-03 fix: hoist config imports above the if-block to avoid F811 redefinition
        from src.aura.config import get_t1_config, get_t2_config, get_t3_config
        if config is None:
            from src.aura.config import load_config
            aura_root = self.bridge_dir.parent if self.bridge_dir else Path(".aura")
            config_path = aura_root / "config.json"
            config = load_config(config_path)
        self._config = config

        # Initialize tier detectors with config
        self.t1 = Tier1FrequencyDetector(patterns_dir=self.patterns_dir, config=get_t1_config(config))
        self.t2 = Tier2CrossDomainDetector(patterns_dir=self.patterns_dir, config=get_t2_config(config))
        self.t3 = Tier3NarrativeArcDetector(patterns_dir=self.patterns_dir, config=get_t3_config(config))

        # Cloud synthesis (optional — works without LLM configured)
        self.cloud = CloudPatternSynthesizer(
            log_dir=self.patterns_dir / "synthesis_logs"
        )

        # Track last run times
        self._run_log_path = self.patterns_dir / "run_log.json"
        self._run_log = self._load_run_log()

    def _load_run_log(self) -> Dict[str, str]:
        """Load last run timestamps for each tier."""
        if self._run_log_path.exists():
            try:
                return json.loads(self._run_log_path.read_text())
            except Exception:
                pass
        return {}

    def _save_run_log(self) -> None:
        """Persist run log."""
        try:
            self._run_log_path.write_text(
                json.dumps(self._run_log, indent=2, default=str)
            )
        except Exception as e:
            logger.error(f"PatternEngine: Failed to save run log: {e}")

    # --- Data Gathering ---

    def _load_trade_journal(self) -> List[Dict[str, Any]]:
        """Load trade outcomes from Buddy's journal."""
        if not self.trade_journal_path.exists():
            return []
        try:
            data = json.loads(self.trade_journal_path.read_text())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("trades", data.get("entries", []))
            return []
        except Exception as e:
            logger.warning(f"PatternEngine: Failed to load trade journal: {e}")
            return []

    def _load_override_events(self) -> List[Dict[str, Any]]:
        """Load override events from the bridge."""
        override_path = self.bridge_dir / "override_events.jsonl"
        if not override_path.exists():
            return []
        try:
            events = []
            with open(override_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return events
        except Exception as e:
            logger.warning(f"PatternEngine: Failed to load overrides: {e}")
            return []

    # --- Tier Execution ---

    def run_t1(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Run Tier 1 (daily) pattern detection.

        Should be called after every conversation or at end of each trading session.

        Args:
            conversations: From self-model graph (get_recent_conversations)
            readiness_history: From self-model graph (get_readiness_history)

        Returns:
            Newly detected or updated T1 patterns
        """
        override_events = self._load_override_events()

        patterns = self.t1.detect(
            conversations=conversations,
            readiness_history=readiness_history,
            override_events=override_events,
        )

        self._run_log["t1_last_run"] = datetime.now(timezone.utc).isoformat()
        self._save_run_log()

        return patterns

    def run_t2(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        anomaly_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectedPattern]:
        """Run Tier 2 (weekly) cross-domain correlation detection.

        Should be called weekly or on-demand via CLI.

        Args:
            conversations: From self-model graph (broader window for T2)
            readiness_history: From self-model graph
            anomaly_context: US-324 — Recent Life_Event nodes from graph

        Returns:
            Newly detected or updated T2 cross-domain patterns
        """
        trade_outcomes = self._load_trade_journal()
        override_events = self._load_override_events()

        patterns = self.t2.detect(
            conversations=conversations,
            readiness_history=readiness_history,
            trade_outcomes=trade_outcomes,
            override_events=override_events,
            anomaly_context=anomaly_context,
        )

        self._run_log["t2_last_run"] = datetime.now(timezone.utc).isoformat()
        self._save_run_log()

        return patterns

    def run_t3(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Run Tier 3 (monthly) narrative arc detection.

        Should be called monthly or on-demand via CLI.

        Args:
            conversations: Full conversation history (multi-week window)
            readiness_history: Full readiness score history

        Returns:
            Newly detected or updated T3 narrative arc patterns
        """
        trade_outcomes = self._load_trade_journal()
        override_events = self._load_override_events()

        patterns = self.t3.detect(
            conversations=conversations,
            readiness_history=readiness_history,
            trade_outcomes=trade_outcomes,
            override_events=override_events,
        )

        self._run_log["t3_last_run"] = datetime.now(timezone.utc).isoformat()
        self._save_run_log()

        # Optionally enrich T3 patterns with cloud synthesis
        if self.cloud.is_available() and patterns:
            self._enrich_with_cloud(patterns)

        return patterns

    def _enrich_with_cloud(self, patterns: List[DetectedPattern]) -> None:
        """Optionally enrich detected patterns with cloud synthesis.

        Only called when cloud is available. Adds narrative depth
        to patterns that have low confidence or complex evidence.
        """
        # Single-pattern explanations for low-confidence patterns
        for pattern in patterns:
            if pattern.confidence < 0.7 and self.cloud.is_available():
                result = self.cloud.synthesize_pattern_explanation(
                    pattern.description,
                    [e.to_dict() for e in pattern.evidence[-5:]],
                )
                if result.success and result.narrative:
                    pattern.description += f" | Cloud insight: {result.narrative[:200]}"
                    pattern.confidence = min(
                        0.95, pattern.confidence + result.confidence_boost
                    )

        # Cross-pattern connections if 3+ patterns detected
        if len(patterns) >= 3 and self.cloud.is_available():
            result = self.cloud.synthesize_pattern_connections(
                [p.to_dict() for p in patterns]
            )
            if result.success:
                logger.info(
                    "Cloud cross-pattern synthesis: %s", result.narrative[:100]
                )

    def _reload_tier_patterns(self) -> Dict[str, str]:
        """US-207/US-312: Reload all tier patterns from disk with validation.

        This ensures each tier sees the latest patterns from previous tiers
        when running in cascade (T1 → T2 → T3). Without this, T2 never
        sees T1's freshly detected patterns because patterns are loaded
        only at __init__ time.

        Returns:
            Dict of tier name → TierReloadResult value
        """
        results = {}
        for tier_name, tier in [("t1", self.t1), ("t2", self.t2), ("t3", self.t3)]:
            try:
                before_count = len(tier.get_active_patterns() or []) if hasattr(tier, 'get_active_patterns') else 0
                tier._load_patterns()
                after_count = len(tier.get_active_patterns() or []) if hasattr(tier, 'get_active_patterns') else 0
                if after_count > 0:
                    results[tier_name] = TierReloadResult.SUCCESS
                elif before_count > 0:
                    results[tier_name] = TierReloadResult.PARTIAL
                    logger.warning("US-312: %s reload lost patterns (%d → %d)", tier_name, before_count, after_count)
                else:
                    results[tier_name] = TierReloadResult.SUCCESS  # No patterns is valid
            except Exception as e:
                results[tier_name] = TierReloadResult.FAILED
                logger.warning("US-312: %s pattern reload FAILED: %s", tier_name, e)
        return results

    def run_all(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        graph=None,
    ) -> Dict[str, List[DetectedPattern]]:
        """Run all tiers and return results grouped by tier.

        US-207/US-312: Each tier reloads patterns from disk before running so it
        sees the latest patterns from previous tiers in the cascade. Reload results
        are validated to detect pattern loss or reload failures.

        Args:
            conversations: From self-model graph
            readiness_history: From self-model graph
            graph: US-324 — Optional SelfModelGraph for anomaly context extraction

        Returns:
            Dict with keys "t1", "t2", and "t3", each containing detected patterns
        """
        t1_patterns = self.run_t1(conversations, readiness_history)

        # US-324: Extract recent Life_Event nodes as anomaly context for T2
        anomaly_context = None
        if graph is not None:
            try:
                from src.aura.core.self_model import NodeType
                from datetime import timedelta
                life_events = graph.get_nodes_by_type(NodeType.LIFE_EVENT)
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                anomaly_context = []
                for node in life_events:
                    if node.properties.get("source") == "anomaly_detector":
                        try:
                            node_ts = datetime.fromisoformat(
                                node.created_at.replace("Z", "+00:00")
                            )
                            if node_ts >= cutoff:
                                anomaly_context.append({
                                    "id": node.id,
                                    "severity": node.properties.get("severity", 0),
                                    "direction": node.properties.get("direction", "unknown"),
                                    "readiness_at_time": node.properties.get("readiness_at_time", 0),
                                    "timestamp": node.created_at,
                                })
                        except (ValueError, TypeError):
                            anomaly_context.append({
                                "id": node.id,
                                "severity": node.properties.get("severity", 0),
                                "direction": node.properties.get("direction", "unknown"),
                                "timestamp": node.created_at,
                            })
                if anomaly_context:
                    logger.info("US-324: Passing %d anomaly events to T2", len(anomaly_context))
            except Exception as e:
                logger.warning("US-324: Failed to extract anomaly context from graph: %s", e)
                anomaly_context = None

        # Reload with validation
        reload1 = self._reload_tier_patterns()
        if reload1.get("t1") == TierReloadResult.FAILED:
            logger.warning("US-312: T1 reload FAILED before T2 run — T2 uses local data only")
        t2_patterns = self.run_t2(conversations, readiness_history, anomaly_context=anomaly_context)

        reload2 = self._reload_tier_patterns()
        for tier_name, status in reload2.items():
            if status == TierReloadResult.FAILED:
                logger.warning("US-312: %s reload FAILED before T3 run", tier_name)
        t3_patterns = self.run_t3(conversations, readiness_history)

        # Store last cascade health
        self._last_cascade_health = {
            "reload_1": reload1,
            "reload_2": reload2,
            "t1_count": len(t1_patterns),
            "t2_count": len(t2_patterns),
            "t3_count": len(t3_patterns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "t1": t1_patterns,
            "t2": t2_patterns,
            "t3": t3_patterns,
        }

    def cascade_health(self) -> Dict[str, Any]:
        """US-312: Get the health status of the last cascade run.

        Returns:
            Dict with per-tier reload status, pattern counts, and timestamp
        """
        if hasattr(self, '_last_cascade_health'):
            return self._last_cascade_health
        return {
            "reload_1": {},
            "reload_2": {},
            "t1_count": 0,
            "t2_count": 0,
            "t3_count": 0,
            "timestamp": None,
        }

    # --- Pattern Queries ---

    def get_all_active_patterns(self) -> List[DetectedPattern]:
        """Get all active patterns across all tiers."""
        patterns = []
        patterns.extend(self.t1.get_active_patterns())
        patterns.extend(self.t2.get_active_patterns())
        patterns.extend(self.t3.get_active_patterns())
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def get_promotable_patterns(self) -> List[DetectedPattern]:
        """Get all patterns ready for rule promotion across all tiers."""
        patterns = []
        patterns.extend(self.t1.get_promotable_patterns())
        patterns.extend(self.t2.get_promotable_patterns())
        patterns.extend(self.t3.get_promotable_patterns())
        return patterns

    def get_cross_engine_patterns(self) -> List[DetectedPattern]:
        """Get only cross-engine (T2+T3) patterns — the novel ones."""
        return [
            p for p in self.get_all_active_patterns()
            if p.domain == PatternDomain.CROSS_ENGINE
        ]

    def get_narrative_arcs(self) -> List[Dict[str, Any]]:
        """Get T3 narrative arc summaries for CLI display."""
        return self.t3.get_arcs_summary()

    def get_significant_correlations(self) -> List[DetectedPattern]:
        """Get statistically significant T2 correlations."""
        return self.t2.get_significant_patterns()

    # --- Status & Reporting ---

    def get_status(self) -> Dict[str, Any]:
        """Get pattern engine status for CLI display."""
        t1_patterns = self.t1.get_active_patterns()
        t2_patterns = self.t2.get_active_patterns()
        t3_patterns = self.t3.get_active_patterns()
        promotable = self.get_promotable_patterns()

        return {
            "t1_patterns": len(t1_patterns),
            "t2_patterns": len(t2_patterns),
            "t3_patterns": len(t3_patterns),
            "promotable": len(promotable),
            "t1_last_run": self._run_log.get("t1_last_run", "never"),
            "t2_last_run": self._run_log.get("t2_last_run", "never"),
            "t3_last_run": self._run_log.get("t3_last_run", "never"),
            "cloud_synthesis": self.cloud.get_status(),
            "top_patterns": [
                {
                    "description": p.description,
                    "tier": p.tier.value,
                    "confidence": p.confidence,
                    "observations": p.observation_count,
                }
                for p in sorted(
                    t1_patterns + t2_patterns + t3_patterns,
                    key=lambda p: p.confidence,
                    reverse=True,
                )[:5]
            ],
        }

    def format_patterns_report(self) -> str:
        """Format a human-readable patterns report for CLI display."""
        status = self.get_status()
        lines = []

        lines.append("═══ Pattern Engine Status ═══")
        lines.append(
            f"T1 (daily):   {status['t1_patterns']} patterns | "
            f"Last run: {status['t1_last_run']}"
        )
        lines.append(
            f"T2 (weekly):  {status['t2_patterns']} patterns | "
            f"Last run: {status['t2_last_run']}"
        )
        lines.append(
            f"T3 (monthly): {status['t3_patterns']} patterns | "
            f"Last run: {status['t3_last_run']}"
        )
        lines.append(f"Promotable:   {status['promotable']} patterns")

        # Cloud synthesis status
        cloud = status.get("cloud_synthesis", {})
        cloud_label = (
            f"{cloud.get('provider', 'none')}/{cloud.get('model', 'none')} "
            f"({cloud.get('daily_calls_remaining', 0)} calls left)"
            if cloud.get("configured")
            else "not configured (local fallback active)"
        )
        lines.append(f"Cloud:        {cloud_label}")

        if status["top_patterns"]:
            lines.append("")
            lines.append("─── Top Patterns ───")
            for i, p in enumerate(status["top_patterns"], 1):
                tier_label = {"t1_daily": "T1", "t2_weekly": "T2", "t3_monthly": "T3"}.get(
                    p["tier"], p["tier"]
                )
                desc = p["description"]
                lines.append(
                    f"  {i}. [{tier_label}] {desc[:80]}..."
                    if len(desc) > 80
                    else f"  {i}. [{tier_label}] {desc}"
                )
                lines.append(
                    f"     Confidence: {p['confidence']:.0%} | "
                    f"Observations: {p['observations']}"
                )

        significant = self.get_significant_correlations()
        if significant:
            lines.append("")
            lines.append("─── Significant Correlations ───")
            for p in significant:
                lines.append(
                    f"  • {p.description[:80]}..."
                    if len(p.description) > 80
                    else f"  • {p.description}"
                )
                lines.append(
                    f"    r={p.correlation_strength:.2f}, "
                    f"p={p.p_value:.3f}, "
                    f"n={p.sample_size}"
                )

        # Narrative arcs (T3)
        arcs = self.get_narrative_arcs()
        if arcs:
            lines.append("")
            lines.append("─── Narrative Arcs (T3) ───")
            for arc in arcs:
                phase_icon = {
                    "building": "📈", "peak": "🔺", "resolving": "📉",
                    "resolved": "✅", "stable": "➡️",
                }.get(arc.get("phase", ""), "❓")
                lines.append(
                    f"  {phase_icon} {arc['arc_type']}: {arc['description']}"
                )
                if arc.get("insight"):
                    lines.append(f"     → {arc['insight'][:100]}")

        return "\n".join(lines)
