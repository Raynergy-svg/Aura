"""Bridge health monitoring — validates bridge files for staleness, schema, and corruption.

US-360: BridgeHealthMonitor checks readiness_signal.json and outcome_signal.json for:
- File existence
- Valid JSON
- Required fields
- Freshness (>30 min = stale)
- Score range validity

Provides check_all() -> BridgeHealthReport and auto_heal() for self-repair.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Required fields for readiness_signal.json
READINESS_REQUIRED_FIELDS = {
    "readiness_score", "cognitive_load", "emotional_state", "timestamp"
}

# Required fields for outcome_signal.json
OUTCOME_REQUIRED_FIELDS = {
    "timestamp"
}

# Staleness threshold (seconds)
STALE_THRESHOLD_SECONDS = 1800  # 30 minutes


@dataclass
class BridgeHealthReport:
    """US-360: Health report for all bridge files."""
    readiness_ok: bool = False
    outcome_ok: bool = False
    readiness_age_seconds: float = float("inf")
    outcome_age_seconds: float = float("inf")
    readiness_score_valid: bool = False
    issues: List[str] = field(default_factory=list)
    overall_health: str = "critical"  # "healthy" | "degraded" | "critical"


class BridgeHealthMonitor:
    """US-360: Monitors bridge files for staleness, schema issues, and corruption."""

    def __init__(self, bridge_dir: Optional[Path] = None):
        """
        Args:
            bridge_dir: Path to the bridge directory containing signal files.
                        Defaults to .aura/bridge/
        """
        self.bridge_dir = Path(bridge_dir) if bridge_dir else Path(".aura/bridge")

    def check_readiness(self) -> Dict:
        """Check readiness_signal.json — existence, schema, freshness."""
        result = {
            "exists": False,
            "valid_json": False,
            "has_required_fields": False,
            "age_seconds": float("inf"),
            "score_valid": False,
            "issues": [],
        }

        path = self.bridge_dir / "readiness_signal.json"

        if not path.exists():
            result["issues"].append("readiness_signal.json missing")
            return result

        result["exists"] = True

        # Check age
        try:
            mtime = path.stat().st_mtime
            result["age_seconds"] = time.time() - mtime
            if result["age_seconds"] > STALE_THRESHOLD_SECONDS:
                result["issues"].append(
                    f"readiness_signal.json stale ({result['age_seconds']:.0f}s old)"
                )
        except OSError as e:
            result["issues"].append(f"Cannot stat readiness_signal.json: {e}")

        # Parse JSON
        try:
            data = json.loads(path.read_text())
            result["valid_json"] = True
        except (json.JSONDecodeError, OSError) as e:
            result["issues"].append(f"readiness_signal.json not valid JSON: {e}")
            return result

        # Check required fields
        missing = READINESS_REQUIRED_FIELDS - set(data.keys())
        if missing:
            result["issues"].append(f"readiness_signal.json missing fields: {missing}")
        else:
            result["has_required_fields"] = True

        # Check score range
        score = data.get("readiness_score")
        if score is not None and 0.0 <= float(score) <= 100.0:
            result["score_valid"] = True
        elif score is not None:
            result["issues"].append(f"readiness_score out of range: {score}")

        return result

    def check_outcome(self) -> Dict:
        """Check outcome_signal.json — existence, schema."""
        result = {
            "exists": False,
            "valid_json": False,
            "has_required_fields": False,
            "age_seconds": float("inf"),
            "issues": [],
        }

        path = self.bridge_dir / "outcome_signal.json"

        if not path.exists():
            result["issues"].append("outcome_signal.json missing")
            return result

        result["exists"] = True

        # Check age
        try:
            mtime = path.stat().st_mtime
            result["age_seconds"] = time.time() - mtime
        except OSError as e:
            result["issues"].append(f"Cannot stat outcome_signal.json: {e}")

        # Parse JSON
        try:
            data = json.loads(path.read_text())
            result["valid_json"] = True
        except (json.JSONDecodeError, OSError) as e:
            result["issues"].append(f"outcome_signal.json not valid JSON: {e}")
            return result

        # Check required fields
        missing = OUTCOME_REQUIRED_FIELDS - set(data.keys())
        if missing:
            result["issues"].append(f"outcome_signal.json missing fields: {missing}")
        else:
            result["has_required_fields"] = True

        return result

    def check_all(self) -> BridgeHealthReport:
        """Run all checks and return a BridgeHealthReport."""
        r_check = self.check_readiness()
        o_check = self.check_outcome()

        all_issues = r_check["issues"] + o_check["issues"]

        readiness_ok = (
            r_check["exists"]
            and r_check["valid_json"]
            and r_check["has_required_fields"]
            and r_check["age_seconds"] <= STALE_THRESHOLD_SECONDS
        )
        outcome_ok = (
            o_check["exists"]
            and o_check["valid_json"]
            and o_check["has_required_fields"]
        )

        # Determine overall health
        if not readiness_ok and r_check["issues"]:
            if not r_check["exists"] or not r_check["valid_json"]:
                overall_health = "critical"
            else:
                overall_health = "degraded"
        elif not outcome_ok:
            overall_health = "degraded"
        elif all_issues:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        # If both missing or corrupt: critical
        if not r_check["exists"] and not o_check["exists"]:
            overall_health = "critical"
        elif not r_check["valid_json"] or (r_check["exists"] and not r_check["valid_json"]):
            overall_health = "critical"

        logger.debug(
            "US-360: Bridge health check — readiness_ok=%s, outcome_ok=%s, health=%s",
            readiness_ok, outcome_ok, overall_health
        )

        return BridgeHealthReport(
            readiness_ok=readiness_ok,
            outcome_ok=outcome_ok,
            readiness_age_seconds=r_check["age_seconds"],
            outcome_age_seconds=o_check["age_seconds"],
            readiness_score_valid=r_check.get("score_valid", False),
            issues=all_issues,
            overall_health=overall_health,
        )

    def auto_heal(self) -> int:
        """Create default files for any missing bridge files.

        Returns:
            Number of files created/healed
        """
        healed = 0
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

        readiness_path = self.bridge_dir / "readiness_signal.json"
        if not readiness_path.exists():
            default_readiness = {
                "readiness_score": 50.0,
                "cognitive_load": "medium",
                "active_stressors": [],
                "override_loss_rate_7d": 0.0,
                "emotional_state": "neutral",
                "confidence_trend": "stable",
                "components": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "conversation_count_7d": 0,
                "fatigue_score": 0.0,
                "model_version": "v1",
            }
            try:
                readiness_path.write_text(json.dumps(default_readiness, indent=2))
                healed += 1
                logger.info("US-360: Auto-healed readiness_signal.json with defaults")
            except OSError as e:
                logger.warning("US-360: Could not create readiness_signal.json: %s", e)

        outcome_path = self.bridge_dir / "outcome_signal.json"
        if not outcome_path.exists():
            default_outcome = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl_today": 0.0,
                "trades_today": 0,
                "success": None,
                "source": "auto_heal",
            }
            try:
                outcome_path.write_text(json.dumps(default_outcome, indent=2))
                healed += 1
                logger.info("US-360: Auto-healed outcome_signal.json with defaults")
            except OSError as e:
                logger.warning("US-360: Could not create outcome_signal.json: %s", e)

        return healed

    def get_health_summary(self) -> str:
        """Return a human-readable health summary string."""
        report = self.check_all()

        health_icon = {"healthy": "OK", "degraded": "WARN", "critical": "CRIT"}.get(
            report.overall_health, "?"
        )

        lines = [f"[{health_icon}] Bridge Health: {report.overall_health.upper()}"]

        r_age = f"{report.readiness_age_seconds:.0f}s" if report.readiness_age_seconds != float("inf") else "N/A"
        o_age = f"{report.outcome_age_seconds:.0f}s" if report.outcome_age_seconds != float("inf") else "N/A"

        r_status = "OK" if report.readiness_ok else "FAIL"
        o_status = "OK" if report.outcome_ok else "FAIL"

        lines.append(f"  readiness_signal.json: {r_status} (age={r_age}, score_valid={report.readiness_score_valid})")
        lines.append(f"  outcome_signal.json:   {o_status} (age={o_age})")

        if report.issues:
            lines.append(f"  Issues ({len(report.issues)}):")
            for issue in report.issues[:5]:  # show max 5 issues
                lines.append(f"    - {issue}")

        return "\n".join(lines)
