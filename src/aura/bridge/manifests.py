"""US-344: Agent Manifests — Capability Discovery and State Advertisement.

Inspired by A2A Agent Cards and ACP capability manifests. Each system
publishes a structured manifest declaring identity, version, operational
state, capabilities, and supported signal fields.  Manifests enable
dynamic field discovery so adding new ReadinessSignal fields no longer
requires code changes on Buddy's side.

File contract:  .aura/bridge/agent_manifests.json
Format:         { "aura": { ... }, "buddy": { ... } }
Locking:        Same atomic-write + fcntl pattern as all bridge files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Staleness threshold: if heartbeat is older than this, state is 'stale'
HEARTBEAT_STALE_SECONDS = 3600  # 1 hour


@dataclass
class AgentManifest:
    """Capability card for one agent (Aura or Buddy)."""

    agent_name: str
    version: str = "1.0.0"
    schema_version: int = 1
    operational_state: str = "active"  # active | degraded | maintenance | offline
    capabilities: List[str] = field(default_factory=list)
    supported_signal_fields: List[str] = field(default_factory=list)
    last_heartbeat: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    _VALID_STATES = {"active", "degraded", "maintenance", "offline", "stale"}

    def __post_init__(self) -> None:
        if not self.last_heartbeat:
            self.last_heartbeat = datetime.now(timezone.utc).isoformat()
        if self.operational_state not in self._VALID_STATES:
            logger.warning(
                "US-344: AgentManifest %s has unknown state '%s' — defaulting to 'active'",
                self.agent_name,
                self.operational_state,
            )
            self.operational_state = "active"

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        """Return True if last_heartbeat is older than HEARTBEAT_STALE_SECONDS."""
        now = now or datetime.now(timezone.utc)
        if not self.last_heartbeat:
            return True
        try:
            hb = datetime.fromisoformat(
                self.last_heartbeat.replace("Z", "+00:00")
            )
            return (now - hb).total_seconds() > HEARTBEAT_STALE_SECONDS
        except (ValueError, AttributeError):
            logger.warning(
                "US-344: AgentManifest %s has malformed heartbeat '%s' — treating as stale",
                self.agent_name,
                self.last_heartbeat,
            )
            return True

    def effective_state(self, now: Optional[datetime] = None) -> str:
        """Operational state, auto-degraded to 'stale' if heartbeat expired."""
        if self.is_stale(now):
            return "stale"
        return self.operational_state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "schema_version": self.schema_version,
            "operational_state": self.operational_state,
            "capabilities": list(self.capabilities),
            "supported_signal_fields": list(self.supported_signal_fields),
            "last_heartbeat": self.last_heartbeat,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentManifest":
        return cls(
            agent_name=data.get("agent_name", "unknown"),
            version=data.get("version", "1.0.0"),
            schema_version=data.get("schema_version", 1),
            operational_state=data.get("operational_state", "active"),
            capabilities=data.get("capabilities", []),
            supported_signal_fields=data.get("supported_signal_fields", []),
            last_heartbeat=data.get("last_heartbeat", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Aura manifest builder — derives capabilities from active modules
# ---------------------------------------------------------------------------

# Capability names mapped to the attribute on ReadinessComputer
_CAPABILITY_MODULE_MAP: Dict[str, str] = {
    "readiness_scoring": "_config",             # Always present
    "bias_detection": "_bias_detector",
    "tilt_detection": "_tilt_detector",
    "decision_cadence": "_cadence_analyzer",
    "anomaly_detection": "_anomaly_detector",
    "trend_analysis": "_trend_analyzer",
    "emotional_regulation": "_regulation_scorer",
    "changepoint_detection": "_changepoint_detector",
    "reliability_scoring": "_reliability_analyzer",
    "graph_topology": "_topology_analyzer",
    "decision_quality": "_dq_scorer",
    "metacognitive_monitoring": "_metacognition_scorer",
    "cognitive_flexibility": "_flexibility_scorer",
    "journal_reflection": "_journal_scorer",
    "override_prediction": "_override_predictor_instance",
    "ml_readiness_v2": "_model",
}

# Fields that Aura can produce in ReadinessSignal
_AURA_SIGNAL_FIELDS: List[str] = [
    "readiness_score",
    "cognitive_load",
    "active_stressors",
    "override_loss_rate_7d",
    "emotional_state",
    "confidence_trend",
    "components",
    "timestamp",
    "conversation_count_7d",
    "confidence_acceleration",
    "fatigue_score",
    "model_version",
    "circadian_multiplier",
    "raw_score",
    "smoothed_score",
    "tilt_score",
    "decision_variability",
    "anomaly_detected",
    "anomaly_severity",
    "bias_scores",
    "override_loss_risk",
    "trend_direction",
    "decision_quality_score",
    "recovery_score",
    "regime_shift_detected",
    "regime_shift_prob",
    "reliability_score",
    # Phase 18 additions
    "calibration_score",
    "low_calibration",
    "signal_weight_recommendation",
]


def build_aura_manifest(
    readiness_computer: Any,
    version: str = "18.0.0",
) -> AgentManifest:
    """Derive an AgentManifest from the current ReadinessComputer state.

    Inspects which optional modules are loaded (non-None) to build the
    capabilities list dynamically.
    """
    capabilities: List[str] = []
    for cap_name, attr_name in _CAPABILITY_MODULE_MAP.items():
        if hasattr(readiness_computer, attr_name):
            val = getattr(readiness_computer, attr_name, None)
            if val is not None:
                capabilities.append(cap_name)

    return AgentManifest(
        agent_name="aura",
        version=version,
        schema_version=1,
        operational_state="active",
        capabilities=sorted(capabilities),
        supported_signal_fields=list(_AURA_SIGNAL_FIELDS),
        metadata={
            "phase": 18,
            "test_count": 633,  # Updated as tests grow
        },
    )


# ---------------------------------------------------------------------------
# Manifest I/O — read/write to .aura/bridge/agent_manifests.json
# ---------------------------------------------------------------------------

_MANIFESTS_FILENAME = "agent_manifests.json"


def write_manifest(
    bridge_dir: Path,
    manifest: AgentManifest,
) -> None:
    """Write (or merge) an agent manifest into agent_manifests.json.

    Uses FeedbackBridge._locked_write for atomic concurrent-safe I/O.
    Preserves other agents' manifests already in the file.
    """
    from src.aura.bridge.signals import FeedbackBridge

    path = bridge_dir / _MANIFESTS_FILENAME
    # Read existing manifests (if any)
    existing: Dict[str, Any] = {}
    raw = FeedbackBridge._locked_read(path)
    if raw:
        try:
            existing = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("US-344: Corrupt agent_manifests.json — overwriting")

    # Merge this agent's manifest
    existing[manifest.agent_name] = manifest.to_dict()
    FeedbackBridge._locked_write(path, json.dumps(existing, indent=2, default=str))


def read_manifests(bridge_dir: Path) -> Dict[str, AgentManifest]:
    """Read all agent manifests from agent_manifests.json.

    Returns dict keyed by agent_name. Missing or corrupt file returns empty dict.
    """
    from src.aura.bridge.signals import FeedbackBridge

    path = bridge_dir / _MANIFESTS_FILENAME
    raw = FeedbackBridge._locked_read(path)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("US-344: Cannot parse agent_manifests.json — returning empty")
        return {}

    result: Dict[str, AgentManifest] = {}
    for name, mdata in data.items():
        try:
            result[name] = AgentManifest.from_dict(mdata)
        except Exception as e:
            logger.warning("US-344: Skipping malformed manifest for '%s': %s", name, e)
    return result


def read_manifest(bridge_dir: Path, agent_name: str) -> Optional[AgentManifest]:
    """Read a single agent's manifest. Returns None if not found."""
    manifests = read_manifests(bridge_dir)
    return manifests.get(agent_name)


def check_schema_compatibility(
    reader_version: int,
    writer_version: int,
    agent_name: str = "unknown",
) -> bool:
    """Check schema version compatibility. Returns True if compatible.

    Logs warning if reader is behind writer — suggests upgrade.
    """
    if reader_version < writer_version:
        logger.warning(
            "US-344: Schema mismatch for '%s': reader v%d < writer v%d "
            "— some signal fields may be unknown. Consider upgrading.",
            agent_name,
            reader_version,
            writer_version,
        )
        return False
    return True
