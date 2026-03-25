"""US-345: Negotiation Protocol — Proposal/Counter/Approve for Bridge Rules.

Replaces unilateral rule creation with a structured negotiation flow
inspired by ACNBP and A2A task lifecycle.

Flow:
  1. Aura detects pattern → creates NegotiationProposal (status=pending)
  2. Buddy reads pending proposals → Accepts, Counter-proposes, or Rejects
  3. On counter: if concession bounds overlap → auto-converge at midpoint
  4. On TTL expiry with no counter → auto-activate (backwards compatible)

All exchanges logged to .aura/bridge/negotiation_log.jsonl.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_NEGOTIATION_LOG = "negotiation_log.jsonl"


@dataclass
class NegotiationProposal:
    """A proposed bridge rule change from one agent to another."""

    proposal_id: str = ""
    proposer: str = "aura"            # "aura" or "buddy"
    action: str = ""                  # rule_type from BridgeRule (e.g., "raise_min_confidence")
    target_parameter: str = ""
    current_value: float = 0.0
    proposed_value: float = 0.0
    operator: str = "set"
    rationale: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    concession_bounds: Dict[str, float] = field(default_factory=dict)
    # concession_bounds: {"min_acceptable": X, "max_concession": Y}
    ttl_hours: float = 4.0
    created_at: str = ""
    expires_at: str = ""
    status: str = "pending"
    # pending | countered | accepted | rejected | expired | auto_activated

    def __post_init__(self) -> None:
        if not self.proposal_id:
            self.proposal_id = f"prop-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now(timezone.utc)
        if not self.created_at:
            self.created_at = now.isoformat()
        if not self.expires_at:
            self.expires_at = (now + timedelta(hours=self.ttl_hours)).isoformat()

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if not self.expires_at:
            return True
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return now > exp
        except (ValueError, AttributeError):
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "proposal",
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "action": self.action,
            "target_parameter": self.target_parameter,
            "current_value": self.current_value,
            "proposed_value": round(self.proposed_value, 4),
            "operator": self.operator,
            "rationale": self.rationale,
            "evidence_ids": list(self.evidence_ids),
            "confidence": round(self.confidence, 3),
            "concession_bounds": {
                k: round(v, 4) for k, v in self.concession_bounds.items()
            },
            "ttl_hours": self.ttl_hours,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NegotiationProposal":
        return cls(
            proposal_id=data.get("proposal_id", ""),
            proposer=data.get("proposer", "aura"),
            action=data.get("action", ""),
            target_parameter=data.get("target_parameter", ""),
            current_value=data.get("current_value", 0.0),
            proposed_value=data.get("proposed_value", 0.0),
            operator=data.get("operator", "set"),
            rationale=data.get("rationale", ""),
            evidence_ids=data.get("evidence_ids", []),
            confidence=data.get("confidence", 0.5),
            concession_bounds=data.get("concession_bounds", {}),
            ttl_hours=data.get("ttl_hours", 4.0),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            status=data.get("status", "pending"),
        )


@dataclass
class CounterProposal:
    """A counter to an existing proposal."""

    counter_id: str = ""
    proposal_id: str = ""
    responder: str = "buddy"
    counter_value: float = 0.0
    rationale: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    concession_bounds: Dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.counter_id:
            self.counter_id = f"counter-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "counter_proposal",
            "counter_id": self.counter_id,
            "proposal_id": self.proposal_id,
            "responder": self.responder,
            "counter_value": round(self.counter_value, 4),
            "rationale": self.rationale,
            "evidence_ids": list(self.evidence_ids),
            "concession_bounds": {
                k: round(v, 4) for k, v in self.concession_bounds.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterProposal":
        return cls(
            counter_id=data.get("counter_id", ""),
            proposal_id=data.get("proposal_id", ""),
            responder=data.get("responder", "buddy"),
            counter_value=data.get("counter_value", 0.0),
            rationale=data.get("rationale", ""),
            evidence_ids=data.get("evidence_ids", []),
            concession_bounds=data.get("concession_bounds", {}),
            created_at=data.get("created_at", ""),
        )


@dataclass
class NegotiationResolution:
    """The final outcome of a negotiation."""

    proposal_id: str = ""
    resolution_type: str = ""
    # accepted | converged | rejected | auto_activated
    agreed_value: float = 0.0
    binding_rule_id: str = ""  # BridgeRule ID created (if any)
    resolved_at: str = ""

    def __post_init__(self) -> None:
        if not self.resolved_at:
            self.resolved_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "resolution",
            "proposal_id": self.proposal_id,
            "resolution_type": self.resolution_type,
            "agreed_value": round(self.agreed_value, 4),
            "binding_rule_id": self.binding_rule_id,
            "resolved_at": self.resolved_at,
        }


# ---------------------------------------------------------------------------
# Convergence Logic
# ---------------------------------------------------------------------------

def compute_convergence(
    proposal_bounds: Dict[str, float],
    counter_bounds: Dict[str, float],
) -> Optional[float]:
    """Find the overlap midpoint between two concession bound ranges.

    Proposal bounds: proposer will accept [min_acceptable, proposed_value]
    Counter bounds:  responder will accept [counter_value, max_concession]

    If the ranges overlap, return the midpoint of the overlap.
    If they don't overlap, return None (negotiation failed).
    """
    p_min = proposal_bounds.get("min_acceptable")
    p_max = proposal_bounds.get("max_concession")
    c_min = counter_bounds.get("min_acceptable")
    c_max = counter_bounds.get("max_concession")

    if any(v is None for v in [p_min, p_max, c_min, c_max]):
        logger.debug("US-345: Incomplete concession bounds — cannot converge")
        return None

    # The overlap range
    overlap_low = max(p_min, c_min)
    overlap_high = min(p_max, c_max)

    if overlap_low > overlap_high:
        logger.info(
            "US-345: No overlap — proposal bounds [%.3f, %.3f] vs counter [%.3f, %.3f]",
            p_min, p_max, c_min, c_max,
        )
        return None

    midpoint = (overlap_low + overlap_high) / 2.0
    logger.info(
        "US-345: Convergence found at %.4f (overlap [%.3f, %.3f])",
        midpoint, overlap_low, overlap_high,
    )
    return midpoint


# ---------------------------------------------------------------------------
# NegotiationEngine
# ---------------------------------------------------------------------------

class NegotiationEngine:
    """Manages the negotiation lifecycle for bridge rule proposals.

    Proposals and counters are stored in-memory and logged to JSONL.
    """

    def __init__(self, bridge_dir: Path) -> None:
        self.bridge_dir = bridge_dir
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = bridge_dir / _NEGOTIATION_LOG
        self._proposals: Dict[str, NegotiationProposal] = {}
        self._counters: Dict[str, CounterProposal] = {}  # keyed by proposal_id
        self._resolutions: Dict[str, NegotiationResolution] = {}
        # Ensure log file exists
        if not self._log_path.exists():
            self._log_path.touch()

    def propose(self, proposal: NegotiationProposal) -> str:
        """Register a new proposal. Returns proposal_id."""
        self._proposals[proposal.proposal_id] = proposal
        self._log_entry(proposal.to_dict())
        logger.info(
            "US-345: Proposal '%s' from %s — %s %s to %.4f",
            proposal.proposal_id,
            proposal.proposer,
            proposal.operator,
            proposal.target_parameter,
            proposal.proposed_value,
        )
        return proposal.proposal_id

    def counter(self, counter: CounterProposal) -> Optional[str]:
        """Register a counter-proposal. Returns counter_id, or None if proposal not found."""
        proposal = self._proposals.get(counter.proposal_id)
        if not proposal:
            logger.warning(
                "US-345: Counter for unknown proposal '%s'", counter.proposal_id
            )
            return None
        if proposal.status not in ("pending",):
            logger.warning(
                "US-345: Cannot counter proposal '%s' — status is '%s'",
                counter.proposal_id,
                proposal.status,
            )
            return None

        proposal.status = "countered"
        self._counters[counter.proposal_id] = counter
        self._log_entry(counter.to_dict())
        logger.info(
            "US-345: Counter '%s' from %s — value %.4f for proposal '%s'",
            counter.counter_id,
            counter.responder,
            counter.counter_value,
            counter.proposal_id,
        )
        return counter.counter_id

    def resolve(self, proposal_id: str) -> Optional[NegotiationResolution]:
        """Attempt to resolve a proposal. Tries convergence if countered.

        Returns NegotiationResolution or None if proposal not found.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return None

        counter = self._counters.get(proposal_id)

        if proposal.status == "countered" and counter:
            # Try convergence via overlapping bounds
            agreed = compute_convergence(
                proposal.concession_bounds,
                counter.concession_bounds,
            )
            if agreed is not None:
                resolution = NegotiationResolution(
                    proposal_id=proposal_id,
                    resolution_type="converged",
                    agreed_value=agreed,
                )
                proposal.status = "accepted"
            else:
                resolution = NegotiationResolution(
                    proposal_id=proposal_id,
                    resolution_type="rejected",
                    agreed_value=0.0,
                )
                proposal.status = "rejected"
        elif proposal.status == "pending":
            # Direct accept (no counter needed)
            resolution = NegotiationResolution(
                proposal_id=proposal_id,
                resolution_type="accepted",
                agreed_value=proposal.proposed_value,
            )
            proposal.status = "accepted"
        else:
            return None

        self._resolutions[proposal_id] = resolution
        self._log_entry(resolution.to_dict())
        return resolution

    def check_expired(self, now: Optional[datetime] = None) -> List[NegotiationResolution]:
        """Check for expired proposals and auto-activate them.

        Returns list of auto-activated resolutions.
        """
        now = now or datetime.now(timezone.utc)
        auto_activated: List[NegotiationResolution] = []

        for pid, proposal in list(self._proposals.items()):
            if proposal.status != "pending":
                continue
            if not proposal.is_expired(now):
                continue

            # Auto-activate at original proposed value (backwards compatible)
            resolution = NegotiationResolution(
                proposal_id=pid,
                resolution_type="auto_activated",
                agreed_value=proposal.proposed_value,
            )
            proposal.status = "auto_activated"
            self._resolutions[pid] = resolution
            self._log_entry(resolution.to_dict())
            auto_activated.append(resolution)
            logger.info(
                "US-345: Auto-activated proposal '%s' at value %.4f (TTL expired)",
                pid,
                proposal.proposed_value,
            )

        return auto_activated

    def get_pending_proposals(self) -> List[NegotiationProposal]:
        """Get all proposals still awaiting response."""
        return [p for p in self._proposals.values() if p.status == "pending"]

    def get_resolutions(self) -> List[NegotiationResolution]:
        """Get all resolved negotiations."""
        return list(self._resolutions.values())

    def get_log_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent entries from negotiation_log.jsonl."""
        from src.aura.bridge.signals import FeedbackBridge

        raw = FeedbackBridge._locked_read(self._log_path)
        if not raw:
            return []
        entries = []
        for line in raw.strip().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries[-limit:]

    def _log_entry(self, entry: Dict[str, Any]) -> None:
        """Append an entry to negotiation_log.jsonl."""
        from src.aura.bridge.signals import FeedbackBridge

        line = json.dumps(entry, default=str) + "\n"
        FeedbackBridge._locked_append(self._log_path, line)
