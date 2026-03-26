"""Bridge signal contracts between Aura and Buddy.

Defines the JSON schemas for bidirectional data flow:
- ReadinessSignal: Human → Domain (already defined in aura.core.readiness)
- OutcomeSignal: Domain → Human (Buddy writes, Aura reads)
- OverrideEvent: Bidirectional (logged by both systems)

All signals are exchanged via JSON files in .aura/bridge/
Both systems can run independently — signals are read when available,
gracefully ignored when not.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

BRIDGE_DIR = Path(".aura/bridge")


@dataclass
class OutcomeSignal:
    """Domain → Human: Buddy's daily trading summary for Aura's pattern engine.

    Written by Buddy after each trade cycle.
    Read by Aura's Tier 2 cross-domain pattern engine.

    US-297: Extended with optional human context fields that Aura populates
    at trade time. Enables per-trade emotional→PnL correlation in T2.
    """

    pnl_today: float = 0.0
    win_rate_7d: float = 0.0
    override_events: List[Dict[str, Any]] = field(default_factory=list)
    regime: str = "NORMAL"
    streak: str = "neutral"  # "winning", "losing", "neutral"
    trades_today: int = 0
    open_positions: int = 0
    max_drawdown_today: float = 0.0
    timestamp: str = ""
    # US-297: Human context enrichment (filled by Aura at trade time)
    emotional_state: Optional[str] = None       # Aura's detected emotional state
    readiness_at_trade: Optional[float] = None  # Readiness score when trade was taken
    cognitive_load: Optional[float] = None      # Cognitive load estimate (0-1)
    active_biases: Optional[Dict[str, float]] = field(default=None)  # Detected biases

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "pnl_today": round(self.pnl_today, 2),
            "win_rate_7d": round(self.win_rate_7d, 4),
            "override_events": self.override_events,
            "regime": self.regime,
            "streak": self.streak,
            "trades_today": self.trades_today,
            "open_positions": self.open_positions,
            "max_drawdown_today": round(self.max_drawdown_today, 2),
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }
        # US-297: Include enrichment fields only when populated
        if self.emotional_state is not None:
            result["emotional_state"] = self.emotional_state
        if self.readiness_at_trade is not None:
            result["readiness_at_trade"] = round(self.readiness_at_trade, 2)
        if self.cognitive_load is not None:
            result["cognitive_load"] = round(self.cognitive_load, 4)
        if self.active_biases is not None:
            result["active_biases"] = {
                k: round(v, 4) for k, v in self.active_biases.items()
            }
        return result


@dataclass
class OverrideEvent:
    """Bidirectional: logged when trader overrides Buddy's signal.

    Both systems learn from this:
    - Buddy logs the market context (pair, direction, outcome)
    - Aura logs the human context (emotional state, conversation topics)
    """

    timestamp: str
    pair: str
    override_type: str          # "took_rejected", "skipped_recommended", "closed_early", "modified_sl_tp"
    buddy_recommendation: str   # What Buddy recommended
    trader_action: str          # What the trader actually did
    outcome: Optional[str] = None       # "win", "loss", or None if still open
    pnl_pips: float = 0.0
    # Human context (filled by Aura)
    emotional_state: str = ""
    cognitive_load: str = ""
    conversation_context: str = ""  # Summary of recent conversation
    # Market context (filled by Buddy)
    regime: str = ""
    confidence_at_time: float = 0.0
    weighted_vote_at_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pair": self.pair,
            "override_type": self.override_type,
            "buddy_recommendation": self.buddy_recommendation,
            "trader_action": self.trader_action,
            "outcome": self.outcome,
            "pnl_pips": self.pnl_pips,
            "emotional_state": self.emotional_state,
            "cognitive_load": self.cognitive_load,
            "conversation_context": self.conversation_context,
            "regime": self.regime,
            "confidence_at_time": self.confidence_at_time,
            "weighted_vote_at_time": self.weighted_vote_at_time,
        }


class BridgeReadError(str, Enum):
    """US-305: Typed read error for distinguishing failure modes."""
    NOT_FOUND = "not_found"
    CORRUPTED = "corrupted"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class BridgeHealthStatus:
    """US-305: Health status for all bridge signal files."""
    readiness: str = "unknown"    # "healthy", "corrupted", "missing", "stale"
    outcome: str = "unknown"
    overrides: str = "unknown"
    rules: str = "unknown"
    details: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "readiness": self.readiness,
            "outcome": self.outcome,
            "overrides": self.overrides,
            "rules": self.rules,
            "details": self.details,
        }


class FeedbackBridge:
    """Manages bidirectional signal flow between Aura and Buddy.

    File-based bridge — both systems read/write JSON files in .aura/bridge/.
    This keeps systems decoupled while enabling data sharing.
    """

    def __init__(self, bridge_dir: Optional[Path] = None):
        self.bridge_dir = bridge_dir or BRIDGE_DIR
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self._outcome_path = self.bridge_dir / "outcome_signal.json"
        self._override_log_path = self.bridge_dir / "override_events.jsonl"
        self._readiness_path = self.bridge_dir / "readiness_signal.json"
        self._ensure_bridge_files()

    def _ensure_bridge_files(self) -> None:
        """Seed absent bridge files with safe defaults so readers never get NOT_FOUND.

        H-01 fix: outcome_signal.json, active_rules.json, and override_events.jsonl
        were never auto-created, leaving FeedbackBridge.read_outcome_signal() returning
        None on every call and blocking T2 correlation and rules engine startup.
        """
        defaults: Dict[Path, str] = {
            self._outcome_path: (
                '{"pnl_today": 0.0, "win_rate_7d": 0.0, "override_events": [], '
                '"regime": "NORMAL", "streak": "neutral", "trades_today": 0, '
                '"open_positions": 0, "max_drawdown_today": 0.0, "timestamp": ""}'
            ),
            self.bridge_dir / "active_rules.json": "[]",
        }
        for path, default in defaults.items():
            if not path.exists():
                self._locked_write(path, default)
                logger.info("Bridge: seeded missing file %s with defaults", path.name)
        # override_events.jsonl — create empty file (JSONL needs no default content)
        if not self._override_log_path.exists():
            self._override_log_path.touch()
            logger.info("Bridge: created empty override_events.jsonl")

    # --- US-202: File-locking helpers for concurrent access safety ---

    @staticmethod
    def _locked_write(path: Path, data: str) -> None:
        """Atomic write with exclusive file lock (fcntl.LOCK_EX).

        Uses a .lock sidecar file to serialize concurrent writers, then
        writes to a temp file and renames atomically so readers never see
        partial content.

        Fix C-01: Added fcntl.LOCK_EX acquisition before writing to prevent
        concurrent writer race conditions on the bridge's critical shared path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(path) + ".lock")
        fd = None
        tmp_path = None
        lock_fd = None
        try:
            # Acquire exclusive lock via sidecar to serialize concurrent writers
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            # Write to temp file then atomically rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".bridge_"
            )
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None  # mark as closed
            # Atomic rename (POSIX guarantees atomicity on same filesystem)
            os.rename(tmp_path, str(path))
            tmp_path = None  # mark as renamed
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    lock_fd.close()
                except OSError:
                    pass

    @staticmethod
    def _locked_read(path: Path) -> Optional[str]:
        """Read with shared file lock (fcntl.LOCK_SH)."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return f.read()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.warning(f"Bridge: locked read failed for {path.name}: {e}")
            return None

    @staticmethod
    def _locked_append(path: Path, line: str) -> None:
        """Append with exclusive file lock (fcntl.LOCK_EX)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.error(f"Bridge: locked append failed for {path.name}: {e}")

    # US-305: Maximum backup files to keep per signal
    MAX_BACKUPS = 3
    # US-305: Stale threshold in seconds (1 hour)
    STALE_THRESHOLD_SECONDS = 3600

    def _backup_file(self, path: Path) -> None:
        """US-305: Rotate backup files (keep last MAX_BACKUPS)."""
        try:
            if not path.exists():
                return
            # Rotate: .bak.3 → delete, .bak.2 → .bak.3, .bak.1 → .bak.2, current → .bak.1
            for i in range(self.MAX_BACKUPS, 1, -1):
                old = Path(f"{path}.bak.{i - 1}")
                new = Path(f"{path}.bak.{i}")
                if old.exists():
                    os.rename(str(old), str(new))
            # Current → .bak.1
            bak1 = Path(f"{path}.bak.1")
            import shutil
            shutil.copy2(str(path), str(bak1))
        except Exception as e:
            logger.warning("US-305: Backup rotation failed for %s: %s", path.name, e)

    def _recover_from_backup(self, path: Path) -> Optional[str]:
        """US-305: Attempt recovery from most recent valid backup."""
        for i in range(1, self.MAX_BACKUPS + 1):
            bak = Path(f"{path}.bak.{i}")
            if bak.exists():
                try:
                    content = bak.read_text()
                    json.loads(content)  # Validate it's valid JSON
                    logger.warning("US-305: Recovered %s from backup .bak.%d", path.name, i)
                    # Restore to main file
                    self._locked_write(path, content)
                    return content
                except Exception:
                    continue
        return None

    def _typed_read(self, path: Path) -> Tuple[Optional[str], Optional[BridgeReadError]]:
        """US-305: Read with typed error reporting."""
        if not path.exists():
            return None, BridgeReadError.NOT_FOUND
        try:
            raw = self._locked_read(path)
            if raw is None:
                return None, BridgeReadError.PERMISSION_DENIED
            # Validate JSON
            json.loads(raw)
            return raw, None
        except json.JSONDecodeError:
            logger.warning("US-305: Corrupted JSON in %s — attempting recovery", path.name)
            recovered = self._recover_from_backup(path)
            if recovered:
                return recovered, None
            return None, BridgeReadError.CORRUPTED
        except PermissionError:
            return None, BridgeReadError.PERMISSION_DENIED
        except (OSError, IOError):
            return None, BridgeReadError.PERMISSION_DENIED

    def _check_stale(self, path: Path) -> bool:
        """US-305: Check if a file hasn't been updated within threshold.

        Fails safe to True (stale) on any error — better to treat a signal
        as stale than to pass corrupted/inaccessible data to Buddy as fresh.
        """
        try:
            if not path.exists():
                return True  # Non-existent file is stale by definition
            mtime = path.stat().st_mtime
            age = datetime.now(timezone.utc).timestamp() - mtime
            return age > self.STALE_THRESHOLD_SECONDS
        except Exception as e:
            logger.warning("_check_stale error for %s, assuming stale: %s", path.name, e)
            return True

    def bridge_health(self) -> BridgeHealthStatus:
        """US-305: Check health of all bridge signal files."""
        status = BridgeHealthStatus()
        files = {
            "readiness": self._readiness_path,
            "outcome": self._outcome_path,
            "overrides": self._override_log_path,
            "rules": self.bridge_dir / "active_rules.json",
        }
        for name, path in files.items():
            if not path.exists():
                setattr(status, name, "missing")
                status.details[name] = "File does not exist"
            else:
                try:
                    raw = path.read_text()
                    if name == "overrides":
                        # JSONL — just check it's non-empty
                        if raw.strip():
                            if self._check_stale(path):
                                setattr(status, name, "stale")
                                status.details[name] = "File not updated recently"
                            else:
                                setattr(status, name, "healthy")
                        else:
                            setattr(status, name, "missing")
                            status.details[name] = "File is empty"
                    else:
                        json.loads(raw)
                        if self._check_stale(path):
                            setattr(status, name, "stale")
                            status.details[name] = "File not updated recently"
                        else:
                            setattr(status, name, "healthy")
                except json.JSONDecodeError:
                    setattr(status, name, "corrupted")
                    status.details[name] = "Invalid JSON content"
                except PermissionError:
                    setattr(status, name, "corrupted")
                    status.details[name] = "Permission denied"
                except Exception as e:
                    setattr(status, name, "corrupted")
                    status.details[name] = str(e)
        return status

    # --- Outcome Signal (Buddy → Aura) ---

    def write_outcome(self, signal: OutcomeSignal) -> None:
        """Buddy writes its trading summary for Aura to read."""
        try:
            # US-305: Backup before overwrite
            self._backup_file(self._outcome_path)
            self._locked_write(
                self._outcome_path,
                json.dumps(signal.to_dict(), indent=2, default=str),
            )
            logger.debug(f"Bridge: wrote outcome signal (PnL: {signal.pnl_today:+.2f})")
        except Exception as e:
            logger.error(f"Bridge: failed to write outcome signal: {e}")

    def read_outcome(self) -> Optional[OutcomeSignal]:
        """Aura reads Buddy's latest trading summary."""
        raw = self._locked_read(self._outcome_path)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return OutcomeSignal(**{
                k: v for k, v in data.items()
                if k in OutcomeSignal.__dataclass_fields__
            })
        except Exception as e:
            logger.warning(f"Bridge: failed to read outcome signal: {e}")
            return None

    # --- Override Events (Bidirectional) ---

    def log_override(self, event: OverrideEvent) -> None:
        """Log an override event (appendable JSONL format)."""
        try:
            self._locked_append(
                self._override_log_path,
                json.dumps(event.to_dict(), default=str) + "\n",
            )
            logger.info(
                f"Bridge: override logged — {event.pair} {event.override_type} "
                f"(emotional: {event.emotional_state or 'unknown'})"
            )
        except Exception as e:
            logger.error(f"Bridge: failed to log override: {e}")

    def get_recent_overrides(self, limit: int = 20) -> List[OverrideEvent]:
        """Read recent override events. US-241: tracks malformed lines."""
        raw = self._locked_read(self._override_log_path)
        if raw is None:
            return []
        try:
            events = []
            total_lines = 0
            skipped_lines = 0
            first_bad_line = None
            for line_num, line in enumerate(raw.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    data = json.loads(line)
                    events.append(OverrideEvent(**{
                        k: v for k, v in data.items()
                        if k in OverrideEvent.__dataclass_fields__
                    }))
                except Exception:
                    skipped_lines += 1
                    if first_bad_line is None:
                        first_bad_line = line_num
                    continue

            # US-241: Log malformed line statistics
            if skipped_lines > 0:
                pct_bad = (skipped_lines / total_lines * 100) if total_lines else 0
                if pct_bad > 20:
                    logger.error(
                        "US-241: %d/%d (%.0f%%) override lines malformed — possible file corruption. "
                        "First bad line: %d", skipped_lines, total_lines, pct_bad, first_bad_line)
                else:
                    logger.warning(
                        "US-241: Skipped %d/%d malformed override lines (first bad: line %d)",
                        skipped_lines, total_lines, first_bad_line)

            return events[-limit:]
        except Exception as e:
            logger.warning(f"Bridge: failed to read overrides: {e}")
            return []

    # --- Readiness Signal (Aura → Buddy) ---

    def read_readiness(self) -> Optional[Dict[str, Any]]:
        """Read Aura's readiness signal (convenience method)."""
        raw = self._locked_read(self._readiness_path)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Bridge: failed to read readiness signal: {e}")
            return None

    # --- Bridge Statistics ---

    def repair_corrupted(self) -> Dict[str, str]:
        """US-311: Attempt to repair corrupted bridge files from backups.

        Returns:
            Dict of filename → result ("repaired", "no_backup", "healthy", "failed")
        """
        results = {}
        health = self.bridge_health()
        for fname, status_attr in [
            ("readiness_signal.json", "readiness"),
            ("outcome_signal.json", "outcome"),
            ("override_events.jsonl", "overrides"),
            ("active_rules.json", "rules"),
        ]:
            status = getattr(health, status_attr)
            if status == "corrupted":
                fpath = self.bridge_dir / fname
                recovered = self._recover_from_backup(fpath)
                results[fname] = "repaired" if recovered else "no_backup"
            elif status == "healthy":
                results[fname] = "healthy"
            else:
                results[fname] = status  # "missing" or "stale"
        return results

    def enrich_outcome_signal(
        self, outcome: Dict[str, Any], human_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """US-323: Enrich an outcome signal with human context from Aura.

        Adds current emotional state, cognitive load, bias counts, override
        risk, decision quality, and readiness-at-trade-time to the outcome
        signal. The enriched signal is written back to outcome_signal.json
        via atomic write, enabling T2 cross-domain correlation between
        human state and trade performance.

        Args:
            outcome: Raw outcome signal dict (from Buddy)
            human_context: Dict from ReadinessComputer.get_last_state_snapshot()

        Returns:
            Enriched outcome dict with human context fields added
        """
        enriched = dict(outcome)  # Preserve original fields

        if human_context:
            enriched["emotional_state_score"] = human_context.get("emotional_state", "unknown")
            enriched["cognitive_load_score"] = human_context.get("cognitive_load_label", "unknown")
            enriched["bias_count"] = len(human_context.get("bias_scores", {}))
            enriched["override_loss_risk"] = round(
                human_context.get("override_loss_risk", 0.0), 4
            )
            enriched["decision_quality_score"] = round(
                human_context.get("decision_quality_score", 0.0), 2
            )
            enriched["readiness_at_trade_time"] = round(
                human_context.get("readiness_score", 0.0), 1
            )
            enriched["graph_context_score"] = round(
                human_context.get("graph_context_score", 0.5), 3
            )
            enriched["tilt_score"] = round(
                human_context.get("tilt_score", 0.0), 3
            )
        else:
            # No cached state — add empty context (backward compatible)
            enriched["emotional_state_score"] = "unknown"
            enriched["cognitive_load_score"] = "unknown"
            enriched["bias_count"] = 0
            enriched["override_loss_risk"] = 0.0
            enriched["decision_quality_score"] = 0.0
            enriched["readiness_at_trade_time"] = 0.0

        # Write enriched signal back to bridge via atomic write
        try:
            self._backup_file(self._outcome_path)
            self._locked_write(
                self._outcome_path,
                json.dumps(enriched, indent=2, default=str),
            )
            logger.info(
                "US-323: Enriched outcome signal — readiness=%.1f, biases=%d, risk=%.2f",
                enriched.get("readiness_at_trade_time", 0),
                enriched.get("bias_count", 0),
                enriched.get("override_loss_risk", 0),
            )
        except Exception as e:
            logger.error("US-323: Failed to write enriched outcome signal: %s", e)

        return enriched

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get the current state of all bridge signals."""
        readiness = self.read_readiness()
        outcome = self.read_outcome()
        overrides = self.get_recent_overrides(limit=5)

        return {
            "readiness_signal": {
                "available": readiness is not None,
                "score": readiness.get("readiness_score") if readiness else None,
                "timestamp": readiness.get("timestamp") if readiness else None,
            },
            "outcome_signal": {
                "available": outcome is not None,
                "pnl_today": outcome.pnl_today if outcome else None,
                "timestamp": outcome.timestamp if outcome else None,
            },
            "override_events": {
                "total_recent": len(overrides),
                "last_override": overrides[-1].to_dict() if overrides else None,
            },
        }
