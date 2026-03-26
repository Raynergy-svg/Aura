"""Persistence Helper — atomic, locked file writes for Aura.

US-273: Reusable locked-write utility extracted from FeedbackBridge._locked_write.
Used by prediction models (readiness_v2, override_predictor) and any other
component that needs crash-safe file persistence.

Falls back to direct write when fcntl is not available (e.g., Windows).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check fcntl availability (not on Windows)
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def atomic_write(path: Path, data: str) -> None:
    """Write data to file atomically with optional file locking.

    Uses temp-file + rename for crash safety. If fcntl is available,
    also acquires an exclusive lock to prevent concurrent write corruption.

    Args:
        path: Target file path
        data: String content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_FCNTL:
        _locked_atomic_write(path, data)
    else:
        _direct_atomic_write(path, data)


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Write a JSON-serializable object atomically.

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default 2)
    """
    data = json.dumps(obj, indent=indent, default=str)
    atomic_write(path, data)


def _locked_atomic_write(path: Path, data: str) -> None:
    """Atomic write with fcntl exclusive lock.

    Fix H-02: The function name implied locking but did not acquire any fcntl lock.
    Two concurrent writers could race: both create separate temp files, both rename,
    one silently overwrites the other's data. Fixed by acquiring LOCK_EX on a .lock
    sidecar file before writing, released after rename completes.
    """
    lock_path = Path(str(path) + ".lock")
    fd = None
    tmp_path = None
    lock_fd = None
    try:
        # Acquire exclusive lock on sidecar to serialize concurrent writers
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".persist_"
        )
        # Write data
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None

        # Atomic rename
        os.rename(tmp_path, str(path))
        tmp_path = None
    finally:
        # Clean up temp file first (data integrity)
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
        # Release lock last — ensure no writers can proceed while cleanup is in progress
        if lock_fd is not None:
            try:
                lock_fileno = lock_fd.fileno()
                fcntl.flock(lock_fileno, fcntl.LOCK_UN)
            except (OSError, ValueError):
                pass  # fileno() can raise ValueError if fd already closed
            try:
                lock_fd.close()
            except OSError:
                pass


def _direct_atomic_write(path: Path, data: str) -> None:
    """Atomic write without locking (Windows fallback).

    Still uses temp + rename for crash safety, just without fcntl lock.
    """
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".persist_"
        )
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.rename(tmp_path, str(path))
        tmp_path = None
        logger.debug("US-273: Direct atomic write (no fcntl) to %s", path)
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
