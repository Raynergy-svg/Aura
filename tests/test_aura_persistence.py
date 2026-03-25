"""Tests for Aura persistence utilities.

US-273: Atomic, locked file writes for crash-safe persistence.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.persistence import atomic_write, atomic_write_json, _HAS_FCNTL


class TestUS273AtomicWrite:
    """US-273: atomic_write creates files safely."""

    def test_write_creates_file(self, tmp_path):
        """atomic_write creates a new file with correct content."""
        target = tmp_path / "test.txt"
        atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_write_overwrites_existing(self, tmp_path):
        """atomic_write replaces existing file content."""
        target = tmp_path / "test.txt"
        target.write_text("old")
        atomic_write(target, "new")
        assert target.read_text() == "new"

    def test_write_creates_parent_dirs(self, tmp_path):
        """atomic_write creates missing parent directories."""
        target = tmp_path / "sub" / "deep" / "file.txt"
        atomic_write(target, "nested")
        assert target.read_text() == "nested"

    def test_write_handles_unicode(self, tmp_path):
        """atomic_write handles unicode content correctly."""
        target = tmp_path / "unicode.txt"
        content = "émotions: 😊 stress: 高い"
        atomic_write(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_write_empty_string(self, tmp_path):
        """atomic_write handles empty string."""
        target = tmp_path / "empty.txt"
        atomic_write(target, "")
        assert target.read_text() == ""

    def test_no_temp_files_left_on_success(self, tmp_path):
        """No .tmp files remain after successful write."""
        target = tmp_path / "clean.txt"
        atomic_write(target, "data")
        tmp_files = list(tmp_path.glob(".persist_*.tmp"))
        assert len(tmp_files) == 0, f"Leftover temp files: {tmp_files}"


class TestUS273AtomicWriteJson:
    """US-273: atomic_write_json serializes and writes JSON atomically."""

    def test_json_write_dict(self, tmp_path):
        """atomic_write_json writes a dict as JSON."""
        target = tmp_path / "data.json"
        obj = {"score": 75, "stressors": ["deadline", "market"]}
        atomic_write_json(target, obj)
        loaded = json.loads(target.read_text())
        assert loaded == obj

    def test_json_write_list(self, tmp_path):
        """atomic_write_json writes a list as JSON."""
        target = tmp_path / "list.json"
        obj = [1, 2, 3, {"nested": True}]
        atomic_write_json(target, obj)
        loaded = json.loads(target.read_text())
        assert loaded == obj

    def test_json_write_with_custom_indent(self, tmp_path):
        """atomic_write_json respects indent parameter."""
        target = tmp_path / "indented.json"
        atomic_write_json(target, {"key": "val"}, indent=4)
        raw = target.read_text()
        assert "    " in raw  # 4-space indent

    def test_json_write_handles_datetime_via_default_str(self, tmp_path):
        """atomic_write_json uses default=str for non-serializable types."""
        from datetime import datetime, timezone
        target = tmp_path / "dt.json"
        obj = {"saved_at": datetime(2026, 3, 23, tzinfo=timezone.utc)}
        atomic_write_json(target, obj)
        loaded = json.loads(target.read_text())
        assert "2026-03-23" in loaded["saved_at"]

    def test_json_roundtrip_model_data(self, tmp_path):
        """Simulates readiness_v2 model save/load roundtrip."""
        target = tmp_path / "model.json"
        model_data = {
            "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bias": 0.05,
            "feature_means": [50.0, 0.6, 0.3],
            "feature_stds": [10.0, 0.2, 0.1],
            "trained": True,
            "train_samples": 100,
        }
        atomic_write_json(target, model_data)
        loaded = json.loads(target.read_text())
        assert loaded["weights"] == model_data["weights"]
        assert loaded["trained"] is True
        assert loaded["train_samples"] == 100


class TestUS273FallbackBehavior:
    """US-273: Platform fallback when fcntl is unavailable."""

    def test_fcntl_flag_is_boolean(self):
        """_HAS_FCNTL is a boolean flag."""
        assert isinstance(_HAS_FCNTL, bool)

    def test_write_works_without_fcntl(self, tmp_path):
        """atomic_write succeeds even when fcntl path is bypassed."""
        target = tmp_path / "nofcntl.txt"
        with patch("aura.persistence._HAS_FCNTL", False):
            atomic_write(target, "fallback mode")
        assert target.read_text() == "fallback mode"

    def test_write_works_with_fcntl(self, tmp_path):
        """atomic_write succeeds when fcntl path is used."""
        target = tmp_path / "withfcntl.txt"
        with patch("aura.persistence._HAS_FCNTL", True):
            atomic_write(target, "locked mode")
        assert target.read_text() == "locked mode"


class TestUS273CrashSafety:
    """US-273: Temp file cleanup on failure scenarios."""

    def test_failed_write_cleans_temp_file(self, tmp_path):
        """If os.rename fails, temp file is cleaned up."""
        target = tmp_path / "fail.txt"
        with patch("aura.persistence.os.rename", side_effect=OSError("rename failed")):
            with pytest.raises(OSError):
                atomic_write(target, "data")
        # No leftover temp files
        tmp_files = list(tmp_path.glob(".persist_*.tmp"))
        assert len(tmp_files) == 0

    def test_target_not_created_on_failure(self, tmp_path):
        """Target file does not exist if write fails."""
        target = tmp_path / "never.txt"
        with patch("aura.persistence.os.rename", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                atomic_write(target, "data")
        assert not target.exists()
