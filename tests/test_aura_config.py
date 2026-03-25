"""Tests for Aura configuration system.

US-272: Configurable pattern thresholds via .aura/config.json.
US-274: Pattern lifecycle end-to-end tests.
US-275: Companion-Bridge integration tests.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.config import load_config, get_t1_config, get_t2_config, get_t3_config, DEFAULTS


# ═══════════════════════════════════════════════════════════════════════
# US-272: Config loading, defaults, partial overrides, tier passthrough
# ═══════════════════════════════════════════════════════════════════════


class TestUS272ConfigLoading:
    """US-272: Config file loading with defaults and validation."""

    def test_missing_config_file_uses_defaults(self, tmp_path):
        """Missing config.json returns all default values."""
        config = load_config(config_path=tmp_path / "nonexistent.json")

        assert config == DEFAULTS
        assert config["t1_stress_frequency_threshold"] == 0.6
        assert config["t2_min_sample_size"] == 5
        assert config["t3_min_weeks_for_arc"] == 4

    def test_empty_config_file_uses_defaults(self, tmp_path):
        """Empty JSON object returns all defaults."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{}")

        config = load_config(config_path=config_path)
        assert config == DEFAULTS

    def test_full_override(self, tmp_path):
        """Config file with all keys overrides all defaults."""
        overrides = {
            "t1_stress_frequency_threshold": 0.8,
            "t1_override_frequency_threshold": 5,
            "t1_readiness_decline_streak": 5,
            "t1_stressor_recurrence_threshold": 0.7,
            "t2_min_sample_size": 10,
            "t2_p_value_threshold": 0.05,
            "t2_min_correlation_strength": 0.4,
            "t3_min_weeks_for_arc": 6,
            "t3_trend_significance": 0.25,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(overrides))

        config = load_config(config_path=config_path)

        for key, value in overrides.items():
            assert config[key] == value, f"{key}: expected {value}, got {config[key]}"

    def test_partial_override_fills_defaults(self, tmp_path):
        """Config with only some keys fills missing ones with defaults."""
        partial = {
            "t1_stress_frequency_threshold": 0.9,
            "t3_min_weeks_for_arc": 8,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(partial))

        config = load_config(config_path=config_path)

        # Overridden
        assert config["t1_stress_frequency_threshold"] == 0.9
        assert config["t3_min_weeks_for_arc"] == 8

        # Defaults
        assert config["t2_min_sample_size"] == DEFAULTS["t2_min_sample_size"]
        assert config["t1_override_frequency_threshold"] == DEFAULTS["t1_override_frequency_threshold"]

    def test_wrong_type_ignored(self, tmp_path, caplog):
        """Config value with wrong type is ignored, default used."""
        bad_types = {
            "t1_stress_frequency_threshold": "not_a_number",
            "t2_min_sample_size": 7,  # valid
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(bad_types))

        import logging
        with caplog.at_level(logging.WARNING):
            config = load_config(config_path=config_path)

        assert config["t1_stress_frequency_threshold"] == DEFAULTS["t1_stress_frequency_threshold"]
        assert config["t2_min_sample_size"] == 7

    def test_unknown_keys_ignored(self, tmp_path, caplog):
        """Unknown config keys are logged but don't cause errors."""
        data = {
            "unknown_key": 42,
            "t1_stress_frequency_threshold": 0.75,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(data))

        import logging
        with caplog.at_level(logging.INFO):
            config = load_config(config_path=config_path)

        assert config["t1_stress_frequency_threshold"] == 0.75
        assert "unknown_key" not in config

    def test_corrupt_json_uses_defaults(self, tmp_path):
        """Corrupt JSON file falls back to all defaults."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{this is not valid json!!")

        config = load_config(config_path=config_path)
        assert config == DEFAULTS

    def test_non_dict_json_uses_defaults(self, tmp_path):
        """JSON array instead of object falls back to defaults."""
        config_path = tmp_path / "config.json"
        config_path.write_text("[1, 2, 3]")

        config = load_config(config_path=config_path)
        assert config == DEFAULTS


class TestUS272TierConfigExtraction:
    """US-272: Config extraction helpers produce correct subsets."""

    def test_t1_config_extraction(self):
        """get_t1_config extracts the 4 T1 keys."""
        config = dict(DEFAULTS)
        t1 = get_t1_config(config)
        assert "stress_frequency_threshold" in t1
        assert "override_frequency_threshold" in t1
        assert "readiness_decline_streak" in t1
        assert "stressor_recurrence_threshold" in t1
        assert len(t1) == 4

    def test_t2_config_extraction(self):
        """get_t2_config extracts the 3 T2 keys."""
        config = dict(DEFAULTS)
        t2 = get_t2_config(config)
        assert "min_sample_size" in t2
        assert "p_value_threshold" in t2
        assert "min_correlation_strength" in t2
        assert len(t2) == 3

    def test_t3_config_extraction(self):
        """get_t3_config extracts the 2 T3 keys."""
        config = dict(DEFAULTS)
        t3 = get_t3_config(config)
        assert "min_weeks_for_arc" in t3
        assert "trend_significance" in t3
        assert len(t3) == 2


class TestUS272TierPassthrough:
    """US-272: Tier constructors accept and use config values."""

    def test_t1_uses_config_threshold(self, tmp_path):
        """T1 detector uses config threshold instead of default."""
        from aura.patterns.tier1 import Tier1FrequencyDetector

        t1 = Tier1FrequencyDetector(
            patterns_dir=tmp_path,
            config={"stress_frequency_threshold": 0.9},
        )
        assert t1.stress_frequency_threshold == 0.9

    def test_t2_uses_config_sample_size(self, tmp_path):
        """T2 detector uses config min_sample_size."""
        from aura.patterns.tier2 import Tier2CrossDomainDetector

        t2 = Tier2CrossDomainDetector(
            patterns_dir=tmp_path,
            config={"min_sample_size": 10},
        )
        assert t2.min_sample_size == 10

    def test_t3_uses_config_weeks(self, tmp_path):
        """T3 detector uses config min_weeks_for_arc."""
        from aura.patterns.tier3 import Tier3NarrativeArcDetector

        t3 = Tier3NarrativeArcDetector(
            patterns_dir=tmp_path,
            config={"min_weeks_for_arc": 8},
        )
        assert t3.min_weeks_for_arc == 8

    def test_engine_passes_config_to_tiers(self, tmp_path):
        """PatternEngine loads config and passes to all three tiers."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "t1_stress_frequency_threshold": 0.85,
            "t2_min_sample_size": 12,
            "t3_min_weeks_for_arc": 6,
        }))

        from aura.config import load_config, get_t1_config, get_t2_config, get_t3_config
        config = load_config(config_path)

        from aura.patterns.engine import PatternEngine
        engine = PatternEngine(
            patterns_dir=tmp_path / "patterns",
            bridge_dir=tmp_path / "bridge",
            config=config,
        )

        assert engine.t1.stress_frequency_threshold == 0.85
        assert engine.t2.min_sample_size == 12
        assert engine.t3.min_weeks_for_arc == 6


# ═══════════════════════════════════════════════════════════════════════
# US-274: Pattern lifecycle end-to-end tests
# ═══════════════════════════════════════════════════════════════════════


class TestUS274PatternLifecycle:
    """US-274: End-to-end pattern detection lifecycle."""

    def test_t1_detects_stress_pattern_after_threshold(self, tmp_path):
        """T1 detects emotional frequency pattern when stress exceeds threshold."""
        from aura.patterns.tier1 import Tier1FrequencyDetector

        t1 = Tier1FrequencyDetector(patterns_dir=tmp_path)

        # 5 of 7 conversations are stressed = 71% > 60% threshold
        conversations = [
            {"emotional_state": "stressed", "detected_stressors": [], "topics": []} for _ in range(5)
        ] + [
            {"emotional_state": "calm", "detected_stressors": [], "topics": []} for _ in range(2)
        ]

        patterns = t1.detect(conversations=conversations, readiness_history=[], override_events=[])
        stress_patterns = [p for p in patterns if "emotional_frequency" in p.pattern_id]
        assert len(stress_patterns) >= 1, "T1 should detect emotional frequency pattern"

    def test_t1_no_pattern_below_threshold(self, tmp_path):
        """T1 does NOT detect pattern when below threshold."""
        from aura.patterns.tier1 import Tier1FrequencyDetector

        t1 = Tier1FrequencyDetector(patterns_dir=tmp_path)

        # 2 of 7 stressed = 29% < 60% threshold
        conversations = [
            {"emotional_state": "stressed", "detected_stressors": [], "topics": []} for _ in range(2)
        ] + [
            {"emotional_state": "calm", "detected_stressors": [], "topics": []} for _ in range(5)
        ]

        patterns = t1.detect(conversations=conversations, readiness_history=[], override_events=[])
        stress_patterns = [p for p in patterns if "emotional_frequency" in p.pattern_id]
        assert len(stress_patterns) == 0, "T1 should NOT detect pattern below threshold"

    def test_t1_configurable_threshold_changes_detection(self, tmp_path):
        """Lowering the threshold detects patterns that wouldn't be detected at defaults."""
        from aura.patterns.tier1 import Tier1FrequencyDetector

        # 3 of 7 stressed = 43% — below default 60% but above 40%
        conversations = [
            {"emotional_state": "stressed", "detected_stressors": [], "topics": []} for _ in range(3)
        ] + [
            {"emotional_state": "calm", "detected_stressors": [], "topics": []} for _ in range(4)
        ]

        # Default threshold = 0.6 → no detection
        t1_default = Tier1FrequencyDetector(patterns_dir=tmp_path / "default")
        default_patterns = t1_default.detect(conversations=conversations, readiness_history=[], override_events=[])
        default_stress = [p for p in default_patterns if "emotional_frequency" in p.pattern_id]
        assert len(default_stress) == 0

        # Lower threshold = 0.4 → detection
        t1_custom = Tier1FrequencyDetector(
            patterns_dir=tmp_path / "custom",
            config={"stress_frequency_threshold": 0.4},
        )
        custom_patterns = t1_custom.detect(conversations=conversations, readiness_history=[], override_events=[])
        custom_stress = [p for p in custom_patterns if "emotional_frequency" in p.pattern_id]
        assert len(custom_stress) >= 1

    def test_t2_returns_empty_on_insufficient_data(self, tmp_path):
        """T2 returns empty when data is below min_sample_size."""
        from aura.patterns.tier2 import Tier2CrossDomainDetector

        t2 = Tier2CrossDomainDetector(patterns_dir=tmp_path)
        # Less than MIN_SAMPLE_SIZE (5) data points
        conversations = [{"emotional_state": "calm"}] * 3
        trade_outcomes = [{"pnl_pips": 10}] * 3

        patterns = t2.detect(conversations=conversations, trade_outcomes=trade_outcomes, readiness_history=[], override_events=[])
        assert len(patterns) == 0

    def test_engine_cascade_t1_to_t2_to_t3(self, tmp_path):
        """PatternEngine.run_all executes T1→T2→T3 in order with reload."""
        from aura.patterns.engine import PatternEngine

        engine = PatternEngine(
            patterns_dir=tmp_path / "patterns",
            bridge_dir=tmp_path / "bridge",
        )
        # run_all with empty data should not crash
        results = engine.run_all(conversations=[], readiness_history=[])
        assert "t1" in results
        assert "t2" in results
        assert "t3" in results


# ═══════════════════════════════════════════════════════════════════════
# US-275: Companion-Bridge integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestUS275CompanionBridgeIntegration:
    """US-275: Companion interaction with bridge files."""

    def test_readiness_signal_written_to_bridge(self, tmp_path):
        """Readiness computation writes signal to bridge path."""
        from aura.cli.companion import AuraCompanion

        bridge_dir = tmp_path / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)

        comp = AuraCompanion(
            db_path=tmp_path / "test.db",
            bridge_dir=bridge_dir,
        )

        # Process a message to trigger readiness update
        comp.process_input("I'm feeling good today, markets look promising")

        signal_path = bridge_dir / "readiness_signal.json"
        assert signal_path.exists(), "Readiness signal should be written to bridge"

        signal_data = json.loads(signal_path.read_text())
        assert "readiness_score" in signal_data
        assert 0 <= signal_data["readiness_score"] <= 100

    def test_companion_handles_missing_bridge_files(self, tmp_path):
        """Companion handles missing bridge files gracefully."""
        from aura.cli.companion import AuraCompanion

        bridge_dir = tmp_path / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        # No files exist — companion should not crash

        comp = AuraCompanion(
            db_path=tmp_path / "test.db",
            bridge_dir=bridge_dir,
        )

        # Process input should work even without bridge files
        result = comp.process_input("Hello, how are things?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_bridge_status_command_with_no_signals(self, tmp_path):
        """/bridge works even with no existing signal files."""
        from aura.cli.companion import AuraCompanion

        bridge_dir = tmp_path / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)

        comp = AuraCompanion(
            db_path=tmp_path / "test.db",
            bridge_dir=bridge_dir,
        )

        result = comp.process_input("/bridge")
        assert "Bridge Status" in result

    def test_outcome_read_returns_default_when_seeded(self, tmp_path):
        """Bridge read_outcome returns a default OutcomeSignal after H-01 fix.

        FeedbackBridge.__init__() now seeds outcome_signal.json with safe defaults
        so downstream readers never receive None on first call. Previously this
        returned None (the buggy behavior). This test reflects the fixed contract.
        """
        from aura.bridge.signals import FeedbackBridge, OutcomeSignal

        bridge = FeedbackBridge(bridge_dir=tmp_path / "bridge")
        outcome = bridge.read_outcome()
        # After H-01 fix: seeded file exists → OutcomeSignal with zero defaults
        assert outcome is not None
        assert isinstance(outcome, OutcomeSignal)
        assert outcome.pnl_today == 0.0
        assert outcome.trades_today == 0
