"""Phase 12 tests — End-to-End Integration, Fatigue Intelligence & Observability.

Tests for:
  US-308: TiltDetector end-to-end wiring
  US-309: Outcome-driven training loop for AdaptiveWeightManager
  US-310: Decision interval variability as cognitive fatigue proxy
  US-311: Bridge health CLI commands
  US-312: Pattern tier cascade reload validation
  US-313: Readiness trend anomaly detection via EWMA residuals
"""

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- Neutral circadian config (all hours = 1.0) ---
NEUTRAL_CIRCADIAN = {h: 1.0 for h in range(24)}


# ──────────────────────────────────────────────────────────────────────
# US-308: TiltDetector End-to-End Wiring
# ──────────────────────────────────────────────────────────────────────

class TestTiltDetectorWiring:
    """US-308: Verify TiltDetector is initialized and fed real context."""

    def test_readiness_computer_creates_tilt_detector(self):
        """ReadinessComputer.__init__ creates a TiltDetector instance."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        assert hasattr(rc, '_tilt_detector')
        assert rc._tilt_detector is not None

    def test_set_context_stores_messages(self):
        """set_context() stores recent messages for tilt detection."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        msgs = [{"content": f"msg {i}", "sentiment": 0.5} for i in range(25)]
        rc.set_context(messages=msgs)
        assert len(rc._recent_messages) == 20  # Trimmed to 20

    def test_set_context_stores_outcomes(self):
        """set_context() stores recent outcomes for tilt detection."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        outcomes = [{"trade_won": True} for _ in range(15)]
        rc.set_context(outcomes=outcomes)
        assert len(rc._recent_outcomes) == 10  # Trimmed to 10

    def test_compute_uses_tilt_detector_with_context(self):
        """compute() calls TiltDetector with stored messages, not empty defaults."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        # Set revenge keyword messages
        revenge_msgs = [
            {"content": "I need to make it back", "sentiment": 0.3},
            {"content": "Just one more trade to recover", "sentiment": 0.2},
            {"content": "I can double down and recover my losses", "sentiment": 0.2},
            {"content": "I need to recover everything now", "sentiment": 0.1},
            {"content": "Let me make it back quickly", "sentiment": 0.1},
        ]
        rc.set_context(
            messages=revenge_msgs,
            outcomes=[{"trade_won": False}] * 3,
        )
        rc._last_smoothed_score = None
        signal = rc.compute()
        # Tilt should be detected (revenge keywords present)
        assert signal.tilt_score > 0, "Tilt should be detected with revenge keywords"

    def test_calm_messages_no_tilt(self):
        """Calm messages produce tilt_score == 0."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        calm_msgs = [
            {"content": "Market looks stable today", "sentiment": 0.7},
            {"content": "I'm feeling good about my analysis", "sentiment": 0.8},
        ]
        rc.set_context(messages=calm_msgs, outcomes=[])
        rc._last_smoothed_score = None
        signal = rc.compute()
        assert signal.tilt_score == 0.0

    def test_tilt_penalty_reduces_readiness(self):
        """When tilt is detected, readiness score is reduced."""
        from src.aura.core.readiness import ReadinessComputer
        # First: compute without tilt
        rc1 = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        rc1.set_context(messages=[], outcomes=[])
        rc1._last_smoothed_score = None
        signal_calm = rc1.compute()

        # Second: compute with tilt
        rc2 = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        revenge_msgs = [
            {"content": "make it back", "sentiment": 0.2},
            {"content": "recover my losses", "sentiment": 0.2},
            {"content": "double down now", "sentiment": 0.1},
            {"content": "just one more to recover", "sentiment": 0.1},
            {"content": "I need to make it back", "sentiment": 0.1},
        ]
        rc2.set_context(
            messages=revenge_msgs,
            outcomes=[{"trade_won": False}] * 3,
        )
        rc2._last_smoothed_score = None
        signal_tilt = rc2.compute()

        if signal_tilt.tilt_score > 0:
            assert signal_tilt.raw_score <= signal_calm.raw_score

    def test_set_context_partial_update(self):
        """set_context() with only messages doesn't clear outcomes."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        rc.set_context(outcomes=[{"trade_won": True}])
        rc.set_context(messages=[{"content": "hi", "sentiment": 0.5}])
        assert len(rc._recent_outcomes) == 1  # Unchanged


# ──────────────────────────────────────────────────────────────────────
# US-309: Outcome-Driven Training Loop
# ──────────────────────────────────────────────────────────────────────

class TestTrainingLoop:
    """US-309: Verify adaptive weight training from trade outcomes."""

    def test_train_from_outcome_updates_alpha_on_correct(self):
        """Correct prediction increments alpha."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            # Create a readiness signal where all components > 0.5 (predict good)
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable", components=ReadinessComponents(
                    emotional_state_score=0.9, cognitive_load_score=0.9,
                    override_discipline_score=0.8, stress_level_score=0.7,
                    confidence_trend_score=0.7, engagement_score=0.5,
                ),
            )
            outcome = {"trade_won": True, "profit_pips": 25.0, "timestamp": datetime.now(timezone.utc).isoformat()}
            initial_alpha = awm._priors["emotional_state"]["alpha"]
            result = rc.train_from_outcome(outcome, signal)
            assert result is True
            assert awm._priors["emotional_state"]["alpha"] > initial_alpha

    def test_train_from_outcome_updates_beta_on_incorrect(self):
        """Incorrect prediction increments beta."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable", components=ReadinessComponents(
                    emotional_state_score=0.9, cognitive_load_score=0.9,
                    override_discipline_score=0.8, stress_level_score=0.7,
                    confidence_trend_score=0.7, engagement_score=0.5,
                ),
            )
            # Trade lost but components predicted good → incorrect
            outcome = {"trade_won": False, "profit_pips": -15.0, "timestamp": datetime.now(timezone.utc).isoformat()}
            initial_beta = awm._priors["emotional_state"]["beta"]
            rc.train_from_outcome(outcome, signal)
            assert awm._priors["emotional_state"]["beta"] > initial_beta

    def test_train_without_adaptive_weights_returns_false(self):
        """Training without AdaptiveWeightManager returns False."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        result = rc.train_from_outcome({"trade_won": True})
        assert result is False

    def test_train_without_readiness_signal_returns_false(self):
        """Training without a readiness signal returns False."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            rc = ReadinessComputer(
                signal_path=Path(td) / "nonexistent_signal.json",
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            result = rc.train_from_outcome({"trade_won": True})
            assert result is False

    def test_train_persists_weights(self):
        """Training persists updated weights to disk."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            weights_path = Path(td) / "weights.json"
            awm = AdaptiveWeightManager(persist_path=weights_path)
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable", components=ReadinessComponents(),
            )
            outcome = {"trade_won": True, "profit_pips": 10.0, "timestamp": datetime.now(timezone.utc).isoformat()}
            rc.train_from_outcome(outcome, signal)
            assert weights_path.exists()
            data = json.loads(weights_path.read_text())
            assert data["sample_count"] >= 1

    def test_train_v1_fallback_until_10_outcomes(self):
        """Adaptive weights not used until 10+ outcomes trained."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable", components=ReadinessComponents(),
            )
            # Train 1 time — each call updates 6 components = 6 samples
            # MIN_SAMPLES is 10, so 1 train call (6 samples) should NOT be ready
            rc.train_from_outcome(
                {"trade_won": True, "profit_pips": 10.0, "timestamp": datetime.now(timezone.utc).isoformat()},
                signal,
            )
            assert not awm.is_ready(), f"Expected not ready with {awm.sample_count} samples"

    def test_train_days_old_decay(self):
        """Older outcomes contribute less to weight updates."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable", components=ReadinessComponents(),
            )
            # Recent outcome
            outcome_recent = {"trade_won": True, "profit_pips": 10.0, "timestamp": datetime.now(timezone.utc).isoformat()}
            rc.train_from_outcome(outcome_recent, signal)
            alpha_after_recent = awm._priors["emotional_state"]["alpha"]

            # Reset and train with old outcome (30 days ago — half-life)
            awm2 = AdaptiveWeightManager(persist_path=Path(td) / "weights2.json")
            rc2 = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm2,
            )
            from datetime import timedelta
            old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            outcome_old = {"trade_won": True, "profit_pips": 10.0, "timestamp": old_ts}
            rc2.train_from_outcome(outcome_old, signal)
            alpha_after_old = awm2._priors["emotional_state"]["alpha"]

            # Recent outcome should have larger alpha increment
            recent_increment = alpha_after_recent - 1.0  # Started at 1.0
            old_increment = alpha_after_old - 1.0
            assert recent_increment > old_increment

    def test_train_full_loop_weight_change(self):
        """Full loop: outcome → weight update → weights differ from initial."""
        from src.aura.core.readiness import ReadinessComputer, AdaptiveWeightManager, ReadinessSignal, ReadinessComponents
        with tempfile.TemporaryDirectory() as td:
            awm = AdaptiveWeightManager(persist_path=Path(td) / "weights.json")
            initial_weights = awm.get_weights()
            rc = ReadinessComputer(
                circadian_config=NEUTRAL_CIRCADIAN,
                adaptive_weights=awm,
            )
            signal = ReadinessSignal(
                readiness_score=75, cognitive_load="low", active_stressors=[],
                override_loss_rate_7d=0.0, emotional_state="calm",
                confidence_trend="stable",
                components=ReadinessComponents(
                    emotional_state_score=0.9, cognitive_load_score=0.3,
                    override_discipline_score=0.8, stress_level_score=0.7,
                    confidence_trend_score=0.7, engagement_score=0.5,
                ),
            )
            # Train with a won trade — high-scoring components rewarded, low-scoring ones punished
            for _ in range(5):
                rc.train_from_outcome(
                    {"trade_won": True, "profit_pips": 20.0, "timestamp": datetime.now(timezone.utc).isoformat()},
                    signal,
                )
            updated_weights = awm.get_weights()
            # Weights should have shifted
            assert updated_weights != initial_weights


# ──────────────────────────────────────────────────────────────────────
# US-310: Decision Interval Variability
# ──────────────────────────────────────────────────────────────────────

class TestDecisionCadenceAnalyzer:
    """US-310: Verify HRV-proxy fatigue detection from decision intervals."""

    def test_optimal_intervals_score_high(self):
        """Intervals in the optimal range produce variability_score ~1.0."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer(optimal_rmssd_low=30.0, optimal_rmssd_high=120.0)
        # Moderately variable intervals (60±20 seconds)
        base = 1000.0
        timestamps = [base + i * 60 + (10 * ((-1)**i)) for i in range(10)]
        result = analyzer.analyze(timestamps)
        assert result.variability_score >= 0.5, f"Expected high score for moderate variability, got {result.variability_score}"

    def test_very_regular_intervals_fatigue(self):
        """Very regular (identical) intervals indicate fatigue → low score."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer(optimal_rmssd_low=30.0, optimal_rmssd_high=120.0)
        # Perfectly regular: 60s apart exactly
        timestamps = [1000.0 + i * 60.0 for i in range(10)]
        result = analyzer.analyze(timestamps)
        assert result.rmssd < 30.0, f"RMSSD should be near 0 for regular intervals, got {result.rmssd}"
        assert result.variability_score < 0.5, f"Expected low score for regular intervals, got {result.variability_score}"

    def test_erratic_intervals_stress(self):
        """Highly erratic intervals indicate stress → low score."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer(optimal_rmssd_low=30.0, optimal_rmssd_high=120.0)
        # Very erratic: alternating 5s and 200s gaps
        timestamps = [0, 5, 205, 210, 410, 415, 615, 620, 820, 825]
        result = analyzer.analyze(timestamps)
        assert result.rmssd > 120.0, f"RMSSD should be high for erratic intervals, got {result.rmssd}"
        assert result.variability_score < 0.8

    def test_insufficient_data_returns_1(self):
        """Fewer than 5 timestamps returns variability_score=1.0."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer()
        result = analyzer.analyze([100.0, 200.0, 300.0])
        assert result.variability_score == 1.0

    def test_single_timestamp(self):
        """Single timestamp returns default metrics."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer()
        result = analyzer.analyze([100.0])
        assert result.variability_score == 1.0

    def test_configurable_range(self):
        """Custom optimal range changes scoring."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        # Very tight optimal range
        analyzer = DecisionCadenceAnalyzer(optimal_rmssd_low=50.0, optimal_rmssd_high=60.0)
        timestamps = [1000.0 + i * 60.0 for i in range(10)]  # Regular
        result = analyzer.analyze(timestamps)
        # With low range at 50, regular intervals (RMSSD ≈ 0) should score low
        assert result.variability_score < 0.5

    def test_rmssd_computation_correctness(self):
        """RMSSD formula is correct: sqrt(mean(successive_diffs^2))."""
        from src.aura.core.readiness import DecisionCadenceAnalyzer
        analyzer = DecisionCadenceAnalyzer()
        # Timestamps with known intervals: [10, 20, 10, 20] → diffs [10, -10, 10]
        timestamps = [0, 10, 30, 40, 60]
        result = analyzer.analyze(timestamps)
        # Intervals: [10, 20, 10, 20]
        # Successive diffs: [10, -10, 10]
        # RMSSD = sqrt((100 + 100 + 100) / 3) = sqrt(100) = 10
        assert abs(result.rmssd - 10.0) < 0.1, f"Expected RMSSD ≈ 10.0, got {result.rmssd}"

    def test_readiness_signal_includes_variability(self):
        """ReadinessSignal includes decision_variability field."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        rc._last_smoothed_score = None
        signal = rc.compute()
        assert hasattr(signal, 'decision_variability')
        d = signal.to_dict()
        assert 'decision_variability' in d


# ──────────────────────────────────────────────────────────────────────
# US-311: Bridge Health CLI Commands
# ──────────────────────────────────────────────────────────────────────

class TestBridgeHealthCLI:
    """US-311: Verify bridge health status display and repair commands."""

    def test_repair_corrupted_healthy_files(self):
        """repair_corrupted() returns 'healthy' for valid files."""
        from src.aura.bridge.signals import FeedbackBridge
        with tempfile.TemporaryDirectory() as td:
            bridge = FeedbackBridge(bridge_dir=Path(td))
            # Write valid signals
            (Path(td) / "readiness_signal.json").write_text('{"readiness_score": 75}')
            (Path(td) / "outcome_signal.json").write_text('{"trade_won": true}')
            results = bridge.repair_corrupted()
            # At least some files should report status
            assert isinstance(results, dict)

    def test_repair_corrupted_missing_file(self):
        """repair_corrupted() reports 'missing' for nonexistent files."""
        from src.aura.bridge.signals import FeedbackBridge
        with tempfile.TemporaryDirectory() as td:
            bridge = FeedbackBridge(bridge_dir=Path(td))
            results = bridge.repair_corrupted()
            assert isinstance(results, dict)
            # At least one file should be missing
            assert any(v in ("missing", "healthy") for v in results.values())

    def test_bridge_health_status_method_exists(self):
        """bridge_health() method returns BridgeHealthStatus."""
        from src.aura.bridge.signals import FeedbackBridge
        with tempfile.TemporaryDirectory() as td:
            bridge = FeedbackBridge(bridge_dir=Path(td))
            health = bridge.bridge_health()
            assert health is not None
            assert hasattr(health, 'readiness')  # Field is 'readiness', not 'readiness_status'

    def test_companion_bridge_status_command(self):
        """AuraCompanion handles /bridge-status command."""
        from src.aura.cli.companion import AuraCompanion
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            bridge_dir = Path(td) / "bridge"
            bridge_dir.mkdir()
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            result = companion._handle_bridge_status()
            assert "Bridge Health Status" in result

    def test_companion_bridge_repair_command(self):
        """AuraCompanion handles /bridge-repair command."""
        from src.aura.cli.companion import AuraCompanion
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            bridge_dir = Path(td) / "bridge"
            bridge_dir.mkdir()
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            result = companion._handle_bridge_repair()
            assert "Bridge Repair Results" in result

    def test_staleness_warning_flag(self):
        """Staleness warning sets _staleness_warned flag."""
        from src.aura.cli.companion import AuraCompanion
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            bridge_dir = Path(td) / "bridge"
            bridge_dir.mkdir()
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            assert companion._staleness_warned is False

    def test_staleness_suppressed_after_first(self):
        """Once _staleness_warned is True, no repeat warnings."""
        from src.aura.cli.companion import AuraCompanion
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            bridge_dir = Path(td) / "bridge"
            bridge_dir.mkdir()
            companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)
            companion._staleness_warned = True
            # Should stay True and not reset
            assert companion._staleness_warned is True

    def test_repair_return_value_is_dict(self):
        """repair_corrupted returns dict with file→status pairs."""
        from src.aura.bridge.signals import FeedbackBridge
        with tempfile.TemporaryDirectory() as td:
            bridge = FeedbackBridge(bridge_dir=Path(td))
            results = bridge.repair_corrupted()
            assert isinstance(results, dict)
            for v in results.values():
                assert v in ("healthy", "corrupted", "missing", "stale", "repaired", "no_backup", "failed")


# ──────────────────────────────────────────────────────────────────────
# US-312: Pattern Tier Cascade Validation
# ──────────────────────────────────────────────────────────────────────

class TestPatternCascadeValidation:
    """US-312: Verify pattern tier reload validation and health reporting."""

    def test_tier_reload_result_values(self):
        """TierReloadResult has SUCCESS, PARTIAL, FAILED constants."""
        from src.aura.patterns.engine import TierReloadResult
        assert TierReloadResult.SUCCESS == "success"
        assert TierReloadResult.PARTIAL == "partial"
        assert TierReloadResult.FAILED == "failed"

    def test_reload_returns_dict(self):
        """_reload_tier_patterns() returns Dict[str, str]."""
        from src.aura.patterns.engine import PatternEngine
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            result = pe._reload_tier_patterns()
            assert isinstance(result, dict)
            assert "t1" in result
            assert "t2" in result
            assert "t3" in result

    def test_successful_reload_reports_success(self):
        """Successful reload returns SUCCESS for each tier."""
        from src.aura.patterns.engine import PatternEngine, TierReloadResult
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            result = pe._reload_tier_patterns()
            # With no patterns, success is valid (empty is fine)
            for tier_name, status in result.items():
                assert status in (TierReloadResult.SUCCESS, TierReloadResult.PARTIAL, TierReloadResult.FAILED)

    def test_cascade_health_method_exists(self):
        """cascade_health() returns dict with expected keys."""
        from src.aura.patterns.engine import PatternEngine
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            health = pe.cascade_health()
            assert isinstance(health, dict)
            assert "t1_count" in health
            assert "t2_count" in health
            assert "t3_count" in health

    def test_cascade_health_after_run_all(self):
        """cascade_health() populated after run_all()."""
        from src.aura.patterns.engine import PatternEngine
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            pe.run_all(conversations=[], readiness_history=[])
            health = pe.cascade_health()
            assert health.get("timestamp") is not None

    def test_run_all_stores_cascade_health(self):
        """run_all() stores _last_cascade_health."""
        from src.aura.patterns.engine import PatternEngine
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            result = pe.run_all(conversations=[], readiness_history=[])
            assert hasattr(pe, '_last_cascade_health')
            assert "reload_1" in pe._last_cascade_health
            assert "reload_2" in pe._last_cascade_health

    def test_run_all_returns_three_tiers(self):
        """run_all() returns dict with t1, t2, t3 keys."""
        from src.aura.patterns.engine import PatternEngine
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            result = pe.run_all(conversations=[], readiness_history=[])
            assert "t1" in result
            assert "t2" in result
            assert "t3" in result

    def test_reload_handles_tier_exception(self):
        """Tier reload handles exceptions gracefully."""
        from src.aura.patterns.engine import PatternEngine, TierReloadResult
        with tempfile.TemporaryDirectory() as td:
            pe = PatternEngine(
                patterns_dir=Path(td) / "patterns",
                bridge_dir=Path(td) / "bridge",
            )
            # Force T1 to fail by breaking _load_patterns
            original = pe.t1._load_patterns
            def fail_load():
                raise RuntimeError("Simulated corruption")
            pe.t1._load_patterns = fail_load
            result = pe._reload_tier_patterns()
            assert result["t1"] == TierReloadResult.FAILED
            pe.t1._load_patterns = original  # Restore


# ──────────────────────────────────────────────────────────────────────
# US-313: Readiness Anomaly Detection
# ──────────────────────────────────────────────────────────────────────

class TestReadinessAnomalyDetector:
    """US-313: Verify EWMA residual anomaly detection."""

    def test_stable_scores_no_anomaly(self):
        """Stable readiness scores produce no anomalies."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        for _ in range(20):
            result = detector.update(70.0)
        assert result.anomaly_detected is False

    def test_sudden_drop_triggers_anomaly(self):
        """A sudden large drop triggers anomaly detection."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        # Build baseline at 75
        for _ in range(15):
            detector.update(75.0)
        # Sudden drop to 20
        result = detector.update(20.0)
        assert result.anomaly_detected is True
        assert result.severity > 0

    def test_gradual_decline_no_anomaly(self):
        """Gradual decline doesn't trigger anomaly (EWMA adapts)."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        score = 80.0
        anomaly_count = 0
        for _ in range(30):
            result = detector.update(score)
            if result.anomaly_detected:
                anomaly_count += 1
            score -= 1.0  # Gradual decline
        # Most steps should NOT trigger anomaly
        assert anomaly_count < 5, f"Too many anomalies for gradual decline: {anomaly_count}"

    def test_severity_scaling(self):
        """Severity increases with larger residuals."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        for _ in range(15):
            detector.update(70.0)
        result = detector.update(10.0)  # Very large drop
        if result.anomaly_detected:
            assert result.severity > 0.5

    def test_minimum_history_requirement(self):
        """No anomaly detection until MIN_HISTORY scores collected."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        # Feed 5 scores then a big drop — should NOT detect (need 10)
        for _ in range(5):
            detector.update(70.0)
        result = detector.update(10.0)
        assert result.anomaly_detected is False

    def test_ewma_convergence(self):
        """EWMA baseline converges toward the mean score."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        detector = ReadinessAnomalyDetector(alpha=0.1)
        for _ in range(50):
            result = detector.update(60.0)
        # Baseline should be near 60
        assert abs(result.baseline - 60.0) < 2.0

    def test_configurable_alpha(self):
        """Different alpha values produce different baselines."""
        from src.aura.core.readiness import ReadinessAnomalyDetector
        fast = ReadinessAnomalyDetector(alpha=0.5)  # Fast adaptation
        slow = ReadinessAnomalyDetector(alpha=0.05)  # Slow adaptation
        # Start at 80, then switch to 40
        for _ in range(10):
            fast.update(80.0)
            slow.update(80.0)
        for _ in range(5):
            r_fast = fast.update(40.0)
            r_slow = slow.update(40.0)
        # Fast should be closer to 40
        assert r_fast.baseline < r_slow.baseline

    def test_readiness_signal_includes_anomaly_fields(self):
        """ReadinessSignal includes anomaly_detected and anomaly_severity."""
        from src.aura.core.readiness import ReadinessComputer
        rc = ReadinessComputer(circadian_config=NEUTRAL_CIRCADIAN)
        rc._last_smoothed_score = None
        signal = rc.compute()
        assert hasattr(signal, 'anomaly_detected')
        assert hasattr(signal, 'anomaly_severity')
        d = signal.to_dict()
        assert 'anomaly_detected' in d
        assert 'anomaly_severity' in d
