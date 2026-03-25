"""Tests for Phase 6 defensive hardening changes.

US-265: Tests covering input validation, bounds checking, buffer caps,
type safety, and observability improvements from US-260 through US-264.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.core.conversation_processor import ConversationProcessor, ConversationSignals
from aura.prediction.readiness_v2 import ReadinessModelV2, ReadinessTrainingSample


# ═══════════════════════════════════════════════════════════════════════
# US-261: Input validation and bounds in ConversationProcessor
# ═══════════════════════════════════════════════════════════════════════


class TestUS261InputValidation:
    """US-261: process_message() validates input and enforces bounds."""

    def test_empty_message_returns_neutral(self):
        """Empty string message returns neutral ConversationSignals."""
        proc = ConversationProcessor()
        signals = proc.process_message("")
        assert isinstance(signals, ConversationSignals)
        # Should not have been appended to session messages
        assert len(proc._session_messages) == 0

    def test_whitespace_message_returns_neutral(self):
        """Whitespace-only message returns neutral ConversationSignals."""
        proc = ConversationProcessor()
        signals = proc.process_message("   \t\n  ")
        assert isinstance(signals, ConversationSignals)
        assert len(proc._session_messages) == 0

    def test_normal_message_works(self):
        """Normal messages still work as before."""
        proc = ConversationProcessor()
        signals = proc.process_message("I'm feeling good today")
        assert isinstance(signals, ConversationSignals)
        assert len(proc._session_messages) == 1

    def test_oversized_message_truncated(self, caplog):
        """Messages over MAX_MESSAGE_LENGTH are truncated with warning."""
        proc = ConversationProcessor()
        huge_msg = "x" * 20000
        with caplog.at_level(logging.WARNING):
            signals = proc.process_message(huge_msg)

        assert isinstance(signals, ConversationSignals)
        # Message should have been stored truncated
        assert len(proc._session_messages) == 1
        assert len(proc._session_messages[0]["content"]) == proc.MAX_MESSAGE_LENGTH

        # Should log the truncation
        truncation_logs = [r for r in caplog.records if "US-261" in r.message and "truncated" in r.message]
        assert len(truncation_logs) == 1

    def test_session_messages_capped(self):
        """_session_messages is capped at MAX_SESSION_MESSAGES."""
        proc = ConversationProcessor()
        # Fill beyond the cap
        for i in range(proc.MAX_SESSION_MESSAGES + 50):
            proc.process_message(f"msg {i}")

        assert len(proc._session_messages) <= proc.MAX_SESSION_MESSAGES

    def test_message_count_reflects_cap(self):
        """message_count in signals reflects actual count after cap."""
        proc = ConversationProcessor()
        for i in range(10):
            signals = proc.process_message(f"message {i}")

        # message_count should match stored messages
        assert signals.message_count == len(proc._session_messages)


# ═══════════════════════════════════════════════════════════════════════
# US-262: Unbounded growth fixes
# ═══════════════════════════════════════════════════════════════════════


class TestUS262UnboundedGrowth:
    """US-262: Training buffer cap and message history cap."""

    def test_training_buffer_capped(self, tmp_path):
        """readiness_v2 training buffer is capped at MAX_BUFFER_SIZE."""
        model = ReadinessModelV2(
            model_path=tmp_path / "model.json",
            min_samples=99999,  # Prevent auto-train
        )

        # Add more samples than MAX_BUFFER_SIZE
        for i in range(model.MAX_BUFFER_SIZE + 100):
            model.add_training_sample(
                {
                    "emotional_state": 0.5,
                    "cognitive_load": 0.5,
                    "override_discipline": 0.5,
                    "stress_level": 0.5,
                    "confidence_trend": 0.5,
                    "engagement": 0.5,
                },
                trading_outcome_quality=0.5,
            )

        assert len(model._training_buffer) <= model.MAX_BUFFER_SIZE

    def test_training_buffer_keeps_newest(self, tmp_path):
        """Buffer cap drops oldest samples, keeps newest."""
        model = ReadinessModelV2(
            model_path=tmp_path / "model.json",
            min_samples=99999,
        )

        # Add samples with identifiable engagement values
        for i in range(model.MAX_BUFFER_SIZE + 10):
            model.add_training_sample(
                {
                    "emotional_state": 0.5,
                    "cognitive_load": 0.5,
                    "override_discipline": 0.5,
                    "stress_level": 0.5,
                    "confidence_trend": 0.5,
                    "engagement": float(i) / 1000.0,  # Identifiable
                },
                trading_outcome_quality=0.5,
            )

        # Last sample should have the highest engagement
        last_engagement = model._training_buffer[-1].engagement
        expected = float(model.MAX_BUFFER_SIZE + 9) / 1000.0
        assert abs(last_engagement - expected) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# US-260: Exception handling specificity
# ═══════════════════════════════════════════════════════════════════════


class TestUS260ExceptionHandling:
    """US-260: Verify exception handlers are specific, not bare."""

    def test_readiness_write_signal_no_duplicate_handler(self):
        """readiness.py _write_signal has no duplicate exception handler."""
        import inspect
        from aura.core.readiness import ReadinessComputer
        source = inspect.getsource(ReadinessComputer._write_signal)

        # Count "except" occurrences — should have exactly 3 (ImportError, inner, outer)
        except_count = source.count("except ")
        # Old code had unreachable 4th handler; new code should have 3
        assert except_count == 3, f"Expected 3 except clauses, got {except_count}"

        # Should NOT have bare "except Exception" — should be specific
        assert "except Exception" not in source, "Still has bare except Exception"

    def test_conversation_processor_has_max_constants(self):
        """ConversationProcessor has MAX_MESSAGE_LENGTH and MAX_SESSION_MESSAGES."""
        assert hasattr(ConversationProcessor, "MAX_MESSAGE_LENGTH")
        assert hasattr(ConversationProcessor, "MAX_SESSION_MESSAGES")
        assert ConversationProcessor.MAX_MESSAGE_LENGTH > 0
        assert ConversationProcessor.MAX_SESSION_MESSAGES > 0


# ═══════════════════════════════════════════════════════════════════════
# US-264: Companion response type safety
# ═══════════════════════════════════════════════════════════════════════


class TestUS264TypeSafety:
    """US-264: Type-safe outcome access in companion response generation."""

    def test_generate_response_source_inspection(self):
        """_generate_response uses hasattr guard on outcome.streak."""
        import inspect
        from aura.cli.companion import AuraCompanion
        source = inspect.getsource(AuraCompanion._generate_response)
        assert "hasattr(outcome" in source, "Missing hasattr guard on outcome"

    def test_override_predictor_context_sources_from_bridge(self):
        """Override predictor context uses bridge outcome when available."""
        import inspect
        from aura.cli.companion import AuraCompanion
        source = inspect.getsource(AuraCompanion._generate_response)
        # Should reference bridge.read_outcome() for context sourcing
        assert "bridge.read_outcome()" in source or "_outcome" in source


# ═══════════════════════════════════════════════════════════════════════
# US-263: Cloud fallback observability
# ═══════════════════════════════════════════════════════════════════════


class TestUS263Observability:
    """US-263: Fallback methods log their activation."""

    def test_fallback_methods_have_logging(self):
        """All local fallback methods contain US-263 log statements."""
        import inspect
        from aura.patterns.cloud_fallback import CloudPatternSynthesizer
        for method_name in [
            "_local_fallback_explanation",
            "_local_fallback_connections",
            "_local_fallback_override_risk",
        ]:
            method = getattr(CloudPatternSynthesizer, method_name)
            source = inspect.getsource(method)
            assert "US-263" in source, f"{method_name} missing US-263 logging"

    def test_rate_limit_persistence_has_logging(self):
        """_save_daily_count and _load_daily_count have US-263 logging."""
        import inspect
        from aura.patterns.cloud_fallback import CloudPatternSynthesizer
        for method_name in ["_save_daily_count", "_load_daily_count"]:
            source = inspect.getsource(getattr(CloudPatternSynthesizer, method_name))
            assert "US-263" in source, f"{method_name} missing US-263 logging"
