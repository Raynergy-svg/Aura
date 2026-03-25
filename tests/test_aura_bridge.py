"""Unit tests for Aura Bridge subsystem — US-211.

Tests cover:
  FeedbackBridge:
    1. Locked write/read round-trip
    2. Locked read on missing file returns None
    3. Locked append creates and appends
    4. OutcomeSignal write/read
    5. OverrideEvent log/read

  BridgeRule:
    6. Default TTL is ~14 days
    7. Expired rule detection
    8. Malformed expires_at treated as expired (US-206)
    9. Empty expires_at treated as expired (US-206)
    10. extend_ttl refreshes expiry

  BridgeRulesEngine:
    11. Operator precedence: multiply → add → set (US-210)
    12. Value clamping (US-210)
    13. Rule creation from pattern
    14. Stale rule expiry
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.bridge.signals import (
    FeedbackBridge,
    OutcomeSignal,
    OverrideEvent,
)
from src.aura.bridge.rules_engine import (
    BridgeRule,
    BridgeRulesEngine,
)


# ═══════════════════════════════════════════════════════════════════════════
# FeedbackBridge tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bridge(tmp_path):
    return FeedbackBridge(bridge_dir=tmp_path / "bridge")


# ── 1. Locked write/read round-trip ──────────────────────────────────────

def test_locked_write_read_roundtrip(tmp_path):
    path = tmp_path / "test.json"
    data = json.dumps({"key": "value", "number": 42})
    FeedbackBridge._locked_write(path, data)
    result = FeedbackBridge._locked_read(path)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["key"] == "value"
    assert parsed["number"] == 42


# ── 2. Locked read on missing file ───────────────────────────────────────

def test_locked_read_missing_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    result = FeedbackBridge._locked_read(path)
    assert result is None


# ── 3. Locked append ────────────────────────────────────────────────────

def test_locked_append(tmp_path):
    path = tmp_path / "append_test.jsonl"
    FeedbackBridge._locked_append(path, '{"line": 1}\n')
    FeedbackBridge._locked_append(path, '{"line": 2}\n')
    content = path.read_text()
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["line"] == 1
    assert json.loads(lines[1])["line"] == 2


# ── 4. OutcomeSignal write/read ──────────────────────────────────────────

def test_outcome_signal_roundtrip(bridge):
    signal = OutcomeSignal(
        pnl_today=150.50,
        win_rate_7d=0.65,
        regime="TRENDING",
        streak="winning",
        trades_today=3,
    )
    bridge.write_outcome(signal)
    result = bridge.read_outcome()
    assert result is not None
    assert result.pnl_today == 150.50
    assert result.win_rate_7d == 0.65
    assert result.regime == "TRENDING"
    assert result.streak == "winning"
    assert result.trades_today == 3


# ── 5. OverrideEvent log/read ───────────────────────────────────────────

def test_override_log_and_read(bridge):
    event1 = OverrideEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        pair="EUR/USD",
        override_type="took_rejected",
        buddy_recommendation="skip",
        trader_action="buy",
        outcome="loss",
        pnl_pips=-25.0,
    )
    event2 = OverrideEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        pair="GBP/USD",
        override_type="closed_early",
        buddy_recommendation="hold",
        trader_action="close",
        outcome="win",
        pnl_pips=10.0,
    )
    bridge.log_override(event1)
    bridge.log_override(event2)
    overrides = bridge.get_recent_overrides(limit=10)
    assert len(overrides) == 2
    assert overrides[0].pair == "EUR/USD"
    assert overrides[1].pair == "GBP/USD"
    assert overrides[0].pnl_pips == -25.0


# ═══════════════════════════════════════════════════════════════════════════
# BridgeRule tests
# ═══════════════════════════════════════════════════════════════════════════

# ── 6. Default TTL ──────────────────────────────────────────────────────

def test_default_ttl():
    rule = BridgeRule(
        rule_id="test-1",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Test rule",
        adjustment={"parameter": "min_confidence_threshold", "value": 0.7, "operator": "set"},
    )
    created = datetime.fromisoformat(rule.created_at)
    expires = datetime.fromisoformat(rule.expires_at)
    delta = expires - created
    # Should be approximately 14 days
    assert 13 <= delta.days <= 15


# ── 7. Expired rule detection ───────────────────────────────────────────

def test_rule_expired():
    rule = BridgeRule(
        rule_id="test-2",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Test expired",
        adjustment={"parameter": "x", "value": 1, "operator": "set"},
        expires_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )
    assert rule.is_expired() is True


def test_rule_not_expired():
    rule = BridgeRule(
        rule_id="test-3",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Test active",
        adjustment={"parameter": "x", "value": 1, "operator": "set"},
        expires_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
    )
    assert rule.is_expired() is False


# ── 8. Malformed expires_at → expired (US-206) ─────────────────────────

def test_malformed_expires_at_treated_as_expired():
    rule = BridgeRule(
        rule_id="test-malformed",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Malformed expiry",
        adjustment={"parameter": "x", "value": 1, "operator": "set"},
        expires_at="not-a-date-at-all",
    )
    assert rule.is_expired() is True


# ── 9. Empty expires_at → expired (US-206) ──────────────────────────────

def test_empty_expires_at_treated_as_expired():
    rule = BridgeRule(
        rule_id="test-empty",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Empty expiry",
        adjustment={"parameter": "x", "value": 1, "operator": "set"},
        expires_at="",
    )
    # Force empty — __post_init__ sets a default if empty, so override after
    rule.expires_at = ""
    assert rule.is_expired() is True


# ── 10. extend_ttl refreshes ────────────────────────────────────────────

def test_extend_ttl():
    rule = BridgeRule(
        rule_id="test-extend",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Extend test",
        adjustment={"parameter": "x", "value": 1, "operator": "set"},
        expires_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )
    assert rule.is_expired() is True
    rule.extend_ttl(days=14)
    assert rule.is_expired() is False


# ═══════════════════════════════════════════════════════════════════════════
# BridgeRulesEngine tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine(tmp_path):
    return BridgeRulesEngine(rules_path=tmp_path / "active_rules.json")


# ── 11. Operator precedence: multiply → add → set (US-210) ─────────────

def test_operator_precedence(engine):
    """When multiple rules target same param: multiply first, add second, set last."""
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=7)).isoformat()

    # Manually inject rules with different operators
    engine._rules = [
        BridgeRule(
            rule_id="r-set",
            rule_type="raise_min_confidence",
            direction="aura_to_buddy",
            description="Set rule",
            adjustment={"parameter": "min_confidence_threshold", "value": 0.80, "operator": "set"},
            expires_at=future,
        ),
        BridgeRule(
            rule_id="r-add",
            rule_type="reduce_position_risk",
            direction="aura_to_buddy",
            description="Add rule",
            adjustment={"parameter": "min_confidence_threshold", "value": 0.05, "operator": "add"},
            expires_at=future,
        ),
        BridgeRule(
            rule_id="r-multiply",
            rule_type="tighten_rr_ratio",
            direction="aura_to_buddy",
            description="Multiply rule",
            adjustment={"parameter": "min_confidence_threshold", "value": 1.5, "operator": "multiply"},
            expires_at=future,
        ),
    ]

    adjustments = engine.get_buddy_gate_adjustments()
    # Order: multiply first (no existing → skip), add second (sets to 0.05),
    # wait — multiply on nonexistent is skipped. add sets 0.05. set overwrites to 0.80.
    # Actually: multiply requires existing value so it's skipped.
    # add with no existing value: sets 0.05
    # set: overwrites to 0.80
    # Final: 0.80
    assert adjustments["min_confidence_threshold"] == 0.80


# ── 12. Value clamping (US-210) ─────────────────────────────────────────

def test_value_clamping(engine):
    """Values exceeding valid ranges should be clamped."""
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=7)).isoformat()

    engine._rules = [
        BridgeRule(
            rule_id="r-clamp-high",
            rule_type="raise_min_confidence",
            direction="aura_to_buddy",
            description="Too high",
            adjustment={"parameter": "min_confidence_threshold", "value": 5.0, "operator": "set"},
            expires_at=future,
        ),
    ]

    adjustments = engine.get_buddy_gate_adjustments()
    # min_confidence_threshold clamped to [0.0, 1.0]
    assert adjustments["min_confidence_threshold"] == 1.0


def test_value_clamping_low(engine):
    """Values below valid range should be clamped upward."""
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=7)).isoformat()

    engine._rules = [
        BridgeRule(
            rule_id="r-clamp-low",
            rule_type="reduce_position_risk",
            direction="aura_to_buddy",
            description="Too low multiplier",
            adjustment={"parameter": "position_size_multiplier", "value": 0.1, "operator": "set"},
            expires_at=future,
        ),
    ]

    adjustments = engine.get_buddy_gate_adjustments()
    # position_size_multiplier clamped to [0.3, 2.0]
    assert adjustments["position_size_multiplier"] == 0.3


# ── 13. Rule creation from pattern ──────────────────────────────────────

def test_create_rule_from_pattern(engine):
    rule = engine.create_rule_from_pattern(
        pattern_id="pat-001",
        pattern_description="Emotional drift detected over 3 days",
        pattern_domain="human",
        pattern_confidence=0.75,
        observation_count=4,
    )
    assert rule is not None
    assert rule.rule_type == "raise_min_confidence"
    assert rule.direction == "aura_to_buddy"
    assert rule.confidence == 0.75
    assert len(engine.get_active_rules()) == 1


def test_create_rule_reinforces_existing(engine):
    """Second pattern of same type should reinforce, not duplicate."""
    engine.create_rule_from_pattern(
        pattern_id="pat-001",
        pattern_description="Emotional drift detected",
        pattern_domain="human",
        pattern_confidence=0.70,
        observation_count=3,
    )
    engine.create_rule_from_pattern(
        pattern_id="pat-002",
        pattern_description="Emotional drift continues",
        pattern_domain="human",
        pattern_confidence=0.80,
        observation_count=2,
    )
    active = engine.get_active_rules()
    assert len(active) == 1  # Same rule, reinforced
    assert active[0].confidence > 0.70  # Confidence should have increased
    assert len(active[0].source_pattern_ids) == 2


# ── 14. Stale rule expiry ───────────────────────────────────────────────

def test_expire_stale_rules(engine):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    engine._rules = [
        BridgeRule(
            rule_id="r-active",
            rule_type="raise_min_confidence",
            direction="aura_to_buddy",
            description="Active",
            adjustment={"parameter": "x", "value": 1, "operator": "set"},
            expires_at=future,
        ),
        BridgeRule(
            rule_id="r-stale",
            rule_type="reduce_position_risk",
            direction="aura_to_buddy",
            description="Stale",
            adjustment={"parameter": "y", "value": 1, "operator": "set"},
            expires_at=past,
        ),
    ]

    expired_count = engine.expire_stale_rules()
    assert expired_count == 1
    assert len(engine.get_active_rules()) == 1
    assert engine.get_active_rules()[0].rule_id == "r-active"
