"""Tests for the Pattern DSL Engine (src/aura/evolution/dsl.py).

Covers:
  - DSLEvaluator: all operators, var resolution, depth limit
  - DSLValidator: field references, range checks, arity, depth, node count
  - PatternSpec: serialization roundtrip
  - build_context_from_signals: mock signals to context mapping
  - SEED_PATTERNS: evaluation against sample contexts
"""

import json
import pytest
from datetime import datetime, timedelta, timezone

from src.aura.evolution.dsl import (
    DSLEvaluationError,
    DSLEvaluator,
    DSLValidator,
    FIELD_REGISTRY,
    PatternSpec,
    SEED_PATTERNS,
    _VALID_FIELDS,
    build_context_from_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return DSLEvaluator()


@pytest.fixture
def validator():
    return DSLValidator()


@pytest.fixture
def sample_context():
    """A realistic context dict for testing pattern evaluation."""
    return {
        "current": {
            "emotional_state": "stressed",
            "stress_score": 0.8,
            "readiness_score": 35.0,
            "cognitive_load": "high",
            "confidence_trend": "falling",
            "tilt_score": 0.7,
            "fatigue_score": 0.6,
            "decision_quality": 30.0,
            "valence": -0.5,
            "arousal": 0.8,
        },
        "biases": {
            "disposition_effect": 0.2,
            "loss_aversion": 0.6,
            "recency_bias": 0.3,
            "confirmation_bias": 0.7,
            "sunk_cost": 0.1,
            "anchoring": 0.55,
            "overconfidence": 0.4,
            "hindsight": 0.1,
            "attribution_error": 0.05,
        },
        "outcome": {
            "pnl_today": -150.0,
            "win_rate_7d": 0.3,
            "streak": "losing",
        },
        "overrides": {
            "count_1h": 2,
            "count_24h": 5,
            "count_7d": 12,
            "loss_rate_7d": 0.65,
        },
    }


@pytest.fixture
def calm_context():
    """A calm/healthy context where most patterns should NOT fire."""
    return {
        "current": {
            "emotional_state": "calm",
            "stress_score": 0.2,
            "readiness_score": 85.0,
            "cognitive_load": "low",
            "confidence_trend": "rising",
            "tilt_score": 0.05,
            "fatigue_score": 0.1,
            "decision_quality": 75.0,
            "valence": 0.6,
            "arousal": 0.3,
        },
        "biases": {
            "disposition_effect": 0.0,
            "loss_aversion": 0.1,
            "recency_bias": 0.05,
            "confirmation_bias": 0.1,
            "sunk_cost": 0.0,
            "anchoring": 0.05,
            "overconfidence": 0.1,
            "hindsight": 0.0,
            "attribution_error": 0.0,
        },
        "outcome": {
            "pnl_today": 200.0,
            "win_rate_7d": 0.7,
            "streak": "winning",
        },
        "overrides": {
            "count_1h": 0,
            "count_24h": 1,
            "count_7d": 2,
            "loss_rate_7d": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# DSLEvaluator -- operator tests
# ---------------------------------------------------------------------------

class TestDSLEvaluatorOperators:
    """Test each operator supported by DSLEvaluator."""

    def test_gt_true(self, evaluator):
        ctx = {"current": {"stress_score": 0.8}}
        assert evaluator.evaluate({">": [{"var": "current.stress_score"}, 0.5]}, ctx) is True

    def test_gt_false(self, evaluator):
        ctx = {"current": {"stress_score": 0.3}}
        assert evaluator.evaluate({">": [{"var": "current.stress_score"}, 0.5]}, ctx) is False

    def test_gte_boundary(self, evaluator):
        ctx = {"x": 5}
        assert evaluator.evaluate({">=": [{"var": "x"}, 5]}, ctx) is True
        assert evaluator.evaluate({">=": [{"var": "x"}, 6]}, ctx) is False

    def test_lt_true(self, evaluator):
        ctx = {"current": {"readiness_score": 30}}
        assert evaluator.evaluate({"<": [{"var": "current.readiness_score"}, 40]}, ctx) is True

    def test_lte_boundary(self, evaluator):
        ctx = {"x": 5}
        assert evaluator.evaluate({"<=": [{"var": "x"}, 5]}, ctx) is True
        assert evaluator.evaluate({"<=": [{"var": "x"}, 4]}, ctx) is False

    def test_eq_string(self, evaluator):
        ctx = {"current": {"emotional_state": "calm"}}
        assert evaluator.evaluate(
            {"==": [{"var": "current.emotional_state"}, "calm"]}, ctx
        ) is True
        assert evaluator.evaluate(
            {"==": [{"var": "current.emotional_state"}, "stressed"]}, ctx
        ) is False

    def test_eq_numeric(self, evaluator):
        ctx = {"x": 42}
        assert evaluator.evaluate({"==": [{"var": "x"}, 42]}, ctx) is True

    def test_ne(self, evaluator):
        ctx = {"current": {"emotional_state": "stressed"}}
        assert evaluator.evaluate(
            {"!=": [{"var": "current.emotional_state"}, "calm"]}, ctx
        ) is True
        assert evaluator.evaluate(
            {"!=": [{"var": "current.emotional_state"}, "stressed"]}, ctx
        ) is False

    def test_and_all_true(self, evaluator):
        ctx = {"a": 10, "b": 20}
        cond = {"and": [{">": [{"var": "a"}, 5]}, {">": [{"var": "b"}, 15]}]}
        assert evaluator.evaluate(cond, ctx) is True

    def test_and_one_false(self, evaluator):
        ctx = {"a": 3, "b": 20}
        cond = {"and": [{">": [{"var": "a"}, 5]}, {">": [{"var": "b"}, 15]}]}
        assert evaluator.evaluate(cond, ctx) is False

    def test_or_one_true(self, evaluator):
        ctx = {"a": 3, "b": 20}
        cond = {"or": [{">": [{"var": "a"}, 5]}, {">": [{"var": "b"}, 15]}]}
        assert evaluator.evaluate(cond, ctx) is True

    def test_or_none_true(self, evaluator):
        ctx = {"a": 3, "b": 10}
        cond = {"or": [{">": [{"var": "a"}, 5]}, {">": [{"var": "b"}, 15]}]}
        assert evaluator.evaluate(cond, ctx) is False

    def test_not_true(self, evaluator):
        ctx = {"x": 3}
        assert evaluator.evaluate({"not": [{">": [{"var": "x"}, 5]}]}, ctx) is True

    def test_not_false(self, evaluator):
        ctx = {"x": 10}
        assert evaluator.evaluate({"not": [{">": [{"var": "x"}, 5]}]}, ctx) is False

    def test_in_list(self, evaluator):
        ctx = {"current": {"emotional_state": "anxious"}}
        cond = {"in": [{"var": "current.emotional_state"}, ["stressed", "anxious", "fatigued"]]}
        assert evaluator.evaluate(cond, ctx) is True

    def test_in_list_miss(self, evaluator):
        ctx = {"current": {"emotional_state": "calm"}}
        cond = {"in": [{"var": "current.emotional_state"}, ["stressed", "anxious"]]}
        assert evaluator.evaluate(cond, ctx) is False

    def test_in_string(self, evaluator):
        ctx = {"msg": "hello world"}
        assert evaluator.evaluate({"in": ["world", {"var": "msg"}]}, ctx) is True

    def test_empty_condition(self, evaluator):
        assert evaluator.evaluate({}, {"x": 1}) is False


# ---------------------------------------------------------------------------
# DSLEvaluator -- var resolution
# ---------------------------------------------------------------------------

class TestVarResolution:
    """Test variable path resolution with nested dicts."""

    def test_single_level(self, evaluator):
        ctx = {"x": 42}
        assert evaluator.evaluate({"==": [{"var": "x"}, 42]}, ctx) is True

    def test_two_levels(self, evaluator):
        ctx = {"current": {"tilt_score": 0.8}}
        assert evaluator.evaluate(
            {">": [{"var": "current.tilt_score"}, 0.5]}, ctx
        ) is True

    def test_three_levels(self, evaluator):
        ctx = {"a": {"b": {"c": 99}}}
        assert evaluator.evaluate({"==": [{"var": "a.b.c"}, 99]}, ctx) is True

    def test_missing_path_returns_none(self, evaluator):
        ctx = {"current": {"tilt_score": 0.8}}
        # Comparing None > 0.5 should return False (not crash)
        assert evaluator.evaluate(
            {">": [{"var": "current.nonexistent"}, 0.5]}, ctx
        ) is False

    def test_var_with_default(self, evaluator):
        ctx = {"a": 1}
        result = evaluator._eval_node({"var": ["missing", 42]}, ctx, 0)
        assert result == 42

    def test_var_without_default_missing(self, evaluator):
        ctx = {"a": 1}
        result = evaluator._eval_node({"var": "missing"}, ctx, 0)
        assert result is None


# ---------------------------------------------------------------------------
# DSLEvaluator -- depth enforcement
# ---------------------------------------------------------------------------

class TestDepthEnforcement:
    """Test that excessive nesting is caught."""

    def test_max_depth_raises(self, evaluator):
        # Build a deeply nested condition: not(not(not(...(> x 0)...)))
        node = {">": [{"var": "x"}, 0]}
        for _ in range(25):
            node = {"not": [node]}

        with pytest.raises(DSLEvaluationError, match="depth exceeded"):
            evaluator.evaluate(node, {"x": 1})

    def test_within_depth_limit(self, evaluator):
        # 10 levels should be fine (MAX_DEPTH = 20)
        node = {">": [{"var": "x"}, 0]}
        for _ in range(10):
            node = {"not": [node]}

        # 10 nots around > means result flips: even number of nots = True
        result = evaluator.evaluate(node, {"x": 1})
        assert result is True  # 10 nots = even, so back to True

    def test_unknown_operator_raises(self, evaluator):
        with pytest.raises(DSLEvaluationError, match="Unknown operator"):
            evaluator.evaluate({"foobar": [1, 2]}, {"x": 1})


# ---------------------------------------------------------------------------
# DSLValidator tests
# ---------------------------------------------------------------------------

class TestDSLValidator:
    """Test static validation of PatternSpec condition trees."""

    def test_valid_spec(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={">": [{"var": "current.stress_score"}, 0.5]},
        )
        errors = validator.validate(spec)
        assert errors == []

    def test_missing_rule_id(self, validator):
        spec = PatternSpec(
            rule_id="",
            condition={">": [{"var": "current.stress_score"}, 0.5]},
        )
        errors = validator.validate(spec)
        assert any("rule_id" in e for e in errors)

    def test_empty_condition(self, validator):
        spec = PatternSpec(rule_id="test", condition={})
        errors = validator.validate(spec)
        assert any("condition" in e for e in errors)

    def test_bad_var_reference(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={">": [{"var": "current.nonexistent_field"}, 0.5]},
        )
        errors = validator.validate(spec)
        assert any("Unknown field" in e for e in errors)

    def test_out_of_range_threshold_above(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={">": [{"var": "current.stress_score"}, 1.5]},
        )
        errors = validator.validate(spec)
        assert any("above maximum" in e for e in errors)

    def test_out_of_range_threshold_below(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={">": [{"var": "current.stress_score"}, -0.5]},
        )
        errors = validator.validate(spec)
        assert any("below minimum" in e for e in errors)

    def test_invalid_enum_value(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={"==": [{"var": "current.emotional_state"}, "furious"]},
        )
        errors = validator.validate(spec)
        assert any("not a valid enum" in e for e in errors)

    def test_wrong_arity_comparison(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={">": [{"var": "current.stress_score"}]},
        )
        errors = validator.validate(spec)
        assert any("2 arguments" in e for e in errors)

    def test_wrong_arity_not(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={"not": [
                {">": [{"var": "current.stress_score"}, 0.5]},
                {"<": [{"var": "current.stress_score"}, 0.9]},
            ]},
        )
        errors = validator.validate(spec)
        assert any("1 argument" in e for e in errors)

    def test_unknown_operator(self, validator):
        spec = PatternSpec(
            rule_id="test",
            condition={"xor": [True, False]},
        )
        errors = validator.validate(spec)
        assert any("Unknown operator" in e for e in errors)

    def test_max_depth_exceeded(self, validator):
        # Build 12-deep nesting (limit is 10)
        node = {">": [{"var": "current.stress_score"}, 0.5]}
        for _ in range(12):
            node = {"not": [node]}

        spec = PatternSpec(rule_id="test", condition=node)
        errors = validator.validate(spec)
        assert any("max depth" in e.lower() for e in errors)

    def test_max_nodes_exceeded(self, validator):
        # Build a wide or with 60 comparisons (limit is 50 nodes)
        comparisons = [
            {">": [{"var": "current.stress_score"}, 0.1 + i * 0.01]}
            for i in range(30)
        ]
        spec = PatternSpec(
            rule_id="test",
            condition={"or": comparisons},
        )
        errors = validator.validate(spec)
        assert any("nodes" in e.lower() and "max" in e.lower() for e in errors)

    def test_seed_patterns_all_valid(self, validator):
        """Every seed pattern should pass validation."""
        for pattern in SEED_PATTERNS:
            errors = validator.validate(pattern)
            assert errors == [], f"Seed pattern {pattern.rule_id} has errors: {errors}"


# ---------------------------------------------------------------------------
# PatternSpec serialization roundtrip
# ---------------------------------------------------------------------------

class TestPatternSpecSerialization:
    """Test PatternSpec to_dict/from_dict/to_json roundtrip."""

    def test_roundtrip_dict(self):
        original = PatternSpec(
            rule_id="test_rt",
            version=2,
            generation=3,
            lineage=["parent_a", "parent_b"],
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"target": "readiness_score", "direction": "down", "magnitude": "high"},
            action={"type": "penalty", "adjustment": -10},
            meta={"description": "test", "tier": "t1_daily"},
            fitness={"precision": 0.8, "recall": 0.7},
        )

        d = original.to_dict()
        restored = PatternSpec.from_dict(d)

        assert restored.rule_id == original.rule_id
        assert restored.version == original.version
        assert restored.generation == original.generation
        assert restored.lineage == original.lineage
        assert restored.condition == original.condition
        assert restored.prediction == original.prediction
        assert restored.action == original.action
        assert restored.meta == original.meta
        assert restored.fitness == original.fitness
        assert restored.created_at == original.created_at

    def test_roundtrip_json(self):
        original = PatternSpec(
            rule_id="json_rt",
            condition={"and": [{">": [{"var": "current.stress_score"}, 0.5]}, {"<": [{"var": "current.readiness_score"}, 60]}]},
        )
        json_str = original.to_json()
        data = json.loads(json_str)
        restored = PatternSpec.from_dict(data)
        assert restored.rule_id == "json_rt"
        assert restored.condition == original.condition

    def test_created_at_auto_set(self):
        spec = PatternSpec(rule_id="auto_ts")
        assert spec.created_at != ""
        # Should be parseable as ISO datetime
        datetime.fromisoformat(spec.created_at.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# build_context_from_signals
# ---------------------------------------------------------------------------

class TestBuildContextFromSignals:
    """Test context building from mock signal objects."""

    def test_with_dict_signals(self):
        readiness = {
            "readiness_score": 65.0,
            "emotional_state": "anxious",
            "cognitive_load": "medium",
            "confidence_trend": "falling",
            "tilt_score": 0.3,
            "fatigue_score": 0.2,
            "decision_quality_score": 55.0,
            "bias_scores": {"confirmation_bias": 0.4, "anchoring": 0.3},
        }
        signals = {
            "sentiment_score": 0.3,
            "affect_valence": -0.2,
            "affect_arousal": 0.5,
            "bias_scores": {
                "confirmation_bias": 0.45,
                "loss_aversion": 0.2,
            },
        }
        outcome = {
            "pnl_today": -50.0,
            "win_rate_7d": 0.4,
            "streak": "losing",
        }

        ctx = build_context_from_signals(readiness, signals, outcome)

        assert ctx["current"]["emotional_state"] == "anxious"
        assert ctx["current"]["readiness_score"] == 65.0
        assert ctx["current"]["stress_score"] == pytest.approx(0.7, abs=0.01)
        assert ctx["current"]["valence"] == -0.2
        assert ctx["current"]["arousal"] == 0.5
        assert ctx["outcome"]["pnl_today"] == -50.0
        assert ctx["outcome"]["streak"] == "losing"
        # bias_scores from signals takes priority
        assert ctx["biases"]["confirmation_bias"] == 0.45

    def test_override_counting(self):
        now = datetime.now(timezone.utc)
        overrides = [
            {"timestamp": (now - timedelta(minutes=30)).isoformat(), "outcome": "loss"},
            {"timestamp": (now - timedelta(hours=2)).isoformat(), "outcome": "win"},
            {"timestamp": (now - timedelta(hours=12)).isoformat(), "outcome": "loss"},
            {"timestamp": (now - timedelta(days=3)).isoformat(), "outcome": "loss"},
            {"timestamp": (now - timedelta(days=10)).isoformat(), "outcome": "loss"},
        ]

        ctx = build_context_from_signals(
            readiness={"readiness_score": 50},
            signals={"sentiment_score": 0.5},
            overrides=overrides,
        )

        assert ctx["overrides"]["count_1h"] == 1
        assert ctx["overrides"]["count_24h"] == 3
        assert ctx["overrides"]["count_7d"] == 4
        # 3 losses out of 4 within 7d
        assert ctx["overrides"]["loss_rate_7d"] == pytest.approx(0.75, abs=0.01)

    def test_none_signals_dont_crash(self):
        ctx = build_context_from_signals(
            readiness=None,
            signals=None,
            outcome=None,
            overrides=None,
        )
        assert ctx["current"]["readiness_score"] == 50.0
        assert ctx["current"]["emotional_state"] == "neutral"
        assert ctx["overrides"]["count_1h"] == 0

    def test_context_has_all_registry_paths(self):
        """Every FIELD_REGISTRY path should be resolvable in the context."""
        ctx = build_context_from_signals(
            readiness={"readiness_score": 50},
            signals={"sentiment_score": 0.5},
        )

        evaluator = DSLEvaluator()
        for field_path in _VALID_FIELDS:
            # Resolve var and ensure it's not None
            result = evaluator._eval_node({"var": field_path}, ctx, 0)
            assert result is not None, f"Field {field_path} resolved to None in context"


# ---------------------------------------------------------------------------
# SEED_PATTERNS evaluation against sample contexts
# ---------------------------------------------------------------------------

class TestSeedPatternEvaluation:
    """Test that seed patterns fire/don't fire against known contexts."""

    def test_stress_override_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_stress_override")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_stress_override_silent_when_calm(self, evaluator, calm_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_stress_override")
        assert evaluator.evaluate(pattern.condition, calm_context) is False

    def test_emotional_drift_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_emotional_drift")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_emotional_drift_silent_when_calm(self, evaluator, calm_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_emotional_drift")
        assert evaluator.evaluate(pattern.condition, calm_context) is False

    def test_overtrade_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_overtrade")
        # count_24h=5 < 8 but let's check: count_1h=2 < 3, so neither fires
        # Actually both thresholds are not met for this sample
        # count_1h >= 3 ? no (2 < 3). count_24h >= 8 ? no (5 < 8). So False.
        assert evaluator.evaluate(pattern.condition, sample_context) is False

    def test_overtrade_fires_with_high_counts(self, evaluator):
        ctx = {"overrides": {"count_1h": 4, "count_24h": 10, "count_7d": 20, "loss_rate_7d": 0.5}}
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_overtrade")
        assert evaluator.evaluate(pattern.condition, ctx) is True

    def test_tilt_spiral_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_tilt_spiral")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_tilt_spiral_silent_when_calm(self, evaluator, calm_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_tilt_spiral")
        assert evaluator.evaluate(pattern.condition, calm_context) is False

    def test_bias_compound_fires(self, evaluator, sample_context):
        """Sample context has confirmation_bias=0.7, loss_aversion=0.6, anchoring=0.55 > 0.5."""
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_bias_compound")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_bias_compound_silent_when_calm(self, evaluator, calm_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_bias_compound")
        assert evaluator.evaluate(pattern.condition, calm_context) is False

    def test_impulsive_risk_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_impulsive_risk")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_override_loss_spiral_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_override_loss_spiral")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_cognitive_fatigue_convergence_fires(self, evaluator, sample_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_cognitive_fatigue_convergence")
        assert evaluator.evaluate(pattern.condition, sample_context) is True

    def test_cognitive_fatigue_convergence_silent_when_calm(self, evaluator, calm_context):
        pattern = next(p for p in SEED_PATTERNS if p.rule_id == "seed_cognitive_fatigue_convergence")
        assert evaluator.evaluate(pattern.condition, calm_context) is False

    def test_all_seeds_have_unique_ids(self):
        ids = [p.rule_id for p in SEED_PATTERNS]
        assert len(ids) == len(set(ids)), f"Duplicate seed IDs: {ids}"

    def test_seed_count(self):
        assert len(SEED_PATTERNS) >= 5


# ---------------------------------------------------------------------------
# FIELD_REGISTRY integrity
# ---------------------------------------------------------------------------

class TestFieldRegistry:
    """Test that FIELD_REGISTRY is well-formed."""

    def test_all_fields_have_type(self):
        for path, reg in FIELD_REGISTRY.items():
            assert "type" in reg, f"Field {path} missing 'type'"
            assert reg["type"] in ("float", "int", "enum"), f"Field {path} has unknown type {reg['type']}"

    def test_enum_fields_have_values(self):
        for path, reg in FIELD_REGISTRY.items():
            if reg["type"] == "enum":
                assert "values" in reg, f"Enum field {path} missing 'values'"
                assert len(reg["values"]) > 0, f"Enum field {path} has empty values"

    def test_numeric_fields_have_range(self):
        for path, reg in FIELD_REGISTRY.items():
            if reg["type"] in ("float", "int"):
                assert "min" in reg, f"Numeric field {path} missing 'min'"
                assert "max" in reg, f"Numeric field {path} missing 'max'"
