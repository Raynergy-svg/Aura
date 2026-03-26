"""Pattern DSL Engine -- JsonLogic-based declarative pattern specification.

Aura's L2 self-evolution layer. Patterns are expressed as JsonLogic condition
trees that can be evaluated against live signal data WITHOUT arbitrary code
execution (no eval/exec).

Each PatternSpec describes:
  - WHEN (condition): JsonLogic tree evaluated against signal context
  - THEN (prediction): What signal change the pattern predicts
  - DO (action): What readiness adjustment or gate to apply

Architecture:
  ConversationSignals + ReadinessSignal + OutcomeSignal + OverrideEvents
      -> build_context_from_signals() -> flat context dict
      -> DSLEvaluator.evaluate(condition, context) -> bool
      -> action applied by caller (readiness adjustment, gate, message)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. FIELD_REGISTRY -- canonical DSL field paths, descriptions, valid ranges
# ---------------------------------------------------------------------------

FIELD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -- current signal state --
    "current.emotional_state": {
        "description": "Current emotional state classification",
        "type": "enum",
        "values": ["calm", "anxious", "stressed", "energized", "fatigued", "neutral"],
    },
    "current.stress_score": {
        "description": "Aggregate stress level from conversation signals",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "current.readiness_score": {
        "description": "Composite trader readiness score",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
    },
    "current.cognitive_load": {
        "description": "Cognitive load classification",
        "type": "enum",
        "values": ["low", "medium", "high"],
    },
    "current.confidence_trend": {
        "description": "Direction of trader confidence",
        "type": "enum",
        "values": ["rising", "falling", "stable"],
    },
    "current.tilt_score": {
        "description": "Revenge/tilt trading severity",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "current.fatigue_score": {
        "description": "Decision fatigue severity",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "current.decision_quality": {
        "description": "Composite decision quality score",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
    },
    "current.valence": {
        "description": "Affect valence (-1 negative to +1 positive)",
        "type": "float",
        "min": -1.0,
        "max": 1.0,
    },
    "current.arousal": {
        "description": "Affect arousal (0 calm to 1 activated)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    # -- cognitive biases (9 types) --
    "biases.disposition_effect": {
        "description": "Tendency to sell winners too early / hold losers too long",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.loss_aversion": {
        "description": "Disproportionate focus on downside risk",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.recency_bias": {
        "description": "Overweighting recent events vs historical",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.confirmation_bias": {
        "description": "Seeking validation for existing beliefs",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.sunk_cost": {
        "description": "Reluctance to abandon losing positions due to invested cost",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.anchoring": {
        "description": "Over-reliance on initial reference price",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.overconfidence": {
        "description": "Excessive certainty in predictions",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.hindsight": {
        "description": "Hindsight bias -- believing past events were predictable",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "biases.attribution_error": {
        "description": "Misattributing outcomes to skill vs luck",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    # -- outcome data from Buddy --
    "outcome.pnl_today": {
        "description": "Today's profit/loss in base currency",
        "type": "float",
        "min": None,
        "max": None,
    },
    "outcome.win_rate_7d": {
        "description": "7-day rolling win rate",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
    "outcome.streak": {
        "description": "Current streak classification",
        "type": "enum",
        "values": ["winning", "losing", "neutral"],
    },
    # -- override statistics --
    "overrides.count_1h": {
        "description": "Override count in last 1 hour",
        "type": "int",
        "min": 0,
        "max": None,
    },
    "overrides.count_24h": {
        "description": "Override count in last 24 hours",
        "type": "int",
        "min": 0,
        "max": None,
    },
    "overrides.count_7d": {
        "description": "Override count in last 7 days",
        "type": "int",
        "min": 0,
        "max": None,
    },
    "overrides.loss_rate_7d": {
        "description": "Fraction of overrides that resulted in losses (7 days)",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    },
}

# Convenience set for fast lookup
_VALID_FIELDS = frozenset(FIELD_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 2. PatternSpec -- declarative pattern specification
# ---------------------------------------------------------------------------

@dataclass
class PatternSpec:
    """A declarative pattern specification using JsonLogic conditions.

    Attributes:
        rule_id:     Unique identifier for this pattern rule.
        version:     Schema version (for forward compatibility).
        generation:  Evolutionary generation (0 = seed, 1+ = evolved).
        lineage:     List of ancestor rule_ids this was derived from.
        condition:   JsonLogic condition tree (the WHEN clause).
        prediction:  What the pattern predicts {target, direction, magnitude}.
        action:      What to do when triggered {type, adjustment, component, message}.
        meta:        Descriptive metadata {description, tier, domain, tags}.
        fitness:     Computed during backtesting {accuracy, recall, precision, ...}.
        created_at:  ISO timestamp of creation.
    """

    rule_id: str
    version: int = 1
    generation: int = 0
    lineage: List[str] = field(default_factory=list)
    condition: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, str] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    fitness: Dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "generation": self.generation,
            "lineage": list(self.lineage),
            "condition": self.condition,
            "prediction": self.prediction,
            "action": self.action,
            "meta": dict(self.meta),
            "fitness": dict(self.fitness),
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternSpec":
        return cls(
            rule_id=data["rule_id"],
            version=data.get("version", 1),
            generation=data.get("generation", 0),
            lineage=data.get("lineage", []),
            condition=data.get("condition", {}),
            prediction=data.get("prediction", {}),
            action=data.get("action", {}),
            meta=data.get("meta", {}),
            fitness=data.get("fitness", {}),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# 3. DSLEvaluator -- safe JsonLogic evaluation engine
# ---------------------------------------------------------------------------

class DSLEvaluationError(Exception):
    """Raised when evaluation encounters an unrecoverable error."""


class DSLEvaluator:
    """Evaluate JsonLogic condition trees against a signal context dict.

    Supports: >, >=, <, <=, ==, !=, and, or, not, var, in.
    NO eval(), NO exec(), NO arbitrary code execution.
    Safe recursive evaluation with configurable depth limit.
    """

    MAX_DEPTH = 20

    def evaluate(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a JsonLogic condition tree.

        Args:
            condition: JsonLogic dict, e.g. {">": [{"var": "current.stress_score"}, 0.7]}
            context:   Nested dict of signal values.

        Returns:
            True if condition is satisfied, False otherwise.

        Raises:
            DSLEvaluationError: On malformed conditions or depth overflow.
        """
        if not condition:
            return False
        result = self._eval_node(condition, context, depth=0)
        return bool(result)

    # -- internal recursive evaluator --

    def _eval_node(self, node: Any, context: Dict[str, Any], depth: int) -> Any:
        """Recursively evaluate a JsonLogic node.

        A node is either:
          - A literal (int, float, str, bool, None, list)
          - A JsonLogic operation dict with exactly one key (the operator)
        """
        if depth > self.MAX_DEPTH:
            raise DSLEvaluationError(
                f"Evaluation depth exceeded maximum of {self.MAX_DEPTH}"
            )

        # Literals pass through
        if not isinstance(node, dict):
            return node

        if len(node) == 0:
            return False

        if len(node) != 1:
            raise DSLEvaluationError(
                f"JsonLogic node must have exactly 1 operator key, got {len(node)}: {list(node.keys())}"
            )

        operator = next(iter(node))
        args = node[operator]

        # Dispatch to operator handlers
        handler = self._OPERATORS.get(operator)
        if handler is None:
            raise DSLEvaluationError(f"Unknown operator: {operator!r}")

        return handler(self, args, context, depth)

    # -- operator implementations --

    def _op_var(self, args: Any, context: Dict[str, Any], depth: int) -> Any:
        """Resolve a variable path like 'current.stress_score' from context."""
        if isinstance(args, list):
            path = args[0] if args else ""
            default = args[1] if len(args) > 1 else None
        else:
            path = args
            default = None

        parts = str(path).split(".")
        current = context
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                if default is not None:
                    return default
                return None
        return current

    def _op_gt(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'>' requires 2 arguments, got {len(resolved)}")
        a, b = resolved
        if a is None or b is None:
            return False
        return float(a) > float(b)

    def _op_gte(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'>=' requires 2 arguments, got {len(resolved)}")
        a, b = resolved
        if a is None or b is None:
            return False
        return float(a) >= float(b)

    def _op_lt(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'<' requires 2 arguments, got {len(resolved)}")
        a, b = resolved
        if a is None or b is None:
            return False
        return float(a) < float(b)

    def _op_lte(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'<=' requires 2 arguments, got {len(resolved)}")
        a, b = resolved
        if a is None or b is None:
            return False
        return float(a) <= float(b)

    def _op_eq(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'==' requires 2 arguments, got {len(resolved)}")
        return resolved[0] == resolved[1]

    def _op_ne(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'!=' requires 2 arguments, got {len(resolved)}")
        return resolved[0] != resolved[1]

    def _op_and(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        if not isinstance(args, list):
            raise DSLEvaluationError("'and' requires a list of conditions")
        for item in args:
            if not self._eval_node(item, context, depth + 1):
                return False
        return True

    def _op_or(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        if not isinstance(args, list):
            raise DSLEvaluationError("'or' requires a list of conditions")
        for item in args:
            if self._eval_node(item, context, depth + 1):
                return True
        return False

    def _op_not(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        if isinstance(args, list):
            if len(args) != 1:
                raise DSLEvaluationError(
                    f"'not' requires exactly 1 argument, got {len(args)}"
                )
            return not self._eval_node(args[0], context, depth + 1)
        return not self._eval_node(args, context, depth + 1)

    def _op_in(self, args: Any, context: Dict[str, Any], depth: int) -> bool:
        """Check if value is contained in a list or string."""
        resolved = self._resolve_args(args, context, depth)
        if len(resolved) != 2:
            raise DSLEvaluationError(f"'in' requires 2 arguments, got {len(resolved)}")
        value, collection = resolved
        if collection is None:
            return False
        if isinstance(collection, (list, tuple)):
            return value in collection
        if isinstance(collection, str):
            return str(value) in collection
        return False

    # -- helper --

    def _resolve_args(
        self, args: Any, context: Dict[str, Any], depth: int
    ) -> List[Any]:
        """Resolve a list of JsonLogic argument nodes to their values."""
        if not isinstance(args, list):
            args = [args]
        return [self._eval_node(a, context, depth + 1) for a in args]

    # Operator dispatch table
    _OPERATORS: Dict[str, Any] = {
        "var": _op_var,
        ">": _op_gt,
        ">=": _op_gte,
        "<": _op_lt,
        "<=": _op_lte,
        "==": _op_eq,
        "!=": _op_ne,
        "and": _op_and,
        "or": _op_or,
        "not": _op_not,
        "in": _op_in,
    }


# ---------------------------------------------------------------------------
# 4. DSLValidator -- static validation of PatternSpec
# ---------------------------------------------------------------------------

class DSLValidator:
    """Validate PatternSpec condition trees for well-formedness and safety.

    Checks:
      - All var references exist in FIELD_REGISTRY
      - Numeric thresholds are within valid ranges for their target field
      - Condition tree is well-formed (correct operator arities)
      - Max nesting depth <= 10
      - Max total nodes <= 50
    """

    MAX_DEPTH = 10
    MAX_NODES = 50

    def validate(self, spec: PatternSpec) -> List[str]:
        """Validate a PatternSpec. Returns list of error strings (empty = valid)."""
        errors: List[str] = []

        # Basic structure checks
        if not spec.rule_id:
            errors.append("rule_id is required")
        if not spec.condition:
            errors.append("condition is required (empty dict)")
            return errors

        # Walk the condition tree
        node_count = [0]  # mutable counter in list for nested mutation
        self._validate_node(spec.condition, errors, depth=0, node_count=node_count)

        if node_count[0] > self.MAX_NODES:
            errors.append(
                f"Condition tree has {node_count[0]} nodes, max is {self.MAX_NODES}"
            )

        return errors

    def _validate_node(
        self,
        node: Any,
        errors: List[str],
        depth: int,
        node_count: List[int],
    ) -> None:
        """Recursively validate a condition tree node."""
        node_count[0] += 1

        # Literal values are always valid
        if not isinstance(node, dict):
            return

        if depth > self.MAX_DEPTH:
            errors.append(
                f"Condition tree exceeds max depth of {self.MAX_DEPTH}"
            )
            return

        if len(node) == 0:
            return

        if len(node) != 1:
            errors.append(
                f"JsonLogic node must have exactly 1 operator, got {len(node)}: "
                f"{list(node.keys())}"
            )
            return

        operator = next(iter(node))
        args = node[operator]

        # Check operator is known
        known_ops = {
            "var", ">", ">=", "<", "<=", "==", "!=", "and", "or", "not", "in",
        }
        if operator not in known_ops:
            errors.append(f"Unknown operator: {operator!r}")
            return

        # Validate var references
        if operator == "var":
            path = args[0] if isinstance(args, list) else args
            path = str(path)
            if path not in _VALID_FIELDS:
                errors.append(f"Unknown field reference: {path!r}")
            return

        # Validate arity for comparison operators
        comparison_ops = {">", ">=", "<", "<=", "==", "!="}
        if operator in comparison_ops:
            if not isinstance(args, list) or len(args) != 2:
                errors.append(
                    f"Operator {operator!r} requires exactly 2 arguments"
                )
                return
            # Check range validity when one arg is var and the other is literal
            self._validate_range(args, errors)
            for arg in args:
                self._validate_node(arg, errors, depth + 1, node_count)
            return

        # Validate logic operators
        if operator == "not":
            if isinstance(args, list):
                if len(args) != 1:
                    errors.append(
                        f"'not' requires exactly 1 argument, got {len(args)}"
                    )
                    return
                self._validate_node(args[0], errors, depth + 1, node_count)
            else:
                self._validate_node(args, errors, depth + 1, node_count)
            return

        if operator in ("and", "or"):
            if not isinstance(args, list):
                errors.append(f"'{operator}' requires a list of conditions")
                return
            for item in args:
                self._validate_node(item, errors, depth + 1, node_count)
            return

        if operator == "in":
            if not isinstance(args, list) or len(args) != 2:
                errors.append("'in' requires exactly 2 arguments")
                return
            for arg in args:
                self._validate_node(arg, errors, depth + 1, node_count)
            return

    def _validate_range(self, args: List[Any], errors: List[str]) -> None:
        """If one arg is a var and the other is a literal, check range validity."""
        var_node = None
        literal_val = None

        for arg in args:
            if isinstance(arg, dict) and "var" in arg:
                var_path = arg["var"]
                if isinstance(var_path, list):
                    var_path = var_path[0] if var_path else ""
                var_node = str(var_path)
            elif not isinstance(arg, dict):
                literal_val = arg

        if var_node is None or literal_val is None:
            return

        if var_node not in FIELD_REGISTRY:
            return  # Unknown var is caught elsewhere

        reg = FIELD_REGISTRY[var_node]
        if reg["type"] == "enum":
            if isinstance(literal_val, str) and literal_val not in reg.get("values", []):
                errors.append(
                    f"Value {literal_val!r} is not a valid enum value for {var_node}; "
                    f"valid: {reg.get('values', [])}"
                )
            return

        if not isinstance(literal_val, (int, float)):
            return

        field_min = reg.get("min")
        field_max = reg.get("max")
        if field_min is not None and literal_val < field_min:
            errors.append(
                f"Threshold {literal_val} for {var_node} is below minimum {field_min}"
            )
        if field_max is not None and literal_val > field_max:
            errors.append(
                f"Threshold {literal_val} for {var_node} is above maximum {field_max}"
            )


# ---------------------------------------------------------------------------
# 5. build_context_from_signals -- bridge between Aura signals and DSL context
# ---------------------------------------------------------------------------

def build_context_from_signals(
    readiness: Any,
    signals: Any,
    outcome: Any = None,
    overrides: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Convert Aura's current signal state into a flat context dict for DSL evaluation.

    Args:
        readiness: ReadinessSignal object (or dict with matching keys).
        signals:   ConversationSignals object (or dict with matching keys).
        outcome:   Optional outcome signal dict from Buddy.
        overrides: Optional list of recent override event dicts.

    Returns:
        Nested dict matching FIELD_REGISTRY paths::

            {
                "current": {"stress_score": 0.7, "readiness_score": 45, ...},
                "biases": {"recency_bias": 0.3, "anchoring": 0.65, ...},
                "outcome": {"pnl_today": -25.0, ...},
                "overrides": {"count_1h": 2, ...},
            }
    """
    # Helper to get attribute or dict key
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # -- current signal state --
    # stress_score: we invert sentiment_score (0=very_negative -> stress=1.0)
    raw_sentiment = _get(signals, "sentiment_score", 0.5)
    stress_score = 1.0 - float(raw_sentiment)

    current: Dict[str, Any] = {
        "emotional_state": _get(readiness, "emotional_state", "neutral"),
        "stress_score": round(stress_score, 4),
        "readiness_score": float(_get(readiness, "readiness_score", 50.0)),
        "cognitive_load": _get(readiness, "cognitive_load", "medium"),
        "confidence_trend": _get(readiness, "confidence_trend", "stable"),
        "tilt_score": float(_get(readiness, "tilt_score", 0.0)),
        "fatigue_score": float(_get(readiness, "fatigue_score", 0.0)),
        "decision_quality": float(
            _get(readiness, "decision_quality_score", 0.0)
        ),
        "valence": float(_get(signals, "affect_valence", 0.0)),
        "arousal": float(_get(signals, "affect_arousal", 0.0)),
    }

    # -- biases (prefer signals, fallback to readiness) --
    raw_biases = (
        _get(signals, "bias_scores", None)
        or _get(readiness, "bias_scores", None)
        or {}
    )
    if not isinstance(raw_biases, dict):
        raw_biases = {}

    bias_keys = [
        "disposition_effect",
        "loss_aversion",
        "recency_bias",
        "confirmation_bias",
        "sunk_cost",
        "anchoring",
        "overconfidence",
        "hindsight",
        "attribution_error",
    ]
    biases: Dict[str, float] = {
        k: float(raw_biases.get(k, 0.0)) for k in bias_keys
    }

    # -- outcome data --
    outcome_ctx: Dict[str, Any] = {
        "pnl_today": float(_get(outcome, "pnl_today", 0.0)),
        "win_rate_7d": float(_get(outcome, "win_rate_7d", 0.5)),
        "streak": _get(outcome, "streak", "neutral"),
    }

    # -- override statistics --
    override_list = overrides or []
    now = datetime.now(timezone.utc)

    count_1h = 0
    count_24h = 0
    count_7d = 0
    losses_7d = 0
    total_7d = 0

    for ov in override_list:
        ts_str = _get(ov, "timestamp", None)
        if ts_str is None:
            # No timestamp -- count in all windows
            count_1h += 1
            count_24h += 1
            count_7d += 1
            total_7d += 1
            continue

        try:
            if isinstance(ts_str, str):
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                ts = ts_str
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            count_7d += 1
            total_7d += 1
            continue

        delta = (now - ts).total_seconds()
        if delta <= 3600:
            count_1h += 1
        if delta <= 86400:
            count_24h += 1
        if delta <= 604800:
            count_7d += 1
            total_7d += 1
            if _get(ov, "outcome", None) == "loss":
                losses_7d += 1

    loss_rate = losses_7d / max(total_7d, 1)

    overrides_ctx: Dict[str, Any] = {
        "count_1h": count_1h,
        "count_24h": count_24h,
        "count_7d": count_7d,
        "loss_rate_7d": round(loss_rate, 4),
    }

    return {
        "current": current,
        "biases": biases,
        "outcome": outcome_ctx,
        "overrides": overrides_ctx,
    }


# ---------------------------------------------------------------------------
# 6. SEED_PATTERNS -- hardcoded patterns translated into DSL form
# ---------------------------------------------------------------------------

SEED_PATTERNS: List[PatternSpec] = [
    # 1. Stress + Override -> Readiness Penalty
    PatternSpec(
        rule_id="seed_stress_override",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {">": [{"var": "current.stress_score"}, 0.7]},
                {">=": [{"var": "overrides.count_1h"}, 1]},
            ]
        },
        prediction={
            "target": "readiness_score",
            "direction": "down",
            "magnitude": "moderate",
        },
        action={
            "type": "readiness_penalty",
            "adjustment": -15,
            "component": "stress",
            "message": (
                "High stress with recent override detected "
                "-- applying readiness penalty"
            ),
        },
        meta={
            "description": (
                "High stress combined with recent override "
                "suggests impaired decision-making"
            ),
            "tier": "t1_daily",
            "domain": "cross",
            "tags": "stress,override,penalty",
        },
    ),
    # 2. Emotional Drift -> Confidence Gate
    PatternSpec(
        rule_id="seed_emotional_drift",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {
                    "in": [
                        {"var": "current.emotional_state"},
                        ["stressed", "anxious", "fatigued"],
                    ]
                },
                {"<": [{"var": "current.valence"}, -0.3]},
            ]
        },
        prediction={
            "target": "confidence_trend",
            "direction": "down",
            "magnitude": "significant",
        },
        action={
            "type": "confidence_gate",
            "adjustment": -20,
            "component": "emotional",
            "message": "Sustained negative emotional state -- gating confidence",
        },
        meta={
            "description": (
                "Persistent negative emotional state with low valence "
                "predicts confidence drop"
            ),
            "tier": "t1_daily",
            "domain": "human",
            "tags": "emotion,drift,gate",
        },
    ),
    # 3. Overtrade Detection -> Fatigue Flag
    PatternSpec(
        rule_id="seed_overtrade",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "or": [
                {">=": [{"var": "overrides.count_1h"}, 3]},
                {">=": [{"var": "overrides.count_24h"}, 8]},
            ]
        },
        prediction={
            "target": "fatigue_score",
            "direction": "up",
            "magnitude": "high",
        },
        action={
            "type": "fatigue_flag",
            "adjustment": 0.3,
            "component": "fatigue",
            "message": (
                "Override frequency exceeds safe threshold "
                "-- flagging decision fatigue"
            ),
        },
        meta={
            "description": (
                "Excessive override activity in short windows "
                "indicates overtrading / fatigue"
            ),
            "tier": "t1_daily",
            "domain": "trading",
            "tags": "overtrade,fatigue,override",
        },
    ),
    # 4. Bias Compound -> Amplified Penalty
    PatternSpec(
        rule_id="seed_bias_compound",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {
                    "or": [
                        {">": [{"var": "biases.confirmation_bias"}, 0.5]},
                        {">": [{"var": "biases.anchoring"}, 0.5]},
                        {">": [{"var": "biases.sunk_cost"}, 0.5]},
                        {">": [{"var": "biases.loss_aversion"}, 0.5]},
                        {">": [{"var": "biases.overconfidence"}, 0.5]},
                        {">": [{"var": "biases.recency_bias"}, 0.5]},
                    ]
                },
                {
                    "or": [
                        {">": [{"var": "biases.confirmation_bias"}, 0.5]},
                        {">": [{"var": "biases.anchoring"}, 0.5]},
                        {">": [{"var": "biases.sunk_cost"}, 0.5]},
                        {">": [{"var": "biases.loss_aversion"}, 0.5]},
                        {">": [{"var": "biases.overconfidence"}, 0.5]},
                        {">": [{"var": "biases.recency_bias"}, 0.5]},
                    ]
                },
            ]
        },
        prediction={
            "target": "readiness_score",
            "direction": "down",
            "magnitude": "high",
        },
        action={
            "type": "bias_penalty",
            "adjustment": -20,
            "component": "bias_interaction",
            "message": (
                "Multiple cognitive biases active above threshold "
                "-- amplified readiness penalty"
            ),
        },
        meta={
            "description": (
                "Two or more biases exceeding threshold interact "
                "to amplify poor decision-making"
            ),
            "tier": "t2_weekly",
            "domain": "human",
            "tags": "bias,compound,interaction",
        },
    ),
    # 5. Tilt Spiral -> Trading Pause
    PatternSpec(
        rule_id="seed_tilt_spiral",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {">": [{"var": "current.tilt_score"}, 0.6]},
                {"==": [{"var": "outcome.streak"}, "losing"]},
            ]
        },
        prediction={
            "target": "readiness_score",
            "direction": "down",
            "magnitude": "severe",
        },
        action={
            "type": "trading_pause",
            "adjustment": -40,
            "component": "tilt",
            "message": (
                "Tilt detected during losing streak "
                "-- suggesting trading pause"
            ),
        },
        meta={
            "description": (
                "High tilt score combined with losing streak "
                "suggests revenge trading spiral"
            ),
            "tier": "t1_daily",
            "domain": "cross",
            "tags": "tilt,revenge,pause,spiral",
        },
    ),
    # 6. Low Readiness + High Arousal -> Impulsive Risk
    PatternSpec(
        rule_id="seed_impulsive_risk",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {"<": [{"var": "current.readiness_score"}, 40]},
                {">": [{"var": "current.arousal"}, 0.7]},
                {"!=": [{"var": "current.emotional_state"}, "calm"]},
            ]
        },
        prediction={
            "target": "override_probability",
            "direction": "up",
            "magnitude": "high",
        },
        action={
            "type": "readiness_penalty",
            "adjustment": -25,
            "component": "arousal",
            "message": (
                "Low readiness with high arousal "
                "-- elevated impulsive override risk"
            ),
        },
        meta={
            "description": (
                "Low readiness combined with high emotional arousal "
                "predicts impulsive overrides"
            ),
            "tier": "t1_daily",
            "domain": "human",
            "tags": "impulsive,arousal,low_readiness",
        },
    ),
    # 7. Override Loss Spiral
    PatternSpec(
        rule_id="seed_override_loss_spiral",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {">": [{"var": "overrides.loss_rate_7d"}, 0.6]},
                {">=": [{"var": "overrides.count_7d"}, 3]},
            ]
        },
        prediction={
            "target": "readiness_score",
            "direction": "down",
            "magnitude": "moderate",
        },
        action={
            "type": "readiness_penalty",
            "adjustment": -20,
            "component": "override_quality",
            "message": (
                "Override loss rate exceeds 60% over 7 days "
                "-- pattern of poor manual decisions"
            ),
        },
        meta={
            "description": (
                "Persistent high override loss rate indicates "
                "systematic manual decision failure"
            ),
            "tier": "t2_weekly",
            "domain": "trading",
            "tags": "override,loss,spiral,weekly",
        },
    ),
    # 8. Cognitive Overload + Fatigue Convergence
    PatternSpec(
        rule_id="seed_cognitive_fatigue_convergence",
        version=1,
        generation=0,
        lineage=[],
        condition={
            "and": [
                {"==": [{"var": "current.cognitive_load"}, "high"]},
                {">": [{"var": "current.fatigue_score"}, 0.5]},
                {"<": [{"var": "current.decision_quality"}, 40]},
            ]
        },
        prediction={
            "target": "readiness_score",
            "direction": "down",
            "magnitude": "high",
        },
        action={
            "type": "readiness_penalty",
            "adjustment": -30,
            "component": "cognitive",
            "message": (
                "Cognitive overload converging with fatigue and low decision "
                "quality -- severe impairment"
            ),
        },
        meta={
            "description": (
                "Triple convergence of high cognitive load, fatigue, "
                "and low decision quality"
            ),
            "tier": "t2_weekly",
            "domain": "human",
            "tags": "cognitive,fatigue,convergence,overload",
        },
    ),
]
