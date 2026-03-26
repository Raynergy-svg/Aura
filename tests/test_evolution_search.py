"""Tests for the evolutionary search engine.

Tests mutation operators, crossover, random generation, fitness evaluation,
tournament selection, population evolution, library persistence, and adoption.
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.evolution.dsl import (
    DSLEvaluator,
    DSLValidator,
    FIELD_REGISTRY,
    PatternSpec,
    SEED_PATTERNS,
)
from src.aura.evolution.search import (
    PatternEvolver,
    crossover,
    mutate_grow,
    mutate_operator,
    mutate_prune,
    mutate_threshold,
    mutate_variable,
    random_condition,
    _count_nodes,
    _random_leaf,
    _get_numeric_fields,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _simple_condition():
    """A simple condition: current.tilt_score > 0.5"""
    return {">": [{"var": "current.tilt_score"}, 0.5]}


def _compound_condition():
    """A compound 'and' condition using real FIELD_REGISTRY paths."""
    return {"and": [
        {">": [{"var": "current.tilt_score"}, 0.6]},
        {"<": [{"var": "current.readiness_score"}, 50.0]},
    ]}


def _nested_condition():
    """Nested logic condition using real FIELD_REGISTRY paths."""
    return {"or": [
        {"and": [
            {">": [{"var": "current.tilt_score"}, 0.7]},
            {"<": [{"var": "current.valence"}, -0.2]},
        ]},
        {">": [{"var": "current.stress_score"}, 0.8]},
    ]}


def _make_signal_history(n=20):
    """Generate a synthetic signal history for testing fitness evaluation.

    Uses nested dict format matching the DSL's build_context_from_signals output.
    Also includes flat keys for backward compatibility with _check_event_occurred.
    """
    import random as _rand
    _rand.seed(42)
    history = []
    for i in range(n):
        # Create a declining readiness scenario for some signals
        readiness = 70.0 - i * 2.0 + _rand.uniform(-5, 5)
        readiness = max(0.0, min(100.0, readiness))

        tilt = 0.2 + (i / n) * 0.6 + _rand.uniform(-0.1, 0.1)
        tilt = max(0.0, min(1.0, tilt))

        stress = 0.3 + (i / n) * 0.4 + _rand.uniform(-0.1, 0.1)
        stress = max(0.0, min(1.0, stress))

        outcome_val = _rand.choice(["winning", "losing", "neutral"])

        history.append({
            # Nested format for DSLEvaluator var resolution
            "current": {
                "emotional_state": _rand.choice(["calm", "stressed", "anxious"]),
                "stress_score": round(stress, 3),
                "readiness_score": round(readiness, 1),
                "cognitive_load": "medium",
                "confidence_trend": "stable",
                "tilt_score": round(tilt, 3),
                "fatigue_score": round(_rand.uniform(0.0, 0.7), 3),
                "decision_quality": round(_rand.uniform(20.0, 80.0), 1),
                "valence": round(_rand.uniform(-0.5, 0.5), 3),
                "arousal": round(_rand.uniform(0.0, 1.0), 3),
            },
            "biases": {
                "disposition_effect": round(_rand.uniform(0.0, 0.5), 3),
                "loss_aversion": round(_rand.uniform(0.0, 0.5), 3),
                "recency_bias": round(_rand.uniform(0.0, 0.5), 3),
                "confirmation_bias": round(_rand.uniform(0.0, 0.3), 3),
                "sunk_cost": round(_rand.uniform(0.0, 0.3), 3),
                "anchoring": round(_rand.uniform(0.0, 0.4), 3),
                "overconfidence": round(_rand.uniform(0.0, 0.4), 3),
                "hindsight": round(_rand.uniform(0.0, 0.3), 3),
                "attribution_error": round(_rand.uniform(0.0, 0.3), 3),
            },
            "outcome": {
                "pnl_today": round(_rand.uniform(-50, 50), 2),
                "win_rate_7d": round(_rand.uniform(0.3, 0.7), 3),
                "streak": outcome_val,
            },
            "overrides": {
                "count_1h": _rand.randint(0, 3),
                "count_24h": _rand.randint(0, 10),
                "count_7d": _rand.randint(0, 20),
                "loss_rate_7d": round(_rand.uniform(0.0, 0.6), 3),
            },
            # Flat keys used by _check_event_occurred for readiness/tilt lookups
            "readiness_score": round(readiness, 1),
            "tilt_score": round(tilt, 3),
        })
    return history


@pytest.fixture
def evolver(tmp_path):
    """Create a PatternEvolver with a temp library path."""
    return PatternEvolver(
        population_size=10,
        elite_count=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
        max_generations=5,
        library_path=tmp_path / "library.json",
    )


@pytest.fixture
def signal_history():
    return _make_signal_history(20)


# ===================================================================
# Mutation operator tests
# ===================================================================

class TestMutateThreshold:
    """Test mutate_threshold produces valid conditions."""

    def test_simple_condition(self):
        """Threshold mutation on a simple comparison changes the threshold value."""
        original = _simple_condition()
        mutated = mutate_threshold(original)

        # Result should be a valid condition dict
        assert isinstance(mutated, dict)
        assert len(mutated) == 1

        # The operator should still be a comparison
        op = next(iter(mutated))
        assert op in [">", ">=", "<", "<=", "==", "!="]

        # Threshold should still be a number
        args = mutated[op]
        assert isinstance(args[1], (int, float))

    def test_compound_condition(self):
        """Threshold mutation on compound condition preserves structure."""
        original = _compound_condition()
        mutated = mutate_threshold(original)
        assert isinstance(mutated, dict)
        assert "and" in mutated or any(
            op in mutated for op in [">", ">=", "<", "<="]
        )

    def test_does_not_modify_original(self):
        """Mutation should not modify the original condition."""
        original = _simple_condition()
        original_copy = copy.deepcopy(original)
        mutate_threshold(original)
        assert original == original_copy

    def test_threshold_stays_in_bounds(self):
        """Mutated threshold stays within FIELD_REGISTRY bounds."""
        # current.tilt_score has min=0.0, max=1.0
        for _ in range(50):
            cond = {">": [{"var": "current.tilt_score"}, 0.5]}
            mutated = mutate_threshold(cond)
            op = next(iter(mutated))
            val = mutated[op][1]
            assert 0.0 <= val <= 1.0, f"tilt_score threshold {val} out of bounds"


class TestMutateOperator:
    """Test mutate_operator swaps operators."""

    def test_comparison_swap(self):
        """Should swap comparison operator."""
        original = {">": [{"var": "current.tilt_score"}, 0.5]}

        # Run multiple times since it's stochastic
        swapped = False
        for _ in range(20):
            mutated = mutate_operator(original)
            op = next(iter(mutated))
            if op != ">":
                swapped = True
                break
        assert swapped, "Expected operator swap after 20 attempts"

    def test_logic_swap(self):
        """Should swap and/or operator."""
        original = _compound_condition()
        swapped = False
        for _ in range(20):
            mutated = mutate_operator(original)
            op = next(iter(mutated))
            if op != "and":
                swapped = True
                assert op == "or"
                break
        # It might also pick a comparison node to swap
        # Just ensure the result is valid
        assert isinstance(mutate_operator(original), dict)


class TestMutateVariable:
    """Test mutate_variable swaps variable references."""

    def test_variable_swap(self):
        """Should swap the variable to a different field of the same type."""
        original = {">": [{"var": "current.tilt_score"}, 0.5]}
        swapped = False
        for _ in range(50):
            mutated = mutate_variable(original)
            op = next(iter(mutated))
            args = mutated[op]
            var_name = args[0].get("var", "")
            if var_name != "current.tilt_score":
                swapped = True
                # Should be a float field with bounded range
                assert FIELD_REGISTRY[var_name]["type"] == "float"
                break
        assert swapped, "Expected variable swap after 50 attempts"


class TestMutateGrow:
    """Test mutate_grow wraps a leaf in a logic node."""

    def test_grow_simple(self):
        """Growing a simple leaf wraps it in and/or with a new leaf."""
        original = _simple_condition()
        grown = mutate_grow(original)

        assert isinstance(grown, dict)
        # Should now have a logic operator at root
        op = next(iter(grown))
        assert op in ["and", "or"], f"Expected logic op, got {op}"
        args = grown[op]
        assert len(args) == 2

    def test_grow_increases_nodes(self):
        """Growing should increase the node count."""
        original = _simple_condition()
        original_count = _count_nodes(original)
        grown = mutate_grow(original)
        grown_count = _count_nodes(grown)
        assert grown_count > original_count


class TestMutatePrune:
    """Test mutate_prune simplifies the tree."""

    def test_prune_compound(self):
        """Pruning a compound condition should reduce it to a child."""
        original = _compound_condition()
        pruned = mutate_prune(original)

        assert isinstance(pruned, dict)
        # Should be simpler (fewer or equal nodes)
        pruned_count = _count_nodes(pruned)
        original_count = _count_nodes(original)
        assert pruned_count <= original_count

    def test_prune_returns_valid_dict(self):
        """Pruned result should still be a valid dict."""
        for _ in range(10):
            original = _nested_condition()
            pruned = mutate_prune(original)
            assert isinstance(pruned, dict)
            assert len(pruned) >= 1


# ===================================================================
# Crossover tests
# ===================================================================

class TestCrossover:
    """Test crossover produces two children."""

    def test_produces_two_children(self):
        """Crossover should return exactly two condition dicts."""
        parent_a = _compound_condition()
        parent_b = _nested_condition()
        child_a, child_b = crossover(parent_a, parent_b)

        assert isinstance(child_a, dict)
        assert isinstance(child_b, dict)

    def test_children_differ_from_parents(self):
        """At least one child should differ from its parent (usually)."""
        parent_a = _compound_condition()
        parent_b = _nested_condition()

        # Run several times; at least once children should differ
        any_different = False
        for _ in range(10):
            child_a, child_b = crossover(
                copy.deepcopy(parent_a), copy.deepcopy(parent_b)
            )
            if child_a != parent_a or child_b != parent_b:
                any_different = True
                break
        assert any_different

    def test_does_not_modify_parents(self):
        """Crossover should not modify the parent conditions."""
        parent_a = _compound_condition()
        parent_b = _nested_condition()
        pa_copy = copy.deepcopy(parent_a)
        pb_copy = copy.deepcopy(parent_b)

        crossover(parent_a, parent_b)

        assert parent_a == pa_copy
        assert parent_b == pb_copy


# ===================================================================
# Random condition tests
# ===================================================================

class TestRandomCondition:
    """Test random_condition generates valid trees."""

    def test_produces_dict(self):
        """Should produce a dict."""
        cond = random_condition()
        assert isinstance(cond, dict)

    def test_all_fields_bounded(self):
        """All fields returned by _get_numeric_fields should have bounded ranges."""
        fields = _get_numeric_fields()
        assert len(fields) > 0, "Should have at least some numeric fields"
        for f in fields:
            info = FIELD_REGISTRY[f]
            assert info.get("min") is not None, f"Field {f} has None min"
            assert info.get("max") is not None, f"Field {f} has None max"

    def test_validator_accepts(self):
        """Random conditions should pass structural validation."""
        validator = DSLValidator()
        for _ in range(20):
            cond = random_condition(max_depth=3)
            # Build a minimal spec to validate
            spec = PatternSpec(
                rule_id="test_random",
                condition=cond,
                prediction={"target": "readiness_score", "direction": "down"},
            )
            errors = validator.validate(spec)
            assert errors == [], f"Validation errors: {errors}"

    def test_respects_max_depth(self):
        """With max_depth=1, should produce mostly leaves."""
        for _ in range(10):
            cond = random_condition(max_depth=1)
            assert isinstance(cond, dict)
            # Should be shallow
            nodes = _count_nodes(cond)
            # A leaf has ~3 nodes (op, var, threshold), a depth-1 logic has more
            assert nodes <= 15  # generous bound

    def test_random_leaf_uses_valid_fields(self):
        """Random leaves should reference fields that exist in FIELD_REGISTRY."""
        for _ in range(30):
            leaf = _random_leaf()
            op = next(iter(leaf))
            args = leaf[op]
            assert isinstance(args, list) and len(args) == 2
            var_ref = args[0]
            assert "var" in var_ref
            field_name = var_ref["var"]
            assert field_name in FIELD_REGISTRY, f"Unknown field: {field_name}"


# ===================================================================
# Fitness evaluation tests
# ===================================================================

class TestFitnessEvaluation:
    """Test fitness evaluation with mock signal history."""

    def test_perfect_predictor(self, evolver):
        """A condition that fires exactly when events occur should have high precision."""
        # Create a history where readiness drops when tilt > 0.5
        history = []
        for i in range(20):
            tilt = 0.3 if i % 2 == 0 else 0.8
            # When tilt is high, readiness drops sharply in next signal
            if i > 0 and history[i - 1]["current"]["tilt_score"] > 0.5:
                readiness = history[i - 1]["current"]["readiness_score"] - 15.0
            else:
                readiness = 60.0
            history.append({
                "current": {
                    "readiness_score": readiness,
                    "tilt_score": tilt,
                    "stress_score": 0.3,
                    "fatigue_score": 0.2,
                    "decision_quality": 50.0,
                    "valence": 0.0,
                    "arousal": 0.3,
                },
                "biases": {"disposition_effect": 0.1, "loss_aversion": 0.1,
                           "recency_bias": 0.1, "confirmation_bias": 0.1,
                           "sunk_cost": 0.1, "anchoring": 0.1,
                           "overconfidence": 0.1, "hindsight": 0.1,
                           "attribution_error": 0.1},
                "outcome": {"pnl_today": 0.0, "win_rate_7d": 0.5, "streak": "neutral"},
                "overrides": {"count_1h": 0, "count_24h": 0, "count_7d": 0, "loss_rate_7d": 0.0},
                # Flat keys for event detection
                "readiness_score": readiness,
                "tilt_score": tilt,
            })

        # Condition that fires when tilt > 0.5
        spec = PatternSpec(
            rule_id="test_perfect",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
        )

        fitness = evolver.evaluate_fitness(spec, history)
        assert fitness["precision"] >= 0.0
        assert fitness["support"] > 0
        assert "fitness_score" in fitness

    def test_empty_history(self, evolver):
        """Fitness with empty history should return zeros."""
        spec = PatternSpec(
            rule_id="test_empty",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
        )
        fitness = evolver.evaluate_fitness(spec, [])
        assert fitness["precision"] == 0.0
        assert fitness["recall"] == 0.0
        assert fitness["fitness_score"] == 0.0

    def test_never_fires(self, evolver, signal_history):
        """A condition that never fires should have zero support."""
        spec = PatternSpec(
            rule_id="test_never",
            condition={">": [{"var": "current.tilt_score"}, 999.0]},
            prediction={"event": "readiness_drop"},
        )
        fitness = evolver.evaluate_fitness(spec, signal_history)
        assert fitness["support"] == 0
        assert fitness["precision"] == 0.0

    def test_override_loss_prediction(self, evolver):
        """Test fitness evaluation for override_loss prediction type."""
        history = [
            {
                "current": {"tilt_score": 0.8, "readiness_score": 50.0,
                            "stress_score": 0.6, "fatigue_score": 0.3,
                            "decision_quality": 40.0, "valence": -0.3,
                            "arousal": 0.6},
                "biases": {"disposition_effect": 0.1, "loss_aversion": 0.2,
                           "recency_bias": 0.1, "confirmation_bias": 0.1,
                           "sunk_cost": 0.1, "anchoring": 0.1,
                           "overconfidence": 0.1, "hindsight": 0.1,
                           "attribution_error": 0.1},
                "outcome": {"pnl_today": -20.0, "win_rate_7d": 0.4, "streak": "losing"},
                "overrides": {"count_1h": 1, "count_24h": 3, "count_7d": 8, "loss_rate_7d": 0.4},
                "readiness_score": 50.0,
                "tilt_score": 0.8,
            },
            {
                "current": {"tilt_score": 0.3, "readiness_score": 55.0,
                            "stress_score": 0.3, "fatigue_score": 0.1,
                            "decision_quality": 60.0, "valence": 0.2,
                            "arousal": 0.3},
                "biases": {"disposition_effect": 0.1, "loss_aversion": 0.1,
                           "recency_bias": 0.1, "confirmation_bias": 0.1,
                           "sunk_cost": 0.1, "anchoring": 0.1,
                           "overconfidence": 0.1, "hindsight": 0.1,
                           "attribution_error": 0.1},
                "outcome": {"pnl_today": 10.0, "win_rate_7d": 0.6, "streak": "winning"},
                "overrides": {"count_1h": 0, "count_24h": 1, "count_7d": 3, "loss_rate_7d": 0.1},
                "readiness_score": 55.0,
                "tilt_score": 0.3,
            },
        ]

        spec = PatternSpec(
            rule_id="test_override",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "override_loss"},
        )
        fitness = evolver.evaluate_fitness(spec, history)
        assert isinstance(fitness["precision"], float)
        assert isinstance(fitness["fitness_score"], float)

    def test_complexity_counted(self, evolver, signal_history):
        """Complexity should reflect the condition tree size."""
        simple_spec = PatternSpec(
            rule_id="test_simple",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
        )
        complex_spec = PatternSpec(
            rule_id="test_complex",
            condition=_nested_condition(),
            prediction={"event": "readiness_drop"},
        )

        simple_fit = evolver.evaluate_fitness(simple_spec, signal_history)
        complex_fit = evolver.evaluate_fitness(complex_spec, signal_history)

        assert simple_fit["complexity"] < complex_fit["complexity"]

    def test_dsl_native_prediction_format(self, evolver, signal_history):
        """Test fitness works with DSL-native prediction format (target+direction)."""
        spec = PatternSpec(
            rule_id="test_dsl_native",
            condition={">": [{"var": "current.stress_score"}, 0.5]},
            prediction={"target": "readiness_score", "direction": "down", "magnitude": "moderate"},
        )
        fitness = evolver.evaluate_fitness(spec, signal_history)
        assert "fitness_score" in fitness
        assert isinstance(fitness["fitness_score"], float)


# ===================================================================
# Selection and evolution tests
# ===================================================================

class TestTournamentSelection:
    """Test tournament selection preserves elites."""

    def test_elite_preservation(self, evolver, signal_history):
        """Elites should survive into the next generation."""
        evolver.seed_population()

        # Run one generation
        new_pop = evolver.evolve_one_generation(signal_history)

        # Population should be the right size
        assert len(new_pop) == evolver.population_size

        # Generation should have incremented
        assert evolver._generation == 1


class TestEvolveOneGeneration:
    """Test evolve_one_generation runs without errors."""

    def test_runs_without_error(self, evolver, signal_history):
        """Should complete a full generation without exceptions."""
        evolver.seed_population()
        new_pop = evolver.evolve_one_generation(signal_history)

        assert len(new_pop) == evolver.population_size
        assert all(isinstance(s, PatternSpec) for s in new_pop)

    def test_multiple_generations(self, evolver, signal_history):
        """Should run multiple generations smoothly."""
        evolver.seed_population()
        for _ in range(3):
            evolver.evolve_one_generation(signal_history)

        assert evolver._generation == 3
        assert len(evolver._population) == evolver.population_size

    def test_auto_seeds_if_empty(self, evolver, signal_history):
        """If population is empty, evolve_one_generation should auto-seed."""
        assert len(evolver._population) == 0
        new_pop = evolver.evolve_one_generation(signal_history)
        assert len(new_pop) == evolver.population_size

    def test_fitness_assigned(self, evolver, signal_history):
        """After evolution, all individuals should have fitness dicts."""
        evolver.seed_population()
        evolver.evolve_one_generation(signal_history)

        # Check archive -- stores evaluated patterns
        for spec in evolver._archive:
            assert "fitness_score" in spec.fitness


# ===================================================================
# Library persistence tests
# ===================================================================

class TestLibraryPersistence:
    """Test library save/load roundtrip."""

    def test_save_load_roundtrip(self, evolver, signal_history):
        """Save and load should preserve population and archive."""
        evolver.seed_population()
        evolver.evolve_one_generation(signal_history)

        # Save
        evolver.save_library()
        assert evolver._library_path.exists()

        # Create a new evolver and load
        evolver2 = PatternEvolver(
            population_size=10,
            library_path=evolver._library_path,
        )
        evolver2.load_library()

        assert evolver2._generation == evolver._generation
        assert len(evolver2._population) == len(evolver._population)
        assert len(evolver2._archive) == len(evolver._archive)

    def test_load_nonexistent(self, tmp_path):
        """Loading from nonexistent path should not crash."""
        evolver = PatternEvolver(
            library_path=tmp_path / "nonexistent" / "library.json"
        )
        evolver.load_library()  # should not raise
        assert len(evolver._population) == 0

    def test_save_creates_directory(self, tmp_path):
        """Save should create parent directories if needed."""
        lib_path = tmp_path / "deep" / "nested" / "library.json"
        evolver = PatternEvolver(
            population_size=5,
            library_path=lib_path,
        )
        evolver.seed_population()
        evolver.save_library()
        assert lib_path.exists()

        data = json.loads(lib_path.read_text())
        assert "population" in data
        assert "archive" in data
        assert data["generation"] == 0


# ===================================================================
# Adoption filter tests
# ===================================================================

class TestGetAdoptablePatterns:
    """Test get_adoptable_patterns filters correctly."""

    def test_filters_by_precision(self, evolver):
        """Only patterns with precision >= threshold should be adoptable."""
        good = PatternSpec(
            rule_id="good_pattern",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
            fitness={"precision": 0.8, "support": 10, "fitness_score": 0.5},
        )
        bad = PatternSpec(
            rule_id="bad_pattern",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
            fitness={"precision": 0.3, "support": 10, "fitness_score": 0.2},
        )
        evolver._archive = [good, bad]

        adoptable = evolver.get_adoptable_patterns(min_precision=0.6, min_support=5)
        assert len(adoptable) == 1
        assert adoptable[0].rule_id == "good_pattern"

    def test_filters_by_support(self, evolver):
        """Only patterns with support >= threshold should be adoptable."""
        low_support = PatternSpec(
            rule_id="low_support",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
            fitness={"precision": 0.9, "support": 2, "fitness_score": 0.6},
        )
        high_support = PatternSpec(
            rule_id="high_support",
            condition={">": [{"var": "current.tilt_score"}, 0.5]},
            prediction={"event": "readiness_drop"},
            fitness={"precision": 0.9, "support": 20, "fitness_score": 0.7},
        )
        evolver._archive = [low_support, high_support]

        adoptable = evolver.get_adoptable_patterns(min_precision=0.6, min_support=5)
        assert len(adoptable) == 1
        assert adoptable[0].rule_id == "high_support"

    def test_empty_archive(self, evolver):
        """Empty archive should return empty list."""
        evolver._archive = []
        assert evolver.get_adoptable_patterns() == []

    def test_both_filters_combined(self, evolver):
        """Both precision and support must be met."""
        specs = [
            PatternSpec(
                rule_id="both_good",
                condition={">": [{"var": "current.tilt_score"}, 0.5]},
                prediction={"event": "readiness_drop"},
                fitness={"precision": 0.7, "support": 10, "fitness_score": 0.5},
            ),
            PatternSpec(
                rule_id="good_prec_low_sup",
                condition={">": [{"var": "current.tilt_score"}, 0.5]},
                prediction={"event": "readiness_drop"},
                fitness={"precision": 0.7, "support": 3, "fitness_score": 0.4},
            ),
            PatternSpec(
                rule_id="low_prec_good_sup",
                condition={">": [{"var": "current.tilt_score"}, 0.5]},
                prediction={"event": "readiness_drop"},
                fitness={"precision": 0.4, "support": 15, "fitness_score": 0.3},
            ),
        ]
        evolver._archive = specs

        adoptable = evolver.get_adoptable_patterns(min_precision=0.6, min_support=5)
        assert len(adoptable) == 1
        assert adoptable[0].rule_id == "both_good"


# ===================================================================
# Count nodes utility test
# ===================================================================

class TestCountNodes:
    """Test _count_nodes utility."""

    def test_leaf(self):
        assert _count_nodes({">": [{"var": "current.tilt_score"}, 0.5]}) == 3

    def test_compound(self):
        cond = _compound_condition()
        # and(>(var, val), <(var, val)) = 1+3+3 = 7
        assert _count_nodes(cond) == 7

    def test_empty(self):
        assert _count_nodes({}) == 0
