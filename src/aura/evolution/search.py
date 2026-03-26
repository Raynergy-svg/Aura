"""Evolutionary search engine for Aura behavioral patterns.

Discovers new behavioral patterns through genetic programming over the DSL.
Uses tournament selection, subtree crossover, and five mutation operators
(threshold, operator, variable, grow, prune) to evolve JsonLogic condition
trees that predict readiness drops, override losses, and tilt episodes.

Fitness is computed via walk-forward backtesting against signal history.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.aura.evolution.dsl import (
    DSLEvaluator,
    DSLValidator,
    FIELD_REGISTRY,
    PatternSpec,
    SEED_PATTERNS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator sets used by mutations
# ---------------------------------------------------------------------------

COMPARE_OPS = [">", ">=", "<", "<="]
LOGIC_OPS = ["and", "or"]
PREDICTION_EVENTS = ["readiness_drop", "override_loss", "tilt_episode"]

# Prediction targets (from the real DSL seed patterns) mapped to search event types
# The DSL uses {target, direction, magnitude}; we map target+direction to event types
PREDICTION_TARGET_MAP = {
    ("readiness_score", "down"): "readiness_drop",
    ("confidence_trend", "down"): "readiness_drop",
    ("fatigue_score", "up"): "tilt_episode",
    ("tilt_score", "up"): "tilt_episode",
}

# Lookahead windows per prediction type (number of signals to look ahead)
LOOKAHEAD = {
    "readiness_drop": 3,
    "override_loss": 1,
    "tilt_episode": 5,
}


# ===================================================================
# Helper utilities
# ===================================================================

def _resolve_ctx(ctx: Dict, path: str, default: Any = None) -> Any:
    """Resolve a dotted path from a context dict (nested or flat).

    Supports both nested dicts ({"current": {"readiness_score": 50}})
    and flat dicts ({"readiness_score": 50, "current.readiness_score": 50}).
    """
    # Try flat lookup first
    if path in ctx:
        return ctx[path]
    # Try nested traversal
    parts = path.split(".")
    current = ctx
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _count_nodes(condition: Dict) -> int:
    """Count total nodes in a JsonLogic condition tree."""
    if not isinstance(condition, dict):
        return 1
    if not condition:
        return 0
    op = next(iter(condition))
    args = condition[op]
    if op == "var":
        return 1
    if not isinstance(args, list):
        return 1 + _count_nodes(args)
    return 1 + sum(_count_nodes(a) for a in args)


def _collect_subtrees(condition: Dict) -> List[Tuple[List, int, Dict]]:
    """Collect all subtrees with their parent path.

    Returns list of (path, index_in_parent, subtree) tuples.
    path is the key-chain to reach the parent list, index is position within it.
    The root itself is included with path=[] and index=-1.
    """
    results: List[Tuple[List, int, Dict]] = []

    def _walk(node: Any, path: List, idx: int):
        if isinstance(node, dict) and len(node) == 1:
            results.append((path, idx, node))
            op = next(iter(node))
            args = node[op]
            if isinstance(args, list):
                for i, child in enumerate(args):
                    _walk(child, path + [op], i)

    _walk(condition, [], -1)
    return results


def _get_numeric_fields() -> List[str]:
    """Return field names that are numeric (float or int) with bounded ranges.

    Filters out enum fields and fields where min or max is None (unbounded),
    since those cannot be used for random threshold generation.
    """
    return [
        name for name, info in FIELD_REGISTRY.items()
        if info.get("type") in ("float", "int")
        and info.get("min") is not None
        and info.get("max") is not None
    ]


def _get_fields_of_type(field_type: str) -> List[str]:
    """Return field names matching a given type with bounded ranges."""
    return [
        name for name, info in FIELD_REGISTRY.items()
        if info.get("type") == field_type
        and info.get("min") is not None
        and info.get("max") is not None
    ]


def _random_threshold_for_field(field_name: str) -> float:
    """Generate a random threshold value appropriate for a field."""
    info = FIELD_REGISTRY.get(field_name, {"type": "float", "min": 0.0, "max": 1.0})
    mn = info.get("min")
    mx = info.get("max")

    # Fallback for unbounded fields
    if mn is None:
        mn = 0.0
    if mx is None:
        mx = 100.0 if info.get("type") == "int" else 1.0

    if info.get("type") == "int":
        return float(random.randint(int(mn), int(mx)))
    return round(random.uniform(float(mn), float(mx)), 3)


def _random_leaf() -> Dict:
    """Generate a single random comparison leaf node."""
    field_name = random.choice(_get_numeric_fields())
    op = random.choice(COMPARE_OPS)
    threshold = _random_threshold_for_field(field_name)
    return {op: [{"var": field_name}, threshold]}


def _find_comparisons(condition: Dict, path: Optional[List] = None) -> List[Tuple[List, Dict]]:
    """Find all comparison nodes in a condition tree.

    Returns (path_to_parent_args_list, node) tuples.
    """
    if path is None:
        path = []
    results = []

    if not isinstance(condition, dict) or not condition:
        return results

    op = next(iter(condition))
    args = condition[op]

    if op in COMPARE_OPS:
        results.append((path, condition))
    elif op in LOGIC_OPS or op == "!":
        if isinstance(args, list):
            for i, child in enumerate(args):
                if isinstance(child, dict):
                    results.extend(_find_comparisons(child, path + [(op, i)]))
    return results


def _find_logic_nodes(condition: Dict, path: Optional[List] = None) -> List[Tuple[List, Dict]]:
    """Find all and/or nodes in a condition tree."""
    if path is None:
        path = []
    results = []

    if not isinstance(condition, dict) or not condition:
        return results

    op = next(iter(condition))
    args = condition[op]

    if op in LOGIC_OPS:
        results.append((path, condition))
        if isinstance(args, list):
            for i, child in enumerate(args):
                if isinstance(child, dict):
                    results.extend(_find_logic_nodes(child, path + [(op, i)]))
    return results


# ===================================================================
# Mutation operators
# ===================================================================

def mutate_threshold(condition: Dict) -> Dict:
    """Find a numeric comparison, perturb the threshold by gaussian noise."""
    result = copy.deepcopy(condition)
    comparisons = _find_comparisons(result)

    if not comparisons:
        return result

    _path, node = random.choice(comparisons)
    op = next(iter(node))
    args = node[op]

    if len(args) == 2 and isinstance(args[1], (int, float)):
        # Find the field to get bounds
        field_name = None
        if isinstance(args[0], dict) and "var" in args[0]:
            var_ref = args[0]["var"]
            if isinstance(var_ref, list):
                field_name = var_ref[0]
            else:
                field_name = var_ref

        info = FIELD_REGISTRY.get(field_name, {"min": 0.0, "max": 1.0, "type": "float"})
        mn = info.get("min")
        mx = info.get("max")
        if mn is None:
            mn = 0.0
        if mx is None:
            mx = 100.0 if info.get("type") == "int" else 1.0
        span = float(mx) - float(mn)

        # Gaussian perturbation (10% of range as sigma)
        sigma = span * 0.1
        new_val = args[1] + random.gauss(0, sigma)
        new_val = max(float(mn), min(float(mx), new_val))

        if info.get("type") == "int":
            new_val = float(round(new_val))
        else:
            new_val = round(new_val, 3)

        args[1] = new_val

    return result


def mutate_operator(condition: Dict) -> Dict:
    """Swap a comparison operator or logic operator."""
    result = copy.deepcopy(condition)

    # Coin flip: mutate comparison op or logic op
    comparisons = _find_comparisons(result)
    logic_nodes = _find_logic_nodes(result)

    candidates = []
    if comparisons:
        candidates.append(("compare", comparisons))
    if logic_nodes:
        candidates.append(("logic", logic_nodes))

    if not candidates:
        return result

    kind, nodes = random.choice(candidates)
    _path, node = random.choice(nodes)
    op = next(iter(node))
    args = node[op]

    if kind == "compare":
        new_ops = [o for o in COMPARE_OPS if o != op]
        if new_ops:
            new_op = random.choice(new_ops)
            del node[op]
            node[new_op] = args
    else:
        new_ops = [o for o in LOGIC_OPS if o != op]
        if new_ops:
            new_op = random.choice(new_ops)
            del node[op]
            node[new_op] = args

    return result


def mutate_variable(condition: Dict) -> Dict:
    """Swap a var reference with another of the same type from FIELD_REGISTRY."""
    result = copy.deepcopy(condition)
    comparisons = _find_comparisons(result)

    if not comparisons:
        return result

    _path, node = random.choice(comparisons)
    op = next(iter(node))
    args = node[op]

    if len(args) >= 1 and isinstance(args[0], dict) and "var" in args[0]:
        current_field = args[0]["var"]
        if isinstance(current_field, list):
            current_field = current_field[0]

        info = FIELD_REGISTRY.get(current_field, {"type": "float"})
        same_type = _get_fields_of_type(info["type"])
        alternatives = [f for f in same_type if f != current_field]

        if alternatives:
            new_field = random.choice(alternatives)
            args[0] = {"var": new_field}
            # Also update threshold to be valid for the new field
            if len(args) == 2 and isinstance(args[1], (int, float)):
                args[1] = _random_threshold_for_field(new_field)

    return result


def mutate_grow(condition: Dict) -> Dict:
    """Take a leaf comparison and wrap it in an 'and' with a new random condition."""
    result = copy.deepcopy(condition)
    comparisons = _find_comparisons(result)

    if not comparisons:
        # Whole tree is non-comparison; wrap root
        new_leaf = _random_leaf()
        return {"and": [result, new_leaf]}

    # Pick a random comparison to grow
    path_info, node = random.choice(comparisons)

    new_leaf = _random_leaf()
    logic_op = random.choice(LOGIC_OPS)
    replacement = {logic_op: [copy.deepcopy(node), new_leaf]}

    if not path_info:
        # The node IS the root
        return replacement

    # Navigate to parent and replace
    _replace_at_path(result, path_info, replacement)
    return result


def mutate_prune(condition: Dict) -> Dict:
    """Take an 'and'/'or' node and replace it with one of its children (simplification)."""
    result = copy.deepcopy(condition)
    logic_nodes = _find_logic_nodes(result)

    if not logic_nodes:
        return result

    path_info, node = random.choice(logic_nodes)
    op = next(iter(node))
    args = node[op]

    if not isinstance(args, list) or len(args) == 0:
        return result

    # Pick a random child to keep
    child = random.choice(args)

    if not path_info:
        # The node IS the root; replace entire tree with the child
        if isinstance(child, dict):
            return child
        # If child is a literal, wrap it in a trivial comparison
        return _random_leaf()

    _replace_at_path(result, path_info, child)
    return result


def _replace_at_path(root: Dict, path: List[Tuple[str, int]], replacement: Any) -> None:
    """Navigate a condition tree along path and replace the node at the end."""
    current = root
    for step_idx, (parent_op, child_idx) in enumerate(path):
        if step_idx == len(path) - 1:
            # Replace this child
            args = current.get(parent_op)
            if isinstance(args, list) and child_idx < len(args):
                args[child_idx] = replacement
            return
        else:
            args = current.get(parent_op)
            if isinstance(args, list) and child_idx < len(args):
                current = args[child_idx]
            else:
                return


# ===================================================================
# Crossover operator
# ===================================================================

def crossover(parent_a: Dict, parent_b: Dict) -> Tuple[Dict, Dict]:
    """Select random subtrees from each parent and swap them.

    Returns two children condition trees.
    """
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)

    subtrees_a = _collect_subtrees(child_a)
    subtrees_b = _collect_subtrees(child_b)

    # Filter to swappable subtrees (skip root if it's the only one)
    swappable_a = [s for s in subtrees_a if s[1] >= 0]
    swappable_b = [s for s in subtrees_b if s[1] >= 0]

    if not swappable_a or not swappable_b:
        # Can't do subtree swap, just swap entire trees
        return child_b, child_a

    path_a, idx_a, sub_a = random.choice(swappable_a)
    path_b, idx_b, sub_b = random.choice(swappable_b)

    # Navigate to the parents and swap
    _swap_subtree(child_a, path_a, idx_a, copy.deepcopy(sub_b))
    _swap_subtree(child_b, path_b, idx_b, copy.deepcopy(sub_a))

    return child_a, child_b


def _swap_subtree(root: Dict, path: List[str], idx: int, replacement: Dict) -> None:
    """Navigate root along path (list of operator keys) and replace child at idx."""
    current = root
    for step_op in path:
        args = current.get(step_op)
        if isinstance(args, list) and len(args) > 0:
            # Go deeper (last step of path leads to the parent)
            if step_op == path[-1]:
                break
            # Not yet at target depth, navigate into the child
            # We need to find which child to descend into
            # Path stores operator names sequentially from root to parent
            pass

    # Direct approach: navigate to the container
    current = root
    for i, step_op in enumerate(path):
        if i == len(path) - 1:
            # This is the parent operator; replace child at idx
            args = current.get(step_op)
            if isinstance(args, list) and idx < len(args):
                args[idx] = replacement
            return
        else:
            args = current.get(step_op)
            if isinstance(args, list):
                # Find which child is a dict to descend into
                # Use the next path element to guide
                next_op = path[i + 1]
                for child in args:
                    if isinstance(child, dict) and next_op in child:
                        current = child
                        break
                else:
                    return
            else:
                return

    # If path is empty, try replacing at root level
    op = next(iter(root), None)
    if op:
        args = root.get(op)
        if isinstance(args, list) and idx < len(args):
            args[idx] = replacement


# ===================================================================
# Random pattern generator
# ===================================================================

def random_condition(max_depth: int = 3) -> Dict:
    """Generate a random valid condition tree.

    Used for population seeding and grow mutation.
    """
    return _random_tree(depth=0, max_depth=max_depth)


def _random_tree(depth: int, max_depth: int) -> Dict:
    """Recursively build a random condition tree.

    Generates compact trees that stay within the DSLValidator's MAX_NODES=50 limit.
    Leaf probability increases with depth to keep trees manageable.
    """
    # At max depth, always emit a leaf
    if depth >= max_depth:
        return _random_leaf()

    # Increasing probability of leaf at deeper levels
    leaf_prob = 0.3 + depth * 0.25
    if depth > 0 and random.random() < leaf_prob:
        return _random_leaf()

    # Otherwise, create a logic node with 2 children (not 3, to limit blowup)
    logic_op = random.choice(LOGIC_OPS)
    num_children = 2
    children = [_random_tree(depth + 1, max_depth) for _ in range(num_children)]
    return {logic_op: children}


# ===================================================================
# PatternEvolver
# ===================================================================

class PatternEvolver:
    """Evolutionary search engine for behavioral patterns.

    Uses tournament selection, subtree crossover, and multiple mutation
    operators to evolve JsonLogic condition trees. Fitness is computed
    via walk-forward backtesting against signal history.
    """

    def __init__(
        self,
        population_size: int = 30,
        elite_count: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        max_generations: int = 50,
        library_path: Optional[Path] = None,
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations

        self._population: List[PatternSpec] = []
        self._archive: List[PatternSpec] = []  # Best patterns ever found
        self._generation: int = 0
        self._evaluator = DSLEvaluator()
        self._validator = DSLValidator()
        self._library_path = library_path or Path(".aura/evolution/library.json")

    # ----- Population seeding -----

    def seed_population(self, seeds: Optional[List[PatternSpec]] = None):
        """Initialize population from seed patterns + random patterns."""
        self._population = []
        self._generation = 0

        # Add seed patterns
        source_seeds = seeds if seeds is not None else SEED_PATTERNS
        for spec in source_seeds:
            clone = copy.deepcopy(spec)
            clone.generation = 0
            self._population.append(clone)

        # Fill remaining slots with random patterns
        while len(self._population) < self.population_size:
            spec = self._random_pattern_spec()
            self._population.append(spec)

        # Trim to population_size if seeds exceed it
        self._population = self._population[:self.population_size]

    def _random_pattern_spec(self) -> PatternSpec:
        """Generate a random PatternSpec with a random condition tree."""
        condition = random_condition(max_depth=random.randint(2, 3))
        event = random.choice(PREDICTION_EVENTS)
        rule_id = f"evo_{uuid.uuid4().hex[:8]}"

        return PatternSpec(
            rule_id=rule_id,
            version=1,
            generation=self._generation,
            lineage=["random"],
            condition=condition,
            prediction={"event": event},
            action={"gate": f"auto_{event}", "ttl": random.choice([300, 600, 900])},
            meta={"origin": "random"},
        )

    # ----- Fitness evaluation -----

    def evaluate_fitness(
        self, spec: PatternSpec, signal_history: List[Dict]
    ) -> Dict[str, float]:
        """Walk-forward backtest a single pattern against history.

        Returns: {precision, recall, f1, lift, support, complexity, fitness_score}
        """
        if not signal_history:
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "lift": 0.0, "support": 0, "complexity": 0,
                "fitness_score": 0.0,
            }

        # Support both search-native prediction format {"event": "readiness_drop"}
        # and DSL-native format {"target": "readiness_score", "direction": "down"}
        event_type = spec.prediction.get("event")
        if event_type is None:
            target = spec.prediction.get("target", "readiness_score")
            direction = spec.prediction.get("direction", "down")
            event_type = PREDICTION_TARGET_MAP.get(
                (target, direction), "readiness_drop"
            )
        lookahead = LOOKAHEAD.get(event_type, 3)

        tp = 0
        fp = 0
        fn = 0
        total_positives = 0

        for i, ctx in enumerate(signal_history):
            # Check if the predicted event actually occurs in the lookahead window
            actual_event = self._check_event_occurred(
                event_type, signal_history, i, lookahead
            )
            if actual_event:
                total_positives += 1

            # Evaluate the condition
            fires = False
            try:
                fires = self._evaluator.evaluate(spec.condition, ctx)
            except Exception:
                pass

            if fires and actual_event:
                tp += 1
            elif fires and not actual_event:
                fp += 1
            elif not fires and actual_event:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Lift: precision / base_rate
        base_rate = total_positives / len(signal_history) if signal_history else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0

        support = tp + fp  # number of times the pattern fired
        complexity = _count_nodes(spec.condition)

        # Composite fitness score
        fitness_score = (
            0.35 * precision
            + 0.25 * recall
            + 0.25 * (min(lift, 3.0) / 3.0)  # cap lift contribution
            - 0.15 * (min(complexity, 20) / 20.0)
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "lift": round(lift, 4),
            "support": support,
            "complexity": complexity,
            "fitness_score": round(fitness_score, 4),
        }

    def _check_event_occurred(
        self,
        event_type: str,
        signal_history: List[Dict],
        current_idx: int,
        lookahead: int,
    ) -> bool:
        """Check if the predicted event occurred within the lookahead window."""
        end_idx = min(current_idx + lookahead + 1, len(signal_history))
        window = signal_history[current_idx + 1 : end_idx]

        if not window:
            return False

        current = signal_history[current_idx]

        if event_type == "readiness_drop":
            # Check if readiness_score decreased by 10+ in next signals
            current_score = _resolve_ctx(current, "current.readiness_score")
            if current_score is None:
                current_score = _resolve_ctx(current, "readiness_score", 50.0)
            for future in window:
                future_score = _resolve_ctx(future, "current.readiness_score")
                if future_score is None:
                    future_score = _resolve_ctx(future, "readiness_score", 50.0)
                if float(current_score) - float(future_score) >= 10.0:
                    return True
            return False

        elif event_type == "override_loss":
            # Check if next override event was a loss
            for future in window:
                outcome = _resolve_ctx(future, "outcome.streak")
                if outcome is None:
                    outcome = _resolve_ctx(future, "outcome")
                if outcome == "loss" or outcome == "losing":
                    return True
            return False

        elif event_type == "tilt_episode":
            # Check if tilt_score exceeded 0.5 in next signals
            for future in window:
                tilt = _resolve_ctx(future, "current.tilt_score")
                if tilt is None:
                    tilt = _resolve_ctx(future, "tilt_score", 0.0)
                if float(tilt) > 0.5:
                    return True
            return False

        return False

    # ----- Selection -----

    def _tournament_select(self, k: int = 3) -> PatternSpec:
        """Tournament selection: pick k random individuals, return best."""
        candidates = random.sample(
            self._population, min(k, len(self._population))
        )
        return max(
            candidates,
            key=lambda s: s.fitness.get("fitness_score", 0.0),
        )

    # ----- Evolution -----

    def evolve_one_generation(
        self, signal_history: List[Dict]
    ) -> List[PatternSpec]:
        """Run one generation: evaluate fitness, select, crossover, mutate.

        Args:
            signal_history: List of context dicts with outcome data.

        Returns:
            New population after selection + variation.
        """
        if not self._population:
            self.seed_population()

        # 1. Evaluate fitness for all individuals
        for spec in self._population:
            spec.fitness = self.evaluate_fitness(spec, signal_history)

        # 2. Sort by fitness (descending)
        self._population.sort(
            key=lambda s: s.fitness.get("fitness_score", 0.0), reverse=True
        )

        # 3. Update archive with best patterns
        self._update_archive()

        # 4. Elite preservation
        elites = [copy.deepcopy(s) for s in self._population[: self.elite_count]]

        # 5. Build new population
        new_population: List[PatternSpec] = list(elites)

        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self._population) >= 2:
                # Crossover
                parent_a = self._tournament_select()
                parent_b = self._tournament_select()
                child_a_cond, child_b_cond = crossover(
                    parent_a.condition, parent_b.condition
                )

                child_a = self._make_child(parent_a, child_a_cond, parent_b)
                child_b = self._make_child(parent_b, child_b_cond, parent_a)

                # Possibly mutate children
                if random.random() < self.mutation_rate:
                    child_a.condition = self._apply_random_mutation(child_a.condition)
                if random.random() < self.mutation_rate:
                    child_b.condition = self._apply_random_mutation(child_b.condition)

                new_population.append(child_a)
                if len(new_population) < self.population_size:
                    new_population.append(child_b)
            else:
                # Mutation only
                parent = self._tournament_select()
                child = self._make_child(
                    parent,
                    self._apply_random_mutation(copy.deepcopy(parent.condition)),
                )
                new_population.append(child)

        self._population = new_population[: self.population_size]
        self._generation += 1

        # Update generation on all individuals
        for spec in self._population:
            spec.generation = self._generation

        return list(self._population)

    def _make_child(
        self,
        primary_parent: PatternSpec,
        condition: Dict,
        secondary_parent: Optional[PatternSpec] = None,
    ) -> PatternSpec:
        """Create a child PatternSpec from parent(s) with a new condition."""
        lineage = list(primary_parent.lineage)
        if primary_parent.rule_id not in lineage:
            lineage.append(primary_parent.rule_id)
        if secondary_parent and secondary_parent.rule_id not in lineage:
            lineage.append(secondary_parent.rule_id)

        # Keep lineage bounded
        lineage = lineage[-10:]

        return PatternSpec(
            rule_id=f"evo_{uuid.uuid4().hex[:8]}",
            version=1,
            generation=self._generation + 1,
            lineage=lineage,
            condition=condition,
            prediction=dict(primary_parent.prediction),
            action=dict(primary_parent.action),
            meta={"origin": "evolved"},
        )

    def _apply_random_mutation(self, condition: Dict) -> Dict:
        """Apply a random mutation operator."""
        mutations = [
            mutate_threshold,
            mutate_operator,
            mutate_variable,
            mutate_grow,
            mutate_prune,
        ]
        mutation_fn = random.choice(mutations)
        try:
            result = mutation_fn(condition)
            # Validate the result has at least one operator
            if isinstance(result, dict) and result:
                return result
            return condition
        except Exception:
            logger.debug("Mutation failed, returning original condition")
            return condition

    def _update_archive(self):
        """Update the archive with the best patterns from current population."""
        for spec in self._population:
            score = spec.fitness.get("fitness_score", 0.0)
            if score <= 0:
                continue

            # Check if already in archive (by rule_id)
            existing = next(
                (a for a in self._archive if a.rule_id == spec.rule_id), None
            )
            if existing:
                if score > existing.fitness.get("fitness_score", 0.0):
                    self._archive.remove(existing)
                    self._archive.append(copy.deepcopy(spec))
            else:
                self._archive.append(copy.deepcopy(spec))

        # Keep archive bounded (top 100 by fitness)
        self._archive.sort(
            key=lambda s: s.fitness.get("fitness_score", 0.0), reverse=True
        )
        self._archive = self._archive[:100]

    # ----- Adoption -----

    def get_adoptable_patterns(
        self,
        min_precision: float = 0.6,
        min_support: int = 5,
    ) -> List[PatternSpec]:
        """Return patterns from the archive that pass adoption thresholds."""
        adoptable = []
        for spec in self._archive:
            precision = spec.fitness.get("precision", 0.0)
            support = spec.fitness.get("support", 0)
            if precision >= min_precision and support >= min_support:
                adoptable.append(spec)
        return adoptable

    # ----- Persistence -----

    def save_library(self):
        """Persist archive + current population to library.json."""
        self._library_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generation": self._generation,
            "archive": [s.to_dict() for s in self._archive],
            "population": [s.to_dict() for s in self._population],
        }

        tmp = self._library_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._library_path)

        logger.info(
            "Saved evolution library: %d archive, %d population, gen %d",
            len(self._archive),
            len(self._population),
            self._generation,
        )

    def load_library(self):
        """Load archive + population from library.json."""
        if not self._library_path.exists():
            logger.info("No evolution library found at %s", self._library_path)
            return

        try:
            data = json.loads(self._library_path.read_text())
            self._generation = data.get("generation", 0)
            self._archive = [
                PatternSpec.from_dict(d) for d in data.get("archive", [])
            ]
            self._population = [
                PatternSpec.from_dict(d) for d in data.get("population", [])
            ]
            logger.info(
                "Loaded evolution library: %d archive, %d population, gen %d",
                len(self._archive),
                len(self._population),
                self._generation,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load evolution library: %s", e)
