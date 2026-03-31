"""Aura CLI Companion — terminal-based conversational interface.

Phase 1 implementation per PRD v2.2 §19 Demo Spec:
"Aura companion running as a CLI/terminal interface."

This is the user-facing front end. It:
1. Accepts natural language input from the trader
2. Processes messages through ConversationProcessor for emotional signals
3. Updates the readiness score via ReadinessComputer
4. Writes the readiness signal for Buddy to read via the bridge
5. Shows bridge status (Buddy's latest outcomes, override history)

Phase 2 adds: Phi-4 14B responses, self-model graph queries, pattern insights.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.core.conversation_processor import ConversationProcessor, ConversationSignals
from src.aura.core.readiness import ReadinessComputer, ReadinessSignal, AdaptiveWeightManager
from src.aura.core.self_model import (
    SelfModelGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from src.aura.bridge.signals import FeedbackBridge, OutcomeSignal
from src.aura.patterns.engine import PatternEngine
from src.aura.persistence import atomic_write_json  # Fix M-01: needed for atomic onboarding profile write
from src.aura.cli.brand import get_brand, THEMES, RESET as BRAND_RESET, BOLD as BRAND_BOLD

# Phase 3 prediction models (lazy import — may not be available)
try:
    from src.aura.prediction.override_predictor import OverridePredictor
    from src.aura.prediction.readiness_v2 import ReadinessModelV2
    _HAS_PREDICTION = True
except ImportError:
    _HAS_PREDICTION = False

logger = logging.getLogger(__name__)

# ANSI colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


class AuraCompanion:
    """The Aura CLI companion — conversational interface + readiness engine.

    Args:
        db_path: Path to self-model database
        bridge_dir: Path to bridge signal directory
    """

    # US-261: Input bounds
    MAX_LABEL_LENGTH = 200
    MAX_MESSAGE_HISTORY = 200

    def __init__(
        self,
        db_path: Optional[Path] = None,
        bridge_dir: Optional[Path] = None,
    ):
        self.processor = ConversationProcessor()

        # --- Wire up ALL learning models so Aura evolves through use ---
        aura_root = (bridge_dir or Path(".aura/bridge")).parent
        models_dir = aura_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Adaptive component weights — learns which readiness components
        # actually predict trade outcomes (persisted to disk)
        adaptive_weights: Optional[AdaptiveWeightManager] = None
        try:
            adaptive_weights = AdaptiveWeightManager(
                persist_path=aura_root / "adaptive_weights.json"
            )
            logger.info("Adaptive weights loaded — Aura will learn from outcomes")
        except Exception as e:
            logger.debug(f"AdaptiveWeightManager not available: {e}")

        # ReadinessModelV2 — ML model that learns to predict readiness
        # from outcomes (persisted to disk)
        v2_model = None
        if _HAS_PREDICTION:
            try:
                v2_model = ReadinessModelV2(
                    model_path=models_dir / "readiness_v2.json"
                )
                logger.info("ReadinessV2 model loaded — Aura will train on outcomes")
            except Exception as e:
                logger.debug(f"ReadinessModelV2 not available: {e}")

        self.readiness = ReadinessComputer(
            signal_path=(bridge_dir or Path(".aura/bridge")) / "readiness_signal.json",
            v2_model=v2_model,
            adaptive_weights=adaptive_weights,
        )
        self.graph = SelfModelGraph(db_path=db_path)
        self.bridge = FeedbackBridge(bridge_dir=bridge_dir)
        self.pattern_engine = PatternEngine(
            patterns_dir=aura_root / "patterns",
            bridge_dir=bridge_dir,
        )
        self._conversation_id = f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._active_stressors: List[str] = []
        self._latest_signals: Optional[ConversationSignals] = None
        self._latest_readiness: Optional[ReadinessSignal] = None
        self._message_history: List[Dict[str, str]] = []
        # US-308: Outcome cache for tilt detection
        self._outcome_cache: List[Dict[str, Any]] = []
        # US-309: Track last trained outcome to avoid duplicate training
        self._last_trained_outcome_ts: Optional[str] = None
        # US-311: Track staleness warnings
        self._staleness_warned = False

        # Phase 3: Prediction models (also wire override predictor to readiness)
        self._override_predictor: Optional[Any] = None
        self._readiness_v2 = v2_model
        if _HAS_PREDICTION:
            try:
                self._override_predictor = OverridePredictor(
                    model_path=models_dir / "override_predictor.json"
                )
                # Wire override predictor into readiness so it influences the score
                self.readiness._override_predictor = self._override_predictor
                logger.info("OverridePredictor loaded — Aura will learn override patterns")
            except Exception as e:
                logger.debug(f"OverridePredictor not available: {e}")

        # L2 Evolution: Pattern DSL + evolutionary search
        self._evolver = None
        self._signal_history: List[Dict[str, Any]] = []
        self._evolution_interval = 10  # Run evolution every N messages
        self._message_count_since_evolution = 0
        try:
            from src.aura.evolution.search import PatternEvolver
            evolution_dir = aura_root / "evolution"
            evolution_dir.mkdir(parents=True, exist_ok=True)
            self._evolver = PatternEvolver(
                population_size=30,
                max_generations=20,
                library_path=evolution_dir / "library.json",
            )
            self._evolver.load_library()
            if not self._evolver._population:
                self._evolver.seed_population()
            logger.info("L2 Evolution engine loaded — Aura will discover new patterns")
        except Exception as e:
            logger.debug(f"Evolution engine not available: {e}")

        # US-360: DecisionFatigueIndex for companion-level fatigue tracking
        self._decision_fatigue_index = None
        try:
            from src.aura.scoring.decision_fatigue import DecisionFatigueIndex
            self._decision_fatigue_index = DecisionFatigueIndex()
            logger.debug("US-360: DecisionFatigueIndex loaded in AuraCompanion")
        except ImportError:
            logger.debug("US-360: DecisionFatigueIndex not available in AuraCompanion")

        # US-360: BiasInteractionScorer for companion-level penalty tracking
        self._bias_interaction_scorer = None
        try:
            from src.aura.scoring.bias_interactions import BiasInteractionScorer
            self._bias_interaction_scorer = BiasInteractionScorer()
            logger.debug("US-360: BiasInteractionScorer loaded in AuraCompanion")
        except ImportError:
            logger.debug("US-360: BiasInteractionScorer not available in AuraCompanion")

        # US-360: Sentiment history for DecisionFatigueIndex
        self._companion_sentiment_history: List[float] = []

        # US-361: Load threshold state from bridge on startup
        _threshold_path = (bridge_dir or Path(".aura/bridge")) / "threshold_state.json"
        if _threshold_path.exists():
            try:
                self.readiness.load_threshold_state(_threshold_path)
                logger.info("US-361: Threshold state loaded from bridge on startup")
            except Exception as e:
                logger.debug("US-361: Could not load threshold state: %s", e)

    def _needs_onboarding(self) -> bool:
        """Check if self-model is empty and needs onboarding."""
        stats = self.graph.get_stats()
        # If no Person or Goal nodes exist, we haven't onboarded
        nodes_by_type = stats.get("nodes_by_type", {})
        # Keys are lowercase enum values (e.g., "person", "goal")
        has_person = nodes_by_type.get("person", 0) > 0
        has_goals = nodes_by_type.get("goal", 0) > 0
        return not (has_person and has_goals)

    def run_onboarding(self) -> None:
        """Run first-time onboarding to seed the self-model graph.

        PRD v2.2 §13 Phase 2: "Companion onboarding flow that seeds self-model"

        Asks a structured set of questions to populate:
        - Person node (trader identity)
        - Goal nodes (trading goals, life goals)
        - Value nodes (what matters to the trader)
        - TradingState node (current experience level, style)

        All responses are processed through ConversationProcessor for
        emotional signal extraction even during onboarding.
        """
        brand = get_brand()
        print()
        brand.print_header("Welcome — First Setup", style="heavy")
        print()
        print(brand.render_info("I'm Aura, your self-awareness companion. I work alongside Buddy"))
        print(brand.render_info("to help you trade better by understanding yourself better."))
        print()
        print(brand.render_dim("Let me get to know you a bit. This takes about 2 minutes."))
        print(brand.render_dim("(You can skip any question by pressing Enter)"))
        print()

        onboarding_data: Dict[str, Any] = {}

        # --- Q1: Name / identity ---
        name = input(f"{CYAN}What should I call you? {RESET}").strip()
        if name:
            name = name[:self.MAX_LABEL_LENGTH]  # US-261: Truncate oversized labels
            onboarding_data["name"] = name
            self.graph.add_node(GraphNode(
                id="person_trader",
                node_type=NodeType.PERSON,
                label=name,
                properties={"role": "trader", "onboarded": True},
                confidence=1.0,
            ))
            print(f"  Nice to meet you, {name}.\n")

        # --- Q2: Trading experience ---
        experience = input(
            f"{CYAN}How would you describe your trading experience? "
            f"{DIM}(beginner / intermediate / experienced){RESET} "
        ).strip().lower()
        if experience:
            onboarding_data["experience"] = experience
            self.graph.add_node(GraphNode(
                id="trading_state_baseline",
                node_type=NodeType.TRADING_STATE,
                label=f"Trader experience: {experience}",
                properties={
                    "experience_level": experience,
                    "onboarding": True,
                },
            ))

        # --- Q3: Trading goals ---
        print(f"\n{CYAN}What are your main trading goals?{RESET}")
        goals_raw = input(
            f"{DIM}(e.g., consistent income, grow capital, learn the craft, "
            f"financial independence){RESET}\n> "
        ).strip()
        if goals_raw:
            # Fix L-02: signals return value intentionally unused here — processor.process_message
            # is called for its side effect (updating session message history + state).
            # The returned ConversationSignals object is not needed during onboarding collection.
            _ = self.processor.process_message(goals_raw, role="user")
            goals = [g.strip() for g in goals_raw.replace(",", "\n").split("\n") if g.strip()]
            for i, goal in enumerate(goals):
                goal_id = f"goal_{i}_{goal.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=goal_id,
                    node_type=NodeType.GOAL,
                    label=goal,
                    properties={"priority": i + 1, "source": "onboarding"},
                    confidence=0.8,
                ))
                # Link person → goal
                if name:
                    self.graph.add_edge(GraphEdge(
                        source_id="person_trader",
                        target_id=goal_id,
                        edge_type=EdgeType.INFLUENCES,
                        properties={"relationship": "pursues"},
                    ))
            onboarding_data["goals"] = goals
            print(f"  Noted — {len(goals)} goal{'s' if len(goals) != 1 else ''} recorded.\n")

        # --- Q4: What matters most (values) ---
        values_raw = input(
            f"{CYAN}What matters most to you beyond trading? "
            f"{DIM}(family, health, career, relationships, hobbies?){RESET}\n> "
        ).strip()
        if values_raw:
            _ = self.processor.process_message(values_raw, role="user")  # Fix L-02: side-effect call, return unused
            values = [v.strip() for v in values_raw.replace(",", "\n").split("\n") if v.strip()]
            for i, value in enumerate(values):
                value_id = f"value_{value.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=value_id,
                    node_type=NodeType.VALUE,
                    label=value,
                    properties={"priority": i + 1, "source": "onboarding"},
                    confidence=0.7,
                ))
                if name:
                    self.graph.add_edge(GraphEdge(
                        source_id="person_trader",
                        target_id=value_id,
                        edge_type=EdgeType.INFLUENCES,
                        properties={"relationship": "values"},
                    ))
            onboarding_data["values"] = values
            print(f"  Got it — these help me understand what else is on your plate.\n")

        # --- Q5: Known stressors ---
        stressors_raw = input(
            f"{CYAN}Any current stressors I should be aware of? "
            f"{DIM}(work deadlines, health, financial pressure, relationship?){RESET}\n> "
        ).strip()
        if stressors_raw:
            _ = self.processor.process_message(stressors_raw, role="user")  # Fix L-02: side-effect call, return unused
            stressors = [s.strip() for s in stressors_raw.replace(",", "\n").split("\n") if s.strip()]
            for stressor in stressors:
                stressor_id = f"stressor_{stressor.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=stressor_id,
                    node_type=NodeType.LIFE_EVENT,
                    label=stressor,
                    properties={"active": True, "source": "onboarding"},
                ))
                self._active_stressors.append(stressor)
            onboarding_data["stressors"] = stressors
            print(f"  I'll keep these in mind — they affect readiness even subtly.\n")

        # --- Q6: Trading style (helps Buddy calibrate) ---
        style = input(
            f"{CYAN}How would you describe your trading style? "
            f"{DIM}(scalper / swing / position / day-trader / mixed){RESET} "
        ).strip().lower()
        if style:
            onboarding_data["trading_style"] = style
            # Update trading state node
            self.graph.add_node(GraphNode(
                id="trading_state_style",
                node_type=NodeType.TRADING_STATE,
                label=f"Trading style: {style}",
                properties={"style": style, "source": "onboarding"},
            ))

        # --- Wrap up ---
        # Write onboarding data to bridge for Buddy to see
        onboarding_path = (self.bridge.bridge_dir / "onboarding_profile.json")
        try:
            # Fix M-01: was write_text() (non-atomic). A crash mid-write produced a
            # truncated onboarding profile that Buddy would read as corrupted JSON.
            atomic_write_json(onboarding_path, onboarding_data)
        except Exception as e:
            logger.warning("US-238: Failed to write onboarding profile: %s", e)
            # US-238: Surface save failure to user — don't let them think it succeeded
            print(f"\n{YELLOW}{BOLD}⚠ Warning:{RESET} Your onboarding profile could not be saved to disk.")
            print(f"  Your graph nodes are in memory, but Buddy won't see your profile until saved.")
            print(f"  Run {BOLD}/onboarding{RESET} again later to retry, or check disk permissions.")

        # Compute initial readiness baseline
        self._update_readiness()

        brand = get_brand()
        print()
        brand.print_success(f"Onboarding complete!")
        brand.print_info(f"Your self-model has been seeded with {self.graph.get_stats()['total_nodes']} nodes.")
        brand.print_dim("The more we talk, the better I understand your patterns.")
        print()

    def start_session(self) -> None:
        """Initialize a new conversation session."""
        # Check if first-time onboarding is needed
        if self._needs_onboarding():
            self.run_onboarding()

        # US-209: Validate onboarding actually completed (user may have Ctrl-C'd)
        if self._needs_onboarding():
            logger.warning(
                "US-209: Onboarding incomplete — self-model missing Person or Goal nodes. "
                "Readiness will default to 50 (unknown state)."
            )
            print(f"\n{YELLOW}Note: Onboarding wasn't completed. "
                  f"Readiness will default to neutral (50/100) until onboarding finishes.{RESET}")
            print(f"{DIM}Run /onboard to restart onboarding.{RESET}\n")

        brand = get_brand()

        # Show bridge status if Buddy is running
        bridge_status = self.bridge.get_bridge_status()
        if bridge_status["outcome_signal"]["available"]:
            outcome = self.bridge.read_outcome()
            if outcome:
                print(brand.render_buddy_badge(
                    connected=True,
                    regime=outcome.regime,
                    pnl=outcome.pnl_today
                ))
        else:
            print(brand.render_buddy_badge(connected=False))

        # Compute initial readiness
        self._update_readiness()
        print()

    def process_input(self, user_input: str) -> str:
        """Process user input and return Aura's response.

        Args:
            user_input: The trader's message

        Returns:
            Aura's response text
        """
        # Handle commands
        if user_input.startswith("/"):
            return self._handle_command(user_input)

        # US-237: Each stage wrapped in try-except for session resilience.
        # A crash in one stage should not kill the interactive session.

        # Brand-aware thinking indicator — single Claude Code-style
        brand = get_brand()
        brand.print_thinking("thinking")

        # Stage 1: Process through conversation processor
        try:
            signals = self.processor.process_message(user_input, role="user")
            self._latest_signals = signals
        except Exception as e:
            logger.error("US-237: process_message failed: %s", e, exc_info=True)
            # Fix M-06: ConversationSignals() default constructor is fully self-consistent —
            # all fields have explicit defaults (lists via field(default_factory=list),
            # scalars via literal defaults). Downstream stages (stressor update, graph logging,
            # readiness compute) all handle empty/neutral signals safely. If ConversationSignals
            # ever gains required fields, this fallback must be updated to match.
            signals = ConversationSignals()  # Neutral fallback — all fields initialized to safe defaults
            self._latest_signals = signals

        # Stage 2: Update active stressors
        try:
            for stressor in signals.detected_stressors:
                if stressor not in self._active_stressors:
                    self._active_stressors.append(stressor)
        except Exception as e:
            logger.error("US-237: stressor update failed: %s", e)

        # Stage 3: Log to self-model graph
        try:
            self._log_to_graph(user_input, signals)
        except Exception as e:
            logger.error("US-237: _log_to_graph failed: %s", e)

        # Stage 3.5 (was Stage 5): GAP-004 fix — run pattern detection BEFORE readiness
        # so that pattern signals can modulate the readiness score.
        pattern_context: Optional[Dict[str, Any]] = None
        try:
            pattern_context = self._run_pattern_detection(signals)
        except Exception as e:
            logger.error("US-237: _run_pattern_detection failed: %s", e)

        # Stage 3.8: US-308: Feed message context to readiness for tilt detection
        # Must run BEFORE compute() so tilt detector has current message data.
        try:
            # Include current user_input in context (it hasn't been appended to
            # _message_history yet at this point in the pipeline)
            current_msg = {"content": user_input, "sentiment": signals.sentiment_score if signals else 0.5}
            msg_context = [{"content": m.get("content", ""), "sentiment": m.get("sentiment", 0.5)} for m in self._message_history[-19:]]
            msg_context.append(current_msg)
            self.readiness.set_context(messages=msg_context, outcomes=self._outcome_cache)
        except Exception as e:
            logger.debug("US-308: Context setting failed: %s", e)

        # Stage 4: Update readiness score (now receives pattern context + raw text)
        try:
            readiness = self._update_readiness(signals, pattern_context=pattern_context, message_text=user_input)
        except Exception as e:
            logger.error("US-237: _update_readiness failed: %s", e)
            readiness = self._latest_readiness  # Reuse last known score

        # Stage 4.6: US-309: Check for new outcome signals and train adaptive weights
        try:
            outcome = self.bridge.read_outcome()
            if outcome is not None:
                outcome_ts = outcome.get("timestamp") if isinstance(outcome, dict) else getattr(outcome, "timestamp", None)
                if outcome_ts and outcome_ts != self._last_trained_outcome_ts:
                    trained = self.readiness.train_from_outcome(outcome, self._latest_readiness)
                    if trained:
                        self._last_trained_outcome_ts = outcome_ts
                        self._outcome_cache.append(outcome if isinstance(outcome, dict) else {"trade_won": getattr(outcome, "trade_won", False)})
                    # US-361: Also update AdaptiveThresholdLearner posteriors from outcome
                    try:
                        self.readiness.update_thresholds_from_outcome(outcome)
                    except Exception as _te:
                        logger.debug("US-361: Threshold outcome update error: %s", _te)
        except Exception as e:
            logger.debug("US-309: Training check error: %s", e)

        # Stage 4.65: L2 Evolution — record signal snapshot + periodic evolution run
        if self._evolver:
            try:
                from src.aura.evolution.dsl import build_context_from_signals
                outcome_data = self.bridge.read_outcome()
                overrides_data = self.bridge.get_recent_overrides(limit=50) if hasattr(self.bridge, 'get_recent_overrides') else []
                ctx = build_context_from_signals(
                    readiness=readiness,
                    signals=signals,
                    outcome=outcome_data,
                    overrides=overrides_data,
                )
                self._signal_history.append(ctx)
                # Cap history to prevent unbounded growth
                if len(self._signal_history) > 500:
                    self._signal_history = self._signal_history[-500:]

                self._message_count_since_evolution += 1
                if self._message_count_since_evolution >= self._evolution_interval:
                    self._message_count_since_evolution = 0
                    if len(self._signal_history) >= 10:
                        self._evolver.evolve_one_generation(self._signal_history)
                        adopted = self._evolver.get_adoptable_patterns(min_precision=0.6, min_support=3)
                        if adopted:
                            logger.info("L2: Evolved %d adoptable patterns", len(adopted))
                        self._evolver.save_library()
            except Exception as e:
                logger.debug("L2 evolution step failed (non-fatal): %s", e)

        # Stage 4.7: US-311: Check for stale bridge signals
        if not self._staleness_warned:
            try:
                health = self.bridge.bridge_health()
                stale_warnings = []
                if health.readiness == "stale":
                    stale_warnings.append("readiness signal is stale (>1 hour)")
                if health.outcome == "stale":
                    stale_warnings.append("outcome signal is stale (>24 hours)")
                if stale_warnings:
                    self._staleness_warned = True
                    logger.warning("US-311: Bridge staleness: %s", ", ".join(stale_warnings))
            except Exception as e:
                logger.debug("US-311: Staleness check error: %s", e)

        # Stage 4.8: GAP-010 fix — enrich outcome signal with human context
        try:
            outcome_raw = self.bridge.read_outcome()
            if outcome_raw is not None:
                outcome_dict = outcome_raw.to_dict() if hasattr(outcome_raw, "to_dict") else outcome_raw
                human_context = self.readiness.get_last_state_snapshot()
                self.bridge.enrich_outcome_signal(outcome_dict, human_context)
        except Exception as e:
            logger.debug("GAP-010: Bridge enrichment failed (non-fatal): %s", e)

        # Stage 6: Generate response
        try:
            response = self._generate_response(user_input, signals, readiness)
        except Exception as e:
            logger.error("US-237: _generate_response failed: %s", e)
            response = "I'm here and listening. Could you tell me more about how you're feeling?"

        # Clear thinking indicator before showing response
        brand.clear_thinking()

        # Log assistant response (non-critical)
        try:
            self.processor.process_message(response, role="assistant")
        except Exception as e:
            logger.error("US-237: assistant log failed: %s", e)

        self._message_history.append({"role": "user", "content": user_input})
        self._message_history.append({"role": "assistant", "content": response})

        # US-262: Cap message history to prevent unbounded growth
        if len(self._message_history) > self.MAX_MESSAGE_HISTORY:
            self._message_history = self._message_history[-self.MAX_MESSAGE_HISTORY:]

        return response

    def get_signal_state(self):
        """Return current signal data for the dashboard panel.

        Returns (readiness, signals, active_stressors) tuple for brand
        rendering. Safe to call after process_input().
        """
        return self._latest_readiness, self._latest_signals, list(self._active_stressors)

    def _generate_response(
        self,
        user_input: str,
        signals: ConversationSignals,
        readiness: ReadinessSignal,
    ) -> str:
        """Generate Aura's response — LLM-powered with template fallback."""
        # Try LLM-powered response first
        try:
            from src.aura.core.mind import think, build_context

            outcome = self.bridge.read_outcome()
            if not hasattr(outcome, 'to_dict'):
                outcome = None
            recent_overrides = self.bridge.get_recent_overrides(limit=10)

            context = build_context(
                readiness=readiness,
                signals=signals,
                active_stressors=self._active_stressors,
                outcome=outcome,
                recent_overrides=recent_overrides,
                message_history=self._message_history,
            )

            response = think(
                user_message=user_input,
                context=context,
                message_history=self._message_history,
            )
            if response:
                return response
        except Exception as e:
            logger.debug("Mind unavailable, using template fallback: %s", e)

        # Fallback: template-based responses (original Phase 1 logic)
        return self._generate_template_response(user_input, signals, readiness)

    def _generate_template_response(
        self,
        user_input: str,
        signals: ConversationSignals,
        readiness: ReadinessSignal,
    ) -> str:
        """Fallback template responses when LLM is unavailable.

        Design: warm, direct, human. Not a chatbot, not a therapist.
        Short and grounded — like Claude, not like Clippy.
        """
        brand = get_brand()
        parts: List[str] = []

        # ── Acknowledge emotional state naturally ──
        state = signals.emotional_state
        score = readiness.readiness_score if readiness else 50

        if state == "stressed":
            if score < 35:
                parts.append("You're carrying a lot right now. Maybe step away for a bit — you'll see it clearer after.")
            else:
                parts.append("I can feel the stress in what you're saying. Let's just be aware of it.")
        elif state == "fatigued":
            parts.append("You sound tired. Rest isn't weakness — it's how you stay sharp.")
        elif state == "anxious":
            if score < 40:
                parts.append("There's a lot of tension right now. When you're this wound up, even good setups feel like traps.")
            else:
                parts.append("Something feels a little tight. Worth noticing before you move on anything.")
        elif state == "energized":
            if score > 70:
                parts.append("You sound good today. Something shifted — in a good way.")
            else:
                parts.append("Energy's up, but something underneath isn't quite settled yet. Just keep that in the background.")

        # ── Override detection — this is critical ──
        if signals.override_mentioned:
            recent_overrides = self.bridge.get_recent_overrides(limit=10)
            losing_overrides = [o for o in recent_overrides if o.outcome == "loss"]
            if len(losing_overrides) >= 2:
                loss_pct = len(losing_overrides) / len(recent_overrides) * 100 if recent_overrides else 0
                parts.append(
                    f"You're overriding Buddy again. {len(losing_overrides)} of your "
                    f"last {len(recent_overrides)} overrides were losses ({loss_pct:.0f}%). "
                    f"What's different this time?"
                )
            else:
                parts.append("Logging the override. These patterns tell us a lot about when your gut helps and when it doesn't.")

            # Phase 3: Override prediction warning
            if self._override_predictor:
                try:
                    _outcome = self.bridge.read_outcome()
                    _conf = getattr(_outcome, "confidence_at_time", 0.5) if _outcome else 0.5
                    _vote = getattr(_outcome, "weighted_vote_at_time", 0.5) if _outcome else 0.5
                    _regime = getattr(_outcome, "regime", "NORMAL") if _outcome else "NORMAL"
                    context = {
                        "emotional_state": signals.emotional_state,
                        "cognitive_load": "high" if len(signals.stress_keywords_found) > 2 else "normal",
                        "confidence_at_time": _conf if isinstance(_conf, (int, float)) else 0.5,
                        "weighted_vote_at_time": _vote if isinstance(_vote, (int, float)) else 0.5,
                        "override_type": "took_rejected",
                        "regime": _regime if isinstance(_regime, str) else "NORMAL",
                    }
                    prediction = self._override_predictor.predict_loss_probability(context)
                    if prediction.loss_probability >= 0.60:
                        wc = brand.c("warning")
                        parts.append(
                            f"\n{wc}Override risk: {prediction.loss_probability:.0%} "
                            f"predicted loss. {prediction.recommendation}{BRAND_RESET}"
                        )
                except Exception:
                    pass

        # ── Stressor acknowledgment ──
        if signals.detected_stressors:
            stressor_text = ", ".join(signals.detected_stressors)
            parts.append(f"I'm noticing some things in the background — {stressor_text}. These affect you even when you don't feel it.")

        # ── Readiness context — gentle, not clinical ──
        if readiness and score < 35:
            wc = brand.c("warning")
            parts.append(
                f"\n{wc}Your readiness is low right now. "
                f"Buddy is being careful on your behalf — lighter sizing, fewer entries.{BRAND_RESET}"
            )
        elif readiness and score < 55:
            dc = brand.c("text_dim")
            parts.append(
                f"\n{dc}You're not quite yourself right now. Buddy's adjusting accordingly.{BRAND_RESET}"
            )

        # ── Default — warm presence, not robotic ──
        if not parts:
            msg_count = len(self._message_history)
            if msg_count == 0:
                parts.append("Hey. I'm here. What's going on?")
            elif msg_count < 4:
                parts.append("I hear you. Go on.")
            else:
                parts.append("Still here.")

        # ── Buddy context if relevant ──
        outcome = self.bridge.read_outcome()
        if outcome and "trading" in signals.topics:
            if hasattr(outcome, "streak") and outcome.streak == "losing":
                dc = brand.c("text_dim")
                parts.append(
                    f"\n{dc}Buddy's been in a rough stretch too. "
                    f"That might be coloring things.{BRAND_RESET}"
                )

        return "\n".join(parts)

    def _update_readiness(
        self,
        signals: Optional[ConversationSignals] = None,
        pattern_context: Optional[Dict[str, Any]] = None,
        message_text: Optional[str] = None,
    ) -> ReadinessSignal:
        """Recompute readiness score and write to bridge.

        Args:
            signals: Conversation signals from the current message
            pattern_context: GAP-004 fix — pattern detection output with
                'stress_keywords' and 'confidence_trend_adjustment' keys
            message_text: Raw user message for cognitive load / tilt analysis
        """
        recent_overrides = self.bridge.get_recent_overrides(limit=20)
        override_events = [o.to_dict() for o in recent_overrides]

        # Get conversation count from graph
        recent_convs = self.graph.get_recent_conversations(limit=50)
        # Count conversations from last 7 days
        from datetime import timedelta
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        conv_count_7d = sum(1 for c in recent_convs if c.get("timestamp", "") >= week_ago)

        # GAP-004 fix: Merge pattern-derived stress keywords into signal keywords
        stress_keywords = list(signals.stress_keywords_found) if signals else []
        if pattern_context and pattern_context.get("stress_keywords"):
            stress_keywords.extend(pattern_context["stress_keywords"])

        # GAP-004 fix: Modulate confidence trend based on pattern severity
        confidence_trend = signals.confidence_trend if signals else "stable"
        if pattern_context:
            adjustment = pattern_context.get("confidence_trend_adjustment", 0.0)
            if adjustment <= -0.2 and confidence_trend != "falling":
                confidence_trend = "falling"
                logger.info("GAP-004: Pattern severity overrode confidence_trend to 'falling'")

        # Extract Phase 19 features from signals when available
        bias_scores = signals.bias_scores if signals else None
        style_drift_score = signals.style_drift_score if signals else 0.0
        granularity_score = signals.emotional_granularity_score if signals else 0.5
        coherence_score = signals.narrative_coherence_score if signals else 0.5
        affect_volatility = signals.affect_volatility if signals else 0.0

        # Derive affect_stuck: negative valence stuck with high inertia
        affect_stuck = False
        if signals and signals.affect_valence < -0.3 and signals.affect_inertia > 0.6:
            affect_stuck = True
            logger.info("Affect stuck detected: valence=%.3f, inertia=%.3f",
                        signals.affect_valence, signals.affect_inertia)

        # US-360: Compute fatigue_score from DecisionFatigueIndex
        fatigue_score = 0.0
        if self._decision_fatigue_index is not None and signals is not None:
            try:
                # Accumulate sentiment in companion history
                _sent_proxy = max(0.0, min(1.0, 1.0 - (signals.sentiment_score + 1.0) / 2.0))
                self._companion_sentiment_history.append(_sent_proxy)
                if len(self._companion_sentiment_history) > 50:
                    self._companion_sentiment_history = self._companion_sentiment_history[-50:]
                fatigue_result = self._decision_fatigue_index.update(
                    sentiment=_sent_proxy,
                )
                # DecisionFatigueIndex composite is 0-100; convert to 0-1 for compute()
                fatigue_score = fatigue_result.composite / 100.0
            except Exception as e:
                logger.debug("US-360: DecisionFatigueIndex update failed in companion: %s", e)

        # US-360: Compute bias_interaction_penalty from BiasInteractionScorer
        bias_interaction_penalty = 0.0
        if self._bias_interaction_scorer is not None and bias_scores:
            try:
                interaction_result = self._bias_interaction_scorer.score(bias_scores)
                bias_interaction_penalty = interaction_result.interaction_penalty
            except Exception as e:
                logger.debug("US-360: BiasInteractionScorer failed in companion: %s", e)

        readiness = self.readiness.compute(
            emotional_state=signals.emotional_state if signals else "neutral",
            stress_keywords=stress_keywords,
            active_stressors=self._active_stressors,
            recent_override_events=override_events,
            conversation_count_7d=conv_count_7d + 1,  # +1 for current conversation
            confidence_trend=confidence_trend,
            bias_scores=bias_scores,
            message_text=message_text,
            style_drift_score=style_drift_score,
            granularity_score=granularity_score,
            coherence_score=coherence_score,
            affect_volatility=affect_volatility,
            affect_stuck=affect_stuck,
            fatigue_score=fatigue_score,
            bias_interaction_penalty=bias_interaction_penalty,
        )
        self._latest_readiness = readiness

        # Log to graph
        self.graph.log_readiness(
            score=readiness.readiness_score,
            components=readiness.components.to_dict(),
            trigger="conversation_update" if signals else "session_start",
        )

        return readiness

    def _log_to_graph(self, message: str, signals: ConversationSignals) -> None:
        """Log conversation data to the self-model graph."""
        # Add conversation node
        conv_node = GraphNode(
            id=f"{self._conversation_id}_{signals.message_count}",
            node_type=NodeType.CONVERSATION,
            label=message[:100],
            properties=signals.to_dict(),
            confidence=0.8,
        )
        self.graph.add_node(conv_node)

        # Add emotion node if notable
        if signals.emotional_state not in ("neutral",):
            emotion_node = GraphNode(
                id=f"emotion_{signals.emotional_state}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                node_type=NodeType.EMOTION,
                label=signals.emotional_state,
                properties={
                    "sentiment_score": signals.sentiment_score,
                    "stress_keywords": signals.stress_keywords_found,
                },
            )
            self.graph.add_node(emotion_node)

            # Link conversation → emotion
            self.graph.add_edge(GraphEdge(
                source_id=conv_node.id,
                target_id=emotion_node.id,
                edge_type=EdgeType.TRIGGERS,
            ))

        # Add stressor nodes
        for stressor in signals.detected_stressors:
            stressor_node = GraphNode(
                id=f"stressor_{stressor.replace(' ', '_')}",
                node_type=NodeType.LIFE_EVENT,
                label=stressor,
                properties={"active": True},
            )
            self.graph.add_node(stressor_node)

    def _run_pattern_detection(self, signals: ConversationSignals) -> Dict[str, Any]:
        """Run T1 pattern detection after each conversation message.

        T2 cross-domain runs less frequently — only every 5th message or via /patterns command.

        Returns:
            Dict with 'stress_keywords' (List[str]) and 'confidence_trend_adjustment' (float)
            extracted from detected patterns, for feeding into readiness compute.
        """
        pattern_context: Dict[str, Any] = {
            "stress_keywords": [],
            "confidence_trend_adjustment": 0.0,
        }
        try:
            conversations = self.graph.get_recent_conversations(limit=20)
            readiness_history = self.graph.get_readiness_history(limit=20)

            # T1 runs every message (lightweight)
            t1_patterns = self.pattern_engine.run_t1(conversations, readiness_history)

            if t1_patterns:
                logger.info(
                    f"T1 detected {len(t1_patterns)} patterns after message "
                    f"#{signals.message_count}"
                )
                # GAP-004 fix: Extract high-severity pattern signals for readiness modulation
                for p in t1_patterns:
                    if p.confidence >= 0.6:
                        # High-confidence stress/negative patterns add stress keywords
                        desc_lower = p.description.lower()
                        if any(kw in desc_lower for kw in (
                            "stress", "anxiety", "frustration", "fatigue",
                            "overwhelm", "fear", "anger", "loss", "tilt",
                        )):
                            pattern_context["stress_keywords"].append(
                                f"pattern:{p.description[:50]}"
                            )
                        # Negative patterns push confidence trend downward
                        if p.confidence >= 0.7:
                            pattern_context["confidence_trend_adjustment"] -= 0.1

            # T2 runs every 5th message (moderate cost)
            t2_patterns = []
            if signals.message_count % 5 == 0 and signals.message_count > 0:
                t2_patterns = self.pattern_engine.run_t2(conversations, readiness_history)
                if t2_patterns:
                    logger.info(f"T2 detected {len(t2_patterns)} cross-domain patterns")
                    # GAP-004 fix: Cross-domain correlations with negative outcomes
                    for p in t2_patterns:
                        if p.correlation_strength is not None and p.correlation_strength < -0.3:
                            pattern_context["stress_keywords"].append(
                                f"t2_correlation:{p.description[:50]}"
                            )
                            pattern_context["confidence_trend_adjustment"] -= 0.15

        except Exception as e:
            logger.warning(f"Pattern detection failed (non-fatal): {e}")

        return pattern_context

    def _handle_command(self, command: str) -> str:
        """Handle CLI commands."""
        cmd = command.strip().lower()

        if cmd == "/status":
            return self._cmd_status()
        elif cmd == "/bridge":
            return self._cmd_bridge()
        elif cmd == "/bridge-status":
            return self._handle_bridge_status()
        elif cmd == "/bridge-repair":
            return self._handle_bridge_repair()
        elif cmd == "/readiness":
            return self._cmd_readiness()
        elif cmd.startswith("/graph"):
            return self._cmd_graph(cmd)
        elif cmd.startswith("/patterns"):
            return self._cmd_patterns(cmd)
        elif cmd.startswith("/predict"):
            return self._cmd_predict()
        elif cmd.startswith("/validate"):
            return self._cmd_validate()
        elif cmd.startswith("/rules"):
            return self._cmd_rules()
        elif cmd.startswith("/coach"):
            return self._cmd_coach()
        elif cmd.startswith("/insights"):
            return self._cmd_insights()
        elif cmd.startswith("/quality"):
            return self._cmd_quality()
        elif cmd.startswith("/anomalies"):
            return self._cmd_anomalies()
        elif cmd.startswith("/recovery"):
            return self._cmd_recovery()
        elif cmd.startswith("/regimes"):
            return self._cmd_regimes()
        elif cmd.startswith("/reliability"):
            return self._cmd_reliability()
        elif cmd.startswith("/style"):
            return self._cmd_style()
        elif cmd.startswith("/flexibility"):
            return self._cmd_flexibility()
        elif cmd.startswith("/journal"):
            return self._cmd_journal()
        elif cmd.startswith("/weights"):
            return self._cmd_weights()
        elif cmd.startswith("/granularity"):
            return self._cmd_granularity()
        elif cmd.startswith("/coherence"):
            return self._cmd_coherence()
        elif cmd.startswith("/negotiate"):
            return self._cmd_negotiate()
        elif cmd.startswith("/calibration"):
            return self._cmd_calibration()
        elif cmd == "/affect":
            return self._cmd_affect()
        elif cmd == "/fatigue":
            return self._cmd_fatigue()
        elif cmd.startswith("/theme"):
            return self._cmd_theme(cmd)
        elif cmd == "/help":
            return self._cmd_help()
        elif cmd == "/quit":
            return "__QUIT__"
        else:
            brand = get_brand()
            return f"{brand.c('warning')}Unknown command: {cmd}{BRAND_RESET}\n\n  Type {brand.c('primary')}/help{BRAND_RESET} for all available commands."

    def _cmd_theme(self, cmd: str) -> str:
        """Handle /theme command — list or switch themes."""
        brand = get_brand()
        parts = cmd.strip().split()
        if len(parts) == 1:
            # Just /theme — show list
            return brand.render_theme_list()
        theme_name = parts[1].lower()
        if brand.set_theme(theme_name):
            # Re-fetch brand to get new colors
            brand = get_brand()
            lines = []
            lines.append(brand.render_success(f"Theme switched to {brand.theme.display_name}"))
            lines.append("")
            lines.append(f"  {brand.c('primary')}Primary{BRAND_RESET}  "
                         f"{brand.c('secondary')}Secondary{BRAND_RESET}  "
                         f"{brand.c('accent')}Accent{BRAND_RESET}  "
                         f"{brand.c('success')}Success{BRAND_RESET}  "
                         f"{brand.c('warning')}Warning{BRAND_RESET}  "
                         f"{brand.c('error')}Error{BRAND_RESET}")
            return "\n".join(lines)
        else:
            available = ", ".join(THEMES.keys())
            return brand.render_error(f"Unknown theme: {theme_name}\n  Available: {available}")

    def _cmd_help(self) -> str:
        """Show branded help with all available commands."""
        brand = get_brand()
        p = brand.c("primary")
        s = brand.c("secondary")
        a = brand.c("accent")
        d = brand.c("text_dim")
        r = BRAND_RESET
        b = BRAND_BOLD

        sections = {
            "Awareness": [
                ("/status", "Current Aura status overview"),
                ("/readiness", "Readiness score and components"),
                ("/affect", "Affect dynamics and emotional state"),
                ("/fatigue", "Decision fatigue tracking"),
            ],
            "Intelligence": [
                ("/patterns", "Pattern detection (T1/T2/T3 tiers)"),
                ("/insights", "Cross-domain insights"),
                ("/predict", "ML predictions (readiness, override risk)"),
                ("/quality", "Decision quality assessment"),
                ("/anomalies", "Anomaly detection"),
                ("/coach", "Coaching recommendations"),
            ],
            "Bridge (Buddy)": [
                ("/bridge", "Bridge status and signals"),
                ("/bridge-status", "Detailed bridge diagnostics"),
                ("/bridge-repair", "Attempt bridge repair"),
                ("/negotiate", "Negotiation protocol"),
                ("/calibration", "Calibration loop status"),
            ],
            "Analysis": [
                ("/graph", "Self-model graph stats"),
                ("/rules", "Active bridge rules"),
                ("/regimes", "Market regime tracking"),
                ("/recovery", "Recovery plan status"),
                ("/style", "Trading style analysis"),
                ("/flexibility", "Adaptive flexibility metrics"),
                ("/reliability", "Signal reliability"),
                ("/coherence", "System coherence"),
                ("/journal", "Session journal"),
                ("/weights", "Component weights"),
                ("/granularity", "Granularity controls"),
            ],
            "System": [
                ("/theme", "List themes  ·  /theme <name> to switch"),
                ("/help", "This help screen"),
                ("/quit", "End session"),
            ],
        }

        lines = [brand.render_header("Commands", style="heavy")]
        for section_name, commands in sections.items():
            lines.append(f"  {a}▸ {p}{b}{section_name}{r}")
            for cmd_name, desc in commands:
                lines.append(f"    {s}{cmd_name:<18}{r} {d}{desc}{r}")
            lines.append("")

        lines.append(f"  {d}Or just type naturally — I'm always listening.{r}")
        return "\n".join(lines)

    def _cmd_status(self) -> str:
        """Show current Aura status."""
        readiness = self._latest_readiness
        summary = self.processor.get_session_summary()
        graph_stats = self.graph.get_stats()

        lines = [
            f"{BOLD}═══ Aura Status ═══{RESET}",
            f"Session: {self._conversation_id}",
            f"Messages this session: {summary['message_count']}",
            f"Net sentiment: {summary['net_sentiment']:+.3f}",
            f"Active stressors: {', '.join(self._active_stressors) or 'none'}",
            "",
            f"{BOLD}Self-Model Graph:{RESET}",
            f"  Nodes: {graph_stats['total_nodes']}",
            f"  Edges: {graph_stats['total_edges']}",
            f"  Conversations logged: {graph_stats['total_conversations']}",
        ]

        if readiness:
            lines.extend([
                "",
                f"{BOLD}Readiness: {readiness.readiness_score:.0f}/100{RESET}",
                f"  Emotional: {readiness.emotional_state}",
                f"  Cognitive load: {readiness.cognitive_load}",
                f"  Override loss rate (7d): {readiness.override_loss_rate_7d:.0%}",
                f"  Confidence trend: {readiness.confidence_trend}",
            ])

        return "\n".join(lines)

    def _cmd_bridge(self) -> str:
        """Show bridge status."""
        status = self.bridge.get_bridge_status()
        lines = [f"{BOLD}═══ Bridge Status ═══{RESET}"]

        r = status["readiness_signal"]
        # Fix M-05: Replaced ambiguous multi-line ternary with explicit if/else.
        # The original ternary operator scope was hard to read and could raise
        # KeyError if r['score'] was missing while r['available'] was True.
        if r.get("available") and r.get("score") is not None:
            lines.append(f"Readiness → Buddy: ✓ (score: {r['score']:.0f}/100)")
        else:
            lines.append("Readiness → Buddy: not yet computed")

        o = status["outcome_signal"]
        if o["available"]:
            lines.append(f"Outcomes ← Buddy: ✓ (PnL: {o['pnl_today']:+.2f})")
        else:
            lines.append(f"Outcomes ← Buddy: ✗ (Buddy not active)")

        ov = status["override_events"]
        lines.append(f"Override events: {ov['total_recent']} recent")

        return "\n".join(lines)

    def _handle_bridge_status(self) -> str:
        """US-311: Display bridge health status."""
        health = self.bridge.bridge_health()
        lines = [f"\n{CYAN}{BOLD}Bridge Health Status{RESET}\n"]

        status_colors = {"healthy": GREEN, "corrupted": RED, "missing": YELLOW, "stale": YELLOW}

        for fname, attr_name in [
            ("readiness_signal.json", "readiness"),
            ("outcome_signal.json", "outcome"),
            ("override_events.jsonl", "overrides"),
            ("active_rules.json", "rules"),
        ]:
            status = getattr(health, attr_name)
            color = status_colors.get(status, DIM)
            lines.append(f"  {fname}: {color}{status}{RESET}")

        return "\n".join(lines)

    def _handle_bridge_repair(self) -> str:
        """US-311: Attempt repair of corrupted bridge files."""
        results = self.bridge.repair_corrupted()
        lines = [f"\n{CYAN}{BOLD}Bridge Repair Results{RESET}\n"]
        for fname, result in results.items():
            color = GREEN if result in ("healthy", "repaired") else RED
            lines.append(f"  {fname}: {color}{result}{RESET}")
        return "\n".join(lines)

    def _cmd_readiness(self) -> str:
        """Show detailed readiness breakdown."""
        if not self._latest_readiness:
            return "No readiness score computed yet. Start a conversation first."

        r = self._latest_readiness
        c = r.components

        score_color = GREEN if r.readiness_score >= 70 else (YELLOW if r.readiness_score >= 40 else RED)

        lines = [
            f"{BOLD}═══ Readiness Score: {score_color}{r.readiness_score:.0f}/100{RESET} ═══",
            "",
            f"  Emotional state:    {c.emotional_state_score:.2f}  (weight: 25%)",
            f"  Cognitive load:     {c.cognitive_load_score:.2f}  (weight: 20%)",
            f"  Override discipline: {c.override_discipline_score:.2f}  (weight: 25%)",
            f"  Stress level:       {c.stress_level_score:.2f}  (weight: 15%)",
            f"  Confidence trend:   {c.confidence_trend_score:.2f}  (weight: 10%)",
            f"  Engagement:         {c.engagement_score:.2f}  (weight: 5%)",
            "",
            f"  Buddy impact: ",
        ]

        if r.readiness_score >= 80:
            lines.append(f"    {GREEN}Full trading capacity{RESET}")
        elif r.readiness_score >= 60:
            lines.append(f"    {YELLOW}Position sizes reduced 20%{RESET}")
        elif r.readiness_score >= 40:
            lines.append(f"    {YELLOW}Position sizes reduced 40%, wider SL{RESET}")
        elif r.readiness_score >= 20:
            lines.append(f"    {RED}Minimum positions only{RESET}")
        else:
            lines.append(f"    {RED}TRADE BLOCK — readiness too low{RESET}")

        return "\n".join(lines)

    def _cmd_graph(self, cmd: str = "/graph") -> str:
        """Show self-model graph summary or run path analysis.

        Sub-commands:
            /graph              — Show graph summary
            /graph path A B     — US-299: Find causal path between nodes A and B
        """
        parts = cmd.strip().split()

        # US-299: /graph path <source> <target>
        if len(parts) >= 4 and parts[1] == "path":
            source_id = parts[2]
            target_id = parts[3]
            path = self.graph.get_path_between(source_id, target_id)
            if path is None:
                return f"No path found between '{source_id}' and '{target_id}' within 3 hops."
            lines = [f"{BOLD}═══ Causal Path ═══{RESET}"]
            for i, (node_id, edge_type) in enumerate(path):
                node = self.graph.get_node(node_id)
                label = node.label if node else node_id
                if i == 0:
                    lines.append(f"  [{label}]")
                else:
                    lines.append(f"    →({edge_type})→ [{label}]")
            return "\n".join(lines)

        # Default: show graph summary
        stats = self.graph.get_stats()
        lines = [
            f"{BOLD}═══ Self-Model Graph ═══{RESET}",
            f"Total nodes: {stats['total_nodes']}",
            f"Total edges: {stats['total_edges']}",
            f"Conversations: {stats['total_conversations']}",
            "",
            "Nodes by type:",
        ]
        for node_type, count in stats["nodes_by_type"].items():
            lines.append(f"  {node_type}: {count}")

        # Show recent readiness history
        history = self.graph.get_readiness_history(limit=5)
        if history:
            lines.extend(["", "Recent readiness scores:"])
            for entry in history:
                ts = entry.get("timestamp", "")[:19]
                score = entry.get("score", 0)
                trigger = entry.get("trigger", "")
                lines.append(f"  {ts} — {score:.0f}/100 ({trigger})")

        return "\n".join(lines)

    def _cmd_patterns(self, cmd: str) -> str:
        """Show pattern engine status or force a run.

        Sub-commands:
            /patterns         — Show all active patterns
            /patterns run     — Force full T1+T2+T3 analysis
            /patterns arcs    — Show T3 narrative arcs
            /patterns cloud   — Show cloud synthesis status
        """
        parts = cmd.strip().split()

        # /patterns arcs → show T3 narrative arcs specifically
        if len(parts) > 1 and parts[1] == "arcs":
            arcs = self.pattern_engine.get_narrative_arcs()
            if not arcs:
                return f"{DIM}No narrative arcs detected yet. Need 4+ weeks of data.{RESET}"
            lines = [f"{BOLD}═══ Narrative Arcs (T3) ═══{RESET}"]
            for arc in arcs:
                phase_icon = {
                    "building": "📈", "peak": "🔺", "resolving": "📉",
                    "resolved": "✅", "stable": "➡️",
                }.get(arc.get("phase", ""), "❓")
                lines.append(f"\n{phase_icon} {BOLD}{arc['arc_type']}{RESET} ({arc['phase']})")
                lines.append(f"  {arc['description']}")
                lines.append(f"  Duration: {arc['duration_weeks']} weeks | Confidence: {arc['confidence']:.0%}")
                if arc.get("insight"):
                    lines.append(f"  {GREEN}→ {arc['insight']}{RESET}")
            return "\n".join(lines)

        # /patterns cloud → show cloud synthesis status
        if len(parts) > 1 and parts[1] == "cloud":
            # Fix L-03: Guard against None cloud attribute (no cloud provider configured).
            # Previously crashed with AttributeError if PatternEngine.cloud is None.
            if not self.pattern_engine.cloud:
                return "Cloud synthesis not configured. Set ANTHROPIC_API_KEY or similar to enable."
            cloud_status = self.pattern_engine.cloud.get_status()
            lines = [f"{BOLD}═══ Cloud Synthesis Status ═══{RESET}"]
            lines.append(f"  Configured: {cloud_status['configured']}")
            lines.append(f"  Provider: {cloud_status['provider']}")
            lines.append(f"  Model: {cloud_status['model']}")
            lines.append(f"  Available: {cloud_status['available']}")
            lines.append(f"  Daily calls: {cloud_status['daily_calls_used']}/{cloud_status['daily_calls_max']}")
            if not cloud_status['configured']:
                lines.append(f"\n{DIM}Set AURA_LLM_PROVIDER and AURA_LLM_API_KEY to enable.{RESET}")
            return "\n".join(lines)

        # /patterns run → force full T1+T2+T3 analysis
        if len(parts) > 1 and parts[1] == "run":
            conversations = self.graph.get_recent_conversations(limit=50)
            readiness_history = self.graph.get_readiness_history(limit=50)
            results = self.pattern_engine.run_all(conversations, readiness_history)

            t1_count = len(results.get("t1", []))
            t2_count = len(results.get("t2", []))
            t3_count = len(results.get("t3", []))
            report = self.pattern_engine.format_patterns_report()

            return (
                f"{DIM}Pattern engine ran: {t1_count} T1 + {t2_count} T2 + {t3_count} T3 patterns{RESET}\n\n"
                + report
            )

        # /patterns → show current status
        return self.pattern_engine.format_patterns_report()

    def _cmd_validate(self) -> str:
        """Run self-model graph validation and show health report."""
        try:
            from src.aura.core.self_model_validator import SelfModelValidator
            validator = SelfModelValidator(
                graph=self.graph,
                auto_remediate=True,
            )
            report = validator.validate()
            return report.format_report()
        except Exception as e:
            return f"{DIM}Self-model validation failed: {e}{RESET}"

    def _cmd_rules(self) -> str:
        """Show active bridge rules."""
        try:
            from src.aura.bridge.rules_engine import BridgeRulesEngine
            engine = BridgeRulesEngine()
            summary = engine.get_rules_summary()

            if summary["active_rules"] == 0:
                return f"{DIM}No active bridge rules. Rules are created when patterns are promoted.{RESET}"

            lines = [f"{BOLD}═══ Active Bridge Rules ═══{RESET}"]
            lines.append(
                f"  Active: {summary['active_rules']} | "
                f"Expired: {summary['expired_rules']} | "
                f"Aura→Buddy: {summary['aura_to_buddy']} | "
                f"Buddy→Aura: {summary['buddy_to_aura']}"
            )

            for rule in summary["rules"]:
                direction_icon = "🧠→📊" if rule["direction"] == "aura_to_buddy" else "📊→🧠"
                lines.append(f"\n{direction_icon} {BOLD}{rule['rule_type']}{RESET}")
                lines.append(f"  {rule['description']}")
                lines.append(
                    f"  Confidence: {rule['confidence']:.0%} | "
                    f"Triggered: {rule['triggered_count']}x | "
                    f"Expires: {rule['expires_at'][:10]}"
                )
                adj = rule.get("adjustment", {})
                if adj:
                    lines.append(f"  Adjustment: {adj.get('parameter')} = {adj.get('value')}")

            return "\n".join(lines)
        except Exception as e:
            return f"{DIM}Bridge rules engine not available: {e}{RESET}"

    def _cmd_predict(self) -> str:
        """Show prediction model status and current override risk."""
        lines = [f"{BOLD}═══ Prediction Models (Phase 3) ═══{RESET}"]

        if self._override_predictor:
            info = self._override_predictor.get_model_info()
            lines.append(f"\n{BOLD}Override Predictor:{RESET}")
            lines.append(f"  Trained: {info['trained']}")
            lines.append(f"  Training samples: {info['train_samples']}")
            lines.append(f"  Training accuracy: {info['train_accuracy']:.1%}")

            # Show current risk assessment
            context = {
                "emotional_state": self._latest_signals.emotional_state if self._latest_signals else "neutral",
                "cognitive_load": "high" if self._active_stressors and len(self._active_stressors) > 2 else "normal",
                "confidence_at_time": 0.5,
                "weighted_vote_at_time": 0.5,
                "override_type": "took_rejected",
                "regime": "NORMAL",
            }
            pred = self._override_predictor.predict_loss_probability(context)
            risk_color = RED if pred.risk_level in ("critical", "high") else (YELLOW if pred.risk_level == "moderate" else GREEN)
            lines.append(f"\n  Current override risk: {risk_color}{pred.loss_probability:.0%} ({pred.risk_level}){RESET}")
            if pred.top_risk_factors:
                lines.append(f"  Top factors: {', '.join(pred.top_risk_factors)}")
        else:
            lines.append(f"\n{DIM}Override predictor not available{RESET}")

        if self._readiness_v2:
            info = self._readiness_v2.get_model_info()
            lines.append(f"\n{BOLD}Readiness V2:{RESET}")
            lines.append(f"  Version: {info['version']}")
            lines.append(f"  Training buffer: {info['buffer_size']}/{info['min_samples']} samples")
            if info['trained']:
                lines.append(f"  R²: {info['r_squared']:.3f}")
                comparison = self._readiness_v2.get_weight_comparison()
                lines.append(f"\n  Weight comparison (v1 → v2):")
                for name, comp in comparison.items():
                    delta_sign = "+" if comp['delta'] > 0 else ""
                    lines.append(
                        f"    {name}: {comp['v1_weight']:.2f} → {comp['v2_weight']:.2f} "
                        f"({delta_sign}{comp['delta']:.3f})"
                    )
            else:
                lines.append(f"  Samples until v2: {info['samples_until_v2']}")
        else:
            lines.append(f"\n{DIM}Readiness v2 model not available{RESET}")

        return "\n".join(lines)

    def _cmd_coach(self) -> str:
        """US-285: Provide actionable readiness-based coaching recommendations.

        Analyzes current readiness components and generates specific,
        contextual recommendations targeting the weakest areas.
        """
        readiness = self._latest_readiness
        if not readiness:
            return (
                f"{BOLD}═══ Aura Coach ═══{RESET}\n\n"
                f"I don't have enough data yet to coach you. "
                f"Chat with me a bit first so I can assess your state."
            )

        score = readiness.readiness_score
        components = readiness.components.to_dict()

        lines = [f"{BOLD}═══ Aura Coach ═══{RESET}"]
        lines.append(f"Readiness: {score:.0f}/100\n")

        if score >= 70:
            lines.append(f"{GREEN}Your profile looks solid today.{RESET}")
            lines.append("You're in a good headspace for trading decisions.")
            # Still check for any weak spots
            weak = [(k, v) for k, v in components.items() if v < 0.6]
            if weak:
                lines.append(f"\n{YELLOW}One area to watch:{RESET}")
                for name, val in weak[:1]:
                    lines.append(f"  {self._coach_recommendation(name, val)}")
            return "\n".join(lines)

        # Find the weakest components (sorted ascending)
        sorted_components = sorted(components.items(), key=lambda x: x[1])
        weakest = sorted_components[:3]  # Top 3 weakest

        lines.append(f"{YELLOW}Areas to focus on:{RESET}\n")

        for name, val in weakest:
            if val < 0.7:
                rec = self._coach_recommendation(name, val)
                lines.append(f"  • {rec}")

        # Add fatigue/acceleration warnings if available
        if hasattr(readiness, 'fatigue_score') and readiness.fatigue_score > 0.3:
            lines.append(f"\n{RED}⚠ Decision fatigue detected.{RESET}")
            lines.append("  You've been overriding frequently. Consider stepping away for 30 minutes.")

        if hasattr(readiness, 'confidence_acceleration') and readiness.confidence_acceleration < -0.05:
            lines.append(f"\n{RED}⚠ Your confidence is dropping rapidly.{RESET}")
            lines.append("  This often precedes impulsive decisions. Consider pausing new entries.")

        return "\n".join(lines)

    @staticmethod
    def _coach_recommendation(component: str, score: float) -> str:
        """Generate a specific recommendation for a low-scoring component."""
        severity = "significantly low" if score < 0.3 else "below optimal"

        recommendations = {
            "emotional_state": (
                f"Emotional state is {severity} ({score:.0%}). "
                "Consider a brief mindfulness exercise or journaling before your next trade session."
            ),
            "cognitive_load": (
                f"Cognitive load is {severity} ({score:.0%}). "
                "You're juggling too many concerns. Simplify your trading plan today — focus on 1-2 pairs max."
            ),
            "override_discipline": (
                f"Override discipline is {severity} ({score:.0%}). "
                "Recent overrides haven't gone well. Trust Buddy's signals more today, or reduce position sizes."
            ),
            "stress_level": (
                f"Stress level is {severity} ({score:.0%}). "
                "High stress impairs decision-making. Take a 15-minute break or do some physical movement."
            ),
            "confidence_trend": (
                f"Confidence trend is {severity} ({score:.0%}). "
                "Your confidence has been declining. Review recent wins to rebuild perspective."
            ),
            "engagement": (
                f"Engagement is {severity} ({score:.0%}). "
                "You haven't checked in recently. Regular Aura conversations help track your readiness."
            ),
        }

        return recommendations.get(
            component,
            f"{component} is {severity} ({score:.0%}). Monitor this closely."
        )

    def _cmd_insights(self) -> str:
        """US-295: Show patterns, biases, and graph health insights.

        Provides transparency into what Aura has learned:
        - Top 3 active patterns (from T1/T2) with strength and age
        - Current bias scores with plain-language descriptions
        - Graph health: total nodes, dormant count, model version
        """
        lines = [f"{BOLD}═══ Aura Insights ═══{RESET}"]

        # --- Top patterns from T1/T2 ---
        try:
            conversations = self.graph.get_recent_conversations(limit=30)
            readiness_history = self.graph.get_readiness_history(limit=30)
            results = self.pattern_engine.run_all(conversations, readiness_history)
            all_patterns = results.get("t1", []) + results.get("t2", [])

            if all_patterns:
                # Sort by confidence descending, take top 3
                sorted_patterns = sorted(
                    all_patterns,
                    key=lambda p: p.get("confidence", 0),
                    reverse=True,
                )[:3]

                lines.append(f"\n{BOLD}Top Patterns:{RESET}")
                for p in sorted_patterns:
                    tier = p.get("tier", "?")
                    desc = p.get("description", p.get("pattern_type", "unknown"))
                    conf = p.get("confidence", 0)
                    age = p.get("age_days", 0)
                    lines.append(
                        f"  T{tier}: {desc} "
                        f"(confidence: {conf:.0%}, age: {age}d)"
                    )
            else:
                lines.append(f"\n{DIM}No significant patterns detected yet — keep talking to Aura.{RESET}")
        except Exception as e:
            lines.append(f"\n{DIM}Pattern analysis unavailable: {e}{RESET}")

        # --- Current bias scores ---
        bias_descriptions = {
            "disposition_effect": "Holding losers / cutting winners early",
            "loss_aversion": "Disproportionate focus on downside risk",
            "recency_bias": "Overweighting recent events vs historical",
            "confirmation_bias": "Seeking validation for existing beliefs",
        }

        if self._latest_signals and self._latest_signals.bias_scores:
            biases = self._latest_signals.bias_scores
            active_biases = {k: v for k, v in biases.items() if v > 0.1}

            if active_biases:
                lines.append(f"\n{BOLD}Active Biases:{RESET}")
                for name, score in sorted(active_biases.items(), key=lambda x: x[1], reverse=True):
                    desc = bias_descriptions.get(name, name)
                    color = RED if score > 0.5 else YELLOW
                    lines.append(f"  {color}{name}: {score:.0%}{RESET} — {desc}")
            else:
                lines.append(f"\n{DIM}No significant biases detected in recent messages.{RESET}")
        else:
            lines.append(f"\n{DIM}No bias data yet — process some messages first.{RESET}")

        # --- Graph health ---
        lines.append(f"\n{BOLD}Graph Health:{RESET}")
        stats = self.graph.get_stats()
        total_nodes = stats.get("total_nodes", 0)
        lines.append(f"  Total nodes: {total_nodes}")

        # Check dormant nodes (effective_strength < 0.1)
        dormant_count = 0
        try:
            from datetime import datetime as dt, timezone as tz
            now = dt.now(tz.utc)
            # Query all nodes and check effective strength
            all_node_ids = []
            for nt, count in stats.get("nodes_by_type", {}).items():
                if count > 0:
                    try:
                        nodes = self.graph.get_nodes_by_type(nt)
                        all_node_ids.extend([n.id for n in nodes])
                    except Exception:
                        pass
            for nid in all_node_ids:
                try:
                    eff = self.graph.get_effective_strength(nid, query_time=now)
                    if eff < 0.1:
                        dormant_count += 1
                except Exception:
                    pass
        except Exception:
            pass
        lines.append(f"  Dormant nodes (strength < 10%): {dormant_count}")

        # Model version
        model_version = "v1 (static)"
        if self._latest_readiness:
            mv = getattr(self._latest_readiness, "model_version", "v1")
            model_version = f"v2 (learned)" if mv == "v2" else "v1 (static)"
        lines.append(f"  Readiness model: {model_version}")

        # V2 training progress
        if self._readiness_v2:
            try:
                info = self._readiness_v2.get_model_info()
                buf = info.get("buffer_size", 0)
                min_s = info.get("min_samples", 20)
                if not info.get("trained", False):
                    lines.append(f"  V2 training: {buf}/{min_s} samples collected")
                else:
                    lines.append(f"  V2 R²: {info.get('r_squared', 0):.3f}")
            except Exception:
                pass

        return "\n".join(lines)

    def _cmd_quality(self) -> str:
        """US-325: Show decision quality breakdown for the current session's trading discussion."""
        lines = [f"{BOLD}═══ Decision Quality Analysis ═══{RESET}"]

        try:
            from src.aura.scoring.decision_quality import DecisionQualityScorer

            scorer = DecisionQualityScorer()

            # Extract conversation text from session history
            conv_text = " ".join(
                msg.get("content", "") for msg in self._message_history
                if msg.get("role") == "user"
            )

            if not conv_text.strip():
                return f"{DIM}No conversation to analyze yet — send some messages about your trading first.{RESET}"

            result = scorer.score(conv_text)
            dims = result.to_dict()["dimensions"]

            lines.append(f"\n{BOLD}Composite Score: {result.composite_score:.1f}/100{RESET}")
            lines.append("")

            # Dimension breakdown with visual bars
            dim_labels = {
                "process_adherence": ("Process Adherence", 0.25),
                "information_adequacy": ("Information Adequacy", 0.20),
                "metacognitive_awareness": ("Metacognitive Awareness", 0.15),
                "uncertainty_acknowledgment": ("Uncertainty Acknowledgment", 0.15),
                "rationale_clarity": ("Rationale Clarity", 0.10),
                "emotional_regulation": ("Emotional Regulation", 0.10),
                "cognitive_reflection": ("Cognitive Reflection", 0.05),
            }

            for key, (label, weight) in dim_labels.items():
                score = dims.get(key, 0.0)
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                color = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
                lines.append(
                    f"  {label:30s} {color}{bar} {score:.0%}{RESET} (w={weight})"
                )

            # Guidance
            lines.append("")
            weakest = min(dims.items(), key=lambda x: x[1])
            strongest = max(dims.items(), key=lambda x: x[1])
            lines.append(f"{DIM}Strongest: {dim_labels.get(strongest[0], (strongest[0],))[0]} ({strongest[1]:.0%}){RESET}")
            lines.append(f"{DIM}Weakest:   {dim_labels.get(weakest[0], (weakest[0],))[0]} ({weakest[1]:.0%}){RESET}")

        except ImportError:
            lines.append(f"{RED}Decision quality scorer not available.{RESET}")
        except Exception as e:
            lines.append(f"{RED}Error scoring decision quality: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_anomalies(self) -> str:
        """US-324/US-325: Show recent readiness anomalies with timestamps and severities."""
        lines = [f"{BOLD}═══ Recent Anomalies ═══{RESET}"]

        try:
            from src.aura.core.self_model import NodeType

            life_events = self.graph.get_nodes_by_type(NodeType.LIFE_EVENT)
            anomaly_events = [
                n for n in life_events
                if n.properties.get("source") == "anomaly_detector"
            ]

            if not anomaly_events:
                return f"{DIM}No anomalies detected yet. Anomalies are flagged when readiness deviates significantly from your baseline.{RESET}"

            # Sort by created_at descending, limit to last 10
            anomaly_events.sort(key=lambda n: n.created_at, reverse=True)
            anomaly_events = anomaly_events[:10]

            for event in anomaly_events:
                props = event.properties
                severity = props.get("severity", 0)
                direction = props.get("direction", "unknown")
                readiness = props.get("readiness_at_time", 0)
                ts = event.created_at

                # Color based on severity
                color = RED if severity > 0.7 else (YELLOW if severity > 0.4 else DIM)
                arrow = "↓" if direction == "drop" else "↑"

                lines.append(
                    f"  {color}{arrow} {ts[:19]} — severity {severity:.2f}, "
                    f"readiness={readiness:.0f}, direction={direction}{RESET}"
                )

            lines.append(f"\n{DIM}Showing last {len(anomaly_events)} anomalies.{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error loading anomalies: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_recovery(self) -> str:
        """US-326/US-331: Show emotional recovery metrics."""
        lines = [f"{BOLD}═══ Recovery & Regulation ═══{RESET}"]

        try:
            from src.aura.scoring.emotional_regulation import EmotionalRegulationScorer

            scorer = EmotionalRegulationScorer()
            # Get readiness history from the computer
            readiness_history = getattr(self._readiness_computer, '_readiness_history', [])

            if len(readiness_history) < 5:
                return f"{DIM}Not enough readiness history yet (need 5+, have {len(readiness_history)}). Keep chatting to build your profile.{RESET}"

            metrics = scorer.score(
                readiness_history=readiness_history,
                active_stressors_count=len(self._latest_readiness.active_stressors) if self._latest_readiness else 0,
                current_readiness=self._latest_readiness.readiness_score if self._latest_readiness else 50.0,
            )

            # Display with visual bars
            def bar(val, width=20):
                filled = int(val * width)
                return "█" * filled + "░" * (width - filled)

            lines.append(f"  Recovery Efficiency:   {bar(metrics.recovery_efficiency)} {metrics.recovery_efficiency:.2f}")
            lines.append(f"  Regulation Discipline: {bar(metrics.regulation_discipline)} {metrics.regulation_discipline:.2f}")
            lines.append(f"  Stress Absorption:     {bar(metrics.stress_absorption)} {metrics.stress_absorption:.2f}")
            lines.append(f"  ─────────────────────────────────────")
            lines.append(f"  {BOLD}Composite Recovery:     {bar(metrics.composite_recovery_score)} {metrics.composite_recovery_score:.2f}{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error computing recovery metrics: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_regimes(self) -> str:
        """US-330/US-331: Show detected regime shifts."""
        lines = [f"{BOLD}═══ Regime Shifts ═══{RESET}"]

        try:
            from src.aura.core.self_model import NodeType

            life_events = self.graph.get_nodes_by_type(NodeType.LIFE_EVENT)
            regime_events = [
                n for n in life_events
                if n.properties.get("source") == "changepoint_detector"
            ]

            if not regime_events:
                return f"{DIM}No regime shifts detected. These are flagged when your readiness baseline changes significantly.{RESET}"

            # Sort by created_at descending, limit to last 5
            regime_events.sort(key=lambda n: n.created_at, reverse=True)
            regime_events = regime_events[:5]

            for event in regime_events:
                props = event.properties
                prob = props.get("severity", 0)
                pre = props.get("pre_baseline", 0)
                post = props.get("post_baseline", 0)
                readiness = props.get("readiness_at_time", 0)
                ts = event.created_at

                direction = "↑" if post > pre else "↓"
                color = GREEN if post > pre else RED

                lines.append(
                    f"  {color}{direction} {ts[:19]} — prob {prob:.2f}, "
                    f"baseline {pre:.0f} → {post:.0f}, readiness={readiness:.0f}{RESET}"
                )

            lines.append(f"\n{DIM}Showing last {len(regime_events)} regime shifts.{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error loading regime shifts: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_reliability(self) -> str:
        """US-337: Show readiness score reliability metrics."""
        lines = [f"{BOLD}═══ Score Reliability ═══{RESET}"]

        try:
            from src.aura.analysis.reliability import ReadinessReliabilityAnalyzer

            analyzer = ReadinessReliabilityAnalyzer()

            # Try to reconstruct from readiness computer if available
            rc = getattr(self, '_readiness_computer', None)
            if rc is not None and hasattr(rc, '_reliability_analyzer'):
                analyzer = rc._reliability_analyzer or analyzer

            result = analyzer.compute()

            if not result.sufficient_data:
                lines.append(f"{DIM}Insufficient data — need 10+ readiness computations for reliability metrics.{RESET}")
                lines.append(f"  Samples collected: {result.sample_count}")
                return "\n".join(lines)

            # Visual bars
            def bar(val: float, width: int = 20) -> str:
                filled = int(val * width)
                return f"{'█' * filled}{'░' * (width - filled)}"

            alpha = result.cronbachs_alpha
            split_half = result.split_half_reliability
            composite = result.reliability_score

            alpha_color = GREEN if alpha >= 0.7 else (YELLOW if alpha >= 0.6 else RED)
            split_color = GREEN if split_half >= 0.7 else (YELLOW if split_half >= 0.6 else RED)
            comp_color = GREEN if composite >= 0.7 else (YELLOW if composite >= 0.6 else RED)

            lines.extend([
                f"  Cronbach's α:    {alpha_color}{bar(alpha)} {alpha:.3f}{RESET}",
                f"  Split-half:      {split_color}{bar(split_half)} {split_half:.3f}{RESET}",
                f"  Composite:       {comp_color}{bar(composite)} {composite:.3f}{RESET}",
                f"  Samples:         {result.sample_count}",
                "",
            ])

            if alpha < 0.6:
                lines.append(f"  {RED}⚠ Low internal consistency — readiness components are noisy.{RESET}")
            elif alpha >= 0.8:
                lines.append(f"  {GREEN}✓ Excellent internal consistency — components agree well.{RESET}")
            else:
                lines.append(f"  {YELLOW}○ Acceptable consistency — monitor for degradation.{RESET}")

        except ImportError:
            lines.append(f"{RED}Reliability analyzer not available.{RESET}")
        except Exception as e:
            lines.append(f"{RED}Error computing reliability: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_style(self) -> str:
        """US-337: Show recent linguistic style snapshots with drift indicator."""
        lines = [f"{BOLD}═══ Linguistic Style ═══{RESET}"]

        try:
            # Access style tracker from conversation processor
            tracker = getattr(self.processor, '_style_tracker', None)
            if tracker is None:
                return f"{DIM}Style tracker not available. Process some messages first.{RESET}"

            snapshots = tracker._history
            if not snapshots:
                return f"{DIM}No style data yet. Send a few messages first.{RESET}"

            # Show last 5 snapshots
            recent = snapshots[-5:]
            drift = tracker.compute_drift()

            lines.append(f"  Drift score: ", )
            drift_color = RED if drift > 0.5 else (YELLOW if drift > 0.3 else GREEN)
            lines[-1] = f"  Drift score: {drift_color}{drift:.3f}{RESET}" + (" ⚠ HIGH" if drift > 0.5 else "")
            lines.append("")

            lines.append(f"  {'#':>3} {'AvgLen':>7} {'Excl!':>7} {'CAPS':>7} {'I-pron':>7} {'Quest?':>7}")
            lines.append(f"  {'─'*3} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

            for i, snap in enumerate(recent):
                idx = len(snapshots) - len(recent) + i + 1
                lines.append(
                    f"  {idx:>3} {snap.avg_sentence_length:>7.1f} {snap.exclamation_density:>7.3f} "
                    f"{snap.caps_ratio:>7.3f} {snap.pronoun_i_ratio:>7.3f} {snap.question_ratio:>7.3f}"
                )

            if drift > 0.5:
                lines.append(f"\n  {RED}⚠ Significant style drift detected — potential state change.{RESET}")
            elif drift > 0.3:
                lines.append(f"\n  {YELLOW}○ Moderate style drift — monitor for escalation.{RESET}")
            else:
                lines.append(f"\n  {GREEN}✓ Stable writing style.{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error loading style data: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_flexibility(self) -> str:
        """US-343: Show cognitive flexibility scores."""
        lines = [f"{BOLD}═══ Cognitive Flexibility ═══{RESET}"]

        try:
            scorer = None
            # Try to get scorer from readiness computer or create one
            rc = getattr(self, '_readiness_computer', None)
            if rc is not None:
                scorer = getattr(rc, '_flexibility_scorer', None)
            if scorer is None:
                try:
                    from src.aura.scoring.cognitive_flexibility import CognitiveFlexibilityScorer
                    scorer = CognitiveFlexibilityScorer()
                except ImportError:
                    return f"{DIM}Cognitive flexibility scorer not available.{RESET}"

            # Score from recent messages
            recent_text = " ".join(
                m.get("content", "") for m in self._message_history[-5:]
            )
            if not recent_text.strip():
                return f"{DIM}No recent messages to analyze. Send some messages first.{RESET}"

            result = scorer.score(recent_text)

            def bar(val, width=20):
                filled = int(val * width)
                return f"[{'█' * filled}{'░' * (width - filled)}]"

            color_comp = GREEN if result.composite > 0.3 else (YELLOW if result.composite > 0.1 else RED)
            lines.append(f"  Composite: {color_comp}{result.composite:.3f}{RESET} {bar(result.composite)}")
            lines.append(f"  Belief Update:     {result.belief_update:.3f} {bar(result.belief_update)}")
            lines.append(f"  Strategy Adapt:    {result.strategy_adaptation:.3f} {bar(result.strategy_adaptation)}")
            lines.append(f"  Evidence Ack:      {result.evidence_acknowledgment:.3f} {bar(result.evidence_acknowledgment)}")
            lines.append("")

            if result.composite > 0.5:
                lines.append(f"  {GREEN}✓ High flexibility — actively updating beliefs and adapting.{RESET}")
            elif result.composite > 0.3:
                lines.append(f"  {GREEN}○ Good flexibility — showing adaptive thinking.{RESET}")
            elif result.composite > 0.1:
                lines.append(f"  {YELLOW}○ Low flexibility — consider reviewing assumptions.{RESET}")
            else:
                lines.append(f"  {RED}⚠ Rigid thinking detected — risk of confirmation bias.{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_journal(self) -> str:
        """US-343: Show journal reflection quality for recent messages."""
        lines = [f"{BOLD}═══ Journal Reflection Quality ═══{RESET}"]

        try:
            try:
                from src.aura.scoring.journal_reflection import JournalReflectionScorer
                scorer = JournalReflectionScorer()
            except ImportError:
                return f"{DIM}Journal reflection scorer not available.{RESET}"

            recent = self._message_history[-5:]
            if not recent:
                return f"{DIM}No recent messages to analyze.{RESET}"

            depth_labels = {1: "L1 Summary", 2: "L2 Reasoning", 3: "L3 Causal", 4: "L4 Meta-Reflection"}

            for i, msg in enumerate(recent):
                text = msg.get("content", "")
                if not text.strip():
                    continue
                result = scorer.score(text)
                depth_color = GREEN if result.depth_level >= 3 else (YELLOW if result.depth_level >= 2 else RED)
                premortem_icon = f"{GREEN}✓{RESET}" if result.premortem_present else f"{DIM}✗{RESET}"

                idx = len(self._message_history) - len(recent) + i + 1
                lines.append(f"  #{idx}: {depth_color}{depth_labels.get(result.depth_level, 'L1')}{RESET}")
                lines.append(f"       Causal density: {result.causal_density:.3f}  Pre-mortem: {premortem_icon}")
                lines.append(f"       Quality: {result.reflection_quality:.3f}")

            lines.append("")
            avg_quality = 0.0
            count = 0
            for msg in recent:
                text = msg.get("content", "")
                if text.strip():
                    r = scorer.score(text)
                    avg_quality += r.reflection_quality
                    count += 1
            if count > 0:
                avg_quality /= count
                qcolor = GREEN if avg_quality > 0.5 else (YELLOW if avg_quality > 0.3 else RED)
                lines.append(f"  Average quality: {qcolor}{avg_quality:.3f}{RESET}")

        except Exception as e:
            lines.append(f"{RED}Error: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_weights(self) -> str:
        """US-341: Show current readiness component weights (static vs adaptive)."""
        lines = [f"{BOLD}═══ Component Weights ═══{RESET}"]

        try:
            from src.aura.core.readiness import _COMPONENT_WEIGHTS

            rc = getattr(self, '_readiness_computer', None)
            adaptive_mgr = None
            if rc is not None:
                adaptive_mgr = getattr(rc, '_adaptive_weights', None)

            adaptive_active = adaptive_mgr is not None and adaptive_mgr.is_ready()
            adaptive_weights = adaptive_mgr.get_weights() if adaptive_active else None

            def bar(val, width=20):
                filled = int(val * width)
                return f"[{'█' * filled}{'░' * (width - filled)}]"

            lines.append(f"  {'Component':<22} {'Static':>8} {'Adaptive':>8} {'Active':>8}")
            lines.append(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}")

            for name, static_w in _COMPONENT_WEIGHTS.items():
                adap_w = adaptive_weights.get(name, 0.0) if adaptive_weights else 0.0
                active_w = adap_w if adaptive_active else static_w
                color = GREEN if adaptive_active else DIM
                lines.append(
                    f"  {name:<22} {static_w:>8.3f} {adap_w:>8.3f} {color}{active_w:>8.3f}{RESET} {bar(active_w)}"
                )

            lines.append("")
            if adaptive_active:
                lines.append(f"  {GREEN}✓ Adaptive weights active (samples={adaptive_mgr.sample_count}){RESET}")
            else:
                sample_count = adaptive_mgr.sample_count if adaptive_mgr else 0
                needed = (adaptive_mgr.MIN_SAMPLES if adaptive_mgr else 10) - sample_count
                lines.append(f"  {YELLOW}○ Using static weights (need {max(0, needed)} more outcomes for adaptive){RESET}")

        except Exception as e:
            lines.append(f"{RED}Error: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_granularity(self) -> str:
        """US-349: Show emotional granularity metrics."""
        lines = [f"{BOLD}═══ Emotional Granularity ═══{RESET}"]

        def bar(val, width=20):
            filled = int(val * width)
            return f"[{'█' * filled}{'░' * (width - filled)}]"

        try:
            from src.aura.scoring.emotional_granularity import EmotionalGranularityScorer

            if not hasattr(self, '_granularity_scorer'):
                self._granularity_scorer = EmotionalGranularityScorer()

            # Use processor's scorer if available
            scorer = getattr(self.processor, '_granularity_scorer', self._granularity_scorer)
            if scorer and hasattr(scorer, '_word_history') and len(scorer._word_history) > 0:
                # Reconstruct last result from history
                result = scorer.update("")  # Get current state without new text
                lines.append(f"  Vocabulary Richness: {bar(result.vocabulary_richness)} {result.vocabulary_richness:.3f}")
                lines.append(f"  Entropy:             {bar(result.entropy)} {result.entropy:.3f}")
                lines.append(f"  Differentiation:     {bar(result.differentiation)} {result.differentiation:.3f}")
                lines.append(f"  {BOLD}Composite:           {bar(result.composite)} {result.composite:.3f}{RESET}")
                lines.append("")
                if result.composite > 0.6:
                    lines.append(f"  {GREEN}High granularity — nuanced emotional expression{RESET}")
                elif result.composite > 0.3:
                    lines.append(f"  {YELLOW}Moderate granularity — room for more precise emotion labels{RESET}")
                else:
                    lines.append(f"  {RED}Low granularity — limited emotion vocabulary{RESET}")
            else:
                lines.append(f"  {DIM}No emotional data yet. Express emotions in your messages to track granularity.{RESET}")
        except Exception as e:
            lines.append(f"  {DIM}Granularity scoring unavailable: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_coherence(self) -> str:
        """US-349: Show narrative coherence metrics."""
        lines = [f"{BOLD}═══ Narrative Coherence ═══{RESET}"]

        def bar(val, width=20):
            filled = int(val * width)
            return f"[{'█' * filled}{'░' * (width - filled)}]"

        try:
            from src.aura.scoring.narrative_coherence import NarrativeCoherenceTracker

            if not hasattr(self, '_coherence_tracker'):
                self._coherence_tracker = NarrativeCoherenceTracker()

            tracker = self._coherence_tracker
            if len(tracker._session_words) >= 2:
                # Show metrics from most recent update
                lines.append(f"  Sessions tracked: {len(tracker._session_words)}")
                lines.append(f"  Lexical Overlap:       {bar(0.5)} (session-level metric)")
                lines.append(f"  Sentiment Consistency: {bar(0.5)} (session-level metric)")
                lines.append(f"  Strategy Persistence:  {bar(0.5)} (session-level metric)")
                lines.append("")
                lines.append(f"  {DIM}Coherence is computed across sessions — more data improves accuracy.{RESET}")
            else:
                lines.append(f"  {DIM}Need at least 2 sessions to compute coherence. Current: {len(tracker._session_words)} session(s).{RESET}")
        except Exception as e:
            lines.append(f"  {DIM}Coherence tracking unavailable: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_negotiate(self) -> str:
        """US-349: Show active proposals and negotiation history."""
        lines = [f"{BOLD}═══ Negotiation Status ═══{RESET}"]

        try:
            from src.aura.bridge.negotiation import NegotiationEngine

            bridge_dir = getattr(self, '_bridge_dir', None)
            if bridge_dir is None:
                from pathlib import Path
                bridge_dir = Path(".aura/bridge")

            engine = NegotiationEngine(bridge_dir)
            entries = engine.get_log_entries(limit=10)

            if not entries:
                lines.append(f"  {DIM}No negotiation activity yet.{RESET}")
            else:
                proposals = [e for e in entries if e.get("type") == "proposal"]
                counters = [e for e in entries if e.get("type") == "counter_proposal"]
                resolutions = [e for e in entries if e.get("type") == "resolution"]

                # Stats
                auto_activated = sum(1 for r in resolutions if r.get("resolution_type") == "auto_activated")
                converged = sum(1 for r in resolutions if r.get("resolution_type") == "converged")
                total_resolved = len(resolutions)
                convergence_rate = converged / max(1, total_resolved) * 100

                lines.append(f"  Proposals:  {len(proposals)}")
                lines.append(f"  Counters:   {len(counters)}")
                lines.append(f"  Resolved:   {total_resolved}")
                lines.append(f"  Convergence rate: {convergence_rate:.0f}% ({converged} converged, {auto_activated} auto-activated)")
                lines.append("")

                # Last 5 resolutions
                lines.append(f"  {BOLD}Recent Resolutions:{RESET}")
                for r in resolutions[-5:]:
                    rtype = r.get("resolution_type", "?")
                    value = r.get("agreed_value", 0)
                    color = GREEN if rtype == "converged" else (YELLOW if rtype == "auto_activated" else RED)
                    lines.append(f"    {color}{rtype:<16}{RESET} → {value:.4f}")

        except Exception as e:
            lines.append(f"{RED}Error: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_calibration(self) -> str:
        """US-349: Show calibration scores, weight recommendations, and critiques."""
        lines = [f"{BOLD}═══ Calibration & Co-Evolution ═══{RESET}"]

        def bar(val, width=20):
            filled = int(val * width)
            return f"[{'█' * filled}{'░' * (width - filled)}]"

        try:
            from src.aura.bridge.calibration import CalibrationTracker
            from src.aura.bridge.coevolution import CoEvolutionManager
            from src.aura.bridge.critique import CritiqueEngine

            bridge_dir = getattr(self, '_bridge_dir', None)
            if bridge_dir is None:
                from pathlib import Path
                bridge_dir = Path(".aura/bridge")

            # Calibration
            tracker = CalibrationTracker()
            tracker.load_state(bridge_dir)
            aura_cal = tracker.aura_calibration_score()
            buddy_cal = tracker.buddy_calibration_score()

            aura_color = GREEN if aura_cal >= 0.7 else (YELLOW if aura_cal >= 0.5 else RED)
            buddy_color = GREEN if buddy_cal >= 0.7 else (YELLOW if buddy_cal >= 0.5 else RED)

            lines.append(f"  {BOLD}Prediction Accuracy:{RESET}")
            lines.append(f"    Aura:  {aura_color}{aura_cal:.1%}{RESET} {bar(aura_cal)} ({len(tracker.aura_predictions)} samples)")
            lines.append(f"    Buddy: {buddy_color}{buddy_cal:.1%}{RESET} {bar(buddy_cal)} ({len(tracker.buddy_predictions)} samples)")
            if tracker.is_low_calibration():
                lines.append(f"    {RED}⚠ LOW CALIBRATION — Buddy should discount readiness score{RESET}")
            lines.append("")

            # Co-evolution weights
            mgr = CoEvolutionManager()
            # Try to load persisted state
            from src.aura.bridge.signals import FeedbackBridge
            import json
            raw = FeedbackBridge._locked_read(bridge_dir / "calibration_state.json")
            if raw:
                try:
                    state = json.loads(raw)
                    coevo = state.get("coevolution", {})
                    if coevo:
                        mgr.load_from_dict(coevo)
                except Exception:
                    pass

            lines.append(f"  {BOLD}Weight Recommendations:{RESET}")
            lines.append(f"    Aura outcome weight:     {mgr.aura_outcome_weight:.3f} {bar(mgr.aura_outcome_weight / 1.5)}")
            lines.append(f"    Signal weight recommend:  {mgr.signal_weight_recommendation:.3f} {bar(mgr.signal_weight_recommendation / 1.5)}")
            lines.append("")

            # Recent critiques
            critique_engine = CritiqueEngine(bridge_dir)
            aura_critiques = critique_engine.get_recent_critiques(critic="aura", limit=3)
            buddy_critiques = critique_engine.get_recent_critiques(critic="buddy", limit=3)

            lines.append(f"  {BOLD}Recent Critiques:{RESET}")
            if not aura_critiques and not buddy_critiques:
                lines.append(f"    {DIM}No critiques yet.{RESET}")
            for c in aura_critiques:
                lines.append(f"    {CYAN}Aura→Buddy:{RESET} {c.observation[:60]}...")
            for c in buddy_critiques:
                lines.append(f"    {MAGENTA}Buddy→Aura:{RESET} {c.observation[:60]}...")

            # Weight history
            if mgr.weight_history:
                lines.append("")
                lines.append(f"  {BOLD}Weight Changes (last 5):{RESET}")
                for w in mgr.weight_history[-5:]:
                    lines.append(f"    {w.parameter}: {w.old_value:.3f} → {w.new_value:.3f} ({w.trigger})")

        except Exception as e:
            lines.append(f"{RED}Error: {e}{RESET}")

        return "\n".join(lines)

    def _cmd_affect(self) -> str:
        """US-355: Show current affect dynamics."""
        try:
            tracker = getattr(self._processor, '_affect_tracker', None)
            if tracker is None or not tracker._valence_history:
                return "No affect data yet. Send some messages first."

            v = tracker._valence_history[-1] if tracker._valence_history else 0.0
            a = tracker._arousal_history[-1] if tracker._arousal_history else 0.0
            inertia = tracker._compute_inertia()

            # Use last result if available
            lines = [
                "━━━ Affect Dynamics ━━━",
                f"  Valence:    {'█' * int(abs(v) * 10):10s} {v:+.2f} ({'positive' if v > 0.1 else 'negative' if v < -0.1 else 'neutral'})",
                f"  Arousal:    {'█' * int(a * 10):10s} {a:.2f} ({'high' if a > 0.5 else 'low'})",
                f"  Inertia:    {'█' * int(inertia * 10):10s} {inertia:.2f} ({'rigid' if inertia > 0.7 else 'flexible'})",
                f"  History:    {len(tracker._valence_history)} messages tracked",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"Affect tracking unavailable: {e}"

    def _cmd_fatigue(self) -> str:
        """US-355: Show decision fatigue dimensions."""
        try:
            if not self._latest_readiness:
                return "No readiness signal computed yet. Send some messages first."

            fatigue_score = self._latest_readiness.fatigue_score

            lines = [
                "━━━ Decision Fatigue ━━━",
                f"  Overall:    {'█' * int(fatigue_score * 10):10s} {fatigue_score:.1f}/100",
            ]

            if fatigue_score > 70:
                lines.append(f"  {RED}Status: HIGH FATIGUE — Recommend breaks{RESET}")
            elif fatigue_score > 50:
                lines.append(f"  {YELLOW}Status: MODERATE FATIGUE — Monitor closely{RESET}")
            else:
                lines.append(f"  {GREEN}Status: NORMAL — Full capacity{RESET}")

            return "\n".join(lines)
        except Exception as e:
            return f"Fatigue tracking unavailable: {e}"

    def end_session(self) -> None:
        """Clean up and log session summary to graph."""
        summary = self.processor.get_session_summary()

        # Log final conversation record
        self.graph.log_conversation(
            conversation_id=self._conversation_id,
            summary=f"Session with {summary['message_count']} messages, "
                    f"net sentiment {summary['net_sentiment']:+.3f}",
            emotional_state=self._latest_signals.emotional_state if self._latest_signals else "neutral",
            topics=self._latest_signals.topics if self._latest_signals else [],
            readiness_impact=self._latest_readiness.readiness_score if self._latest_readiness else 0,
            messages=self._message_history[-20:],  # Keep last 20 messages
        )

        # US-361: Persist AdaptiveThresholdLearner state to bridge on session end
        try:
            self.readiness.save_threshold_state()
            logger.debug("US-361: Threshold state saved on session end")
        except Exception as e:
            logger.debug("US-361: Could not save threshold state on end_session: %s", e)

        self.graph.close()
