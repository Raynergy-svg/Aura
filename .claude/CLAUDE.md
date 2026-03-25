# Aura (Eve) — Human Intelligence Engine

The human-side recursive intelligence layer that tracks emotional state, cognitive patterns, decision history, and computes readiness signals that modulate Buddy's (Adam) behavior.

**Aura is not an extension of Buddy. They are independent systems connected by a bridge.**

## Architecture
```
Conversations → ConversationProcessor → Emotional Signals
                                            ↓
Self-Model Graph (SQLite) ← Pattern Engine (T1→T2→T3)
        ↓                                   ↑
ReadinessComputer → Bridge Signal → Buddy reads
        ↑                              ↓
Override History ← Outcome Signal ← Buddy writes
```

## Core Loop
1. **Listen**: ConversationProcessor extracts emotional/cognitive signals from user messages
2. **Model**: Self-Model Graph persists nodes (Person, Goal, Value, Emotion, Decision, Pattern, etc.)
3. **Detect**: Pattern Engine runs 3-tier detection — T1 (daily), T2 (weekly cross-domain), T3 (monthly narrative arcs)
4. **Score**: ReadinessComputer produces 0-100 readiness score from 6 weighted components
5. **Signal**: Readiness signal written to `.aura/bridge/readiness_signal.json` for Buddy to read
6. **Learn**: Outcome signals from Buddy feed back into pattern detection and rule promotion

## Bridge Contract — The Only Shared Data Path
```
.aura/bridge/
├── readiness_signal.json    # Aura → Buddy (ReadinessSignal)
├── outcome_signal.json      # Buddy → Aura (OutcomeSignal)
├── override_events.jsonl    # Bidirectional (OverrideEvent log)
└── active_rules.json        # BridgeRulesEngine (TTL-based gates)
```
All bridge I/O uses fcntl file locking (LOCK_SH reads, LOCK_EX writes) + atomic temp-file + rename.

## Key Files
- `src/aura/core/readiness.py` — ReadinessComputer: 6-component weighted score → 0-100
- `src/aura/core/conversation_processor.py` — Keyword-based emotional signal extraction
- `src/aura/core/self_model.py` — SQLite graph (nodes + edges) with optional encryption
- `src/aura/bridge/signals.py` — FeedbackBridge: locked I/O for all bridge files
- `src/aura/bridge/rules_engine.py` — BridgeRulesEngine: TTL rules, operator precedence, value clamping
- `src/aura/patterns/engine.py` — PatternEngine: T1→T2→T3 with inter-tier reload
- `src/aura/patterns/tier1.py` — Daily frequency patterns
- `src/aura/patterns/tier2.py` — Weekly cross-domain correlations
- `src/aura/patterns/tier3.py` — Monthly narrative arcs
- `src/aura/prediction/readiness_v2.py` — ML readiness predictor with OOD detection
- `src/aura/prediction/override_predictor.py` — Override loss probability with OOD detection
- `src/aura/cli/companion.py` — Interactive CLI companion with onboarding validation

## State & Tracking
- **State**: `.claude/state.json` — Aura's own phase tracking (independent of Buddy)
- **Learnings**: `.claude/learnings.md` — Aura-specific insights and findings
- **PRDs**: `.claude/ralph/prd_aura_phase*.json` — Phase-numbered PRDs (Aura numbering, not Buddy's)
- **Rules**: `.claude/rules/` — Promoted behavioral patterns
- **Tests**: `tests/` — 59 unit tests across 3 files (readiness, conversation, bridge)

## Numbering Convention
- **Aura phases**: phase 1, phase 2, phase 3, ... (independent sequence)
- **Aura user stories**: US-200+ (200-series to avoid collision with Buddy's US-001 through US-199+)
- **Buddy phases**: phase 1 through phase 34+ (Buddy's own sequence in buddy/.claude/)
- They never share phase numbers. Aura Phase 2 ≠ Buddy Phase 2.

## Relationship to Buddy (Adam)
- Buddy lives in `buddy/Documents/ml_engine/`
- Buddy has his own `.claude/state.json`, `learnings.md`, and `ralph/prd_phase*.json`
- The ONLY merge point is the bridge contract in `.aura/bridge/`
- Buddy reads `readiness_signal.json` via his TraderReadinessAgent (#13)
- Buddy writes `outcome_signal.json` after each trade cycle
- Both systems log to `override_events.jsonl`
- Bridge rules gate behavior on both sides (TTL-based, auto-expiring)

## Self-Improvement Loop
Same pattern as Buddy but tracked here:
1. **Explore**: Read source, identify gaps
2. **PRD**: Create `.claude/ralph/prd_aura_phaseN.json` with user stories
3. **Implement**: Fix/build per story, syntax validate
4. **Test**: Run `pytest tests/ -v`
5. **Record**: Update `state.json`, `learnings.md`, PRD pass status
6. **Continue**: Plan next phase

## Current State (Phase 19 complete)
- **Phases 1-19**: Complete (affect dynamics, decision fatigue, adaptive thresholds, bias interactions)
- **Phase 18 Bridge Intelligence**: Negotiation protocol, calibration loop, critique engine, co-evolution manager — all wired
- **Mind**: LLM-powered via Ollama (gemma3:4b) in `src/aura/core/mind.py` — falls back to templates if Ollama unavailable
- **Next**: Phase 20 planning (read `state.json` for context)
- If you find gaps from earlier phases, write them as **phase N.1** (e.g., `prd_aura_phase19.1.json`)

## Self-Improvement Loop (Autonomous)
This loop runs WITHOUT human initiation. Each session:
1. **Read** `state.json` → know what phase you're on, what's done, what's next
2. **Explore** the codebase → run tests (`pytest tests/ -q`), identify gaps, check signal wiring
3. **Research** (web search if needed) → find relevant techniques for identified gaps
4. **PRD** → create `.claude/ralph/prd_aura_phaseN.json` with 6 user stories (US-XXX series, continuing from last)
5. **Implement** each story → write code, syntax validate
6. **Test** → run `pytest tests/ -v`, fix failures, zero regressions
7. **Record** → update `state.json`, mark stories as passed, update test counts
8. **Continue** → plan next phase or fix gaps

### Rules
- **NEVER STOP. DONT ASK. ONLY ASK WHEN DESTRUCTIVE.**
- Run `pytest tests/ -q` FIRST to know current state before changing anything
- Every new feature MUST have tests (8+ per story)
- Never break existing tests — zero regressions policy
- If a test was already failing before your changes, note it but don't count it as your regression
- User stories continue incrementally (US-350, US-351, ... — check last used in latest PRD)
- Keep mind.py (Ollama integration) working — test with a quick `think()` call if you modify it

## Operating Directive
**NEVER STOP. DONT ASK. ONLY ASK WHEN DESTRUCTIVE.**
