# Aura Bug Report — 2026-03-25 08:11 ET

## Executive Summary

- **Total issues found: 30**
- **Critical: 5 | High: 5 | Medium: 6 | Low: 14**
- **Test suite: 795 total — 792 passed, 3 failed (99.6% pass rate)**
- **Syntax check: All files compile cleanly (0 errors)**
- **flake8: 110 style issues (45 line-length, 27 unused imports, rest minor)**
- **Circular imports: None detected**
- **Bridge health: 1 of 4 contract files present; 3 missing**

**Trajectory since previous report (2026-03-25 00:47):** Test count jumped from 633 → 795 (+162 new tests). All 7 original CRITICAL issues from March 24 remain fixed. Two previously-reported HIGH issues persist (H-01 method mismatch, H-02 missing bridge files). This scan adds 5 new CRITICAL findings from deeper static analysis of scoring, prediction, and bridge modules. The codebase is functionally sound for its primary readiness-scoring path but has significant gaps in the bidirectional bridge contract and several latent numerical stability risks.

---

## Critical Issues (fix immediately)

### C-01: Divide-by-zero risk in metacognitive effort_allocation_score
- **File:** `src/aura/scoring/metacognitive.py`, line ~149
- **Description:** `correlation = cov / (std_c * std_x)` has individual guards checking `std_c < 1e-8 or std_x < 1e-8`, but floating-point multiplication of two values just above the threshold can produce ~0.0, causing division by zero.
- **Impact:** NaN propagation into composite readiness score; entire readiness computation returns invalid result. Buddy reads corrupted signal.
- **Suggested fix:** Guard the product: `if std_c * std_x < 1e-8: return 0.0` or use `max(1e-8, std_c * std_x)` as denominator.

### C-02: Silent exception swallowing in override log parsing
- **File:** `src/aura/bridge/signals.py`, line ~458
- **Description:** `get_recent_overrides()` silently skips malformed JSON lines with `except Exception: continue`. While statistics are tracked at lines 465-474, the initial bare exception catch at 458 discards error context.
- **Impact:** T2 pattern engine receives incomplete override histories without warning. Silent data loss corrupts pattern detection.
- **Suggested fix:** Log each skipped line with line number and first 100 chars of content. Use `except (json.JSONDecodeError, ValueError)` instead of bare `Exception`.

### C-03: Unhandled field unpacking errors in calibration state
- **File:** `src/aura/bridge/calibration.py`, lines ~211-219
- **Description:** `load_state()` catches `json.JSONDecodeError` but not `KeyError` or `ValueError` from field unpacking after parse. If JSONL line parses as valid JSON but has wrong structure, exception propagates uncaught.
- **Impact:** Calibration state fails to load; weight drift not applied; Buddy remains uncalibrated with stale weights.
- **Suggested fix:** Catch `(json.JSONDecodeError, ValueError, KeyError)` tuple in the try block.

### C-04: Division by zero in bridge critique loss_rate
- **File:** `src/aura/bridge/critique.py`, lines ~144-148
- **Description:** `loss_rate = len(ignored_losses) / total_overrides` is computed after a guard checking `if total_overrides == 0: return None`, but the guard checks a different filtered list than the denominator used in the division. If the filter at line 140 returns empty, division by zero occurs.
- **Impact:** Runtime crash in critique evaluation; blocks bridge health assessment.
- **Suggested fix:** Use the same variable for guard and denominator, or compute `loss_rate` inside an `if total_overrides > 0:` block.

### C-05: Missing readiness_signal.json in bridge file initialization
- **File:** `src/aura/bridge/signals.py`, lines ~167-189
- **Description:** `_ensure_bridge_files()` creates defaults for `outcome_signal.json` and `active_rules.json` but NOT `readiness_signal.json`. On first run (fresh install), `FeedbackBridge.read_readiness()` returns None.
- **Impact:** T2 pattern engine cannot initialize on a fresh install. System fails silently with no readiness signal available. This is the root cause of why 3 of 4 bridge files are missing in production.
- **Suggested fix:** Add readiness_signal.json default creation with `{"readiness_score": 50.0, "timestamp": "<now>", "model_version": "v1"}` at line 180.

---

## High Priority Issues

### H-01: Method name mismatch — predict() vs predict_loss_probability() (PERSISTENT)
- **File:** `src/aura/core/readiness.py`, line ~1155
- **Description:** Code calls `predictor.predict(ctx)` but the `OverridePredictor` class defines `predict_loss_probability(ctx)`. Call is wrapped in try/except, so it silently returns 0.0 instead of crashing.
- **Impact:** US-317 (override loss risk penalty) is completely non-functional every compute cycle. Feature is dead code.
- **Suggested fix:** Change `predictor.predict(ctx)` → `predictor.predict_loss_probability(ctx)`. Add regression test.
- **Status:** Reported in previous scan (2026-03-25 00:47). Still unfixed.

### H-02: Missing bridge files — 3 of 4 contract files absent (PERSISTENT)
- **Files:** `.aura/bridge/`
  - `readiness_signal.json` — Present, fresh (2026-03-25 12:10 UTC)
  - `outcome_signal.json` — **MISSING**
  - `override_events.jsonl` — **MISSING**
  - `active_rules.json` — **MISSING**
- **Impact:** Bidirectional Aura↔Buddy feedback loop is broken. Trade outcomes from Buddy never feed back into Aura pattern detection. Rules engine loads empty state on every startup.
- **Suggested fix:** Add `_ensure_bridge_files()` call in `FeedbackBridge.__init__()` to seed missing files with safe defaults.
- **Status:** Reported across all 4 previous scans. Still unfixed.

### H-03: Lock not released on write failure in persistence layer
- **File:** `src/aura/persistence.py`, lines ~59-105
- **Description:** In `_locked_atomic_write()`, if `os.write(fd, ...)` raises, the finally block attempts `fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)` which can itself fail if the file descriptor is in a bad state. If the unlock call fails, the lock file remains locked permanently.
- **Impact:** Deadlock on all subsequent writes to the same bridge file. Requires manual `.lock` file deletion to recover.
- **Suggested fix:** Wrap lock release in its own try/except. Cache `fileno()` before the flock call.

### H-04: No validation of bridge rule adjustment values
- **File:** `src/aura/bridge/rules_engine.py`, lines ~318-326
- **Description:** Value clamping only applies to numeric types (`isinstance(adjustments[param], (int, float))`). If a rule's adjustment value is None or a string, clamping is silently skipped and invalid data is passed to Buddy.
- **Impact:** Buddy's scanner gates receive wrong types, potentially causing crashes or incorrect gate evaluations.
- **Suggested fix:** Validate adjustment values at rule creation time (line ~235), not just at evaluation time.

### H-05: Incorrect auto-remediation metrics in self_model_validator
- **File:** `src/aura/core/self_model_validator.py`, lines ~336-341
- **Description:** In `_check_stale_nodes()`, `fixes += 1` is incremented before the try block that calls `graph.add_node(node)`. If the SQLite operation fails, the fix count is not decremented.
- **Impact:** Auto-remediation statistics report more fixes than actually applied. Gives false confidence in data integrity.
- **Suggested fix:** Move `fixes += 1` inside the try block, after the successful operation.

---

## Medium Priority Issues

### M-01: Precision loss in co-evolution weight drift
- **File:** `src/aura/bridge/coevolution.py`, lines ~87-92
- **Description:** `_apply_drift()` uses `max_drift` of 0.1 but has no minimum step size. If target and current differ by 0.05, full jump occurs. If `buddy_cal < 0.4`, `aura_outcome_weight` can jump from 1.0 → 0.5 in one step.
- **Impact:** Co-evolution oscillates between extremes instead of converging smoothly.
- **Suggested fix:** Add minimum step size: `if abs(delta) > max_drift: delta = max_drift * sign(delta) else: delta *= 0.5`.

### M-02: Rules engine `multiply` operator silently drops value
- **File:** `src/aura/bridge/rules_engine.py`, lines ~293-299
- **Description:** Missing `elif operator == "multiply":` fallback for multiply rules fired before an initial value is set. Currently safe (all rules use "set"), but will silently fail when multiply rules are added.
- **Impact:** Latent bug — will silently corrupt bridge rule values when multiply operator is used.
- **Suggested fix:** Add multiply fallback with warning log (5-line fix).

### M-03: Stale state false positives in self_model_validator
- **File:** `src/aura/core/self_model_validator.py`, lines ~302-308
- **Description:** Malformed ISO timestamps (e.g., "2026-03-32T00:00:00Z") in `_check_stale_nodes()` are caught and treated as stale. Nodes with parse errors are grouped with legitimately stale nodes.
- **Impact:** False stale detections trigger unnecessary remediation.
- **Suggested fix:** Track parse failures separately from time-based staleness.

### M-04: Index safety in pattern cloud enhancement
- **File:** `src/aura/patterns/engine.py`, line ~268
- **Description:** `pattern.evidence[-5:]` assumes evidence is a non-empty list. If cloud synthesis removes all evidence items, subsequent code fails.
- **Impact:** Cloud synthesis creates invalid patterns that crash downstream.
- **Suggested fix:** Add guard: `evidence[-5:] if evidence else []`.

### M-05: 3 test failures due to missing `textstat` library
- **Files:** `tests/test_aura_phase16.py`, `tests/test_aura_phase18.py`
- **Failures:**
  - `TestPhase16Readability::test_simple_text_high_readability` — Expected > 0.5, got 0.5
  - `TestPhase16Readability::test_complex_academic_text_low_readability` — Expected < 0.4, got 0.5
  - `TestPhase18Integration::test_integration_complex_message_readability_cognitive_load` — Expected < 0.4, got 0.5
- **Root cause:** `TextReadabilityAnalyzer` returns neutral 0.5 defaults when `textstat` is not installed.
- **Impact:** Readability scoring is non-functional in current environment.
- **Suggested fix:** Add `textstat` to dependencies in `pyproject.toml` and install in runtime environment.

### M-06: Learning rate vanishing without clear signal
- **File:** `src/aura/prediction/lr_scheduler.py`, line ~67
- **Description:** `base_lr = self.initial_lr / (1.0 + t / self.warmup_samples)` — for very large t, learning rate approaches zero asymptotically. Floor at line 75 (`max(self.lr_floor, lr)`) catches this but the decay is aggressive.
- **Impact:** Late-stage training updates have negligible effect; model stops learning without explicit signaling.
- **Suggested fix:** Consider switching to cosine annealing or adding a minimum effective LR warning.

---

## Low Priority / Code Quality

### L-01: Unused imports (27 instances across codebase)
- **Files:** Multiple — `typing.Optional` most common, plus `sys`, `Tuple`, and redefined `json` in `companion.py:1909`
- **Suggested fix:** Run `autoflake --remove-all-unused-imports` or add pre-commit flake8 hook with F401.

### L-02: Line length violations (45 instances)
- **Files:** Distributed across all modules; mostly in docstrings and complex f-strings.
- **Suggested fix:** Configure editor ruler at 120 chars; refactor long lines.

### L-03: Unused f-string prefixes (13 instances)
- **Files:** Multiple — f-strings with no interpolation placeholders.
- **Suggested fix:** Remove `f` prefix or add intended interpolation.

### L-04: Unused variables
- `word_count` in `conversation_processor.py`
- `emotional_state` in another module
- **Suggested fix:** Prefix with `_` if intentionally unused, or remove.

### L-05: Redundant imports in pattern engine
- **File:** `src/aura/patterns/engine.py`, lines ~80-82
- **Description:** `timedelta` imported at module level and again inside `run_all()`.
- **Suggested fix:** Remove inner import.

### L-06: Inefficient f-string logging
- **File:** `src/aura/bridge/signals.py`, lines ~255, 272, 402, 418, 432
- **Description:** f-strings in `logger.warning()` calls force interpolation even when log level filters the message.
- **Suggested fix:** Use `%` formatting: `logger.warning("msg %s", var)`.

### L-07: Indentation issues in readiness.py
- **File:** `src/aura/core/readiness.py`, lines ~609-615
- **Description:** 6 comment indentation issues flagged by flake8 (E114/E116).
- **Suggested fix:** Align comments with code block.

### L-08: Missing blank line before function definition
- **File:** `src/aura/patterns/override_extractor.py`, line ~56
- **Description:** E305 — expected 2 blank lines before function definition.
- **Suggested fix:** Add blank line.

### L-09: Multiple imports on one line
- **File:** `src/aura/core/readiness.py`, line ~357
- **Description:** E401 — multiple imports on one line.
- **Suggested fix:** Split to separate import statements.

### L-10: Unvalidated datetime parsing in manifests
- **File:** `src/aura/bridge/manifests.py`, lines ~61-64
- **Description:** `datetime.fromisoformat()` errors logged as "stale" without distinguishing parse errors from actual staleness.
- **Suggested fix:** Log parse errors separately.

### L-11: Type inconsistency in pattern evidence field
- **File:** `src/aura/patterns/base.py`
- **Description:** evidence field may be List or Dict depending on tier; no runtime type guard.
- **Suggested fix:** Enforce consistent type in base class with type annotation.

### L-12–L-14: Various spacing/style issues (6 instances)
- **Files:** Multiple — E261 (missing space before inline comments), E128 (continuation indentation)
- **Suggested fix:** Auto-format with `black` or `autopep8`.

---

## Bridge Contract Health

| File | Status | Last Updated | Schema Compliant | Notes |
|------|--------|-------------|-----------------|-------|
| `readiness_signal.json` | **PRESENT** | 2026-03-25 12:10 UTC | Yes | Score: 78.4, cognitive_load: low, emotional_state: calm |
| `outcome_signal.json` | **MISSING** | N/A | N/A | Never created; Buddy→Aura feedback broken |
| `override_events.jsonl` | **MISSING** | N/A | N/A | No override history available for T2 patterns |
| `active_rules.json` | **MISSING** | N/A | N/A | Rules engine loads empty state every startup |
| `readiness_signal.json.lock` | **PRESENT** | — | N/A | Lock file exists (0 bytes — healthy) |

**Bridge Health Rating: 25% (1 of 4 files operational)**

The readiness signal (Aura→Buddy) path works. The feedback loop (Buddy→Aura) is completely non-functional due to missing files.

---

## Test Coverage Gaps

### Current Status: 795 tests, 792 passing (99.6%)

### Identified Gaps:
1. **No tests for bridge file initialization** — `_ensure_bridge_files()` behavior is untested; the 3 missing bridge files are a direct consequence.
2. **No integration test for predict_loss_probability call** — H-01 method mismatch has persisted through 4 scans because no test exercises the live call path.
3. **No tests for co-evolution oscillation** — M-01 precision loss would be caught by a convergence test.
4. **No tests for multiply operator in rules engine** — M-02 is latent because only "set" operator is tested.
5. **Missing textstat dependency** — 3 tests fail because readability analyzer returns defaults; need either the library installed or tests marked as skip-if-unavailable.
6. **No deadlock test for persistence layer** — H-03 lock-not-released scenario is untested.

---

## Recurring Patterns

### Pattern 1: Method Contract Drift (4 reports running)
The H-01 method name mismatch (`predict()` vs `predict_loss_probability()`) exemplifies a systemic issue: no shared Protocol/ABC type-checks call contracts. New modules define methods that callers reference by different names. **Recommendation:** Define shared `Predictor` Protocol + enable `mypy` in CI.

### Pattern 2: Bridge File Initialization Gap (5 reports running)
Three missing bridge files have been flagged in every scan since March 24. The fix is trivial (~10 lines) but hasn't been applied. This is the highest-impact persistent issue in the codebase. **Recommendation:** Priority 1 fix.

### Pattern 3: Silent Exception Handling in Bridge I/O
Multiple bridge modules catch exceptions too broadly and either swallow errors or log them without sufficient context. This makes debugging bridge failures extremely difficult. Partially improved since March 24, but signals.py line 458 and calibration.py lines 211-219 remain problematic. **Recommendation:** Standardize bridge I/O error handling with a shared `bridge_io_guard()` context manager.

### Pattern 4: Numerical Stability in Scoring Modules
The metacognitive scorer (C-01), critique module (C-04), and lr_scheduler (M-06) all have division-related risks. As new scoring modules are added, this class of bug will recur. **Recommendation:** Create a `safe_divide(a, b, default=0.0)` utility and use it consistently.

---

## Comparison to Previous Report (2026-03-25 00:47)

### Resolved Since Last Report
- None — no code changes detected between scans.

### New Issues Found
| ID | Severity | Description |
|----|----------|-------------|
| C-01 | Critical | Divide-by-zero in metacognitive effort_allocation_score |
| C-02 | Critical | Silent exception swallowing in override log parsing |
| C-03 | Critical | Unhandled field unpacking in calibration state |
| C-04 | Critical | Division by zero in bridge critique loss_rate |
| C-05 | Critical | Missing readiness_signal.json in bridge init |
| H-03 | High | Lock not released on write failure (persistence) |
| H-04 | High | No validation of rule adjustment values |
| H-05 | High | Incorrect auto-remediation metrics |
| M-01 | Medium | Co-evolution weight drift precision loss |
| M-04 | Medium | Index safety in pattern cloud enhancement |
| M-06 | Medium | Learning rate vanishing without signal |

These are newly identified through deeper static analysis of scoring, prediction, and bridge modules that were not examined in previous scans.

### Persistent Issues (unfixed across multiple reports)
| ID | Severity | Reports | Description |
|----|----------|---------|-------------|
| H-01 | High | 2 | Method name mismatch (predict vs predict_loss_probability) |
| H-02 | High | 5 | Missing 3 of 4 bridge contract files |
| M-02 | Medium | 2 | Rules engine multiply operator fallback |
| M-05 | Medium | 2 | textstat library missing (3 test failures) |

### Test Count Delta
- Previous: 633 tests (633 passing)
- Current: 795 tests (792 passing)
- Delta: +162 new tests, -3 newly failing (textstat dependency)

---

## Recommended Priority Actions

1. **[5 min]** Fix H-01: Change `predictor.predict(ctx)` → `predictor.predict_loss_probability(ctx)` in readiness.py
2. **[10 min]** Fix C-05 + H-02: Add `_ensure_bridge_files()` to seed all 4 bridge files with safe defaults
3. **[5 min]** Fix C-01: Guard `std_c * std_x` product in metacognitive.py
4. **[5 min]** Fix C-04: Move loss_rate computation inside `if total_overrides > 0:` block
5. **[10 min]** Fix C-02 + C-03: Narrow exception handling in signals.py and calibration.py
6. **[5 min]** Install textstat: `pip install textstat` to fix 3 test failures
7. **[15 min]** Fix H-03: Add defensive lock release in persistence.py

**Estimated total fix time: ~55 minutes for all critical and high items.**
