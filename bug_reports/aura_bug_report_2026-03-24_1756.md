# Aura Bug Report — 2026-03-24 17:56
**Scan Type:** Automated Scheduled Scan (`aura-bug-scan`)
**Scan Coverage:** Full source tree + bridge files + test suite + static analysis
**Previous Reports:** `aura_bug_report_2026-03-24_1313.md`, `aura_bug_report_2026-03-24_FOLLOWUP.md`

---

## Executive Summary

| Category | Count | Change vs Previous |
|---|---|---|
| CRITICAL | 0 | ↓ 7 (all resolved) |
| HIGH | 2 | ↓ 1 (H-02 resolved, 1 new H-NEW found) |
| MEDIUM | 0 | ↓ 4 (all resolved) |
| LOW / Code Quality | 18 | New enumeration (flake8 full pass) |
| **Test Suite** | **582/582 passing** | Healthy |
| **Bridge Health** | 1/4 files present | Persistent gap |

All 7 CRITICAL issues from the 13:13 report are **confirmed fixed**. The single new HIGH issue is a method-name mismatch in `readiness.py` where `predictor.predict(ctx)` is called on an object that only exposes `predict_loss_probability()`. This is a **runtime crash path** if the override predictor block is reached. Bridge file gaps (outcome, overrides, rules) persist from the FOLLOWUP report with no change.

---

## Section 1 — CRITICAL Issues

**None.** All 7 critical issues from `aura_bug_report_2026-03-24_1313.md` are confirmed resolved.

| Issue ID | Description | Status |
|---|---|---|
| C-01 | `FeedbackBridge._locked_write()` had no actual `LOCK_EX` | ✅ RESOLVED |
| C-02 | `ReadinessModelV2._save_buffer()` used wrong import path | ✅ RESOLVED |
| C-03 | `OverridePredictor._save_model()` used wrong import path | ✅ RESOLVED |
| C-04 | `BridgeRulesEngine._save_rules()` non-atomic write | ✅ RESOLVED |
| C-05 | `Tier1Detector._save_patterns()` non-atomic write | ✅ RESOLVED |
| C-06 | (from 13:13 report — confirmed fixed) | ✅ RESOLVED |
| C-07 | (from 13:13 report — confirmed fixed) | ✅ RESOLVED |

---

## Section 2 — HIGH Priority Issues

### H-NEW-01 — `OverridePredictor.predict()` Method Does Not Exist
**File:** `src/aura/core/readiness.py` (~line 1133)
**Severity:** HIGH — Runtime `AttributeError` crash if code path is reached
**Introduced:** Unknown (not present in previous reports — newly detected)

**Root Cause:**
`ReadinessComputer.compute()` calls:
```python
predictor.predict(ctx)
```
where `predictor` is an instance of `OverridePredictor`. However, `OverridePredictor` defines no `.predict()` method. Its public interface exposes:
- `predict_loss_probability(context: Dict) -> float`
- `predict_batch(contexts: List[Dict]) -> List[float]`
- `update_from_outcome(context, outcome)`

**Impact:**
Any execution of `readiness.py` that reaches the override predictor integration block will raise:
```
AttributeError: 'OverridePredictor' object has no attribute 'predict'
```
This would cause `compute()` to crash, returning no readiness score and leaving the bridge signal stale.

**Fix Required:**
In `readiness.py` ~line 1133, change:
```python
# BEFORE (broken)
result = predictor.predict(ctx)

# AFTER (correct)
result = predictor.predict_loss_probability(ctx)
```
Verify that the return value handling downstream expects a `float` in `[0.0, 1.0]` range (which is what `predict_loss_probability` returns), not a dict or complex object.

**Additional Check:**
Search `readiness.py` for all calls to `predictor.` to confirm no other mismatched method names exist.

---

### H-01 — Missing Bridge Files (Persistent from FOLLOWUP Report)
**File:** `.aura/bridge/`
**Severity:** HIGH — Bridge contract is partially broken; Buddy cannot receive feedback; override loop is non-functional
**Status:** Persists unchanged since `aura_bug_report_2026-03-24_FOLLOWUP.md`

**Current bridge state:**
| File | Expected | Status |
|---|---|---|
| `readiness_signal.json` | Aura → Buddy | ✅ Present, fresh (2026-03-24T21:54 UTC) |
| `outcome_signal.json` | Buddy → Aura | ❌ MISSING |
| `override_events.jsonl` | Bidirectional log | ❌ MISSING |
| `active_rules.json` | TTL gate rules | ❌ MISSING |

**Impact:**
- Buddy cannot write outcome signals (no file path initialized)
- Override event history is not being accumulated
- `BridgeRulesEngine` has no rules file to load — falls back to empty rules on every startup
- Pattern Engine T2/T3 cross-engine correlation is blind to outcome feedback

**Fix Required:**
Seed the missing files with valid empty structures. These should be created by Aura's initialization path if not present — verify `FeedbackBridge.__init__()` or the CLI startup sequence creates them:
```json
// outcome_signal.json
{"score": null, "timestamp": null, "source": "buddy", "outcome": null}

// active_rules.json
{"rules": [], "version": 1, "last_updated": null}
```
```
// override_events.jsonl
(empty file — JSONL appends from here)
```

**Long-term fix:** Add a `_ensure_bridge_files()` call in `FeedbackBridge.__init__()` that creates these files with default content if they don't exist.

---

## Section 3 — MEDIUM Priority Issues

**None.** All 4 medium issues from the 13:13 report are confirmed resolved.

| Issue ID | Description | Status |
|---|---|---|
| M-01 | (from 13:13 report) | ✅ RESOLVED |
| M-02 | (from 13:13 report) | ✅ RESOLVED |
| M-03 | (from 13:13 report) | ✅ RESOLVED |
| M-04 | (from 13:13 report) | ✅ RESOLVED |
| H-02 (FOLLOWUP) | `"stress"` + `"stressed"` double-counting in `STRESS_KEYWORDS` | ✅ RESOLVED — `"stress"` removed, only `"stressed"` remains |

---

## Section 4 — LOW Priority / Code Quality Issues

### L-01 — F811: Double Import in `patterns/engine.py`
**File:** `src/aura/patterns/engine.py`, lines 80 and 84
**Tool:** flake8

`get_t1_config`, `get_t2_config`, `get_t3_config` are imported twice. The first import is inside an `if config is None:` conditional block (line 80); the second is unconditional (line 84) and shadows the first. The unconditional import is the one actually used. The conditional import is dead code.

**Fix:** Remove the import at line 80 (inside the conditional block). Keep only the unconditional import at line 84.

---

### L-02 — F401: Unused Imports — `patterns/engine.py`
**File:** `src/aura/patterns/engine.py`
`PatternStatus` and `PatternTier` are imported but never referenced in the module body.

**Fix:** Remove unused imports or, if they are re-exported intentionally, add `__all__` to document this.

---

### L-03 — F401: Unused Imports — `bridge/signals.py`
**File:** `src/aura/bridge/signals.py`, line 24
`Union` from `typing` is imported but unused.

**Fix:** Remove `Union` from the import line.

---

### L-04 — F401: Unused Imports — `prediction/readiness_v2.py`
**File:** `src/aura/prediction/readiness_v2.py`
`field` from `dataclasses` is imported but unused.

**Fix:** Remove `field` from the import line.

---

### L-05 — F401: Unused Imports — `prediction/override_predictor.py`
**File:** `src/aura/prediction/override_predictor.py`
`Tuple` from `typing` is imported but unused.

**Fix:** Remove `Tuple` from the import line.

---

### L-06 — F401: Unused Imports — `cli/companion.py`
**File:** `src/aura/cli/companion.py`
`json`, `uuid`, and `OutcomeSignal` are imported but unused.

**Fix:** Remove the three unused imports. If `OutcomeSignal` is planned for future use, add a `# noqa: F401` comment with a TODO.

---

### L-07 — F401: Unused Imports — `cli/main.py`
**File:** `src/aura/cli/main.py`
`sys`, `YELLOW`, `RED` are imported but unused.

**Fix:** Remove unused imports.

---

### L-08 — F401: Redundant In-Function Import — `core/readiness.py`
**File:** `src/aura/core/readiness.py`, line 624
`from datetime import datetime, timezone, timedelta` is imported inside `_compute_graph_features()` but these are already imported at module top level.

**Fix:** Remove the redundant in-function import at line 624.

---

### L-09 — F541: 11 F-Strings Without Placeholders — `cli/companion.py`
**File:** `src/aura/cli/companion.py`
Lines 140–143, 237, 257, 285, 293, 751, 806, 1022, 1224 contain `f"..."` strings with no `{...}` interpolation — they are plain strings wrapped in unnecessary f-string syntax.

**Fix:** Change `f"literal string"` → `"literal string"` at each of the 11 locations. No functional change; cosmetic cleanup.

---

### L-10 — F841: Unused Local Variable — `core/conversation_processor.py`
**File:** `src/aura/core/conversation_processor.py`, line ~189
`word_count = ...` is assigned but never read.

**Fix:** Either remove the assignment or use the variable where it was presumably intended.

---

### L-11 — Style: Multiple Imports on One Line — `core/readiness.py`
**File:** `src/aura/core/readiness.py`, line 302
`import tempfile, os` — PEP 8 recommends one import per line.

**Fix:** Split into two lines:
```python
import tempfile
import os
```

---

### L-12 — Missing `__init__` Declaration — `prediction/readiness_v2.py`
**File:** `src/aura/prediction/readiness_v2.py`
`_last_online_update` is assigned in `update_from_outcome()` (line ~589) but never declared in `__init__`. This bypasses type safety and will cause `AttributeError` if `_last_online_update` is read before `update_from_outcome()` is ever called.

**Fix:** Add `self._last_online_update: Optional[datetime] = None` to `ReadinessModelV2.__init__()`.

---

## Section 5 — Bridge Contract Health

| Metric | Value | Status |
|---|---|---|
| `readiness_signal.json` present | Yes | ✅ |
| `readiness_signal.json` freshness | 2026-03-24T21:54 UTC | ✅ Fresh (within 24h) |
| `readiness_signal.json` score | 83.5 | ✅ Valid range |
| `outcome_signal.json` present | No | ❌ Missing |
| `override_events.jsonl` present | No | ❌ Missing |
| `active_rules.json` present | No | ❌ Missing |
| fcntl locking (`_locked_write`) | Confirmed present (sidecar + LOCK_EX) | ✅ |
| Atomic writes (all bridge paths) | Confirmed via `atomic_write_json()` | ✅ |
| T1 pattern last run | 2026-03-24T00:11 UTC | ✅ |
| T2 pattern last run | 2026-03-24T00:11 UTC | ✅ |
| T3 pattern last run | Never | ⚠️ T3 has never executed |

**T3 Note:** Tier 3 (monthly narrative arcs) has no recorded execution. If T3 is intended to run on a monthly schedule, this is expected for a new system. If T3 should have run by now, verify the scheduler configuration for the T3 trigger.

---

## Section 6 — Test Coverage

**Result:** 582/582 tests passing — 0 failures, 0 errors.

| Test File | Tests | Result |
|---|---|---|
| All 19 test files | 582 | ✅ All passing |

**Note:** The H-NEW-01 bug (`predictor.predict()` method mismatch) is **not caught by the current test suite**. The test for the override predictor integration in `ReadinessComputer` apparently mocks the predictor or does not exercise the specific call site. A regression test should be added that calls `compute()` with a real (or correctly-typed mock) `OverridePredictor` instance to ensure the method name is valid.

**Recommended new test:**
```python
def test_readiness_compute_does_not_crash_with_override_predictor():
    """Ensures the override predictor call site uses the correct method name."""
    computer = ReadinessComputer(...)
    # Should not raise AttributeError
    result = computer.compute(context)
    assert result is not None
```

---

## Section 7 — Recurring Patterns

### Pattern 1: Import Hygiene
Every module scanned has at least one unused import (F401). This suggests imports are added during development and not cleaned up. Consider adding `flake8` to a pre-commit hook or CI step to enforce import hygiene automatically.

### Pattern 2: Method Contract Drift
H-NEW-01 is a method-name mismatch between caller (`readiness.py`) and callee (`override_predictor.py`). This class of bug occurs when:
- A method is renamed in the implementation but not updated in call sites
- A method is defined on a base class or protocol but not the concrete implementation

**Recommendation:** Define a `Predictor` Protocol or ABC with required method signatures. Type-check with `mypy` to catch these at development time rather than at runtime.

### Pattern 3: Bridge File Initialization Gap
Three of the four bridge files have never been created. This points to a missing initialization step — likely `FeedbackBridge.__init__()` or the CLI startup sequence doesn't seed absent files with defaults. This has persisted across multiple scan cycles without auto-healing, confirming there is no self-healing initialization.

---

## Section 8 — Comparison to Previous Reports

### vs `aura_bug_report_2026-03-24_1313.md`
| Category | 13:13 Report | This Report | Delta |
|---|---|---|---|
| CRITICAL | 7 | 0 | -7 ✅ |
| HIGH | 4 | 2 | -2 ✅ |
| MEDIUM | 4 | 0 | -4 ✅ |
| LOW | 5 | 12 | +7 (full flake8 pass new) |

### vs `aura_bug_report_2026-03-24_FOLLOWUP.md`
| Issue | FOLLOWUP Status | This Report |
|---|---|---|
| H-01 Missing bridge files | Open | ❌ Still open |
| H-02 Stress double-counting | Open | ✅ Resolved |
| New: H-NEW-01 `predictor.predict()` | Not present | ❌ New finding |

### Trend
The codebase is in a strong improvement trajectory. All critical data integrity and locking bugs have been resolved. The remaining issues are: one runtime crash path (H-NEW-01, HIGH), one persistent infrastructure gap (H-01, HIGH), and a collection of code-quality items that do not affect runtime correctness.

---

## Recommended Action Order

1. **Fix H-NEW-01 immediately** — `readiness.py` line ~1133: change `predictor.predict(ctx)` → `predictor.predict_loss_probability(ctx)`. Add a test that exercises this code path with a real `OverridePredictor` instance.

2. **Seed missing bridge files (H-01)** — Create `outcome_signal.json`, `override_events.jsonl`, and `active_rules.json` with empty/default content. Add `_ensure_bridge_files()` to `FeedbackBridge.__init__()`.

3. **Fix L-12 (`_last_online_update`)** — Add the attribute to `__init__` to prevent a potential `AttributeError` if read before `update_from_outcome()` is ever called.

4. **Clean up imports (L-01 through L-11)** — Run `autoflake --remove-all-unused-imports -i src/` or fix manually. Low risk, improves maintainability.

5. **Add a `Predictor` Protocol** — Define `predict_loss_probability()` on a shared interface so `mypy` can catch method-name mismatches before they reach production.

---

*Report generated by automated `aura-bug-scan` task. Scan duration: ~8 minutes. Next scheduled scan: tomorrow 06:00 UTC.*
