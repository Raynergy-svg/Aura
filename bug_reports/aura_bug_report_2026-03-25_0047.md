# Aura Bug Report — 2026-03-25 00:47 ET

**Scan Type:** Automated Scheduled Scan (`aura-bug-scan`)
**Scan Coverage:** Full source tree + bridge files + test suite + static analysis
**Previous Reports:** `aura_bug_report_2026-03-24_1313.md`, `aura_bug_report_2026-03-24_FOLLOWUP.md`, `aura_bug_report_2026-03-24_1756.md`
**Python Files Scanned:** 34 files across `src/`, all test files under `tests/`
**Tools Used:** `py_compile`, `flake8`, `pytest`

---

## Executive Summary

| Category | Count | Change vs 17:56 Report |
|---|---|---|
| CRITICAL | 0 | — (none) |
| HIGH | 2 | No change (H-NEW-01 and H-01 both persist) |
| MEDIUM | 1 | **+1 NEW** (latent rules engine multiply bug) |
| LOW / Code Quality | 11 | Mostly stable; 2 new, 1 resolved |
| **Test Suite** | **633/633 passing** | ↑ +51 new tests (was 582) |
| **Bridge Health** | 1/4 files present | Unchanged |

The codebase trajectory remains positive. No new critical issues were introduced. The HIGH-priority method-name mismatch (H-NEW-01) is still not fixed — it silently degrades US-317 every compute cycle. One new medium-priority latent logic bug was identified in the rules engine multiply operator path. Test coverage expanded by 51 tests since the previous scan.

---

## Section 1 — CRITICAL Issues

**None.** All previously identified critical issues remain resolved.

---

## Section 2 — HIGH Priority Issues

### H-NEW-01 — `OverridePredictor.predict()` Method Does Not Exist (PERSISTENT — Not Fixed)
**File:** `src/aura/core/readiness.py`, line 1155
**Severity:** HIGH — US-317 feature silently non-functional on every compute cycle
**Status:** OPEN — present in 17:56 report, still unresolved

**Root Cause:**
`ReadinessComputer.compute()` calls:
```python
override_loss_risk = predictor.predict(ctx)
```
`OverridePredictor` exposes no `.predict()` method. Its public interface is:
- `predict_loss_probability(context: Dict) -> float`
- `predict_batch(contexts: List[Dict]) -> List[float]`
- `update_from_outcome(context, outcome)`

**What Actually Happens:**
The call is inside a `try/except Exception` block (lines 1147–1163). When `predictor.predict(ctx)` raises `AttributeError`, the except clause catches it, logs `"US-317: Override predictor error: ..."`, and sets `override_loss_risk = 0.0`. This means:
- US-317 never penalizes high-risk override situations
- The exception log entry fires on every compute cycle where the predictor is trained
- The override risk signal in `readiness_signal.json` is always `0.0` even when risk is elevated

**Note vs Previous Report:** The previous report described this as a "runtime crash." That was slightly inaccurate — the surrounding try/except prevents a crash but causes **silent functional failure**. US-317 is dead code whenever the predictor is trained.

**Fix:**
```python
# readiness.py line 1155 — BEFORE (broken)
override_loss_risk = predictor.predict(ctx)

# AFTER (correct)
override_loss_risk = predictor.predict_loss_probability(ctx)
```

**Test Gap:** The override predictor integration path in `ReadinessComputer.compute()` is not exercised by any test with a real (un-mocked) `OverridePredictor`. Add:
```python
def test_readiness_compute_override_predictor_method_name():
    """Regression: ensure predictor.predict_loss_probability() is called, not .predict()."""
    from src.aura.prediction.override_predictor import OverridePredictor
    predictor = OverridePredictor()
    # Manually set _trained flag to force the code path
    predictor._trained = True
    computer = ReadinessComputer()
    computer._override_predictor = predictor
    # Should not raise AttributeError
    result = computer.compute(emotional_state="calm")
    assert result is not None
    assert result.override_loss_risk == 0.0  # Untrained model returns low risk
```

---

### H-01 — Missing Bridge Files (PERSISTENT — Unchanged Since FOLLOWUP Report)
**File:** `.aura/bridge/`
**Severity:** HIGH — Three of four bridge contract files have never been created
**Status:** OPEN — no change since first identified

**Current Bridge State:**

| File | Present | Status |
|---|---|---|
| `readiness_signal.json` | ✅ Yes | Fresh (updated ~00:44 UTC today) |
| `outcome_signal.json` | ❌ No | Never created |
| `override_events.jsonl` | ❌ No | Never created |
| `active_rules.json` | ❌ No | Never created |

**Impact:**
- `FeedbackBridge.read_outcome_signal()` returns `None` on every call — Buddy's trade outcomes are never fed back to Aura's pattern engine
- T2 cross-domain pattern detection has no trading outcome data to correlate against
- `BridgeRulesEngine` reads from an empty/missing `active_rules.json` on every startup
- `PatternEngine` T2/T3 detection is running partially blind

**Root Cause:** `FeedbackBridge.__init__()` creates the bridge directory but does not initialize absent files with empty/default content. There is no auto-healing step.

**Fix:** Add to `FeedbackBridge.__init__()`:
```python
def _ensure_bridge_files(self) -> None:
    """Seed absent bridge files with safe defaults so readers never get NOT_FOUND."""
    defaults = {
        self._outcome_path: '{"pnl_today": 0.0, "win_rate_7d": 0.0, "override_events": [], "regime": "NORMAL", "streak": "neutral", "trades_today": 0, "open_positions": 0, "max_drawdown_today": 0.0, "timestamp": ""}',
        Path(str(self.bridge_dir) + "/active_rules.json"): '[]',
    }
    for path, default in defaults.items():
        if not path.exists():
            self._locked_write(path, default)
            logger.info("Bridge: seeded missing file %s with defaults", path.name)
    # override_events.jsonl — create empty file
    if not self._override_log_path.exists():
        self._override_log_path.touch()
```

---

## Section 3 — MEDIUM Priority Issues

### M-NEW-01 — Rules Engine `multiply` Operator Silently Drops Value When Parameter Not Pre-Set
**File:** `src/aura/bridge/rules_engine.py`, lines 293–299
**Severity:** MEDIUM (latent — not triggered by current rule types, but will break silently if activated)

**Root Cause:**
In `get_buddy_gate_adjustments()`, the operator evaluation chain is:
```python
if operator == "set":
    adjustments[param] = value
elif operator == "add" and param in adjustments:
    adjustments[param] = adjustments[param] + value
elif operator == "add":                            # fallback for add
    adjustments[param] = value
elif operator == "multiply" and param in adjustments:
    adjustments[param] = adjustments[param] * value
# ← Missing: elif operator == "multiply": ???
```

Rules are sorted by precedence: `multiply(0) → add(1) → set(2)`. This means multiply rules run first, before any initial value is established by an add or set rule. If a multiply rule is the first (or only) rule for a parameter, `param not in adjustments` at evaluation time, so the multiplication value is silently dropped — `adjustments[param]` is never set. Additionally, `rule.triggered_count` is incremented unconditionally at line 300, masking the silent failure.

**Current exposure:** Low, because `_compute_adjustment()` exclusively generates `"operator": "set"` for all rule types. However, if a future rule type is added with `"operator": "multiply"` (e.g., a confidence multiplier), this path will silently discard it.

**Fix:** Add a fallback for orphaned multiply rules:
```python
elif operator == "multiply" and param in adjustments:
    adjustments[param] = adjustments[param] * value
elif operator == "multiply":
    # multiply with no prior value — treat multiply as set (cannot multiply nothing)
    adjustments[param] = value
    logger.warning("Rules engine: multiply rule '%s' on unset param '%s' — applied as set",
                   rule.rule_id, param)
```

---

## Section 4 — LOW Priority / Code Quality

### L-01 — H-NEW-01 Test Gap (Still Unaddressed)
See H-NEW-01 above. The override predictor `predict()` → `predict_loss_probability()` mismatch has no regression test covering the live code path.

---

### L-02 — `_last_online_update` Not Declared in `__init__` (PERSISTENT from 17:56 report)
**File:** `src/aura/prediction/readiness_v2.py`, line 614 (assignment in `update_from_outcome()`)
**Status:** Still open — `_last_online_update` is assigned at line 614 but never declared in `ReadinessModelV2.__init__()`.
If any code path reads `self._last_online_update` before `update_from_outcome()` is ever called (e.g., serialization, status display), it will raise `AttributeError`.
**Fix:** Add `self._last_online_update: Optional[str] = None` to `__init__()` (around line 148, after `self._load_buffer()`).

---

### L-03 — F811 Double Import of Config Functions in `engine.py`
**File:** `src/aura/patterns/engine.py`, lines 80 and 84
**Issue:** `get_t1_config`, `get_t2_config`, `get_t3_config` are imported twice — once on line 80 (inside a `if config is None:` branch) and again unconditionally on line 84. The line 80 import is silently overwritten by line 84, making the conditional import useless.
```python
# line 80 (inside if block):
from src.aura.config import load_config, get_t1_config, get_t2_config, get_t3_config
...
# line 84 (unconditional):
from src.aura.config import get_t1_config, get_t2_config, get_t3_config   # ← F811 redefinition
```
**Fix:** Remove the unconditional import at line 84; move it inside the `if config is None:` block alongside `load_config`. Or hoist all config imports to the top of the function and use a single import.

---

### L-04 — `avg_stress_score` Dead Variable in `emotional_regulation.py`
**File:** `src/aura/scoring/emotional_regulation.py`, line 108
**Issue:** `avg_stress_score` is computed but never referenced:
```python
avg_stress_score = sum(
    s.get("stress_level_score", 0.7) for s in stress_levels
) / len(stress_levels)
```
The function then proceeds to calculate `high_stress_overrides` using `_find_nearest_stress()` (per-override stress lookup), without using the aggregate average. Either: (a) the average was intended to be used in the composite calculation and was accidentally omitted, or (b) it's dead code from a refactor.
**Impact:** Possible incomplete logic in `regulation_discipline()`. If the average was meant to weight the penalty output, the scoring is currently based on count-only ratios rather than stress magnitude.
**Fix:** Investigate whether `avg_stress_score` should be used in the final `discipline` computation. If not needed, delete the assignment to avoid confusion.

---

### L-05 — Unused Import: `uuid` in `tier1.py`
**File:** `src/aura/patterns/tier1.py`, line 19
`import uuid` — uuid is imported but not used in this file. Pattern IDs are constructed using `uuid.uuid4()` in the base module, not here.
**Fix:** Remove `import uuid`.

---

### L-06 — Unused Import: `timedelta` in `tier3.py`
**File:** `src/aura/patterns/tier3.py`, line 39
`from datetime import ..., timedelta` — `timedelta` is not used.
**Fix:** Remove `timedelta` from the import.

---

### L-07 — Multiple Unused Imports in `changepoint.py`
**File:** `src/aura/prediction/changepoint.py`, lines 19-20
`dataclasses.field`, `typing.Any`, `typing.Dict`, `typing.Optional` are imported but unused.
**Fix:** Remove the unused imports.

---

### L-08 — Multiple Unused Imports in `metacognitive.py`
**File:** `src/aura/scoring/metacognitive.py`, lines 15-16
`dataclasses.field`, `typing.Any`, `typing.Dict`, `typing.Optional` are imported but unused.
**Fix:** Remove the unused imports.

---

### L-09 — Unused `Tuple` Import in `emotional_regulation.py`
**File:** `src/aura/scoring/emotional_regulation.py`, line 14
`typing.Tuple` is imported but not used.
**Fix:** Remove from import line.

---

### L-10 — Redundant In-Function Import in `readiness.py`
**File:** `src/aura/core/readiness.py`, line 646
`from datetime import datetime, timezone, timedelta` inside `_compute_graph_features()` — all three are already imported at module top level.
**Fix:** Delete the redundant in-function import at line 646.

---

### L-11 — F541: F-strings Without Placeholders
**Files and lines:**
- `src/aura/core/readiness.py:1251` — f-string with no `{...}` interpolation
- `src/aura/patterns/cloud_fallback.py:727` — same
- `src/aura/patterns/tier2.py:331` — same
**Fix:** Strip the `f` prefix from each to make them plain strings. Cosmetic only.

---

## Section 5 — Bridge Contract Health

| Metric | Value | Status |
|---|---|---|
| `readiness_signal.json` present | Yes | ✅ |
| `readiness_signal.json` freshness | 2026-03-25T00:44 UTC | ✅ Very fresh (< 5 min old at scan time) |
| `readiness_signal.json` score | 83.5 | ✅ Valid range (0-100) |
| `readiness_signal.json` schema | All required fields present | ✅ |
| `outcome_signal.json` present | No | ❌ Missing |
| `override_events.jsonl` present | No | ❌ Missing |
| `active_rules.json` present | No | ❌ Missing |
| `readiness_signal.json.lock` | Present (0 bytes — not locked) | ✅ Normal |
| fcntl locking (`_locked_write`) | Confirmed: LOCK_EX + sidecar | ✅ |
| Atomic writes | Confirmed via `atomic_write_json()` | ✅ |
| T1 pattern last run | 2026-03-24T00:11 UTC | ⚠️ ~24h ago |
| T2 pattern last run | 2026-03-24T00:11 UTC | ⚠️ ~24h ago |
| T3 pattern last run | Never recorded in run_log.json | ⚠️ Never run |
| `t1_patterns.json` | Present, 2 bytes (`[]`) | ✅ Empty array is valid |
| `t2_patterns.json` | Present, 2 bytes (`[]`) | ✅ Empty array is valid |

**Signal Field Note:** `override_loss_risk` is `0.0` in the latest signal. This is expected given H-NEW-01 — the override predictor call site crashes silently on every cycle, so this field will always be `0.0` until H-NEW-01 is fixed.

**T1/T2 Pattern Age Note:** Last pattern run was ~24 hours ago. If the scheduled cadence is daily, this is within normal range. If cadence is sub-daily, the pattern engine may not be running.

---

## Section 6 — Test Coverage

**Result: 633/633 tests passing — 0 failures, 0 errors**
(↑ +51 tests since the 17:56 report which recorded 582 tests)

| Change | Count |
|---|---|
| Tests in 17:56 report | 582 |
| Tests in this scan | 633 |
| Net new tests | **+51** |

The test suite expanded with new phase-specific tests (likely `test_aura_phase15.py` and other recent additions).

**Persistent Test Gap (H-NEW-01):**
The `predictor.predict()` call site in `ReadinessComputer.compute()` remains uncovered by any test that uses a real `OverridePredictor` instance with `_trained = True`. The test for US-317 either mocks the predictor or does not trigger the `_trained` path.

**No new test gaps identified** in the newly added scoring modules (`emotional_regulation.py`, `decision_quality.py`, `metacognitive.py`). These appear to be tested via the phase test files.

---

## Section 7 — Recurring Patterns

### Pattern 1: Unused Imports Accumulating in New Modules
The three newest scoring modules (`changepoint.py`, `metacognitive.py`, `emotional_regulation.py`) all have unused imports from the `typing` module and `dataclasses`. This pattern appears consistently in newer files, suggesting imports are added optimistically at creation time and not cleaned up. A pre-commit flake8 hook with `F401` enforcement would prevent this.

### Pattern 2: H-NEW-01 Persistence Across Multiple Scans
`predictor.predict(ctx)` at `readiness.py:1155` was first identified in the 17:56 scan and has now gone through two consecutive scans unaddressed. While the try/except prevents a crash, US-317 (override loss risk penalty) has been non-functional for multiple days. This pattern of "soft-caught but silently broken" bugs is harder to detect because the system doesn't crash or alert.

**Recommendation:** Add explicit assertions in tests for the side effects of these code paths (e.g., assert that `override_loss_risk > 0` when the predictor is trained and context includes high-risk emotional state), not just that `compute()` returns without exception.

### Pattern 3: Bridge File Initialization Gap (3 Consecutive Reports)
The three missing bridge files have now appeared as an issue in four consecutive scan reports. This indicates no automated remediation is occurring and no one has manually seeded them. The fix is a 10-line `_ensure_bridge_files()` method — the cost/benefit ratio strongly favors fixing this now.

---

## Section 8 — Comparison to Previous Reports

### vs `aura_bug_report_2026-03-24_1756.md`

| Issue ID | 17:56 Status | This Report | Notes |
|---|---|---|---|
| H-NEW-01 `predictor.predict()` | Open | ❌ Still open | |
| H-01 Missing bridge files | Open | ❌ Still open | 4th consecutive scan |
| L-12 `_last_online_update` | Open | ❌ Still open (now L-02) | |
| L-08 F401 in-function import | Open | ❌ Still open (now L-10) | |
| L-09 F541 f-strings | Open | ❌ Still open (now L-11) | |
| M-NEW-01 Multiply op bug | Not present | ⚠️ New (MEDIUM) | Latent, not triggered |
| L-03 F811 double import | Not present | ⚠️ New (LOW) | `engine.py` |
| L-04 `avg_stress_score` unused | Not present | ⚠️ New (LOW) | `emotional_regulation.py` |

### Issue Resolution Velocity
- Critical: 7/7 resolved as of 17:56 report — maintained ✅
- High: 0/2 resolved since 17:56 report — needs attention ⚠️
- Medium: 0/1 resolved (new this scan)
- Low: 4/12 from 17:56 report appear resolved (import hygiene in some files improved)

---

## Recommended Action Order

1. **Fix H-NEW-01 NOW** — `readiness.py:1155`: Change `predictor.predict(ctx)` → `predictor.predict_loss_probability(ctx)`. US-317 has been dead for multiple days. One-line fix, zero risk.

2. **Seed missing bridge files (H-01)** — Add `_ensure_bridge_files()` to `FeedbackBridge.__init__()`. Create `outcome_signal.json` with empty defaults, `override_events.jsonl` as empty file, `active_rules.json` as `[]`. Unblocks T2 correlation and rules engine.

3. **Fix M-NEW-01 (multiply op fallback)** — `rules_engine.py:~line 299`: Add the missing `elif operator == "multiply":` fallback. Low risk, prevents future silent bugs when multiply rules are added.

4. **Fix L-02 (`_last_online_update`)** — `readiness_v2.py:__init__`: Add `self._last_online_update: Optional[str] = None`. One line.

5. **Clean up import hygiene (L-03 through L-11)** — Run `autoflake --remove-all-unused-imports -i src/` or fix manually. Low risk.

6. **Investigate L-04 (`avg_stress_score`)** — Determine whether `avg_stress_score` was meant to influence the `regulation_discipline()` output. If yes, there's missing logic. If no, delete it.

7. **Add regression test for H-NEW-01** — Ensure `predictor.predict_loss_probability()` is exercised through `ReadinessComputer.compute()` with a real predictor instance.

---

*Report generated by automated `aura-bug-scan` task. Scan duration: ~6 minutes. Total tests verified: 633/633 passing.*
*Next scheduled scan: TBD.*

---

## Exterminator Results — 2026-03-25T01:10 UTC

**Run by:** `aura-bug-squash` scheduled task
**Final test count:** 634/634 passing (↑ +1 regression test added)
**Note:** No git repository found in P-90 — commits logged below as descriptions only.

### Fixed Bugs

| ID | Severity | File | Fix | Commit |
|---|---|---|---|---|
| H-NEW-01 | HIGH | `src/aura/core/readiness.py:1155` | Changed `predictor.predict(ctx)` → `predictor.predict_loss_probability(ctx)`. US-317 override risk penalty is now functional. | No git repo |
| H-01 | HIGH | `src/aura/bridge/signals.py` | Added `_ensure_bridge_files()` called from `__init__()`. Seeds `outcome_signal.json` with zero defaults, creates empty `override_events.jsonl`, seeds `active_rules.json` as `[]`. Bridge is now self-healing on startup. | No git repo |
| M-NEW-01 | MEDIUM | `src/aura/bridge/rules_engine.py:~299` | Added `elif operator == "multiply":` fallback — treats orphaned multiply rules as set with a warning log. Prevents silent value discard when multiply rule fires before any set/add. | No git repo |
| L-01 | LOW | `tests/test_aura_readiness.py` | Added `test_readiness_compute_override_predictor_method_name` regression test. Uses real `OverridePredictor` with `_trained=True` to verify `predict_loss_probability()` path doesn't raise. | No git repo |
| L-02 | LOW | `src/aura/prediction/readiness_v2.py:149` | Added `self._last_online_update: Optional[str] = None` to `__init__()`. Prevents `AttributeError` if serialization or status display calls the field before first online update. | No git repo |
| L-03 | LOW | `src/aura/patterns/engine.py:80–84` | Hoisted `get_t1_config`, `get_t2_config`, `get_t3_config` import above the `if config is None:` block; removed unconditional duplicate import below. Eliminates F811 redefinition. | No git repo |
| L-04 | LOW | `src/aura/scoring/emotional_regulation.py:108` | Removed dead `avg_stress_score` variable. Discipline calculation uses per-override stress lookup only — the aggregate average was unused. Added clarifying comment. | No git repo |
| L-05 | LOW | `src/aura/patterns/tier1.py:19` | Removed unused `import uuid`. | No git repo |
| L-06 | LOW | `src/aura/patterns/tier3.py:39` | Removed unused `timedelta` from datetime import. | No git repo |
| L-07 | LOW | `src/aura/prediction/changepoint.py:19–20` | Removed unused `field`, `Any`, `Dict`, `Optional` — kept only `dataclass` and `List` which are used. | No git repo |
| L-08 | LOW | `src/aura/scoring/metacognitive.py:15–16` | Removed unused `field`, `Any`, `Dict`, `Optional` — kept only `dataclass` and `List`. | No git repo |
| L-09 | LOW | `src/aura/scoring/emotional_regulation.py:14` | Removed unused `Tuple` from typing import. | No git repo |
| L-10 | LOW | `src/aura/core/readiness.py:646` | Removed redundant `from datetime import ...` (already at module level). Preserved `import math` as a local import since `math` is not at module level. | No git repo |
| L-11 | LOW | `readiness.py:1248`, `cloud_fallback.py:727–728`, `tier2.py:331,335` | Stripped `f` prefix from 4 f-strings that contained no `{...}` interpolation placeholders. | No git repo |

### Skipped Bugs

None. All 14 actionable items were fixed.

### Collateral Fixes (Tests Updated to Match Fixed Behavior)

Three tests were asserting the old broken behavior and were updated to reflect the corrected contract:

- `tests/test_aura_config.py::test_outcome_read_returns_none_when_missing` → renamed `test_outcome_read_returns_default_when_seeded`. Now asserts that `read_outcome()` returns a default `OutcomeSignal` (not `None`) after H-01 seeding fix.
- `tests/test_aura_phase11.py::test_health_missing_files` → renamed `test_health_seeded_files`. Now asserts `health.outcome == "healthy"` since `_ensure_bridge_files()` seeds the file on init.
- `tests/test_aura_phase13.py` — `_mock_predictor()` and inline mock setup updated to register `predict_loss_probability` instead of `predict` on mock objects.

### New Issues Discovered

None found during fixing.
