# Aura Bug Report — 2026-03-24 13:13 UTC

**Scan Type**: Comprehensive autonomous bug scan
**Codebase**: P-90 / Aura (Eve) Human Intelligence Engine
**Files Scanned**: 20+ source files across `src/aura/`
**Syntax Check**: All files pass `python -m py_compile` (zero syntax errors)
**Test Suite**: `pytest` not installed in environment — tests could not be run
**Previous Reports**: None (first scan)

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 7 |
| HIGH | 4 |
| MEDIUM | 6 |
| LOW | 3 |
| **Total** | **20** |

---

## CRITICAL Issues

### C-01: `_locked_write()` Missing fcntl LOCK_EX Lock
**File**: `src/aura/bridge/signals.py` — `_locked_write()` method
**Impact**: Data corruption risk under concurrent access

The docstring claims "exclusive file lock (fcntl.LOCK_EX)" but the implementation uses only temp-file + `os.rename` — no actual `fcntl.LOCK_EX` is acquired before writing. Meanwhile, `_locked_read()` and `_locked_append()` correctly use `fcntl.LOCK_SH` and `fcntl.LOCK_EX` respectively.

This means a concurrent reader (Buddy) can read a partially-renamed file, or two concurrent writers can race on temp file creation. The bridge contract is the only shared data path between Aura and Buddy — corruption here breaks both systems.

**Fix**: Acquire `fcntl.LOCK_EX` on the target file (or a `.lock` sidecar) before writing the temp file and renaming.

---

### C-02: Import Path Bug in `readiness_v2.py`
**File**: `src/aura/prediction/readiness_v2.py` — line 615
**Impact**: `ImportError` crash at runtime

```python
# Line 615 (BROKEN):
from aura.persistence import atomic_write_json

# Line 570 (CORRECT, same file):
from src.aura.persistence import atomic_write_json
```

The `save_model()` method uses the wrong import path (`aura.persistence` instead of `src.aura.persistence`). Since this is a lazy import inside the method, the error only surfaces when saving a trained model — it passes syntax checks and basic tests but crashes in production use.

**Fix**: Change line 615 to `from src.aura.persistence import atomic_write_json`.

---

### C-03: Import Path Bug in `override_predictor.py`
**File**: `src/aura/prediction/override_predictor.py` — line 557
**Impact**: `ImportError` crash at runtime

Same issue as C-02. The `save_model()` method uses `from aura.persistence import atomic_write_json` instead of the correct `from src.aura.persistence import atomic_write_json`.

**Fix**: Change line 557 to `from src.aura.persistence import atomic_write_json`.

---

### C-04: Non-Atomic Write in `rules_engine.py`
**File**: `src/aura/bridge/rules_engine.py` — `_save_rules()` method
**Impact**: Bridge rule corruption on crash

Uses `write_text()` directly instead of the project's `atomic_write_json()` utility. The rules engine manages TTL-based cross-engine gates — if the file is corrupted mid-write, both Aura and Buddy lose their behavioral constraints.

This directly violates `.claude/rules/improvement.md` JSON Safety Gates: "ALWAYS write JSON atomically: write to .tmp file first, then os.rename() to final path."

**Fix**: Replace `self._rules_path.write_text(json.dumps(...))` with `atomic_write_json(self._rules_path, data)`.

---

### C-05: Non-Atomic Write in All Pattern Tier `_save_patterns()` Methods
**Files**:
- `src/aura/patterns/tier1.py` — `_save_patterns()`
- `src/aura/patterns/tier2.py` — `_save_patterns()`
- `src/aura/patterns/tier3.py` — `_save_patterns()`
**Impact**: Pattern data corruption on crash

All three tier files use `write_text()` directly to persist detected patterns. A crash or power loss during the write corrupts the pattern history, losing weeks of accumulated detection data.

**Fix**: Use `atomic_write_json()` from `src.aura.persistence` in all three files.

---

### C-06: Silent Exception Suppression in `rule_promoter.py`
**File**: `src/aura/patterns/rule_promoter.py` — lines 213-214
**Impact**: Silent data loss, impossible to debug

```python
except Exception:
    pass  # No logging at all
```

The `_load_promoted_ids()` method catches all exceptions silently when parsing JSON from the promotion log. This directly violates `.claude/rules/improvement.md`: "NEVER use bare except: or except Exception: pass — always log the error."

If the promotion log is corrupted, pattern IDs are silently lost, potentially causing duplicate promotions or missed promotions.

**Fix**: Add `logger.warning("Failed to parse promotion log entry: %s", e)` and continue.

---

### C-07: Non-Atomic Writes in `cloud_fallback.py`
**File**: `src/aura/patterns/cloud_fallback.py`
**Impact**: Rate limit state and synthesis log corruption

Two separate non-atomic write locations:
1. **Lines 783-786**: `count_path.write_text()` for daily call count — no atomic write
2. **Lines 809-818**: `open(self._log_path, "a")` for synthesis log — no fcntl lock

Additionally, **lines 378-379** have a state consistency bug: `self._daily_call_count += 1` happens before `self._save_daily_count()`. If the save fails, in-memory count drifts from persisted count, causing rate limit bypass or false enforcement.

**Fix**: Use `atomic_write_json()` for count file; use `_locked_append()` pattern (with fcntl.LOCK_EX) for log file; wrap increment + save in try/except with rollback.

---

## HIGH Issues

### H-01: Missing Module Import in `override_extractor.py`
**File**: `src/aura/patterns/override_extractor.py` — lines 64-65
**Impact**: `ModuleNotFoundError` at runtime

```python
from src.recursive_intelligence.learner import LearningEntry
```

The module `src.recursive_intelligence.learner` does not exist in the P-90 codebase. This import is inside the `extract()` method (lazy), so it passes import-time checks but will crash when override extraction is actually invoked.

**Fix**: Either create the missing module, add a try/except with graceful fallback, or remove the dependency.

---

### H-02: Missing fcntl Lock in `persistence.py` Atomic Write
**File**: `src/aura/persistence.py` — lines 59-87
**Impact**: Race condition with concurrent access

The `_locked_atomic_write()` function name implies it uses fcntl locking, but the implementation only does temp-file + rename without acquiring any lock. While atomic rename prevents partial reads, it does not prevent two concurrent writers from creating conflicting temp files.

**Fix**: Acquire `fcntl.LOCK_EX` on a `.lock` file before writing, release after rename.

---

### H-03: `read_latest_signal()` Crash on Unexpected JSON Keys
**File**: `src/aura/core/readiness.py` — `read_latest_signal()` method
**Impact**: `TypeError` crash on deserialization

The method constructs `ReadinessSignal(components=components, **data)` where `data` comes from JSON with `components` already popped. If the JSON contains any unexpected key not in the `ReadinessSignal` dataclass, this raises `TypeError: __init__() got an unexpected keyword argument`. No schema validation is performed before deserialization.

**Fix**: Filter `data` to only known `ReadinessSignal` fields before passing as `**kwargs`, or wrap in try/except with a warning.

---

### H-04: Non-Atomic Append in `rule_promoter.py`
**File**: `src/aura/patterns/rule_promoter.py` — lines 196-197
**Impact**: Promotion audit trail corruption

Uses `open(self._log_path, "a")` without fcntl locking. If multiple promotion events fire concurrently (e.g., T1 and T2 both promoting patterns in the same engine run), the JSONL file can have interleaved writes.

**Fix**: Use the `_locked_append()` pattern from `signals.py` with `fcntl.LOCK_EX`.

---

## MEDIUM Issues

### M-01: Onboarding Profile Written Non-Atomically
**File**: `src/aura/cli/companion.py` — line 269
**Impact**: Corrupted onboarding profile if crash during write

```python
onboarding_path.write_text(json.dumps(onboarding_data, indent=2, default=str))
```

Uses direct `write_text()` instead of atomic write. If the process crashes mid-write, Buddy reads a truncated onboarding profile.

**Fix**: Use `atomic_write_json()`.

---

### M-02: Missing Error Handling in CLI `main.py` Bridge Status
**File**: `src/aura/cli/main.py` — around line 115
**Impact**: CLI crash on corrupted bridge files

`companion.bridge.get_bridge_status()` is called without try/except. If bridge signal files are corrupted, the CLI crashes instead of showing a degraded status.

**Fix**: Wrap in try/except with graceful "bridge status unavailable" message.

---

### M-03: Insufficient Error Context in Cloud Fallback Logging
**File**: `src/aura/patterns/cloud_fallback.py` — line 415
**Impact**: Hard to diagnose synthesis failures

```python
logger.warning("Cloud synthesis failed: %s", e)
```

Logs only the exception message, not the stack trace or input context. Per improvement.md: "ALWAYS include context in error logs: function name, input parameters, stack trace."

**Fix**: Use `logger.warning("Cloud synthesis failed: %s", e, exc_info=True)`.

---

### M-04: Missing JSON Schema Validation in `rule_promoter.py`
**File**: `src/aura/patterns/rule_promoter.py` — lines 201-215
**Impact**: Silent ignoring of malformed promotion log entries

`json.loads(line)` output is used without validating expected keys. `entry.get("pattern_id")` assumes the key might exist but doesn't warn if the entire entry structure is wrong.

**Fix**: Add schema validation and log warning for malformed entries.

---

### M-05: `_cmd_bridge()` Conditional Formatting Bug
**File**: `src/aura/cli/companion.py` — lines 681-684
**Impact**: Potential formatting error

```python
lines.append(f"Readiness → Buddy: {'✓' if r['available'] else '✗'} "
             f"(score: {r['score']:.0f}/100)" if r['available'] else
             f"Readiness → Buddy: not yet computed")
```

The ternary operator scope is ambiguous — Python's operator precedence means `lines.append(X if condition else Y)`, which is correct, but the multi-line formatting makes it easy to misread and hard to maintain. Additionally, if `r['score']` is missing when `r['available']` is True, this raises `KeyError`.

**Fix**: Restructure as explicit if/else block.

---

### M-06: `process_message()` Return Value Not Validated
**File**: `src/aura/cli/companion.py` — lines 343-349
**Impact**: Fallback `ConversationSignals()` may have unexpected defaults

When `process_message()` fails (Stage 1 fallback), the code creates a bare `ConversationSignals()`. If the default constructor doesn't initialize all fields needed by downstream stages (readiness computation, graph logging), those stages may fail with `AttributeError`.

**Fix**: Ensure `ConversationSignals()` default constructor is fully self-consistent, or use a dedicated "neutral fallback" factory method.

---

## LOW Issues

### L-01: Missing Type Hints on Callback Parameters
**Files**: `cloud_fallback.py` (line 355), various
**Impact**: Type checking tools cannot verify correct usage

Some function parameters accepting callbacks lack type annotations (e.g., `parse_fn` should be `Callable[[str], SynthesisResult]`).

---

### L-02: Unused `signals` Variable in Onboarding
**File**: `src/aura/cli/companion.py` — lines 181, 209, 236
**Impact**: Wasted computation, no functional bug

```python
signals = self.processor.process_message(goals_raw, role="user")
```

The `signals` return value is assigned but never used in the onboarding flow. The emotional processing still happens (side effects logged), but the returned signals object is discarded.

---

### L-03: `_cmd_patterns` Cloud Status Accesses Without Guard
**File**: `src/aura/cli/companion.py` — line 815
**Impact**: AttributeError if cloud module not initialized

```python
cloud_status = self.pattern_engine.cloud.get_status()
```

If `PatternEngine.cloud` is `None` (e.g., no cloud provider configured), this crashes. Should check `if self.pattern_engine.cloud:` first.

---

## Systemic Patterns

### Pattern 1: Non-Atomic Writes Are Pervasive
**7 locations** across 6 files use direct `write_text()` or `open()` for JSON persistence instead of the project's own `atomic_write_json()` utility. The utility exists in `src/aura/persistence.py` but most modules don't use it. This is the single most impactful class of bug in the codebase.

**Affected files**: signals.py, rules_engine.py, tier1.py, tier2.py, tier3.py, cloud_fallback.py, companion.py

**Recommendation**: Audit every `write_text()` and `open(..., "w")` call in the codebase and replace with `atomic_write_json()` or `atomic_write()`.

### Pattern 2: Inconsistent Import Paths (`src.aura` vs `aura`)
Two files use `from aura.persistence` (wrong) while the rest correctly use `from src.aura.persistence`. This likely stems from copy-paste during module development. A project-wide import linter rule would prevent recurrence.

### Pattern 3: Lock Naming Mismatch
Functions named `_locked_write()` and `_locked_atomic_write()` don't actually acquire locks. This is worse than not having locks — developers assume the locking is handled and don't add their own guards.

---

## Recommendations (Priority Order)

1. **Fix C-02 and C-03 immediately** — import path bugs will crash model save operations
2. **Add fcntl LOCK_EX to `_locked_write()`** — the bridge is the critical shared data path
3. **Replace all `write_text()` calls with `atomic_write_json()`** — 7 locations, high corruption risk
4. **Add JSON schema validation to signal deserialization** (H-03) — prevents crash on unexpected keys
5. **Install pytest and verify test suite passes** — unable to confirm test health in this scan
6. **Add a CI lint rule** to flag `write_text()` and bare `except: pass` patterns

---

## Environment Notes

- Python 3.x syntax checks: **PASS** (all files compile cleanly)
- flake8: not installed (could not run static analysis)
- pytest: not installed (test suite not executed)
- No previous bug reports found for comparison

---

*Report generated by automated Aura bug scan — 2026-03-24 13:13 UTC*
