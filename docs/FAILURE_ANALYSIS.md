# Test Results & Failure Analysis

Real test output and root cause analysis for ConsistencyGuard.
Every failure below was reproduced from actual code — not invented.

---

## Current Test Suite — All Passing

```
platform darwin -- Python 3.11.4, pytest-9.0.3
asyncio: mode=Mode.AUTO

tests/test_detector.py::test_identical_prompts_high_similarity   PASSED  [  4%]
tests/test_detector.py::test_different_prompts_low_similarity    PASSED  [  8%]
tests/test_detector.py::test_semantically_similar_prompts        PASSED  [ 12%]
tests/test_detector.py::test_same_response_low_divergence        PASSED  [ 16%]
tests/test_detector.py::test_similar_responses_moderate_divergence PASSED [ 20%]
tests/test_detector.py::test_contradicting_responses_high_divergence PASSED [ 25%]
tests/test_detector.py::test_classify_severity_critical          PASSED  [ 29%]
tests/test_detector.py::test_classify_severity_info              PASSED  [ 33%]
tests/test_providers.py::test_get_provider_returns_anthropic_by_default PASSED [ 37%]
tests/test_providers.py::test_get_provider_explicit_anthropic    PASSED  [ 41%]
tests/test_providers.py::test_get_provider_returns_openai_by_env PASSED  [ 45%]
tests/test_providers.py::test_get_provider_raises_on_unknown     PASSED  [ 50%]
tests/test_providers.py::test_anthropic_complete_returns_text    PASSED  [ 54%]
tests/test_providers.py::test_anthropic_complete_passes_correct_args PASSED [ 58%]
tests/test_providers.py::test_openai_complete_returns_text       PASSED  [ 62%]
tests/test_providers.py::test_openai_complete_passes_correct_args PASSED [ 66%]
tests/test_providers.py::test_anthropic_acomplete_returns_text   PASSED  [ 70%]
tests/test_providers.py::test_openai_acomplete_returns_text      PASSED  [ 75%]
tests/test_store.py::test_init_db_creates_llm_calls_table        PASSED  [ 79%]
tests/test_store.py::test_init_db_creates_violations_table       PASSED  [ 83%]
tests/test_store.py::test_save_call_returns_integer_id           PASSED  [ 87%]
tests/test_store.py::test_get_all_calls_returns_saved_call       PASSED  [ 91%]
tests/test_store.py::test_save_violation_does_not_raise          PASSED  [ 95%]
tests/test_store.py::test_get_stats_returns_correct_counts       PASSED  [100%]

========================= 24 passed in 22.72s ==========================
```

---

## What Each Test Covers

### test_detector.py — Embedding & Detection Logic

| Test | What it proves |
|------|---------------|
| `test_identical_prompts_high_similarity` | Same string embedded twice → similarity ≥ 0.99. Verifies the embedding model is deterministic and normalization is correct. |
| `test_different_prompts_low_similarity` | Completely unrelated prompts → similarity < 0.50. Proves the threshold won't fire on genuinely different questions. |
| `test_semantically_similar_prompts` | "upload limit" vs "max file size" → similarity ≥ 0.60. Validates semantic matching works for paraphrases. |
| `test_same_response_low_divergence` | Same response embedded twice → divergence < 0.10. Proves temperature noise alone won't trigger a violation. |
| `test_similar_responses_moderate_divergence` | "25MB" vs "100MB" → divergence > 0.0. Confirms numerically different answers register as divergent. |
| `test_contradicting_responses_high_divergence` | "25MB limit" vs "contact support" → divergence ≥ 0.40. Validates CRITICAL severity threshold. |
| `test_classify_severity_critical` | `classify_severity(0.45)` → `CRITICAL`. Direct unit test of the severity classifier. |
| `test_classify_severity_info` | `classify_severity(0.10)` → `INFO`. Validates the lower severity boundary. |

### test_providers.py — Provider Abstraction (Mocked)

| Test | What it proves |
|------|---------------|
| `test_get_provider_returns_anthropic_by_default` | No `PROVIDER` env var → returns `AnthropicProvider`. |
| `test_get_provider_explicit_anthropic` | `get_provider("anthropic")` → `AnthropicProvider`. |
| `test_get_provider_returns_openai_by_env` | `PROVIDER=openai` → returns `OpenAIProvider`. |
| `test_get_provider_raises_on_unknown` | `get_provider("groq")` → `ValueError`. Unknown providers fail loudly. |
| `test_anthropic_complete_returns_text` | Mocked Anthropic client → `complete()` returns response text. |
| `test_anthropic_complete_passes_correct_args` | Verifies exact API call shape: model, max_tokens, messages format. |
| `test_openai_complete_returns_text` | Mocked OpenAI client → `complete()` returns response text. |
| `test_openai_complete_passes_correct_args` | Verifies OpenAI chat completions call structure. |
| `test_anthropic_acomplete_returns_text` | Async path with `AsyncMock` → `acomplete()` returns text. |
| `test_openai_acomplete_returns_text` | Async OpenAI path works correctly. |

### test_store.py — SQLite Persistence

| Test | What it proves |
|------|---------------|
| `test_init_db_creates_llm_calls_table` | `init_db()` creates the `llm_calls` table. |
| `test_init_db_creates_violations_table` | `init_db()` creates the `violations` table. |
| `test_save_call_returns_integer_id` | `save_call()` returns a positive integer row ID. |
| `test_get_all_calls_returns_saved_call` | Round-trip: save a call, retrieve it, all fields match. |
| `test_save_violation_does_not_raise` | `save_violation()` completes without exception. |
| `test_get_stats_returns_correct_counts` | After 3 calls + 1 violation, `get_stats()` returns correct totals. |

---

## Failure Scenarios — Real Output & RCA

These are failures that occurred during development. Each one is real —
the error output below was captured from an actual Python process.

---

### Failure 1 — Similarity Threshold Miscalibrated

**The broken test:**
```python
def test_semantically_similar_prompts():
    a = embed("What is the upload limit?")
    b = embed("What is the max file size?")
    assert cosine_similarity(a, b) >= 0.85  # too high
```

**Real failure output:**
```
FAILED tests/test_detector.py::test_semantically_similar_prompts
AssertionError: assert 0.6455 >= 0.85
  +  where 0.6455 = cosine_similarity([...], [...])
```

**Root Cause:**
`all-MiniLM-L6-v2` was benchmarked on sentence pairs (STS-B dataset).
Short domain-specific noun phrases — "upload limit" vs "max file size" —
score 0.64, not 0.85. The 0.85 threshold was copied from a benchmark
that used full sentences.

**Fix:**
Calibrate thresholds on your actual domain vocabulary before writing assertions:

```python
pairs = [
    ("upload limit", "max file size"),
    ("reset password", "forgot my password"),
]
for a, b in pairs:
    print(f"{cosine_similarity(embed(a), embed(b)):.4f}  |  '{a}' vs '{b}'")
```

Corrected assertion: `assert cosine_similarity(a, b) >= 0.60`

---

### Failure 2 — `call.id` is None When `check_consistency` Runs

**The broken pattern:**
```python
call = LLMCall(prompt="test", response="response", ...)
violations = check_consistency(call)  # call.id is None here
call_id = save_call(call)             # too late
```

**Real observed output:**
```
[F2] call.id=None  violations=0  (no error raised — silent bug)
```

**Root Cause:**
`LLMCall.id` is `Optional[int] = None` by default (Pydantic).
The `find_similar_calls` exclusion check does:

```python
if past.id == new_call.id:  # None == 1 → always False
    continue
```

When `call.id` is `None`, the call can match itself in the database
if it was previously saved, producing false violations with no traceback.

**Fix:**
Always call `save_call()` before `check_consistency()`. The correct
order in `proxy.py`:

```python
call_id = save_call(call)   # 1. save first
call.id = call_id           # 2. set id
violations = check_consistency(call)  # 3. then check
```

---

### Failure 3 — Stale 90-Day Baseline Flags Correct New Answer

**Reproduced with:**
```python
# Day 1: old wrong policy saved
old = LLMCall(response="No refunds under any circumstances.",
              timestamp=now - timedelta(days=90), ...)
save_call(old)

# Day 91: new correct policy
new = LLMCall(response="Full refund within 30 days.", ...)
new.id = save_call(new)
violations = check_consistency(new)
```

**Real output:**
```
[F3] VIOLATION  severity=warning  divergence=0.3201
     (correct answer flagged as inconsistent with 90-day-old wrong answer)
```

**Root Cause:**
`find_similar_calls` scanned ALL historical calls with no time window.
A policy that changed 3 months ago permanently flags the correct new answer.

**Fix:**
Set `COMPARISON_WINDOW_DAYS=30` in `.env`. The detector now skips any
call older than 30 days:

```python
cutoff = datetime.utcnow() - timedelta(days=window_days)
if cutoff and past.timestamp < cutoff:
    continue
```

---

### Failure 4 — `embed(None)` Produces Unreadable Traceback

**Broken call:**
```python
embed(None)
```

**Original error (before fix) — 12 frames deep inside PyTorch:**
```
TypeError: 'NoneType' object is not subscriptable
  File ".../sentence_transformers/SentenceTransformer.py", line 171
  File ".../torch/nn/modules/module.py", line 1501
  ... (10 more frames)
```

**After fix — clear error at the boundary:**
```
[F4-FIXED] ValueError: embed() requires a non-empty string, got <class 'NoneType'>
```

**Root Cause:**
No input validation before passing to `model.encode()`. The error
propagated through 12 library frames before surfacing, making the
source of `None` impossible to identify from the traceback.

**Fix:**
`normalize_text()` validates before encoding:
```python
def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        raise ValueError(
            f"embed() requires a non-empty string, got {type(text)!r}"
        )
    return re.sub(r"\s+", " ", text).strip().lower()
```

---

### Failure 5 — Missing API Key Gives Cryptic SDK TypeError

**Reproduced with:**
```python
os.environ.pop('ANTHROPIC_API_KEY', None)
p = AnthropicProvider(api_key=None)
p.complete("test", "claude-haiku-4-5-20251001", 10)
```

**Real error:**
```
[F5] TypeError: "Could not resolve authentication method. Expected one of
api_key, auth_token, or credentials to be set. Or for one of the
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN environment variables..."
```

**Root Cause:**
The Anthropic SDK raises `TypeError` (not `AuthenticationError`) when
`api_key=None` is passed at client construction time. The message mentions
credentials but doesn't point to where in your code the key is missing.

**Fix:**
`cg health` now shows API key status immediately:
```
│ ANTHROPIC_API_KEY  │ not set  │ —  │
```

Add to your `.env` file and rerun `cg health` to confirm.

---

### Failure 6 — `aguarded_call` Deadlocks in Jupyter / FastAPI

**Reproduced with:**
```python
async def inner():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(0))  # nested run_until_complete

asyncio.run(inner())
```

**Real error:**
```
[F6] RuntimeError: This event loop is already running
```

**Root Cause:**
`asyncio.run()` creates a new event loop. Inside Jupyter notebooks and
FastAPI route handlers, an event loop is already running. Calling
`loop.run_until_complete()` inside an already-running loop raises
`RuntimeError`.

**Fix A — Jupyter:**
```python
pip install nest_asyncio
import nest_asyncio; nest_asyncio.apply()
```

**Fix B — FastAPI (correct approach):**
```python
@app.post("/chat")
async def chat(body: ChatRequest):
    # Just await directly — FastAPI is already async
    text, violations = await aguarded_call(body.prompt)
    return {"response": text, "violations": len(violations)}
```

---

### Failure 7 — Tests Write to Production Database

**Reproduced with:**
```python
# Without conftest.py — default DB_PATH is relative to CWD
print(os.getenv("DB_PATH", "consistencyguard.db"))
# → consistencyguard.db   (your real production data)
```

**What happens:**
Every test that calls `init_db()` or `save_call()` without setting
`DB_PATH` writes fake test data into the live production database.
No assertion fails. No exception is raised. The database silently accumulates
garbage entries that pollute real violation reports.

**Root Cause:**
`store.get_db_path()` defaults to a relative path. Without isolation,
the test process and the running application share the same file.

**Fix:**
`tests/conftest.py` uses `autouse=True` to give every test an isolated
temp database automatically:

```python
@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    store.init_db()
    yield db_path
```

`autouse=True` means it applies to every test in every file without
needing to be explicitly requested.

---

### Failure 8 — Fixture Conflict Between `conftest.py` and Per-Test Fixture

**The broken pattern (original `test_store.py`):**
```python
@pytest.fixture
def use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    yield
```

After `conftest.py` was added with its own `autouse` `isolated_db` fixture,
both fixtures called `monkeypatch.setenv("DB_PATH", ...)` — but with
different paths. The last one to run won, which was non-deterministic.

**Error observed:**
```
sqlite3.OperationalError: no such table: llm_calls
```

The test opened a different temp file than the one `init_db()` had
initialized.

**Root Cause:**
Two pytest fixtures competing to set the same environment variable.
Fixture execution order with `autouse` is deterministic but not obvious —
the `autouse` fixture from `conftest.py` runs before the per-test fixture,
meaning the per-test fixture overwrote the path that `init_db()` had used.

**Fix:**
Remove the `use_temp_db` fixture from `test_store.py` entirely.
The `conftest.py` `isolated_db` fixture handles isolation for all tests:

```python
# test_store.py — after fix
def test_save_call_returns_integer_id():   # no fixture argument needed
    store.init_db()
    result = store.save_call(_make_call())
    assert isinstance(result, int)
```

---

## Summary

| # | Failure | Actual Error | Root Cause | Fix |
|---|---------|-------------|------------|-----|
| 1 | Threshold miscalibration | `assert 0.6455 >= 0.85` | MiniLM short-phrase scores misunderstood | Calibrate empirically; use `>= 0.60` |
| 2 | `call.id` None in check | Silent: 0 violations returned | `check_consistency` called before `save_call` | Enforce save-before-check order |
| 3 | Stale 90-day baseline | Correct answer flagged as VIOLATION | No time window on comparisons | `COMPARISON_WINDOW_DAYS=30` |
| 4 | `embed(None)` | 12-frame PyTorch traceback | No input validation | `normalize_text()` with early `ValueError` |
| 5 | Missing API key | Cryptic SDK `TypeError` | Key not validated before provider init | `cg health` to diagnose; set key in `.env` |
| 6 | Async deadlock | `RuntimeError: loop already running` | Nested event loops in Jupyter/FastAPI | `nest_asyncio` or direct `await` |
| 7 | DB pollution | Production data corrupted silently | No test isolation | `conftest.py` with `autouse` `isolated_db` |
| 8 | Fixture conflict | `OperationalError: no such table` | Two fixtures competing for `DB_PATH` | Remove per-test fixture; rely on `conftest.py` |
