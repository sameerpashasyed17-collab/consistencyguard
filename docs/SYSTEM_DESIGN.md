# ConsistencyGuard — System Design

---

## Problem Statement

LLM agents in production can return different answers to semantically identical
prompts across time. No HTTP error is raised. No alert fires. Existing
observability tools record what was said but never compare it against prior
responses to the same question. ConsistencyGuard is purpose-built to detect
this specific class of silent failure.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Application                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ guarded_call(prompt, agent_id)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Proxy Layer                          │
│                       proxy.py                               │
│                                                             │
│  1. normalize_text(prompt)                                  │
│  2. embed(prompt)          ──► Embedder (local MiniLM)      │
│  3. save_call(call)        ──► SQLite Store                 │
│  4. check_consistency(call)──► Detector                     │
│  5. call LLM provider      ──► Anthropic / OpenAI           │
│  6. fire_webhook(violation)──► Webhook (optional)           │
│  7. return (text, violations)                               │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    Embedder        SQLite         Detector        Webhook
   embedder.py      store.py      detector.py    webhooks.py
```

---

## Component Design

### 1. Embedder (`embedder.py`)

**Responsibility:** Convert a text string into a fixed-length vector.

**Model:** `all-MiniLM-L6-v2` via `sentence-transformers`
- 384-dimensional output vector
- Runs entirely on CPU — no GPU required
- ~10ms per embedding on modern hardware
- Apache 2.0 license — no usage restrictions

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Local model, not API | Zero cost per call, no prompt data leaves the machine |
| MiniLM over larger models | Fast enough for real-time use; larger models add latency without meaningful accuracy gain for short domain prompts |
| Lazy singleton | Model loads once on first call, not at import time — avoids startup cost if embedding is never used |
| `normalize_text()` before encoding | Collapses whitespace, lowercases — prevents tokenization artifacts from inflating divergence on identical semantic content |

**Bottleneck:** Single-threaded CPU encoding. At high concurrency, this becomes
a queue. Mitigation: `aguarded_call()` wraps embedding in `run_in_executor` to
avoid blocking the event loop.

---

### 2. Store (`store.py`)

**Responsibility:** Persist LLM calls, embeddings, and violations. Answer
similarity queries.

**Technology:** SQLite (single file, zero ops)

**Schema:**

```sql
CREATE TABLE llm_calls (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt    TEXT NOT NULL,
    response  TEXT NOT NULL,
    model     TEXT,
    agent_id  TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding TEXT NOT NULL   -- JSON-serialized float list
);

CREATE TABLE violations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id_new         INTEGER,
    call_id_ref         INTEGER,
    severity            TEXT,
    prompt_similarity   REAL,
    response_divergence REAL,
    explanation         TEXT,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Similarity scan:** O(n) over all stored embeddings. Acceptable up to ~50,000
calls. Beyond that, latency grows linearly — see Scaling section.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| SQLite over a vector DB | Zero infrastructure, one file, no ops overhead. Sufficient for the problem at this scale. |
| JSON-serialized embeddings | Avoids binary blob complexity; NumPy deserialization is fast |
| `COMPARISON_WINDOW_DAYS` | Prevents a correct answer updated 90 days ago from flagging today's correct answer as inconsistent |
| `agent_id` column | Keeps agent-specific call history isolated for per-agent reporting |

---

### 3. Detector (`detector.py`)

**Responsibility:** Given a new LLM call, find past calls with similar prompts
and check whether the responses have diverged materially.

**Algorithm:**

```
Step 1 — Find similar prompts
  For each past call in the window:
    similarity = cosine_similarity(new_embedding, past_embedding)
    if similarity >= SIMILARITY_THRESHOLD (default 0.92):
      candidate for comparison

Step 2 — Measure response divergence
  For each candidate:
    new_resp_embedding  = embed(new_response)
    past_resp_embedding = embed(past_response)
    divergence = 1.0 - cosine_similarity(new_resp_emb, past_resp_emb)

Step 3 — Classify severity
  if divergence >= 0.40  → CRITICAL
  if divergence >= 0.25  → WARNING
  if divergence >= 0.10  → INFO
  else                   → no violation
```

**Why cosine similarity over other distance metrics:**
- Scale-invariant — response length does not affect the score
- Works directly on normalized sentence embeddings
- Standard for semantic similarity tasks

**Threshold calibration:**
The default thresholds (0.92 prompt similarity, 0.25 divergence) were
calibrated empirically using `all-MiniLM-L6-v2` on domain-specific short
phrases. Key observation: short noun phrases score lower than full sentences
on this model. See `docs/FAILURE_ANALYSIS.md` Failure 1 for details.

---

### 4. Proxy (`proxy.py`)

**Responsibility:** The single entry point that orchestrates all components.
Wraps an LLM call with consistency checking.

**Sync flow:**

```python
def guarded_call(prompt, agent_id, provider):
    embedding = embed(prompt)                    # ~10ms local
    call = LLMCall(prompt, embedding, agent_id)
    call_id = save_call(call)                    # write to SQLite
    call.id = call_id
    violations = check_consistency(call)         # similarity scan
    for v in violations:
        save_violation(v)
        fire_webhook(v)                          # optional, non-blocking
    response = provider.complete(prompt, model)  # actual LLM call
    return response, violations
```

**Critical ordering:** `save_call()` must run before `check_consistency()`.
The detector excludes the current call from its own similarity scan using
`call.id`. If `call.id` is `None` (not yet saved), the call can match itself
— producing a false violation. See `docs/FAILURE_ANALYSIS.md` Failure 2.

**Async variant (`aguarded_call`):**
- Embedding runs in `loop.run_in_executor()` — offloads CPU work from the event loop
- LLM call uses `await provider.acomplete()` — truly non-blocking
- Webhook uses `await afire_webhook()` — non-blocking alert dispatch

---

### 5. Provider Abstraction (`providers.py`)

**Responsibility:** Decouple the proxy from any specific LLM API.

**Interface:**

```python
class BaseProvider:
    def complete(self, prompt: str, model: str, max_tokens: int) -> str: ...
    async def acomplete(self, prompt: str, model: str, max_tokens: int) -> str: ...
```

**Implementations:** `AnthropicProvider`, `OpenAIProvider`

**Factory:** `get_provider(name)` reads the `PROVIDER` env var. Adding a new
provider requires implementing the two methods and registering the class in
`get_provider()` — no changes anywhere else in the codebase.

**Retry:** Both providers use `tenacity` with exponential backoff (3 attempts,
1s initial wait, 2× multiplier). Applied at the provider level, not the proxy
level — the proxy sees a clean success or a final failure.

---

### 6. Webhooks (`webhooks.py`)

**Responsibility:** POST a JSON payload to a configured URL on every violation.

**Design:** Fire-and-forget. Both sync (`fire_webhook`) and async
(`afire_webhook`) variants catch all exceptions silently — a webhook failure
never surfaces to the caller. A monitoring side-channel must not break the
primary call path.

**Payload:**

```json
{
  "event": "consistency_violation",
  "severity": "critical",
  "agent_id": "support-bot",
  "new_prompt": "...",
  "new_response": "...",
  "ref_response": "...",
  "prompt_similarity": 1.0,
  "response_divergence": 0.51,
  "explanation": "...",
  "timestamp": "2026-05-21T10:00:00"
}
```

---

## Data Flow — End to End

```
App calls guarded_call("What is the upload limit?", agent_id="support-bot")
  │
  ├─ normalize_text()         → "what is the upload limit?"
  ├─ embed()                  → [0.12, -0.34, 0.08, ...] (384 dims)
  ├─ save_call()              → row id=42 written to llm_calls
  ├─ check_consistency(id=42)
  │     ├─ scan llm_calls     → finds id=7 (similarity=1.00, same prompt)
  │     ├─ embed(response_7)  → vector A
  │     ├─ embed(response_42) → vector B
  │     ├─ divergence         → 1.0 - cosine(A, B) = 0.51
  │     └─ severity           → CRITICAL
  ├─ save_violation()         → written to violations table
  ├─ fire_webhook()           → POST to WEBHOOK_URL (if set)
  └─ provider.complete()      → calls Anthropic/OpenAI API
        └─ returns text to caller alongside [ConsistencyViolation(...)]
```

---

## Scaling Analysis

### Current design — SQLite, O(n) scan

| Calls in DB | Scan time | Verdict |
|---|---|---|
| 1,000 | < 1ms | Fine |
| 10,000 | ~5ms | Fine |
| 50,000 | ~20ms | Acceptable |
| 200,000 | ~80ms | Noticeable |
| 1,000,000 | ~400ms | Too slow |

### When to scale beyond SQLite

Trigger point: **>100k calls** or **>50 concurrent writers**.

**Option A — aiosqlite**
Async SQLite writes. Removes the single-writer bottleneck for async workloads.
No schema change required.

**Option B — pgvector (PostgreSQL)**
Replace the JSON embedding scan with a native ANN index. Scales to millions
of calls with sub-millisecond similarity search. Requires migrating the schema
and re-embedding historical data.

**Option C — Dedicated vector DB (Qdrant, Weaviate)**
Purpose-built for embedding search at scale. Adds operational complexity and
a new infrastructure dependency. Warranted only at very high call volumes.

### Concurrency model

| Scenario | Current behaviour | Fix |
|---|---|---|
| Single async app (FastAPI) | Works — use `aguarded_call` | No change needed |
| Multi-process workers | SQLite write lock contention | Migrate to PostgreSQL |
| Jupyter notebook | `asyncio.run()` conflict | `pip install nest_asyncio` |

---

## Security Considerations

| Risk | Current State | Mitigation |
|---|---|---|
| Prompts stored as plaintext | SQLite file on local disk | Pre-processing redaction hook (planned) |
| No authentication on DB | Local file, OS-level access control | Sufficient for single-tenant local use |
| Webhook endpoint receives violation data | HTTPS POST only | Validate `WEBHOOK_URL` is HTTPS in production |
| API keys in environment | `.env` file, gitignored | Never commit `.env`; use secrets manager in CI |

---

## Key Design Trade-offs

| Trade-off | Decision | Reasoning |
|---|---|---|
| Local embeddings vs API embeddings | Local | Zero cost, zero latency variance, data stays on machine |
| SQLite vs vector DB | SQLite | Zero ops, sufficient at target scale, trivially portable |
| Sync + async vs async-only | Both | Sync covers scripts/notebooks; async covers FastAPI/production |
| Silent webhook failure vs propagating | Silent | Monitoring must never break the primary call path |
| O(n) scan vs ANN index | O(n) | Fast enough to 50k calls; ANN adds complexity without benefit at this scale |
| Full response buffering vs streaming | Buffer | Streaming responses cannot be embedded until complete |

---

## Known Limitations and Roadmap

| Limitation | Impact | Planned Fix |
|---|---|---|
| No PII scrubbing | Prompts stored as plaintext | Pre-processing redaction hook |
| SQLite single-writer | Bottleneck at high concurrency | pgvector / aiosqlite |
| No multi-tenancy | All agents share one DB | `tenant_id` column + row filtering |
| No streaming support | Buffers full response before checking | Async tail check |
| No prompt template support | Variable values affect similarity | Template registry |
| Embedding model drift | Re-embedding needed after model upgrade | Migration tooling |
