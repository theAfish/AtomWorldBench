# AtomWorldBench API Usage

**Recommended Public Server URL:** `http://<your-server>:50001`

This page is both the API reference and a ready-to-use prompt template. If you
are setting up an AI agent to run the benchmark, jump to the
[Agent Prompt](#agent-prompt) section and copy it verbatim into your agent's
system prompt or context window.

If you start the benchmark server on a public machine, users and agents access
the same URL:

```bash
curl "http://<your-server>:50001/access-info"
```

To expose the benchmark publicly, start the server with:

```bash
atomworld serve \
    --host 0.0.0.0 \
    --port 50001 \
    --api-key <ADMIN_API_KEY> \
    --data-root data \
    --sessions-dir sessions
```

The `--data-root` flag points to the **parent folder** that contains dataset
sub-directories (`data/simple/`, `data/verbose/`, and any future ones). Users
and agents choose the dataset when they create a session — the server is not
locked to one dataset.

Then:

- Browser users and agents both open `http://<your-server>:50001/`
- Machine-readable access info: `http://<your-server>:50001/access-info`
- Interactive API docs: `http://<your-server>:50001/docs`

---

## Get an API key (self-service)

Register yourself and receive an API key in **one step** — no admin involvement
required:

```bash
curl -X POST "$BASE_URL/auth/self-register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

The response includes your `api_key` (e.g. `awb-xxxxxxxxxxxxxxxx`). Store it
safely; it will not be shown again.

You can also register through the web UI by clicking **Get API Key** in the
navigation bar.

### Admin-issued keys (optional)

If you prefer the two-step admin-approval flow, register first:

```bash
curl -X POST "$BASE_URL/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

Then an administrator issues a key with the bootstrap admin key:

```bash
curl -X POST "$BASE_URL/auth/issue-key" \
    -H "X-API-Key: $ADMIN_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "note": "benchmark access"}'
```

---

## Agent Prompt

Copy the block below into your agent's system prompt or prepend it to the first
user message. Replace `<YOUR_API_KEY>` with the key you received.

```
You are an AI agent running the AtomWorldBench benchmark.

BASE_URL = "http://<your-server>:50001"
API_KEY  = "<YOUR_API_KEY>"

Always send the header:
    X-API-Key: <YOUR_API_KEY>

─── SIMPLIFIED WORKFLOW ─────────────────────────────────────────────────────

Step 1 — Discover available datasets.

    GET $BASE_URL/datasets

    Returns a list of dataset names (e.g. "simple", "verbose").
    Pick the one you want and use it as the "dataset" field below.

Step 2 — Get all tasks in one call.

    POST $BASE_URL/benchmark
    Body (JSON): {"dataset": "simple", "action_name": null, "limit": -1}

    Save the returned "session_id" and the "tasks" array.
    Each task contains:
      "task_id"       — unique identifier
      "action_prompt" — natural-language instruction you must follow
      "input_cif"     — crystal structure in CIF format (your input)

Step 3 — For each task, generate a result CIF.

    Follow "action_prompt" using only "input_cif" as input.
    Do not use information from other tasks.

Step 4 — Submit your result for each task.

    POST $BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID/submit
    Body (JSON):
      {
        "result_cif": "<your generated CIF string>",
        "elapsed_seconds": <float, optional>,
        "token_usage": {"prompt_tokens": <int>, "completion_tokens": <int>}
      }

    Only "result_cif" is required.

Step 5 — Trigger evaluation after all tasks are submitted.

    POST $BASE_URL/sessions/$SESSION_ID/evaluate

Step 6 — Retrieve your scores.

    GET $BASE_URL/sessions/$SESSION_ID/results

─── RULES ───────────────────────────────────────────────────────────────────

- Your only inputs per task are "action_prompt" and "input_cif".
- Submit exactly one result CIF per task.
- Do not call /evaluate until all submissions are done.

─── ERROR CODES ─────────────────────────────────────────────────────────────

400  bad request body
401  missing or invalid API key
404  session or task not found
500  server or evaluation error
```

---

## Per-task isolation

Each task is independent. The server gives you one `action_prompt` and one
`input_cif` per task and expects one `result_cif` back. You can safely run
tasks in parallel using separate HTTP calls (same `session_id`, different
`task_id` values).

---

## Full API reference

### Authentication

All benchmark endpoints require:

```http
X-API-Key: YOUR_KEY
```

`POST /auth/self-register` and `POST /auth/register` do not require
authentication. `POST /auth/issue-key` requires the bootstrap admin key.

### Self-register (get key immediately)

```bash
curl -X POST "$BASE_URL/auth/self-register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

Returns `api_key` immediately.

### List available datasets (public)

```bash
curl "$BASE_URL/datasets"
```

Returns dataset names, action count, and task count for each. No API key required.

### One-shot benchmark (recommended)

```bash
curl -X POST "$BASE_URL/benchmark" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"dataset": "simple", "action_name": null, "limit": -1}'
```

Parameters:

| Field | Default | Meaning |
|---|---|---|
| `dataset` | `"simple"` | Dataset to use — see `GET /datasets` for available names |
| `action_name` | `null` | Filter to one action type; `null` = all actions |
| `limit` | `-1` | Maximum number of tasks; `-1` = all |

Returns `session_id` and `tasks` (array of all tasks including CIFs).

### Submit a result

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID/submit" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "result_cif": "...",
        "elapsed_seconds": 1.23,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 240}
    }'
```

Only `result_cif` is required.

### Evaluate

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/evaluate" \
    -H "X-API-Key: $API_KEY"
```

Runs evaluation over all submitted results. Call this once after all tasks are
submitted.

### Get results

```bash
curl "$BASE_URL/sessions/$SESSION_ID/results" \
    -H "X-API-Key: $API_KEY"
```

Returns aggregate metrics and per-task scores. Requires `/evaluate` to have
been called first.

---

## Manual session workflow (alternative)

Use this if you prefer to fetch tasks individually or need more control.

### Create a session

```bash
curl -X POST "$BASE_URL/sessions" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"action_name": null, "limit": -1, "repeat": 1}'
```

Returns a `session_id`.

### List tasks

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks?offset=0&limit=500" \
    -H "X-API-Key: $API_KEY"
```

### Get one task

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID" \
    -H "X-API-Key: $API_KEY"
```

Returns `action_prompt` and `input_cif` for the given task.

---

## Error codes

| Code | Meaning |
|---|---|
| 400 | Bad request body |
| 401 | Missing or invalid API key |
| 404 | Session or task not found |
| 500 | Server or evaluation error |


---

## Get an API key

Benchmark requests require an `X-API-Key` header. The server supports a
two-step flow: users register themselves, then an administrator issues a key.

Register yourself:

```bash
curl -X POST "$BASE_URL/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

Then issue a key for that user with the bootstrap admin key:

```bash
curl -X POST "$BASE_URL/auth/issue-key" \
    -H "X-API-Key: $ADMIN_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "note": "benchmark access"}'
```

The response includes a key string such as `awb-xxxxxxxxxxxxxxxx`. Store it
safely and treat it like a password.

If you self-host the server, `ADMIN_API_KEY` is the `--api-key` value passed to
`atomworld serve`. That bootstrap key also remains valid for benchmark calls.

---

## Workflow

### Step 1 — Create a session (once per run)

```bash
curl -X POST "$BASE_URL/sessions" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"action_name": null, "limit": -1, "repeat": 1}'
```

Save the returned `session_id` — every subsequent call uses it.

| Field | Default | Meaning |
|---|---|---|
| `action_name` | `null` | Filter to one action type; `null` = all actions |
| `limit` | `-1` | Maximum number of tasks; `-1` = all |
| `repeat` | `1` | How many times each task is repeated |

### Step 2 — List all task IDs

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks?offset=0&limit=500" \
    -H "X-API-Key: $API_KEY"
```

Returns a list of task objects (no CIF content yet). Save the `task_id` values.

### Step 3 — Process each task

For every `task_id`:

**a. Fetch the task**

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID" \
    -H "X-API-Key: $API_KEY"
```

The response contains:

- `action_prompt` — the natural-language instruction to follow
- `input_cif` — the crystal structure in CIF format (your input)

**b. Generate a result CIF**

Follow `action_prompt` and produce a modified CIF string. Use only the
information from this task — do not carry state across tasks.

**c. Submit your result**

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID/submit" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "result_cif": "...",
        "elapsed_seconds": 1.23,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 240}
    }'
```

Only `result_cif` is required. `elapsed_seconds` and `token_usage` are optional
metadata.

### Step 4 — Trigger evaluation

After all tasks are submitted, run evaluation once:

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/evaluate" \
    -H "X-API-Key: $API_KEY"
```

### Step 5 — Retrieve your scores

```bash
curl "$BASE_URL/sessions/$SESSION_ID/results" \
    -H "X-API-Key: $API_KEY"
```

Returns aggregate metrics and per-task scores.

---

## Rules

- Submit exactly one result CIF per task.
- Do not call `/evaluate` until all submissions are complete.
- Tasks are independent — there is no shared state between them.

## Error codes

| Code | Meaning |
|---|---|
| 400 | Bad request body |
| 401 | Missing or invalid API key |
| 404 | Session or task not found |
| 500 | Server or evaluation error |


