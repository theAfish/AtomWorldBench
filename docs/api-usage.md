# AtomWorldBench API Usage & Agent Prompt

**Recommended Public Server URL:** `http://<your-server>:50001`

This page is both the API reference and a ready-to-use prompt template. If you are
setting up an AI agent to run the benchmark, jump to the
[Agent Prompt](#agent-prompt) section and copy it verbatim into your agent's
system prompt or context window.

If you start the benchmark server directly on a public machine, users and
agents can begin from the same URL:

```bash
curl "http://<your-server>:50001/access-info"
```

That endpoint returns the live machine-readable benchmark access information for
the running server.

To expose the benchmark publicly, start the server **once**:

```bash
atomworld serve \
    --host 0.0.0.0 \
    --port 50001 \
    --api-key <ADMIN_API_KEY> \
    --data-root data \
    --sessions-dir sessions
```

Then:

- Browser users can open `http://<your-server>:50001/`
- Agents can call `http://<your-server>:50001/access-info`
- Interactive API docs are at `http://<your-server>:50001/docs`

`--api-key` sets the bootstrap **admin** key (for `/admin/*` endpoints only). Regular
users do not need it — they self-register and receive their own key in one step.

---

## Get an API key

Benchmark requests require an `X-API-Key` header. Registration is self-service —
no administrator action is needed.

```bash
curl -X POST "$BASE_URL/auth/self-register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

The response includes a ready-to-use `api_key` (e.g. `awb-xxxxxxxxxxxxxxxx`). Store it
safely and treat it like a password. There is no second step.

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

    Save the returned "session_id" and the "tasks" list.
    Each task contains: task_id, action_prompt, input_cif, action_type,
    frame_index, repeat_index.

Step 3 — For each task (process one at a time):

    a. Read the task fields already in the list from Step 2:
         "action_prompt"  — the natural-language instruction you must follow
         "input_cif"      — the crystal structure in CIF format (your input)

    b. Generate a result CIF.
       Follow "action_prompt" and produce a modified CIF string.
       Do not use any information from other tasks.

    c. Submit your result.
       POST $BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID/submit
       Body (JSON):
         {
           "result_cif": "<your generated CIF string>",
           "elapsed_seconds": <float, optional>,
           "token_usage": {"prompt_tokens": <int>, "completion_tokens": <int>}
         }

       Only "result_cif" is required.

Step 4 — Trigger evaluation after all tasks are submitted.

    POST $BASE_URL/sessions/$SESSION_ID/evaluate

Step 5 — Retrieve your scores.

    GET $BASE_URL/sessions/$SESSION_ID/results

─── RULES ───────────────────────────────────────────────────────────────────

- Process one task at a time. Do not read ahead.
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
`input_cif` per task, and expects one `result_cif` back. There is no shared
state between tasks, so you can safely:

- Spawn a fresh agent context (empty conversation history) for each task.
- Run tasks in parallel using separate HTTP calls (same session, different
  `task_id` values).

The session is only a server-side container for grouping submissions and
running evaluation once at the end. From the agent's point of view, each task
is a standalone question-answer pair.

---

## Full API reference

### Authentication

Public endpoints (no key required): `POST /auth/self-register`, `GET /datasets`, `GET /access-info`, `GET /healthz`.

All benchmark endpoints require:

```http
X-API-Key: YOUR_KEY
```

### Self-register and get a key

```bash
curl -X POST "$BASE_URL/auth/self-register" \
    -H "Content-Type: application/json" \
    -d '{"username": "alice", "email": "alice@example.com", "organization": "WMML"}'
```

Returns `username`, `email`, `organization`, `api_key`, and `created_at`. The
`api_key` is immediately usable — no second step.

### List available datasets (public)

```bash
curl "$BASE_URL/datasets"
```

Returns a list of dataset names, action count, and task count for each. Use the
`name` field as the `dataset` value in `/benchmark`.

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

Returns `session_id` and the full `tasks` list (each with `task_id`,
`action_prompt`, `input_cif`, `action_type`, `frame_index`, `repeat_index`).
The ground-truth output is **never sent to the client** — evaluation runs server-side.

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

Runs evaluation over all submitted results. Call this once, after all tasks are
submitted.

### Get results

```bash
curl "$BASE_URL/sessions/$SESSION_ID/results" \
    -H "X-API-Key: $API_KEY"
```

Returns aggregate metrics and per-task scores. Requires `/evaluate` to have
been called first.

### Other session endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/sessions/{id}` | Session status and progress |
| `GET`  | `/sessions/{id}/tasks` | Paginated task list (no CIF content) |
| `GET`  | `/sessions/{id}/tasks/{tid}` | Task details + input CIF (alternative to /benchmark payload) |

---

## Admin-only operations

The following endpoints require the bootstrap admin key (`--api-key` passed at
server startup). These are **not part of the regular user workflow**.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/admin/users` | List all registered users |
| `GET`  | `/admin/keys` | List all issued API keys |
| `POST` | `/auth/register` | Register a user without issuing a key (admin use) |
| `POST` | `/auth/issue-key` | Manually issue a key for an existing user |
