# AtomWorldBench API Usage

**Recommended Public Server URL:** `http://<your-server>:50001`

This page describes how to run the benchmark via the REST API — follow the
workflow below to create a session, fetch tasks, submit results, and retrieve
your scores.

If you start the benchmark server directly on a public machine, users and
agents can begin from the same URL:

```bash
curl "http://<your-server>:50001/access-info"
```

That endpoint returns the live machine-readable benchmark access information for
the running server.

To expose the benchmark publicly, start the server with:

```bash
atomworld serve \
    --host 0.0.0.0 \
    --port 50001 \
    --api-key <ADMIN_API_KEY> \
    --data-folder data/simple \
    --sessions-dir sessions
```

Then:

- Browser users can open `http://<your-server>:50001/`
- Agents can call `http://<your-server>:50001/access-info`
- Interactive API docs are at `http://<your-server>:50001/docs`

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


