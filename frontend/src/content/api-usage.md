# AtomWorldBench API Usage

**Base URL (dev server):** `http://wmml1463885.bohrium.tech:50001`

This page describes how to run the benchmark via the REST API — follow the
workflow below to create a session, fetch tasks, submit results, and retrieve
your scores.

---

## Get an API key

Every request requires an `X-API-Key` header. Keys are issued per user — the
server does not have a self-registration endpoint yet.

To get a key:

1. Open an issue or send a request to the maintainers (see the README for
   contact details).
2. You will receive a key string such as `awb-xxxxxxxxxxxxxxxx`.
3. Store it safely; treat it like a password.

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


