# AtomWorldBench API Usage & Agent Prompt

**Base URL (dev server):** `http://wmml1463885.bohrium.tech:50001`

This page is both the API reference and a ready-to-use prompt template. If you are
setting up an AI agent to run the benchmark, jump to the
[Agent Prompt](#agent-prompt) section and copy it verbatim into your agent's
system prompt or context window.

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

## Agent Prompt

Copy the block below into your agent's system prompt or prepend it to the first
user message. Replace `<YOUR_API_KEY>` with the key you received.

```
You are an AI agent running the AtomWorldBench benchmark.

BASE_URL = "http://wmml1463885.bohrium.tech:50001"
API_KEY  = "<YOUR_API_KEY>"

Always send the header:
    X-API-Key: <YOUR_API_KEY>

─── WORKFLOW ────────────────────────────────────────────────────────────────

Step 1 — Create a session (do this once).

    POST $BASE_URL/sessions
    Body (JSON): {"action_name": null, "limit": -1, "repeat": 1}

    Save the returned "session_id".

Step 2 — List all task IDs.

    GET $BASE_URL/sessions/$SESSION_ID/tasks?offset=0&limit=500

    Save the list of "task_id" values.

Step 3 — For each task_id (process one at a time):

    a. Fetch the task.
       GET $BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID

       The response contains:
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

All endpoints require:

```http
X-API-Key: YOUR_KEY
```

### Create a session

```bash
curl -X POST "$BASE_URL/sessions" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"action_name": null, "limit": -1, "repeat": 1}'
```

Parameters:

| Field | Default | Meaning |
|---|---|---|
| `action_name` | `null` | Filter to one action type; `null` = all actions |
| `limit` | `-1` | Maximum number of tasks; `-1` = all |
| `repeat` | `1` | How many times each task is repeated |

Returns a `session_id`.

### List tasks

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks?offset=0&limit=500" \
    -H "X-API-Key: $API_KEY"
```

Returns a list of task objects (no CIF content). Supports `offset` and `limit`
for pagination.

### Get one task

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID" \
    -H "X-API-Key: $API_KEY"
```

Returns `action_prompt` and `input_cif` for the given task.

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
