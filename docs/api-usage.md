# Remote Benchmark Usage

Use AtomWorldBench through a hosted base URL. The client only needs to fetch tasks, submit result CIFs, and read scores. Session storage, evaluation, and filesystem handling stay on the server.

## What you need

- Base URL, for example `https://your-benchmark-host`
- API key

Send this header on every request:

```http
X-API-Key: YOUR_SECRET_KEY
```

## Workflow

### 1. Create a session

```bash
curl -X POST "$BASE_URL/sessions" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"action_name": null, "limit": -1, "repeat": 1}'
```

This returns a `session_id`.

### 2. Get the task list

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks?offset=0&limit=500" \
    -H "X-API-Key: $API_KEY"
```

Each task item includes a `task_id`.

### 3. Get one task

```bash
curl "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID" \
    -H "X-API-Key: $API_KEY"
```

The response includes:

- `action_prompt`
- `input_cif`

### 4. Submit your result CIF

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/tasks/$TASK_ID/submit" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "result_cif": "YOUR_GENERATED_CIF",
        "elapsed_seconds": 1.23,
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 240
        }
    }'
```

Only `result_cif` is required.

### 5. Run evaluation

```bash
curl -X POST "$BASE_URL/sessions/$SESSION_ID/evaluate" \
    -H "X-API-Key: $API_KEY"
```

### 6. Get results

```bash
curl "$BASE_URL/sessions/$SESSION_ID/results" \
    -H "X-API-Key: $API_KEY"
```

## Minimal client logic

1. Create one session.
2. List tasks in that session.
3. For each task, read `action_prompt` and `input_cif`.
4. Generate one output CIF.
5. Submit one output per task.
6. Call evaluation once all intended submissions are done.
7. Read final metrics from the results endpoint.

## Notes

- `action_name` may be omitted or set to `null` to include all actions.
- `limit = -1` means all tasks.
- Task listing supports `offset` and `limit`.
- Call `/evaluate` before requesting `/results`.

## Error codes

- `400` bad request
- `401` missing or invalid API key
- `404` session or task not found
- `500` server or evaluation error
