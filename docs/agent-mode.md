# Agent Mode

Agent mode lets you evaluate any external program — a Python script, a compiled binary, or an entire AI agent — without touching the benchmark codebase. Pass `--agent_cli` instead of `--model` and the CLI handles the rest.

## CLI contract

Your agent will be called once per task with a single argument:

```
<agent_cli> \
    --instruction   <str>    # natural-language manipulation instruction
```

The benchmark creates an isolated temporary directory for each task, places `structure.cif` directly inside it, and sets that directory as the agent's working directory. The agent can read the input from `structure.cif` (relative to its cwd) and must produce the output file (`result.cif` by default) anywhere inside the working directory — including subdirectories. After the agent exits the benchmark searches recursively for the output file.

Stdout and stderr are captured to `logs/task_<N>_repeat_<M>.log` under the results folder.

In agent mode, inference also persists each produced CIF to `generated_cifs/task_<frame>_repeat_<repeat>.cif`; evaluation reads these files directly (file-based evaluation path) rather than re-parsing generated text.

## Running agent mode

```bash
# Sequential (default)
atomworld benchmark \
    --agent_cli "python examples/my_agent/run.py" \
    -f data/simple/ -a add_atom_action

# Parallel — run 8 agent subprocesses concurrently (-b controls concurrency in agent mode)
atomworld benchmark \
    --agent_cli "python examples/my_agent/run.py" \
    --timeout 120 \
    -b 8 \
    -f data/simple/ -a add_atom_action
```

!!! info "`-b` in agent mode"
    Sets the number of concurrent agent subprocesses (analogous to batch size in LLM mode). Every task runs in its own isolated temporary directory, so parallelism is safe.