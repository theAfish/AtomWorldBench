# TODO.md — AtomWorld Benchmark

## Project Overview

AtomWorld evaluates LLMs and agents on **crystal structure manipulation** tasks.
The benchmark has two action families and two evaluation modes:

**Action families** (= "task types"):
- `simple` — index-based actions (e.g. `AddAtomAction`, `SwapAtomsAction`)
- `verbose` — motif-centric actions (e.g. `AddMotifAction`, `ChangeElementAction`)
- `active` — multi-step / agentic tasks (planned, not yet populated)

**Evaluation modes:**
- `llm` — prompt LLM with CIF + instruction, parse CIF from response
- `agent` — pass workspace files to an external agent CLI, read `result.cif`

---

## Current State (already implemented)

### CLI
```
atomworld generate    # generate HDF5 datasets from raw CIF files
atomworld benchmark   # run inference + evaluation (LLM mode)
atomworld eval        # evaluation only (skip inference)
atomworld draw        # plot metrics distributions
```

### Data Format (v2 — canonical)
Tasks are stored as **JSON files**, one file per action class:
```
data/
  simple/
    AddAtomAction.json
    RemoveAtomAction.json
    SwapAtomsAction.json
    ... (one per simple action)
  verbose/          ← to be populated (Phase 1)
  active/           ← to be populated (Phase 4)
```
Each file is a JSON array. Record schema:
```json
{
  "action_type":   "AddAtomAction",
  "problem_id":    "<uuid>",
  "mp_id":         "mp-1013842",
  "action_prompt": "Add one Ba atom at ...",
  "input":         "<CIF string>",
  "output":        "<CIF string>"
}
```
`input`/`output` are the raw CIF strings. The dataloader maps them to the
internal column names `input_cif`/`output_cif` transparently.

**Legacy format (v1):** `src/data/<action>.hdf5` + `.csv` files are still
recognised by the dataloader for backwards compatibility but are no longer
the canonical source.

### LLM Mode Pipeline (working)
```
data/simple/<ActionName>.json
  → load_data() in src/utils/dataloader.py  (auto-detects JSON v2)
  → AtomWorldInferencer (src/benchmark/inference/inferencer.py)
  → LLM API  (src/config/models.yaml configures model keys)
  → AtomWorldEvaluator (src/benchmark/evaluation/atomworld_evaluator.py)
```

### Results Structure
```
results/AtomWorld/<category>/<action_name>/<model>/<timestamp>/
  evaluation_results.csv   # per-task results
  evaluation_wrongs.csv    # failed tasks only
  metrics.json             # aggregate stats
  logs/                    # inference logs
```

### Evaluation Metrics (implemented in `src/benchmark/evaluation/metrics.py`)
- `correct` — structure matches within tolerance
- `max_dist` — max per-atom displacement
- `rmsd` — root mean square displacement
- `wrong_type`: `OutputFormatError`, `CIFParsingError`, `AtomCountMismatch`

### Standalone Evaluate (for use as RL reward / judge)
```python
from atomworld import evaluate
result = evaluate(target_cif=..., generated_output=...)
# result.correct, result.rmsd, result.max_dist
```

### Existing Tests
```
src/tests/
  test_actions.py
  test_cif_action_generator.py
  test_dataloader_action_resolution.py
  test_extract_data.py
  test_benchmark_config.py
  test_call_api.py
  test_load_model.py
  test_parser.py
```

---

## Phase 1 — Verbose Dataset Generation

Verbose actions (`AddMotifAction`, `ChangeElementAction`, etc.) exist in
`src/atomworld/actions/verbose/` but `data/verbose/` is empty — no JSON files
have been generated yet.

### 1.1 Verify `atomworld generate` writes JSON v2
Check that `CIFActionGenerator` / `main()` in `src/data_generation/cif_action_generator.py`
outputs files matching the v2 schema (`action_type`, `problem_id`, `mp_id`,
`action_prompt`, `input`, `output`) and writes them to `data/verbose/`.
Update the generator's output path / schema if needed.

### 1.2 Generate verbose action datasets
```
atomworld generate \
  --cif_folder <raw_cif_dir> \
  --actions AddMotifAction RemoveMotifAction ... \
  --output_folder data/verbose/ \
  --num_samples N
```

Available verbose actions (from `src/data_generation/cif_action_generator.py`):
`AddMotifAction`, `RemoveMotifAction`, `ReplaceMotifAction`, `TranslateMotifAction`,
`RotateMotifAction`, `SwapMotifAction`, `ResizeMotifAction`,
`ChangeElementAction`, `LatticeTransformAction`, `MakeSupercellAction`, `RotateStructureAction`

### 1.3 Validate generated data
- Confirm JSON files appear in `data/verbose/` with correct schema fields
- Run `atomworld benchmark -f data/verbose/ -a AddMotifAction -m deepseek_chat -b 1 -n 1` as smoke test

---

## Phase 2 — Agent Mode

Agent mode is **not yet implemented**. The LLM mode pipeline must be extended.

### 2.1 CLI Contract (what agents must implement)
```
<agent_cli> \
  --workspace_dir <path>   # contains structure.cif
  --instruction <string>
  --output_dir <path>      # agent writes result.cif here
```

### 2.2 Workspace Materialization
Add `AgentWorkspaceMaterializer` (e.g. in `src/benchmark/inference/`):
- reads one row from HDF5 (`input_cif`, `action_prompt`)
- writes `<tmp_dir>/structure.cif`
- returns `tmp_dir` path

### 2.3 Agent Runner
Add `AgentInferencer` extending `BaseInferencer` (`src/benchmark/inference/base_inferencer.py`):
- for each task: materialize workspace → spawn agent CLI subprocess → read `result.cif`
- enforce timeout (fail task with `TimeoutError` if exceeded)
- capture stdout/stderr to `logs/`
- return list of `{task_id, predicted_cif, status}` dicts (same shape as LLM mode)

### 2.4 Result Storage
```
results/AtomWorld/<category>/<action_name>/<agent_name>/<timestamp>/
  evaluation_results.csv
  evaluation_wrongs.csv
  metrics.json
  logs/
```
Reuse `AtomWorldEvaluator` unchanged — it only needs the `predicted_cif` field.

### 2.5 CLI Extension
Add `--agent_cli` and `--timeout` arguments to `src/utils/args.py`:
```
atomworld benchmark \
  --agent_cli "python examples/my_agent/run.py" \
  --timeout 120 \
  -f data/simple/ -a add_atom_action
```
When `--agent_cli` is given, use `AgentInferencer` instead of `AtomWorldInferencer`.

---

## Phase 3 — Parallel Execution

Currently inference is sequential. Tasks are independent and safe to parallelize.

### 3.1 Add `--parallel N` flag to benchmark CLI
- `N=1` (default): current sequential behaviour
- `N>1`: use `concurrent.futures.ProcessPoolExecutor` (or `ray` — already an optional dep)

### 3.2 Scope
- Applies to both LLM mode (API calls) and agent mode (subprocess spawning)
- Each worker writes its own temp inference JSON; merge before evaluation

---

## Phase 4 — Active (Multi-Step) Tasks

`results/AtomWorld/active/` and `data/active/` both exist but are empty.
This tier is for multi-step / agentic tasks that cannot be evaluated by
single-step CIF comparison.

### 4.1 Define active task format
Decide evaluation strategy (e.g. trajectory scoring, final-state check, LLM-as-judge).
Active tasks will also use the unified JSON v2 schema; add any extra fields
(e.g. `steps`, `goal`) as optional keys alongside `input`/`output`.

### 4.2 Add active action classes
Implement or register action classes whose `get_action_category()` returns `"active"`.
Generated data goes to `data/active/<ActionName>.json`.

### 4.3 Evaluation for active tasks
Likely requires a separate evaluator; defer until action design is settled.

---

## Phase 5 — Documentation & Examples

### 5.1 Agent Integration Guide
Create `docs/agent_integration.md`:
- CLI contract (workspace layout, output format)
- How to run: `atomworld benchmark --agent_cli ...`
- Minimal Python agent template

### 5.2 Example Agent
Create `examples/simple_agent/`:
```
examples/simple_agent/
  run.py          # reads structure.cif + --instruction, calls LLM, writes result.cif
  README.md
```

### 5.3 Update README
- Add quick-start for verbose benchmarking once data exists
- Document `--agent_cli` flag after Phase 2 is done

---

## Phase 6 — Additional Tests

### 6.1 Agent mode tests
- `test_agent_inferencer.py`: mock subprocess, check result.cif is read correctly
- `test_workspace_materializer.py`: check `structure.cif` written from HDF5 row

### 6.2 Verbose action tests
- Extend `test_cif_action_generator.py` to cover verbose action classes

### 6.3 Parallel execution tests
- Ensure results are identical to sequential mode at `--parallel 1`

---

## Reference: Actual CLI Usage

```bash
# Generate simple action data (writes to data/simple/ in JSON v2 format)
atomworld generate --cif_folder <dir> --actions AddAtomAction --num_samples 500

# Run LLM benchmark (simple) — data/ folder auto-detected as JSON v2
atomworld benchmark -f data/simple/ -a add_atom_action -m deepseek_chat -b 50 -n -1

# Run LLM benchmark (verbose) — once data/verbose/ is populated
atomworld benchmark -f data/verbose/ -a add_motif_action -m deepseek_chat -b 50 -n -1

# Evaluate only (skip inference)
atomworld eval --inference_file results/.../inference/results.json

# Draw metrics plot
atomworld draw -i results/AtomWorld/simple/add_atom_action/deepseek_chat/<ts>/

# Model config lives in src/config/models.yaml
# Add new models there — no code changes needed
```

---

## Directory Map (actual)

```
data/                          ← canonical task data (JSON v2, repo root)
  simple/
    AddAtomAction.json         ← {action_type, problem_id, mp_id, action_prompt, input, output}
    RemoveAtomAction.json
    ... (10 simple actions)
  verbose/                     ← empty, waiting for Phase 1
  active/                      ← empty, waiting for Phase 4
  _raw_data/                   ← source CIF files used during generation

src/
  atomworld/actions/
    simple/          ← AddAtomAction, RemoveAtomAction, ...
    verbose/         ← AddMotifAction, ChangeElementAction, ...
  benchmark/
    inference/       ← BaseInferencer, AtomWorldInferencer  [+ AgentInferencer TODO]
    evaluation/      ← AtomWorldEvaluator, metrics.py
  data_generation/   ← CIFActionGenerator, BaseDataGenerator
  data/              ← legacy HDF5+CSV files (v1); still supported, no longer canonical
  config/models.yaml ← model keys + API config
  utils/
    dataloader.py    ← auto-detects v1 (CSV+HDF5) vs v2 (JSON); maps input→input_cif, output→output_cif
    args.py, logger.py, extract_data.py, visualization.py
  tests/

results/AtomWorld/
  simple/<action>/<model>/<timestamp>/
  verbose/           ← empty, waiting for Phase 1
  active/            ← empty, waiting for Phase 4
```
