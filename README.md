# AtomWorld

Benchmark & toolkit for evaluating LLMs on 3D crystal-structure manipulation.

<p align="center">
  <img src="docs/img/main1.png" width="50%">
</p>

> *"Forget the messy details, I just need a model that can play Lego with atoms."* ⚛️🤖

---

## Installation

**Lightweight** — just the evaluator:

```bash
pip install git+https://github.com/theAfish/AtomWorldBench.git
```

**Full toolkit** — CLI, inference, data generation, plotting, etc.:

```bash
pip install "atomworld[all] @ git+https://github.com/theAfish/AtomWorldBench.git"
```

For development:

```bash
git clone https://github.com/theAfish/AtomWorldBench.git
cd AtomWorldBench
pip install -e ".[dev]"
```

<details>
<summary>Optional dependency groups</summary>

| Extra          | What it adds                                                   |
|----------------|----------------------------------------------------------------|
| `[benchmark]`  | openai, pandas, h5py, tqdm, pyyaml, fastapi, uvicorn, pydantic |
| `[datagen]`    | ase, mp-api, scipy, pandas                                      |
| `[models]`     | transformers, sentencepiece, torch                              |
| `[all]`        | All of the above + ray                                          |
| `[dev]`        | `[all]` + pytest                                                |

</details>

---

## Python API

### Evaluate (for RL / reward functions)

The core `evaluate` function works with just the lightweight install:

```python
from atomworld import evaluate

result = evaluate(
    target_cif=ground_truth_cif_string,
    generated_output=model_output,
)
print(result.correct)    # True / False
print(result.wrong_type) # None, "OutputFormatError", "CIFParsingError", "AtomCountMismatch", "StructureMismatch"
print(result.rmsd)       # float (Å) if correct, else None
print(result.max_dist)   # float (Å) if correct, else None
```

**Example use as an RL reward function:**

```python
from atomworld import evaluate

def score(prompts, completions, *, solution, **kwargs):
    rewards = []
    for sol, comp in zip(solution, completions):
        r = evaluate(target_cif=sol, generated_output=comp)
        rewards.append(1.0 if r.correct else 0.0)
    return rewards
```

### Data loading

```python
from atomworld import load_data

df = load_data("./path/to/dataset", action_name="add_atom_action")
# DataFrame with columns: input_cif, action_prompt, output_cif
```

### Run benchmark from Python

```python
from atomworld import BenchmarkRunner
```

### Actions

AtomWorld has two action families, both under `atomworld.actions`.

#### Simple actions (index-based)

Lightweight, no extra dependencies. Each action addresses atoms by their index in the ASE `Atoms` list and exposes `apply_random()` for one-shot dataset generation.

```python
from ase.io import read
from atomworld.actions import AddAtomAction, ChangeAtomAction, SwapAtomsAction
import numpy as np

atoms = read("my_structure.cif")
rng = np.random.default_rng(42)

action, result = AddAtomAction.apply_random(atoms, rng=rng)
print(action.describe())  # natural-language prompt for the LLM
print(result)             # new Atoms object (ground truth)
```

All simple action classes:

| Class | Description |
|---|---|
| `AddAtomAction` | Insert a new atom at a random position |
| `RemoveAtomAction` | Remove a randomly chosen atom |
| `MoveAtomAction` | Translate one atom by a random vector |
| `MoveTowardsAtomAction` | Move an atom towards another atom |
| `MoveSelectedAtomsAction` | Move a subset of atoms |
| `MoveAroundAtomAction` | Move an atom in a shell around another atom |
| `MoveAllAction` | Translate all atoms uniformly |
| `ChangeAtomAction` | Change an atom's element |
| `SwapAtomsAction` | Swap two atoms |
| `InsertBetweenAtomsAction` | Insert an atom between two existing atoms |
| `RotateAroundAtomAction` | Rotate an atom around a pivot |
| `RotateWholeAction` | Rotate the whole structure |
| `DeleteBelowAtomAction` | Remove atoms below a threshold height |
| `DeleteAroundAtomAction` | Remove atoms within a radius |
| `SuperCellAction` | Build a supercell |

#### Verbose actions (motif-based)

Richer, motif-centric API that produces more expressive natural-language descriptions. Requires `scipy`. You first describe *what* you want to operate on as a **motif**, then pass it to an action.

**Structure actions** — operate on the whole structure, no motif needed:

```python
from ase.io import read
from atomworld.actions.verbose import ChangeElementAction

atoms = read("my_structure.cif")

# Explicit construction
action = ChangeElementAction(operated_atoms=atoms, from_element="Fe", to_element="Co")
result = action.execute()        # new Atoms object
print(action.describe())         # "replace all atoms of element Fe with Co."

# Random construction
action = ChangeElementAction.get_random_one(atoms, seed=42)
result = action.execute()
print(action.describe())
```

All structure action classes:

| Class | Description |
|---|---|
| `ChangeElementAction` | Replace or remove all atoms of an element |
| `LatticeTransformAction` | Apply a linear transformation to the lattice |
| `MakeSupercellAction` | Build a supercell |
| `RotateStructureAction` | Rotate the whole structure |

**Motif actions** — first create a motif, then act on it:

```python
from ase.io import read
from atomworld.actions.verbose import SiteMotif, ClusterMotif, AddMotifAction, RemoveMotifAction

atoms = read("my_structure.cif")

# Detect a random existing atom as a SiteMotif, then remove it
site = SiteMotif.detect_random_one(atoms, seed=0)
action = RemoveMotifAction(operated_motif=site)
result = action.execute()
print(action.describe())

# Create an additive SiteMotif (new atom to insert) and add it at an absolute position
new_fe = SiteMotif.detect_random_one(atoms, seed=1, additive_mode=True,
                                      additive_mode_allowed_symbols=["Fe"])
action = AddMotifAction(
    operated_motif=new_fe,
    operated_atoms=atoms,
    at_position=[1.5, 2.0, 3.0],
)
result = action.execute()
print(action.describe())
```

All motif types:

| Class | Description |
|---|---|
| `SiteMotif` | A single lattice site / atom |
| `ClusterMotif` | A group of atoms |
| `BondMotif` | A pair of bonded atoms |
| `SphereRegionMotif` | A spherical region in space |
| `BoxRegionMotif` | A box-shaped region in space |

All motif action classes:

| Class | Description |
|---|---|
| `AddMotifAction` | Insert a motif into the structure |
| `RemoveMotifAction` | Delete the atoms of a motif |
| `ReplaceMotifAction` | Replace a motif with another |
| `TranslateMotifAction` | Translate the atoms of a motif |
| `RotateMotifAction` | Rotate a motif about its centroid |
| `SwapMotifAction` | Swap two motifs |
| `ResizeMotifAction` | Scale bond lengths within a motif |

> **CLI note:** The `atomworld generate` command supports both action families. Use `--verbose` to generate datasets with the verbose (motif-based) action family. Without this flag it defaults to the simple (index-based) family.

---

## CLI

```bash
atomworld [generate|benchmark|eval|draw|serve] [options]
```

### Quick examples

```bash
# Generate dataset from CIF files
atomworld generate -c ./cifs -o ./dataset -n 1000

# Run full LLM benchmark (inference + evaluation)
atomworld benchmark -f ./dataset -a move_atom_action -m deepseek_chat -o ./results

# Evaluate existing inference results
atomworld eval -f ./dataset -a move_atom_action -i ./inference_results.json -o ./results

# Plot RMSD / max-distance distributions
atomworld draw -i ./results/evaluation_results.json

# Start the agent benchmark API server
atomworld serve --api-key mykey --data-folder data/simple --sessions-dir sessions
```

### Benchmark arguments

```
atomworld benchmark -f DATA -a ACTION -m MODEL [-b BATCH] [-n NUM_BATCH] [-o OUTPUT]
```

| Flag                      | Description                                           |
|---------------------------|-------------------------------------------------------|
| `-f`                      | Data folder (JSON or CSV+HDF5 format)                 |
| `-a`                      | Action name (see below)                               |
| `-m`                      | Model key from `config/models.yaml`                   |
| `-b`                      | Batch size (default: 50)                              |
| `-n`                      | Number of batches (default: all)                      |
| `-o`                      | Output directory                                      |
| `-c`                      | Model config YAML (default: `config/models.yaml`)     |
| `--repeat`                | Repeat each sample N times                            |
| `--skip_inference`        | Evaluate only (needs `--inference_file`)              |
| `--inference_file` / `-i` | Path to inference results JSON                        |
| `--keep_inference`        | Keep inference JSON after evaluation                  |
| `--start_index`           | Resume from sample index                              |

### Available actions

**AtomWorld:** `add_atom_action`, `change_atom_action`, `delete_around_atom_action`, `delete_below_atom_action`, `insert_between_atoms_action`, `move_around_atom_action`, `move_atom_action`, `move_selected_atoms_action`, `move_towards_atom_action`, `remove_atom_action`, `rotate_around_atom_action`, `swap_atoms_action`, `super_cell_action`, `rotate_whole_action`, `move_all_action`

**PointWorld:** `move`, `move_towards`, `insert_between`, `rotate_around`

---

### Adding your own model

Implement your model class in `src/models/` and add its config to `config/models.yaml`. Built-in backends: OpenAI, Azure OpenAI, HuggingFace, vLLM.

---

## Agent Benchmarking via API

Instead of wrapping your agent in a CLI shim, AtomWorldBench exposes a REST API. Your agent framework fetches tasks over HTTP, runs them however it likes, and submits results back. This works identically whether the server is running locally or hosted remotely.

### Start the server

```bash
atomworld serve \
  --api-key mykey \
  --data-folder data/simple \
  --sessions-dir sessions \
  --host 0.0.0.0 \
  --port 8000
```

All requests must include the header `X-API-Key: mykey`.

### What your agent receives per task

For each task the API returns a JSON object with:

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | e.g. `"task_0_repeat_0"` |
| `action_prompt` | `string` | Natural-language instruction, e.g. `"Add one Ba atom at [6.5465 4.7379 3.9022]."` |
| `input_cif` | `string` | Full CIF text of the input crystal structure |
| `action_type` | `string` | Action class name, e.g. `"AddAtomAction"` |
| `frame_index` | `int` | Index of the task in the dataset |
| `repeat_index` | `int` | Repeat number (0-based) |

The ground-truth output is **never sent to the client** — evaluation runs server-side.

### What your agent sends back

```json
{
  "result_cif": "<full CIF text of the modified structure>",
  "elapsed_seconds": 4.2,
  "token_usage": {"prompt_tokens": 1200, "completion_tokens": 800, "total_tokens": 2000}
}
```

`elapsed_seconds` and `token_usage` are optional metadata used for cost reporting.

### Complete workflow

```python
import requests

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "mykey"}

# 1. Create a session
session = requests.post(f"{BASE}/sessions", headers=HEADERS, json={
    "action_name": "AddAtomAction",
    "limit": 100,
    "repeat": 1,
}).json()
sid = session["session_id"]

# 2. Fetch task list (no CIF content, for overview)
tasks = requests.get(f"{BASE}/sessions/{sid}/tasks", headers=HEADERS).json()

# 3. For each task: fetch full details, run agent, submit result
for task_meta in tasks:
    tid = task_meta["task_id"]

    # Fetch task with input CIF
    task = requests.get(f"{BASE}/sessions/{sid}/tasks/{tid}", headers=HEADERS).json()
    input_cif   = task["input_cif"]
    instruction = task["action_prompt"]

    # --- run your agent here ---
    result_cif = my_agent.run(input_cif=input_cif, instruction=instruction)
    # ---------------------------

    requests.post(f"{BASE}/sessions/{sid}/tasks/{tid}/submit", headers=HEADERS, json={
        "result_cif": result_cif,
        "elapsed_seconds": 3.7,
    })

# 4. Trigger evaluation
requests.post(f"{BASE}/sessions/{sid}/evaluate", headers=HEADERS)

# 5. Retrieve metrics
results = requests.get(f"{BASE}/sessions/{sid}/results", headers=HEADERS).json()
print(results["metrics"]["summary"])
```

### Server-side storage layout

```
sessions/
  {session_id}/
    session.json                    ← session state, tasks, submissions
    results/
      evaluation_results.json       ← written after POST /evaluate
      generated_cifs/
        task_0_repeat_0.cif         ← each submitted result CIF
        task_1_repeat_0.cif
        ...
```

### API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a benchmark session |
| `GET`  | `/sessions/{id}` | Session status and progress |
| `GET`  | `/sessions/{id}/tasks` | Paginated task list (no CIF) |
| `GET`  | `/sessions/{id}/tasks/{tid}` | Task details + input CIF |
| `POST` | `/sessions/{id}/tasks/{tid}/submit` | Submit result CIF |
| `POST` | `/sessions/{id}/evaluate` | Trigger evaluation |
| `GET`  | `/sessions/{id}/results` | Retrieve evaluation metrics |

Interactive docs are available at `http://localhost:8000/docs` while the server is running.

---

## Data Generation

```bash
# Generate per-action JSON datasets from CIF files
atomworld generate --cif_folder ./cifs --output_dir ./dataset --num_samples 1000

# (Optional) Download structures from Materials Project
python src/scripts/download_random_mp_data.py --api_key YOUR_KEY --out_path ./cifs --num_entries 500
```

You can also apply actions programmatically — see the [Actions](#actions) section above for full examples with both simple and verbose APIs.

---

## Contributing

Contributions welcome — please open an issue or pull request.

## Results Dashboard (GitHub Pages)

An interactive dashboard visualising benchmark results is hosted at:

> **https://theafish.github.io/AtomWorldBench/**

### Generating / updating the dashboard data

After adding new benchmark results, regenerate the data file and push:

```bash
# From the repo root
python src/scripts/generate_gh_pages_data.py
```

This reads every `results/AtomWorld/simple/<model>/<action>/<timestamp>/evaluation_results.json`
and writes `docs/data/simple_metrics.json`.  Commit and push both files.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{lv2025atomworldbenchmarkevaluatingspatial,
      title={AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials}, 
      author={Taoyuze Lv and Alexander Chen and Fengyu Xie and Chu Wu and Jeffrey Meng and Dongzhan Zhou and Bram Hoex and Zhicheng Zhong and Tong Xie},
      year={2025},
      eprint={2510.04704},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2510.04704}, 
}
```
