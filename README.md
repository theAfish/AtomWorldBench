# AtomWorld

Benchmark & toolkit for evaluating LLMs on 3D crystal-structure manipulation.

<p align="center">
  <img src="docs/img/main1.png" width="50%">
</p>

> *"Forget the messy details, I just need a model that can play Lego with atoms."* ⚛️🤖

---

## Installation

**Lightweight** — just the evaluator (only needs `numpy` + `pymatgen`):

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

| Extra          | What it adds                                |
|----------------|---------------------------------------------|
| `[benchmark]`  | openai, pandas, h5py, tqdm, pyyaml          |
| `[datagen]`    | ase, mp-api, scipy, pandas                   |
| `[models]`     | transformers, sentencepiece, torch            |
| `[viz]`        | matplotlib, seaborn                           |
| `[all]`        | All of the above + ray                        |
| `[dev]`        | `[all]` + pytest                              |

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

---

## CLI

```bash
atomworld [generate|benchmark|eval|draw] [options]
```

### Quick examples

```bash
# Generate dataset from CIF files
atomworld generate -c ./cifs -o ./dataset -n 1000

# Run full benchmark (inference + evaluation)
atomworld benchmark -f ./dataset -a move_atom_action -m deepseek_chat -o ./results

# Evaluate existing inference results
atomworld eval -f ./dataset -a move_atom_action -i ./inference_results.json -o ./results

# Plot RMSD / max-distance distributions
atomworld draw -i ./results/evaluation_results.json
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
| `--plot`                  | Generate histogram after evaluation                   |

### Available actions

**AtomWorld:** `add_atom_action`, `change_atom_action`, `delete_around_atom_action`, `delete_below_atom_action`, `insert_between_atoms_action`, `move_around_atom_action`, `move_atom_action`, `move_selected_atoms_action`, `move_towards_atom_action`, `remove_atom_action`, `rotate_around_atom_action`, `swap_atoms_action`, `super_cell_action`, `rotate_whole_action`, `move_all_action`

**PointWorld:** `move`, `move_towards`, `insert_between`, `rotate_around`

### Adding your own model

Implement your model class in `src/models/` and add its config to `config/models.yaml`. Built-in backends: OpenAI, Azure OpenAI, HuggingFace, vLLM.

---

## Data Generation

```bash
# Generate per-action JSON datasets from CIF files
atomworld generate --cif_folder ./cifs --output_dir ./dataset --num_samples 1000

# (Optional) Download structures from Materials Project
python src/scripts/download_random_mp_data.py --api_key YOUR_KEY --out_path ./cifs --num_entries 500
```

You can also apply actions programmatically:

```python
from ase.io import read
from atom_world.actions import AddAtomAction
import numpy as np

atoms = read("my_structure.cif")
rng = np.random.default_rng(42)
action, result = AddAtomAction.apply_random(atoms, rng=rng)
print(action)
# Add one Fe atom at the Cartesian coordinate [1.23 4.56 7.89] to the cif file.
```

---

## Contributing

Contributions welcome — please open an issue or pull request.

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
