# Atom World

Testing LLMs' ability on operating 3D atomic structures.

<p align="center">
  <img src="docs/img/main1.png" width="50%">
</p>


> *"Forget the messy details, I just need a model that can play Lego with atoms."* ⚛️🤖


---

## Table of Contents

- [Installation](#installation)
- [Usage of the Bench](#usage)
  - [Unified CLI](#unified-cli)
  - [Run the Benchmark](#run-the-benchmark)
    - [Available Benchmarks](#available-benchmarks)
    - [Available Actions](#available-actions)
    - [StructProp Task](#structprop-task)
  - [Analyze the Results](#analyze-the-results)
  - [Construct Your Own Data](#construct-your-own-data-with-mp-api)
- [Usage of the Gym]
  - under construction
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)


---

## Installation

```bash
pip install -e .
```

---

## Usage of the Bench

If you want to run the benchmark for your own model, implement your model in `src/models/` and corresponding parameters in `config/models.yaml`. Currently, we have implemented openai_model, azure_openai_model, huggingface_model, and vllm_model.

### Unified CLI

After installation (`pip install -e .`), you can use the unified CLI:

```bash
atomworld [generate|benchmark|eval|draw] [options]
```

Core commands:

- `atomworld generate ...`: generate AtomWorld JSON datasets.
- `atomworld benchmark ...`: run inference + evaluation (or evaluation-only if `--skip_inference` is provided).
- `atomworld eval ...`: evaluation-only shortcut (automatically adds `--skip_inference`; requires `--inference_file` or `-i`).
- `atomworld draw -i [evaluation_results.json]`: draw RMSD / max-distance distributions.

Examples:

```bash
# Generate dataset
atomworld generate -c ./src/data/_raw_data/training_cifs -o ./src/data/train_data -n 1000

# Run benchmark (inference + evaluation)
atomworld benchmark -f data -a move_atom_action -m deepseek_chat -c config/models.yaml -o results

# Evaluate only from existing inference output
atomworld eval -f data -a move_atom_action -m deepseek_chat -c config/models.yaml -o results -i path/to/inference_results.json

# Draw plots from evaluation results
atomworld draw -i path/to/evaluation_results.json
```

### Run the Benchmark

The benchmark supports two modes: a **legacy runner** (combined inference + evaluation) and a **separated pipeline** (independent inference and offline evaluation, aligned with v2 architecture).

#### Legacy Runner (Combined)

```bash
python ./src/run_benchmark.py -t [benchmark_type] -m [model_name] -a [action_name] -b [batch_size] -n [num_batch]
```

**Arguments:**

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `benchmark_type` | Benchmark to run. See [Available Benchmarks](#available-benchmarks).        |
| `model_name`     | Model to test (e.g., `deepseek_chat`).                                  |
| `action_name`    | Action to test (see [Available Actions](#available-actions)). Only for AtomWorld and PointWorld. |
| `batch_size`     | Number of parallel LLM calls (default: 50).                                 |
| `num_batch`      | Number of batches to test (default: all data).                              |
| `--repeat`       | Repeat each frame/task this many times (default: 1).                        |
| `results_folder` | Optional output directory override. Otherwise a timestamped folder is created. |
| `restart_from_index` | Resume an interrupted job starting from the provided index. |
| `--plot`         | Produce a max-dist histogram after evaluation (AtomWorld, PointWorld, CIFGen). |
| `--data_path`    | Override the default dataset location. For AtomWorld/PointWorld/CIFGen pass a folder; for CIFRepair pass a CSV file. |
| `--input_cifs_file` | (AtomWorld only) Alternate `input_cifs` HDF5 filename to load from the data path. |

Use `--repeat` when you need multiple independent generations per frame. For example, `--repeat 10` will call the LLM ten times for every input frame while preserving the frame index in the output CSVs.

#### Separated Pipeline (v2 Style)

For AtomWorld, inference and evaluation can also be run independently via the new separated pipeline:

```bash
# Run inference only
python ./src/run_benchmark.py -f [data_folder] -a [action_name] -c [config] -o [output]

# Skip inference and evaluate existing results
python ./src/run_benchmark.py -f [data_folder] -a [action_name] -c [config] -o [output] --skip_inference --inference_file [path/to/inference_results.json]
```

| Argument             | Description                                                       |
|----------------------|-------------------------------------------------------------------|
| `-f`                 | Data folder containing CIF action data (CSV+HDF5 or JSON format). |
| `-a`                 | Action name to evaluate.                                          |
| `-c`                 | Model config YAML file path.                                      |
| `-o`                 | Output directory for results.                                     |
| `--skip_inference`   | Skip inference and only run evaluation.                           |
| `--inference_file`   | Path to existing inference results JSON.                          |
| `--keep_inference`   | Keep inference results after evaluation.                          |

---

#### Available Benchmarks

- `atomworld`: AtomWorld
- `pointworld`: PointWorld
- `cifgen`: CIFGen
- `cifrepair`: CIFRepair

> For the StructProp task, see below.

---

#### Available Actions

**AtomWorld:**

- add_atom_action
- change_atom_action
- delete_around_atom_action
- delete_below_atom_action
- insert_between_atoms_action
- move_around_atom_action
- move_atom_action
- move_selected_atoms_action
- move_towards_atom_action
- remove_atom_action
- rotate_around_atom_action
- swap_atoms_action
- super_cell_action
- rotate_whole_action
- move_all_action

**PointWorld:**

- move
- move_towards
- insert_between
- rotate_around

---

#### Use custom datasets

Pass `--data_path` to point the runner at your own dataset without changing the existing directory layout. The flag is benchmark-agnostic:

- **AtomWorld / PointWorld**: provide a folder that contains the CSV/HDF5 pairs (and optional `input_cifs` file). When `--data_path` is set you can also supply action names that are not in the built-in lists (e.g., `insert_between_atoms_action_natoms`).
- **CIFGen**: provide a folder containing your `.cif` files plus the accompanying `description_cards.json`.
- **CIFRepair**: provide the path to a CSV file with the `original_cif`, `modified_cif`, and `removed_value` columns.

For AtomWorld-specific datasets you can additionally set `--input_cifs_file` to choose a different HDF5 file that stores the base `input_cifs` mapping.

Example (PowerShell) using a custom AtomWorld dataset stored under `./data/analysis_data`:

```powershell
python .\src\run_benchmark.py -t atomworld -m deepseek_chat -a insert_between_atoms_action_natoms --data_path .\src\data\analysis_data --input_cifs_file analysis_input_natoms.hdf5
```

---

### StructProp Task

To get CIFs from LLM for StructProp:

```bash
python ./src/struct_prop_bench/inferring.py -m [model_name] -p [property] -b [batch_size] -n [num_batch]
```

Then run your own calculation pipelines. The results should be saved with the format similar to `./results/StructPropBench/dft_statistics.csv` in order to use the `./src/scripts/analyze_structprop_results.py` for final metrics. Or you can modify the analysis script for your own results.

---

### Analyze the Results

In the new codes, the results are saved in `./results/[BenchmarkType]/[ModelName]/[ActionName]/[Timestamp]/`. The `evaluation_results.csv` contains the correct results, and `evaluation_wrongs.csv` contains the incorrect ones. `metrics.json` contains the summary of the metrics. Every record now includes a `frame_index` (and `repeat_index`) column so you can group statistics per original dataset frame, especially when running with `--repeat > 1`.

#### Evaluate JSON Results

If you have results in a JSON format (e.g., collected from users or other sources) instead of running the full benchmark workflow, you can use the `evaluate_json_results.py` script to evaluate them.

**Input Format:**
The input JSON file should be a list of objects, where each object has the following fields:
- `instruction`: The instruction given to the model.
- `input`: The input CIF structure.
- `output`: The target CIF structure (ground truth).
- `response`: The model's generated response (containing the generated CIF).

**Usage:**

```bash
python src/scripts/evaluate_json_results.py [input_file] --output_file [output_file]
```

**Arguments:**
- `input_file`: Path to the input JSON file.
- `--output_file`: Path to save the detailed evaluation results (JSON).
- `--summary_file`: (Optional) Path to save the summary statistics (JSON). If not provided, a summary file will be created next to the output file.


Plotting after evaluation
------------------------

You can now request an automatic max_dist histogram to be generated after a benchmark run by adding the `--plot` flag to `run_benchmark.py`. The runner supports plotting for `atomworld`, `pointworld`, and `cifgen` benchmarks. The plot is saved to the same results folder as `evaluation_results.csv` and will not open an interactive window by default.

You can also draw plots later from a saved `evaluation_results.json` file with:

```bash
atomworld draw -i path/to/evaluation_results.json
```

Examples:

```powershell
python .\src\run_benchmark.py -t atomworld -m deepseek_chat -a move_atom_action -b 10 -n 1 --plot
python .\src\run_benchmark.py -t cifgen -m deepseek_chat -b 10 -n 1 --plot
```


---

### Construct Your Own Data with mp-api

Each action class now has a built-in `randomize(atoms, rng, config)` classmethod that samples valid random parameters, and an `apply_random(atoms, rng, config, copy)` classmethod that randomizes + executes in one step. This replaces the old reflection-based `ActionInputGenerator`.

**Quick example:**

```python
from ase.io import read
from atom_world.actions import AddAtomAction
import numpy as np

atoms = read("my_structure.cif")
rng = np.random.default_rng(42)
action, result = AddAtomAction.apply_random(atoms, rng=rng)
print(action)  # Add one Fe atom at the Cartesian coordinate [1.23 4.56 7.89] to the cif file.
```

**Full pipeline:**

1. (Optional) Download random structures:
	```bash
	python src/scripts/download_random_mp_data.py --api_key [YOUR_API_KEY] --out_path [path] --min_natoms [min_atoms] --max_natoms [max_atoms] --num_entries [total_entries]
	```
    The input CIFs we used are available in `./src/data/input_cifs.zip`.

2. Generate data — **Option A** (legacy CSV+HDF5 output):
	```bash
	python src/atom_world/data_generator.py
	```
	Then convert CIF folders to HDF5:
	```bash
	python src/scripts/convert_cifs_to_h5.py
	```

3. Generate data — **Option B** (v2 JSON output):
	```bash
	python src/data_generation/cif_action_generator.py --cif_folder [path] --output_dir [path] --num_samples [N]
  # or
  atomworld generate --cif_folder [path] --output_dir [path] --num_samples [N]
	```
	This produces per-action JSON files (e.g., `AddAtomAction.json`) with inline CIF strings.

4. Put the generated data in `./src/data/` (or use `--data_path`). The data loader auto-detects CSV+HDF5 vs JSON format.
---


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
```
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
