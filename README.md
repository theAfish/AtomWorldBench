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
atomworld generate -c ./path/to/cifs/folder -o ./path/to/dataset -n 1000

# Run benchmark (inference + evaluation)
atomworld benchmark -f ./path/to/dataset -a move_atom_action -m deepseek_chat -o ./path/to/results

# Evaluate only from existing inference output
atomworld eval -f ./path/to/dataset -a move_atom_action -m deepseek_chat -o ./path/to/results -i ./path/to/inference_results.json

# Draw plots from evaluation results
atomworld draw -i ./path/to/evaluation_results.json
```

### Run the Benchmark

```bash
atomworld benchmark -f [data_folder] -a [action_name] -m [model] -b [batch_size] -n [num_batch] -o [output]
```

**Arguments:**

| Argument                  | Description                                                                   |
|---------------------------|-------------------------------------------------------------------------------|
| `-f`                      | Data folder containing CIF action data (CSV+HDF5 or JSON format).             |
| `-a`                      | Action name (see [Available Actions](#available-actions)).                     |
| `-m`                      | Model name (defined in `config/models.yaml`).                                  |
| `-b`                      | Batch size — number of parallel LLM calls (default: 50).                       |
| `-n`                      | Number of batches to run (default: all data).                                  |
| `-o`                      | Output directory for results.                                                  |
| `-c`                      | Model config YAML path (default: `config/models.yaml`).                        |
| `--repeat`                | Repeat each sample N times (default: 1).                                       |
| `--skip_inference`        | Skip inference and evaluate an existing results file.                          |
| `--inference_file` / `-i` | Path to existing inference results JSON (required with `--skip_inference`).    |
| `--keep_inference`        | Retain the inference results file after evaluation.                            |
| `--start_index`           | Resume inference from this sample index.                                       |
| `--plot`                  | Generate a max-dist histogram after evaluation.                                |
| `--input_cifs_file`       | (AtomWorld only) Alternate HDF5 filename for base input CIFs.                  |

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

Pass `-f` to point the benchmark at any dataset folder. Action names not in the built-in list are accepted. Use `--input_cifs_file` to specify an alternate HDF5 filename for the base input CIFs (AtomWorld only).

```bash
atomworld benchmark -f ./path/to/my_dataset -a my_custom_action -m deepseek_chat -o ./results
```

---

### StructProp Task

```bash
python ./src/struct_prop_bench/inferring.py -m [model_name] -p [property] -b [batch_size] -n [num_batch]
```

Then run your calculation pipelines and save results matching the format of `./results/StructPropBench/dft_statistics.csv`. Use `./src/scripts/analyze_structprop_results.py` for final metrics.

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
