# Atom World Bench

Testing LLMs' ability on operating 3D atomic structures.


## Installation for dev

```
pip install -e .
```

# Atom World Bench

Testing LLMs' ability on operating 3D atomic structures.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Run the Benchmark](#run-the-benchmark)
  - [Analyze the Results](#analyze-the-results)
  - [Construct Your Own Data](#construct-your-own-data-with-mp-api)
- [Available Benchmarks](#available-benchmarks)
- [Available Actions](#available-actions)

---

## Installation

```bash
pip install -e .
```

---

## Usage

If you want to run the benchmark for your own model, implement your model in `src/models/` and corresponding parameters in `config/models.yaml`. Currently, we have implemented openai_model, azure_openai_model, huggingface_model, and vllm_model.

### Run the Benchmark

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

---

## Available Benchmarks

- `atomworld`: AtomWorld
- `pointworld`: PointWorld
- `cifgen`: CIFGen
- `cifrepair`: CIFRepair

> For the StructProp task, see below.

---

## Available Actions

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

**PointWorld:**

- move
- move_towards
- insert_between
- rotate_around

---

### StructProp Task

To get CIFs from LLM for StructProp:

```bash
python ./src/struct_prop_bench/inferring.py -m [model_name] -p [property] -b [batch_size] -n [num_batch]
```

Then run your own calculation pipelines. The results should be saved with the format similar to `./results/StructPropBench/dft_statistics.csv` in order to use the `./src/scripts/analyze_structprop_results.py` for final metrics. Or you can modify the analysis script for your own results.

---

### Analyze the Results

```bash
python ./src/scripts/analyze_results.py
```

---

### Construct Your Own Data with mp-api

**The actions and data_generator are currently under refactoring.** The current pipeline will be updated soon. If you want to construct your own data, you can follow the steps below:

1. Download random structures:
	```bash
	python src/scripts/download_random_mp_data.py --api_key [YOUR_API_KEY] --out_path [path] --min_natoms [min_atoms] --max_natoms [max_atoms] --num_entries [total_entries]
	```
2. Generate data:
	```bash
	python src/atom_world/data_generator.py
	```
3. Convert to h5:
	```bash
	python src/scripts/convert_cifs_to_h5.py
	```

---

