# Atom World

Testing LLMs' ability on operating 3D atomic structures.

<p align="center">
  <img src="docs/img/main1.png" width="50%">
</p>


> *"Forget the messy details, I just need a model that can play Lego with atoms."* ⚛️🤖


Please refer to v1 branch for the codebase corresponding to the paper: https://arxiv.org/abs/2510.04704.

This branch contains the latest updates and new features with improved benchmark formulations.

---

## Table of Contents

- [Installation](#installation)
- [Usage of the Bench](#usage)
  - [Run the Benchmark](#run-the-benchmark)
    - [Available Actions](#available-actions)
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

If you want to run the benchmark for your own model, implement your model in `AtomWorldBench/models/` and corresponding parameters in `config/llm_api_config.yaml`. Currently, we have implemented openai_model, azure_openai_model, huggingface_model, and vllm_model.

### Run the Benchmark

```bash
atomworld benchmark -m [model_name] -a [action_name] -b [batch_size] -n [num_batch]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-m`, `--model` | Model to test (e.g., `deepseek_chat`). | `deepseek_chat` |
| `-a`, `--action_name` | Action to test (see [Available Actions](#available-actions)). If not provided, runs all actions. | `None` |
| `-b`, `--batch_size` | Number of parallel LLM calls. | `50` |
| `-n`, `--num_batch` | Number of batches to test. -1 means all data. | `-1` |
| `-f`, `--data_folder` | Path to data folder. | `AtomWorldBench/data` |
| `-c`, `--config_path` | Path to config file. | `AtomWorldBench/config/llm_api_config.yaml` |
| `-o`, `--output_folder` | Folder to save results. | `results` |
| `-s`, `--start_index` | Start index for inference. | `0` |
| `-r`, `--repeat` | Repeat count for each sample. | `1` |
| `--skip_inference` | Skip inference and only run evaluation. | `False` |
| `--inference_file` | Path to inference results file (if skipping inference). | `None` |
| `--keep_inference` | Whether to keep inference files. | `False` |

---

#### Available Actions

**AtomWorld:**

- AddMotifAction
- ChangeElementAction
- LatticeTransformAction
- MakeSupercellAction
- RemoveMotifAction
- ReplaceMotifAction
- ResizeMotifAction
- RotateMotifAction
- SwapMotifAction
- TranslateMotifAction

---

### Analyze the Results

In the new codes, the results are saved in `./results/[BenchmarkType]/[ModelName]/[ActionName]/[Timestamp]/`. The `evaluation_results.json` contains the results.

You can visualize the results using the `visualize` command:

```bash
atomworld visualize -i [path_to_evaluation_results.json] -o [output_folder]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--input_file` | Path to `evaluation_results.json` file. | Required |
| `-o`, `--output_folder` | Output folder for plots. | Same as input file folder |


---

### Construct Your Own Data with mp-api

**The actions and data_generator are currently under refactoring.** The current pipeline will be updated soon. If you want to construct your own data, you can follow the steps below:

1. (Optional) Download random structures:
	```bash
	python AtomWorldBench/scripts/download_random_mp_data.py --api_key [YOUR_API_KEY] --out_path [path] --min_natoms [min_atoms] --max_natoms [max_atoms] --num_entries [total_entries]
	```
    The input CIFs we used are available in `./src/data/input_cifs.zip`.
2. Generate data:
	```bash
	atomworld generate -c [cif_folder] -o [output_dir] -a [action_names] -n [num_samples]
	```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-c`, `--cif_folder` | Path to folder containing input CIF files. | Required |
| `-o`, `--output_dir` | Directory to save generated JSON files. | Required |
| `-a`, `--action_names` | List of action names to generate. If not provided, uses all ready actions. | `None` |
| `-n`, `--num_samples` | Number of samples per action. | `1000` |
| `--max_attempts` | Max attempts to generate a valid sample. | `10` |
| `--seed` | Random seed. | `75` |
| `--no_random` | Disable random shuffling of structures. | `False` |
| `--allow_repeat` | Allow repeating structures across samples. | `False` |
---


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
```
@misc{lv2025atomworldbenchmarkevaluatingspatial,
      title={AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials}, 
      author={Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Yingheng Wang, Zhicheng Zhong, Tong Xie},
      year={2025},
      eprint={2510.04704},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2510.04704}, 
}
```
