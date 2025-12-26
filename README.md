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

If you want to run the benchmark for your own model, implement your model in `AtomWorldBench/models/` and corresponding parameters in `config/models.yaml`. Currently, we have implemented openai_model, azure_openai_model, huggingface_model, and vllm_model.

### Run the Benchmark

```bash
atomworld benchmark -m [model_name] -a [action_name] -b [batch_size] -n [num_batch]
```

**Arguments:**

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `model_name`     | Model to test (e.g., `deepseek_chat`).                                  |
| `action_name`    | Action to test (see [Available Actions](#available-actions)). Only for AtomWorld and PointWorld. |
| `batch_size`     | Number of parallel LLM calls (default: 50).                                 |
| `num_batch`      | Number of batches to test (default: all data).                              |
| `output_folder`  | Folder to save results (default: `./results/`).                            |
| `keep_inference` | Whether to keep inference files (default: False).                           |

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
	atomworld data_generator -a [action_name] --input_cif_folder [path_to_input_cifs] --output_json_path [output_json_path] --num_samples [num_samples] (--no_random) (--allow_repeat)
	```
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
