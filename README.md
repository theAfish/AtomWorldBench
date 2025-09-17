# Atom World Bench

Testing LLMs' ability on operating 3D atomic structures.


## Installation for dev

```
pip install -e .
```

## Usage

If you want to run the benchmark for your own model, please add your model in `src/models/` folder and corresponding parameters in `config/models.yaml`.

### Run the benchmark

```
python ./src/run_benchmark.py -t [benchmark_type] -m [model_name] -a [action_name] -b [batch_size] -n [num_batch]
```

`benchmark_type` is used for selecting which benchmark you want to run. Current availiable benchmarks:

- `atomworld`:    AtomWorld  
- `pointworld`:   PointWorld
- `cifgen`:       CIFGen
- `cifrepair`:    CIFRepair

`model_name` is the model you want to test. For example, `deepseek_reasoner`

`action_name` is the action to test. This argument is only for **AtomWorld** and **PointWorld**. Please ignore this for running other benchmarks.

Current avaliable actions:

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

For **PointWorld**, 4 actions are implemented:

- move
- move_towards
- insert_between
- rotate_around

`batch_size` is the number of parallel LLM calls. Default 50

`num_batch` is the number of batches for test. Default is the whole dataset

### Analyze the results

Please run the `./src/scripts/analyze_results.py` to analyze the AtomWorld bench results


### How to construct your own data with mp-api

Firstly, run the `src/scripts/download_random_mp_data.py` to obtain random structures from the Materials Project. You can DIY the code to selectively download desired structures according to the mp-api's document.

Then move to the folder where your data is downloaded/stored. Run `src/atom_world/data_generator.py` with your own settings. You will get a csv and a folder of output_cifs.

Finally, collect the folder into a single h5 file using the code `src/scripts/convert_cifs_to_h5.py`. Remember to set the file dir and the chosen actions.

