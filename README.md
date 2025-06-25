## Atom World Bench

Testing LLMs' ability on operating 3D atomic structures.


## Installation

```
pip install .
```

## Usage

# Run the benchmark

```
python ./src/scripts/run_benchmark.py -m [model_name] -a [action_name] -b [batch_size] -n [num_batch]
```

`model_name` is the model you want to test. For example, `deepseek_reasoner`

`action_name` is the action to test. Current avaliable actions:

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

`batch_size` is the number of parallel LLM calls. Default 50

`num_batch` is the number of batches for test.

# Analyze the results

Please run the `./src/scripts/analyze_results.py`

