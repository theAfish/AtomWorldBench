# CLI

```bash
atomworld [generate|benchmark|eval|draw] [options]
```

## Quick examples

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

## Benchmark arguments

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
| `--agent_cli`             | Shell command for an external agent (enables agent mode) |
| `--timeout`               | Per-task timeout in seconds for agent mode (default: 120) |

## Available actions

**AtomWorld:** `add_atom_action`, `change_atom_action`, `delete_around_atom_action`, `delete_below_atom_action`, `insert_between_atoms_action`, `move_around_atom_action`, `move_atom_action`, `move_selected_atoms_action`, `move_towards_atom_action`, `remove_atom_action`, `rotate_around_atom_action`, `swap_atoms_action`, `super_cell_action`, `rotate_whole_action`, `move_all_action`

**PointWorld:** `move`, `move_towards`, `insert_between`, `rotate_around`

## Adding your own model

Implement your model class in `src/models/` and add its config to `config/models.yaml`. Built-in backends: OpenAI, Azure OpenAI, HuggingFace, vLLM.