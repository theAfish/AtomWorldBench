# Python API

## Evaluate

The core `evaluate` function works with just the lightweight install — ideal for RL reward functions:

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

### Use as an RL reward function

```python
from atomworld import evaluate

def score(prompts, completions, *, solution, **kwargs):
    rewards = []
    for sol, comp in zip(solution, completions):
        r = evaluate(target_cif=sol, generated_output=comp)
        rewards.append(1.0 if r.correct else 0.0)
    return rewards
```

## Data loading

```python
from atomworld import load_data

df = load_data("./path/to/dataset", action_name="add_atom_action")
# DataFrame with columns: input_cif, action_prompt, output_cif
```

## Run benchmark from Python

```python
from atomworld import BenchmarkRunner
```