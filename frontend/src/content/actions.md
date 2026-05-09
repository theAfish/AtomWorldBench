# Actions

AtomWorld has two action families, both under `atomworld.actions`.

## Simple actions (index-based)

Lightweight, no extra dependencies. Each action addresses atoms by their index in the ASE `Atoms` list and exposes `apply_random()` for one-shot dataset generation.

```python
from ase.io import read
from atomworld.actions import AddAtomAction, ChangeAtomAction, SwapAtomsAction
import numpy as np

atoms = read("my_structure.cif")
rng = np.random.default_rng(42)

action, result = AddAtomAction.apply_random(atoms, rng=rng)
print(action.describe())  # natural-language prompt for the LLM
print(result)             # new Atoms object (ground truth)
```

### All simple action classes

| Class | Description |
|---|---|
| `AddAtomAction` | Insert a new atom at a random position |
| `RemoveAtomAction` | Remove a randomly chosen atom |
| `MoveAtomAction` | Translate one atom by a random vector |
| `MoveTowardsAtomAction` | Move an atom towards another atom |
| `MoveSelectedAtomsAction` | Move a subset of atoms |
| `MoveAroundAtomAction` | Move an atom in a shell around another atom |
| `MoveAllAction` | Translate all atoms uniformly |
| `ChangeAtomAction` | Change an atom's element |
| `SwapAtomsAction` | Swap two atoms |
| `InsertBetweenAtomsAction` | Insert an atom between two existing atoms |
| `RotateAroundAtomAction` | Rotate an atom around a pivot |
| `RotateWholeAction` | Rotate the whole structure |
| `DeleteBelowAtomAction` | Remove atoms below a threshold height |
| `DeleteAroundAtomAction` | Remove atoms within a radius |
| `SuperCellAction` | Build a supercell |

## Verbose actions (motif-based)

Richer, motif-centric API that produces more expressive natural-language descriptions. Requires `scipy`. You first describe *what* you want to operate on as a **motif**, then pass it to an action.

### Structure actions

Operate on the whole structure, no motif needed:

```python
from ase.io import read
from atomworld.actions.verbose import ChangeElementAction

atoms = read("my_structure.cif")

# Explicit construction
action = ChangeElementAction(operated_atoms=atoms, from_element="Fe", to_element="Co")
result = action.execute()        # new Atoms object
print(action.describe())         # "replace all atoms of element Fe with Co."

# Random construction
action = ChangeElementAction.get_random_one(atoms, seed=42)
result = action.execute()
print(action.describe())
```

#### All structure action classes

| Class | Description |
|---|---|
| `ChangeElementAction` | Replace or remove all atoms of an element |
| `LatticeTransformAction` | Apply a linear transformation to the lattice |
| `MakeSupercellAction` | Build a supercell |
| `RotateStructureAction` | Rotate the whole structure |

### Motif actions

First create a motif, then act on it:

```python
from ase.io import read
from atomworld.actions.verbose import SiteMotif, ClusterMotif, AddMotifAction, RemoveMotifAction

atoms = read("my_structure.cif")

# Detect a random existing atom as a SiteMotif, then remove it
site = SiteMotif.detect_random_one(atoms, seed=0)
action = RemoveMotifAction(operated_motif=site)
result = action.execute()
print(action.describe())

# Create an additive SiteMotif (new atom to insert) and add it at an absolute position
new_fe = SiteMotif.detect_random_one(atoms, seed=1, additive_mode=True,
                                      additive_mode_allowed_symbols=["Fe"])
action = AddMotifAction(
    operated_motif=new_fe,
    operated_atoms=atoms,
    at_position=[1.5, 2.0, 3.0],
)
result = action.execute()
print(action.describe())
```

#### All motif types

| Class | Description |
|---|---|
| `SiteMotif` | A single lattice site / atom |
| `ClusterMotif` | A group of atoms |
| `BondMotif` | A pair of bonded atoms |
| `SphereRegionMotif` | A spherical region in space |
| `BoxRegionMotif` | A box-shaped region in space |

#### All motif action classes

| Class | Description |
|---|---|
| `AddMotifAction` | Insert a motif into the structure |
| `RemoveMotifAction` | Delete the atoms of a motif |
| `ReplaceMotifAction` | Replace a motif with another |
| `TranslateMotifAction` | Translate the atoms of a motif |
| `RotateMotifAction` | Rotate a motif about its centroid |
| `SwapMotifAction` | Swap two motifs |
| `ResizeMotifAction` | Scale bond lengths within a motif |

!!! tip "CLI note"
    The `atomworld generate` command supports both action families. Use `--verbose` to generate datasets with the verbose (motif-based) action family. Without this flag it defaults to the simple (index-based) family.