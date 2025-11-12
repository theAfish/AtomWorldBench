"""Version 2 of actions: decoupled from Atoms in __init__, provides randomize/execute/apply_random.

Contract:
 - randomize(atoms, rng=None) -> dict: sample parameters for this action based on the given atoms.
 - execute(atoms, /, **params) -> Atoms: apply action (mutates provided atoms) and return it.
 - apply_random(atoms, rng=None, copy=True) -> Atoms: convenience: randomize + execute on a copy by default.

This file implements the same actions as the legacy `actions.py` but as a separate module (for safe migration).
"""
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
import random
from typing import Optional, Dict, Any, List


# Reasonable default hyperparameters for random sampling. Callers can override by
# passing a `config` dict to any action's constructor; the action will merge it
# with these defaults.
DEFAULT_CONFIG: Dict[str, Any] = {
    # radii used by "around"/"delete around" actions (angstrom)
    "radius_min": 1.0,
    "radius_max": 3.0,
    # distance used by MoveTowards / general distances (angstrom)
    "distance_min": 0.2,
    "distance_max": 2.0,
    # displacement scale for random d_pos generation (angstrom)
    "dpos_scale": 0.5,
    # distance ratio when inserting between atoms (fraction along the vector)
    "distance_ratio_min": 0.2,
    "distance_ratio_max": 0.8,
    # angle range in degrees for rotations
    "angle_min": 15.0,
    "angle_max": 345.0,
    # probability of including the center atom in DeleteBelow
    "include_self_prob": 0.5,
    # for MoveSelected: select up to this fraction of atoms (at least 1)
    "move_selected_max_fraction": 0.3,
    # symbol pool (exclude the placeholder at index 0)
    "symbol_pool": list(chemical_symbols[1:]),
    # number of decimal places to round floating point values
    "decimal_places": 3,
}


class BaseActionV2:
    """Base interface for actions (v2).

    Notes:
    - Implementations should not modify `atoms` in `randomize`.
    - `execute` may mutate the supplied `atoms` instance. Callers who want safety should pass `atoms.copy()`.
    - Implementations should set `self._last_params` inside `randomize` (or `execute`) so `__str__` can report them.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Merge provided config with defaults (shallow merge)
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config: Dict[str, Any] = cfg
        self._last_params: Optional[Dict[str, Any]] = None

    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        raise NotImplementedError()

    def apply_random(self, atoms: Atoms, rng: Optional[np.random.Generator] = None, copy: bool = True) -> Atoms:
        rng = rng if rng is not None else np.random.default_rng()
        params = self.randomize(atoms, rng=rng)
        self._last_params = params
        target = atoms.copy() if copy else atoms
        return self.execute(target, **params)

    def __str__(self):
        if self._last_params is None:
            return f"{self.__class__.__name__} (not randomized)"
        return f"{self.__class__.__name__} with params={self._last_params}"


# Concrete actions (single atom)
class AddAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        rng = rng if rng is not None else np.random.default_rng()
        symbol = rng.choice(self.config.get('symbol_pool', chemical_symbols[1:]))
        frac = rng.random(3)
        position = np.dot(frac, atoms.get_cell())
        position = np.round(position, decimals=self.config.get('decimal_places', 3))
        params = {"symbol": symbol, "position": position}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        symbol = params["symbol"]
        position = params["position"]
        new_atom = Atoms(symbols=symbol, positions=[position])
        atoms += new_atom
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        symbol = self._last_params["symbol"]
        position = self._last_params["position"]
        return f"Add one {symbol} atom at the Cartesian coordinate {position} to the cif file."


class RemoveAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms to remove")
        rng = rng if rng is not None else np.random.default_rng()
        index = int(rng.integers(0, len(atoms)))
        params = {"index": index}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        index = params["index"]
        if not (0 <= index < len(atoms)):
            raise IndexError("Index out of bounds for atom removal.")
        del atoms[index]
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        return f"Remove the atom at index {index} from the cif file."


class MoveAtomActionV2(BaseActionV2):
    def __init__(self, dpos_scale: float = 2.0):
        super().__init__()
        self.dpos_scale = dpos_scale

    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms to move")
        rng = rng if rng is not None else np.random.default_rng()
        index = int(rng.integers(0, len(atoms)))
        dpos_scale = float(self.config.get('dpos_scale', self.dpos_scale))
        d_pos = rng.normal(scale=dpos_scale, size=3)
        d_pos = np.round(d_pos, decimals=self.config.get('decimal_places', 3))
        params = {"index": index, "d_pos": d_pos}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        index = params["index"]
        d_pos = params["d_pos"]
        if not (0 <= index < len(atoms)):
            raise IndexError("Index out of bounds for atom movement.")
        atoms[index].position += d_pos
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        d_pos = self._last_params["d_pos"]
        return f"Move the atom at index {index} by {d_pos} angstrom in the cif file. "


class ChangeAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms to change")
        rng = rng if rng is not None else np.random.default_rng()
        index = int(rng.integers(0, len(atoms)))
        # the symbol should be different from the current one
        current_symbol = atoms[index].symbol
        symbol_pool = [s for s in self.config.get('symbol_pool', chemical_symbols[1:]) if s != current_symbol]
        symbol = rng.choice(symbol_pool)
        params = {"index": index, "symbol": symbol}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        index = params["index"]
        symbol = params["symbol"]
        if not (0 <= index < len(atoms)):
            raise IndexError("Index out of bounds for atom modification.")
        atoms[index].symbol = symbol
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        symbol = self._last_params["symbol"]
        return f"Change the atom at index {index} into {symbol} in the cif file."


# double-atom actions
class SwapAtomsActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to swap")
        rng = rng if rng is not None else np.random.default_rng()
        max_tries = 20
        for _ in range(max_tries):
            idx1 = int(rng.integers(0, len(atoms)))
            idx2 = int(rng.integers(0, len(atoms)))
            if idx1 != idx2 and atoms[idx1].symbol != atoms[idx2].symbol:
                params = {"index1": idx1, "index2": idx2}
                self._last_params = params
                return params
        raise ValueError("Could not find two atoms of different types to swap.")

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        i1, i2 = params["index1"], params["index2"]
        if not ((0 <= i1 < len(atoms)) and (0 <= i2 < len(atoms))):
            raise IndexError("Index out of bounds for atom swapping.")
        atoms[i1].symbol, atoms[i2].symbol = atoms[i2].symbol, atoms[i1].symbol
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index1 = self._last_params["index1"]
        index2 = self._last_params["index2"]
        return f"Swap atoms at indices {index1} and {index2} in the cif file."


class InsertBetweenAtomsActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to insert between")
        rng = rng if rng is not None else np.random.default_rng()
        idx1 = int(rng.integers(0, len(atoms)))
        idx2 = int(rng.integers(0, len(atoms)))
        while idx2 == idx1:
            idx2 = int(rng.integers(0, len(atoms)))
        symbol = rng.choice(self.config.get('symbol_pool', chemical_symbols[1:]))
        dr_min = float(self.config.get('distance_ratio_min', 0.2))
        dr_max = float(self.config.get('distance_ratio_max', 0.8))
        distance_ratio = float(rng.uniform(dr_min, dr_max))
        distance_ratio = round(distance_ratio, self.config.get('decimal_places', 3))
        params = {"index1": idx1, "index2": idx2, "symbol": symbol, "distance_ratio": distance_ratio}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        i1, i2 = params["index1"], params["index2"]
        if not ((0 <= i1 < len(atoms)) and (0 <= i2 < len(atoms))):
            raise IndexError("Index out of bounds for inserting atom between two atoms.")
        rel = atoms[i2].position - atoms[i1].position
        position = atoms[i1].position + rel * params["distance_ratio"]
        new_atom = Atoms(symbols=params["symbol"], positions=[position])
        atoms += new_atom
        atoms.set_pbc(atoms.get_pbc())
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index1 = self._last_params["index1"]
        index2 = self._last_params["index2"]
        symbol = self._last_params["symbol"]
        distance_ratio = self._last_params["distance_ratio"]
        return f"Insert a {symbol} atom in the line between atoms at indices {index1} and {index2}, at {distance_ratio:.2f} fraction along the line in the cif file."


class MoveTowardsAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) < 2:
            raise ValueError("Not enough atoms for MoveTowardsAtomAction")
        rng = rng if rng is not None else np.random.default_rng()
        i1 = int(rng.integers(0, len(atoms)))
        i2 = int(rng.integers(0, len(atoms)))
        while i2 == i1:
            i2 = int(rng.integers(0, len(atoms)))
        dmin = float(self.config.get('distance_min', 0.1))
        dmax = float(self.config.get('distance_max', 3.0))
        distance = float(rng.uniform(dmin, dmax))
        distance = round(distance, self.config.get('decimal_places', 3))
        params = {"index1": i1, "index2": i2, "distance": distance}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        i1, i2 = params["index1"], params["index2"]
        dist = params["distance"]
        if not ((0 <= i1 < len(atoms)) and (0 <= i2 < len(atoms))):
            raise IndexError("Index out of bounds for moving towards another atom.")
        direction = atoms[i2].position - atoms[i1].position
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Cannot move towards an atom at the same position.")
        direction /= norm
        atoms[i1].position += direction * dist
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index1 = self._last_params["index1"]
        index2 = self._last_params["index2"]
        distance = self._last_params["distance"]
        return f"Move the atom at index {index1} towards the atom at index {index2} by {distance} angstrom in the cif file."


# multiple-atom actions (examples)
class DeleteBelowAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to delete below")
        rng = rng if rng is not None else np.random.default_rng()
        max_tries = 20
        for _ in range(max_tries):
            idx = int(rng.integers(0, len(atoms)))
            z_threshold = atoms[idx].position[2]
            indices_below = [i for i, atom in enumerate(atoms) if atom.position[2] < z_threshold and i != idx]
            if indices_below:
                include_self = bool(rng.choice([True, False]))
                params = {"index": idx, "include_self": include_self}
                self._last_params = params
                return params
        raise ValueError("Could not find an atom with atoms below it to delete.")

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        idx = params["index"]
        include_self = params.get("include_self", False)
        if not (0 <= idx < len(atoms)):
            raise IndexError("Index out of bounds for DeleteBelowAtomAction")
        z_threshold = atoms[idx].position[2]
        indices_to_delete = [i for i, atom in enumerate(atoms) if atom.position[2] < z_threshold]
        if not include_self:
            indices_to_delete = [i for i in indices_to_delete if i != idx]
        if not indices_to_delete:
            raise ValueError("No atoms below the specified atom to delete.")
        atoms = atoms[[i for i in range(len(atoms)) if i not in indices_to_delete]]
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        include_self = self._last_params.get("include_self", False)
        return f"Delete all atoms whose z coordinate is lower than the atom at index {index} in the cif file." + (" Including itself" if include_self else " Excluding itself") + " and atoms with the same z coordinate."


class DeleteAroundAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        idx = int(rng.integers(0, len(atoms)))
        rmin = float(self.config.get('radius_min', 1.0))
        rmax = float(self.config.get('radius_max', 4.0))
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, self.config.get('decimal_places', 3))
        params = {"index": idx, "radius": radius}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        idx = params["index"]
        radius = params["radius"]
        if not (0 <= idx < len(atoms)):
            raise IndexError("Index out of bounds for deleting around an atom.")
        distances = atoms.get_distances(idx, range(len(atoms)), mic=True)
        indices_to_delete = np.where(distances < radius)[0]
        if len(indices_to_delete) == 0:
            raise ValueError("No atoms found within the specified radius to delete.")
        atoms = atoms[[i for i in range(len(atoms)) if i not in indices_to_delete]]
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        radius = self._last_params["radius"]
        return f"Delete all atoms within {radius} angstrom around the atom at index {index} in the cif file."


class MoveSelectedAtomsActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        frac = float(self.config.get('move_selected_max_fraction', 0.3))
        max_count = max(1, int(len(atoms) * frac))
        count = int(rng.integers(1, max_count + 1))
        indices = list(rng.choice(len(atoms), size=count, replace=False))
        dpos_scale = float(self.config.get('dpos_scale', 2.0))
        d_pos = rng.normal(scale=dpos_scale, size=3)
        d_pos = np.round(d_pos, decimals=self.config.get('decimal_places', 3))
        params = {"indices": indices, "d_pos": d_pos}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        indices = params["indices"]
        d_pos = params["d_pos"]
        for index in indices:
            if not (0 <= index < len(atoms)):
                raise IndexError(f"Index {index} out of bounds for atom movement.")
            atoms[index].position += d_pos
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        indices = [int(i) for i in self._last_params["indices"]]
        d_pos = self._last_params["d_pos"].tolist()
        return f"Move all the atoms at indices {indices} by {d_pos} angstrom in the cif file."


class MoveAroundAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        idx = int(rng.integers(0, len(atoms)))
        rmin = float(self.config.get('radius_min', 1.0))
        rmax = float(self.config.get('radius_max', 4.0))
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, self.config.get('decimal_places', 3))
        dpos_scale = float(self.config.get('dpos_scale', 2.0))
        d_pos = rng.normal(scale=dpos_scale, size=3)
        d_pos = np.round(d_pos, decimals=self.config.get('decimal_places', 3))
        params = {"index": idx, "radius": radius, "d_pos": d_pos}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        idx = params["index"]
        radius = params["radius"]
        d_pos = params["d_pos"]
        if not (0 <= idx < len(atoms)):
            raise IndexError("Index out of bounds for moving around an atom.")
        distances = atoms.get_distances(idx, range(len(atoms)), mic=True)
        indices_to_move = np.where(distances < radius)[0]
        for i in indices_to_move:
            atoms[i].position += d_pos
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        radius = self._last_params["radius"]
        d_pos = self._last_params["d_pos"]
        return f"Move all surrounding atoms within {radius} angstrom around the center atom at index {index} by {d_pos} angstrom in the cif file."


class RotateAroundAtomActionV2(BaseActionV2):
    def randomize(self, atoms: Atoms, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        idx = int(rng.integers(0, len(atoms)))
        rmin = float(self.config.get('radius_min', 1.0))
        rmax = float(self.config.get('radius_max', 4.0))
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, self.config.get('decimal_places', 3))
        amin = float(self.config.get('angle_min', 45.0))
        amax = float(self.config.get('angle_max', 315.0))
        angle = float(rng.uniform(amin, amax))
        angle = round(angle, self.config.get('decimal_places', 3))
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        axis = axes[int(rng.integers(0, len(axes)))]
        params = {"index": idx, "radius": radius, "angle": angle, "axis": axis}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        idx = params["index"]
        radius = params["radius"]
        angle = params["angle"]
        axis = params["axis"]
        if not (0 <= idx < len(atoms)):
            raise IndexError("Index out of bounds for rotating around an atom.")
        center_position = atoms[idx].position
        distances = atoms.get_distances(idx, range(len(atoms)), mic=True)
        indices_to_rotate = np.where(distances < radius)[0]
        indices_to_rotate = [i for i in indices_to_rotate if i != idx]
        if not indices_to_rotate:
            return atoms
        sub_atoms = atoms[indices_to_rotate]
        sub_atoms.rotate(angle, axis/np.linalg.norm(axis), center=center_position)
        for idx_local, sub_atom in zip(indices_to_rotate, sub_atoms):
            atoms[idx_local].position = sub_atom.position
        return atoms

    def __str__(self):
        if self._last_params is None:
            return super().__str__()
        index = self._last_params["index"]
        radius = self._last_params["radius"]
        angle = self._last_params["angle"]
        axis = self._last_params["axis"]
        return f"Rotate all surrounding atoms within {radius} angstrom of the center atom at index {index} by {angle} degree around the axis {axis} in the cif file. The rotation should following the right-hand rule."



