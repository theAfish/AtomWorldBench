"""Simple (index-based) move atom actions."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config, _safe_radius


class MoveAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, d_pos: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.d_pos = d_pos

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            self.atoms[self.index].position += self.d_pos
            return self.atoms
        raise IndexError("Index out of bounds for atom movement.")

    def __str__(self):
        return (
            f"Move the atom at index {self.index} by {self.d_pos}"
            " angstrom in the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to move")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        d_pos = rng.normal(scale=cfg["dpos_scale"], size=3)
        d_pos = np.round(d_pos, decimals=cfg["decimal_places"])
        return {"atoms": atoms, "index": index, "d_pos": d_pos}


class MoveTowardsAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int, distance: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.distance = distance

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            direction = self.atoms[self.index2].position - self.atoms[self.index1].position
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("Cannot move towards an atom at the same position.")
            direction /= norm
            self.atoms[self.index1].position += direction * self.distance
            return self.atoms
        raise IndexError("Index out of bounds for moving towards another atom.")

    def __str__(self):
        return (
            f"Move the atom at index {self.index1} towards the atom at index"
            f" {self.index2} by {self.distance} angstrom in the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms for MoveTowardsAtomAction")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        distance = float(rng.uniform(cfg["distance_min"], cfg["distance_max"]))
        distance = round(distance, cfg["decimal_places"])
        return {
            "atoms": atoms,
            "index1": int(idx1),
            "index2": int(idx2),
            "distance": distance,
        }


class MoveSelectedAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, indices: list[int], d_pos: np.ndarray):
        super().__init__(atoms)
        self.indices = indices
        self.d_pos = d_pos

    def execute(self):
        for index in self.indices:
            if 0 <= index < len(self.atoms):
                self.atoms[index].position += self.d_pos
            else:
                raise IndexError(f"Index {index} out of bounds for atom movement.")
        return self.atoms

    def __str__(self):
        return (
            f"Move atoms at indices {self.indices} by {self.d_pos}"
            " angstrom in the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        count = int(rng.integers(1, len(atoms) + 1))
        indices = list(rng.choice(len(atoms), size=count, replace=False))
        d_pos = rng.normal(scale=cfg["dpos_scale"], size=3)
        d_pos = np.round(d_pos, decimals=cfg["decimal_places"])
        return {
            "atoms": atoms,
            "indices": [int(i) for i in indices],
            "d_pos": d_pos,
        }


class MoveAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float, d_pos: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.radius = radius
        self.d_pos = d_pos

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            distances = self.atoms.get_distances(
                self.index, range(len(self.atoms)), mic=True
            )
            indices_to_move = np.where(distances < self.radius)[0]
            for i in indices_to_move:
                self.atoms[i].position += self.d_pos
            return self.atoms
        raise IndexError("Index out of bounds for moving around an atom.")

    def __str__(self):
        return (
            f"Move all surrounding atoms within {self.radius} angstrom around the"
            f" center atom at index {self.index} by {self.d_pos} angstrom in the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg["radius_min"], cfg["radius_max"])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg["decimal_places"])
        d_pos = rng.normal(scale=cfg["dpos_scale"], size=3)
        d_pos = np.round(d_pos, decimals=cfg["decimal_places"])
        return {"atoms": atoms, "index": index, "radius": radius, "d_pos": d_pos}


class MoveAllAction(BaseAction):
    def __init__(self, atoms: Atoms, d_pos: np.ndarray):
        super().__init__(atoms)
        self.d_pos = d_pos

    def execute(self):
        unwrapped_positions = self.atoms.get_positions(wrap=False)
        self.atoms.set_positions(unwrapped_positions + self.d_pos)
        return self.atoms

    def __str__(self):
        return (
            f"Move all the atoms in the structure by {self.d_pos} angstrom in the cif"
            " file. Do not need to consider periodic boundary conditions. Please keep"
            " the cell and the order of atoms unchanged."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        d_pos = rng.normal(scale=cfg["dpos_scale"], size=3)
        d_pos = np.round(d_pos, decimals=cfg["decimal_places"])
        return {"atoms": atoms, "d_pos": d_pos}
