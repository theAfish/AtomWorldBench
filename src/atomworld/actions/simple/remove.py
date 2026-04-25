"""Simple (index-based) remove / delete atom actions."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config, _safe_radius


class RemoveAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int):
        super().__init__(atoms)
        self.index = index

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            del self.atoms[self.index]
            return self.atoms
        raise IndexError("Index out of bounds for atom removal.")

    def __str__(self):
        return (
            f"Remove the atom at index {self.index} from the cif file."
            " The indices of atoms are started from 0."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to remove")
        rng = rng if rng is not None else np.random.default_rng()
        index = int(rng.integers(0, len(atoms)))
        return {"atoms": atoms, "index": index}


class DeleteBelowAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, include_self: bool = False):
        super().__init__(atoms)
        self.index = index
        self.include_self = include_self

    def execute(self):
        z_threshold = self.atoms[self.index].position[2]
        indices_to_delete = [
            i for i, atom in enumerate(self.atoms)
            if atom.position[2] < z_threshold
        ]
        if not self.include_self:
            indices_to_delete = [i for i in indices_to_delete if i != self.index]
        if indices_to_delete:
            self.atoms = self.atoms[
                [i for i in range(len(self.atoms)) if i not in indices_to_delete]
            ]
            return self.atoms
        raise ValueError("No atoms below the specified atom to delete.")

    def __str__(self):
        suffix = (
            " Including itself" if self.include_self else " Excluding itself"
        ) + " and atoms with the same z coordinate."
        return (
            f"Delete all atoms whose z coordinate is lower than the atom at"
            f" index {self.index} in the cif file.{suffix}"
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            return None
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(20):
            idx = int(rng.integers(0, len(atoms)))
            z_threshold = atoms[idx].position[2]
            indices_below = [
                i for i, a in enumerate(atoms)
                if a.position[2] < z_threshold and i != idx
            ]
            if indices_below:
                include_self = bool(rng.random() < 0.5)
                return {"atoms": atoms, "index": idx, "include_self": include_self}
        return None


class DeleteAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float):
        super().__init__(atoms)
        self.index = index
        self.radius = radius

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            distances = self.atoms.get_distances(
                self.index, range(len(self.atoms)), mic=True
            )
            indices_to_delete = np.where(distances < self.radius)[0]
            if len(indices_to_delete) > 0:
                self.atoms = self.atoms[
                    [i for i in range(len(self.atoms)) if i not in indices_to_delete]
                ]
                return self.atoms
            raise ValueError("No atoms found within the specified radius to delete.")
        raise IndexError("Index out of bounds for deleting around an atom.")

    def __str__(self):
        return (
            f"Delete all atoms within {self.radius} angstrom around the atom"
            f" at index {self.index} in the cif file."
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
        return {"atoms": atoms, "index": index, "radius": radius}
