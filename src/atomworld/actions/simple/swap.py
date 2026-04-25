"""Simple (index-based) swap and insert atom actions."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config


class SwapAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            self.atoms[self.index1].symbol, self.atoms[self.index2].symbol = (
                self.atoms[self.index2].symbol,
                self.atoms[self.index1].symbol,
            )
            return self.atoms
        raise IndexError("Index out of bounds for atom swapping.")

    def __str__(self):
        return (
            f"Swap the spatial positions of atoms at indices {self.index1} and"
            f" {self.index2} in the cif file. The indices of atoms are started from 0."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to swap")
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(20):
            idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
            if atoms[int(idx1)].symbol != atoms[int(idx2)].symbol:
                return {"atoms": atoms, "index1": int(idx1), "index2": int(idx2)}
        raise ValueError("Could not find two atoms of different types to swap.")


class InsertBetweenAtomsAction(BaseAction):
    def __init__(
        self,
        atoms: Atoms,
        index1: int,
        index2: int,
        symbol: str,
        distance_ratio: float,
    ):
        super().__init__(atoms)
        self.index1 = index1 % len(atoms)
        self.index2 = index2 % len(atoms)
        self.symbol = symbol
        self.distance_ratio = distance_ratio
        self.distance = 0

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            relative_position = (
                self.atoms[self.index2].position - self.atoms[self.index1].position
            )
            self.distance = np.linalg.norm(relative_position) * self.distance_ratio
            position = (
                self.atoms[self.index1].position + relative_position * self.distance_ratio
            )
            new_atom = Atoms(symbols=self.symbol, positions=[position])
            self.atoms += new_atom
            self.atoms.set_pbc(self.atoms.get_pbc())
            return self.atoms
        raise IndexError("Index out of bounds for inserting atom between two atoms.")

    def __str__(self):
        return (
            f"Insert a {self.symbol} atom in the line between atoms at indices"
            f" {self.index1} and {self.index2}, and the inserted atom must be"
            f" {self.distance:.2f} angstrom from atom at {self.index1} in the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to insert between")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        symbol = str(rng.choice(cfg["symbol_pool"]))
        dr = float(rng.uniform(cfg["distance_ratio_min"], cfg["distance_ratio_max"]))
        dr = round(dr, cfg["decimal_places"])
        return {
            "atoms": atoms,
            "index1": int(idx1),
            "index2": int(idx2),
            "symbol": symbol,
            "distance_ratio": dr,
        }
