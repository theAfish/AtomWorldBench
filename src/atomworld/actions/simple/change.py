"""Simple (index-based) change element action."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config


class ChangeAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, symbol: str):
        super().__init__(atoms)
        self.index = index
        self.new_symbol = symbol

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            self.atoms[self.index].symbol = self.new_symbol
            return self.atoms
        raise IndexError("Index out of bounds for atom modification.")

    def __str__(self):
        return (
            f"Change the atom at index {self.index} into {self.new_symbol} in the cif"
            " file. The indices of atoms are started from 0."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to change")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        current_symbol = atoms[index].symbol
        pool = [s for s in cfg["symbol_pool"] if s != current_symbol]
        symbol = str(rng.choice(pool))
        return {"atoms": atoms, "index": index, "symbol": symbol}
