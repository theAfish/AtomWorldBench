"""Simple (index-based) add atom action."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config


class AddAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, symbol: str, position: np.ndarray):
        super().__init__(atoms)
        self.symbol = symbol
        self.position = position

    def execute(self):
        new_atom = Atoms(symbols=self.symbol, positions=[self.position])
        self.atoms += new_atom
        return self.atoms

    def __str__(self):
        return (
            f"Add one {self.symbol} atom at the Cartesian coordinate"
            f" {self.position} to the cif file."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        symbol = str(rng.choice(cfg["symbol_pool"]))
        frac = rng.random(3)
        position = np.dot(frac, atoms.get_cell())
        position = np.round(position, decimals=cfg["decimal_places"])
        return {"atoms": atoms, "symbol": symbol, "position": position}
