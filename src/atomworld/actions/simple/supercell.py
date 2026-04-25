"""Simple supercell action (index-free, cell-level)."""
import numpy as np
import random
from typing import List, Tuple, Union

from .base import BaseAction


class SuperCellAction(BaseAction):
    """Action to create a supercell by scaling the lattice vectors."""

    def __init__(self, atoms=None, supercell_size: Union[Tuple[int, int, int], List[int], None] = None):
        if atoms is not None:
            super().__init__(atoms)
        if supercell_size is None:
            self._random_initialize()
        elif isinstance(supercell_size, (list, tuple)):
            if len(supercell_size) != 3:
                raise ValueError("supercell_size must have exactly three elements.")
            self.supercell_size = tuple(supercell_size)
        else:
            raise TypeError("supercell_size must be a tuple or list of three integers.")

    def _random_initialize(self):
        while True:
            n1, n2, n3 = random.randint(1, 4), random.randint(1, 4), random.randint(1, 4)
            if n1 * n2 * n3 <= 8 and not (n1 == 1 and n2 == 1 and n3 == 1):
                self.supercell_size = (n1, n2, n3)
                break

    def execute(self):
        self.atoms = self.atoms * self.supercell_size
        return self.atoms

    def __str__(self):
        mul = random.choice(["x", "*", "X", " by "])
        s = self.supercell_size
        return f"Create a supercell with the size {s[0]}{mul}{s[1]}{mul}{s[2]}."

    @classmethod
    def randomize(cls, atoms=None, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(50):
            n1 = int(rng.integers(1, 5))
            n2 = int(rng.integers(1, 5))
            n3 = int(rng.integers(1, 5))
            if n1 * n2 * n3 <= 8 and not (n1 == 1 and n2 == 1 and n3 == 1):
                return {"supercell_size": (n1, n2, n3)}
        return {"supercell_size": (2, 1, 1)}

    @classmethod
    def apply_random(cls, atoms, rng=None, config=None, copy=True):
        rng = rng if rng is not None else np.random.default_rng()
        target = atoms.copy() if copy else atoms
        params = cls.randomize(target, rng=rng, config=config)
        action = cls(atoms=target, **params)
        result = action.execute()
        return action, result
