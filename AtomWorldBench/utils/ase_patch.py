# Monkey patch for ASE Atoms equality comparison to include positions, cell, and pbc.
# This is necessary because the default __eq__ method in ASE Atoms only compares with array_a == array_b,
# which does not work for floating point arrays like positions and cell.
import numpy as np
from ase import Atoms

def patched_atoms_eq(self, other):
    if not isinstance(other, Atoms):
        return False
    a = self.arrays
    b = other.arrays
    return (
        len(self) == len(other) and
        np.allclose(a['positions'], b['positions']) and
        np.array_equal(a['numbers'], b['numbers']) and
        np.allclose(self.cell, other.cell) and
        np.array_equal(self.pbc, other.pbc)
    )

# Apply monkey patch.
Atoms.__eq__ = patched_atoms_eq