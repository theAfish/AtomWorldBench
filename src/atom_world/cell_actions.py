from atom_world.actions import BaseAction
import numpy as np
import random
from typing import Optional, List, Tuple, Union

class CellAction(BaseAction):
    """Base class for actions that modify the simulation cell."""
    
    def __init__(self):
        super().__init__()



class SuperCellAction(CellAction):
    """Action to create a supercell by scaling the lattice vectors."""
    
    def __init__(
            self, 
            supercell_size: Union[Tuple[int, int, int], List[int], None] = None
        ):
        """
        Initialize the SuperCellAction with scale factors for each lattice vector.
        
        Args:
            supercell_size (tuple of int): Scaling factors for (a, b, c) lattice vectors.
        """
        if supercell_size is None:
            self.random_initialize()
        elif isinstance(supercell_size, list):
            if len(supercell_size) != 3:
                raise ValueError("supercell_size list must have exactly three elements.")
            self.supercell_size = tuple(supercell_size)
        elif isinstance(supercell_size, tuple):
            if len(supercell_size) != 3:
                raise ValueError("supercell_size tuple must have exactly three elements.")
            self.supercell_size = supercell_size

    def random_initialize(self):
        """Randomly initialize supercell size with integers between 2 and 3."""
        while True:
        # Generate three random integers between 1 and 4 (inclusive)
            n1 = random.randint(1, 4)
            n2 = random.randint(1, 4)
            n3 = random.randint(1, 4)

            # Check the conditions
            product_ok = (n1 * n2 * n3) <= 8
            not_all_ones = not (n1 == 1 and n2 == 1 and n3 == 1)

            # If all conditions are met, return the list
            if product_ok and not_all_ones:
                self.supercell_size = (n1, n2, n3)
                break

    def execute(self, atoms):
        """Apply the supercell transformation to the atoms."""
        atoms = atoms * self.supercell_size
        return atoms
    
    def __str__(self):
        multiply_symbols = ['×', 'x', '*', 'X', ' by ']
        mul_symbol = random.choice(multiply_symbols)
        return f"Create a supercell with the size {self.supercell_size[0]}{mul_symbol}{self.supercell_size[1]}{mul_symbol}{self.supercell_size[2]}."

    @classmethod
    def randomize(cls, atoms=None, rng=None, config=None):
        """Sample random supercell parameters."""
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
        action = cls(**params)
        result = action.execute(target)
        return action, result
    

# Example usage:
if __name__ == "__main__":
    from ase import Atoms
    from ase.visualize import view

    # Create a simple cubic structure
    a = 3.5  # Lattice constant
    atoms = Atoms('NaCl', positions=[(0, 0, 0), (a/2, a/2, a/2)],
                  cell=[a, a, a], pbc=True)

    # Visualize the original structure
    print("Original structure:")
    view(atoms)

    # Apply the SuperCellAction to create a 2x2x2 supercell
    action = SuperCellAction(supercell_size=None)
    supercell_atoms = action.execute(atoms)

    # Visualize the supercell structure
    print("Supercell structure:")
    view(supercell_atoms)

    print(action)  # Print the action description