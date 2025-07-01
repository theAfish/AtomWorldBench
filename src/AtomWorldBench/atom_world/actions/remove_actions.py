"""Actions to remove atoms."""
from numpy.typing import ArrayLike
from pymatgen.core import Structure

from .base import BaseAction


class RemoveMultiAtomsAction(BaseAction):
    """Action to remove multiple atoms from a structure."""

    def __init__(self, structure: Structure, indices: ArrayLike[int]):
        """Initialize the action with a structure and indices of atoms to remove.

        Args:
            structure (pymatgen.core.structure.Structure): The structure to be modified.
            indices (ArrayLike[int]): List of indices of atoms to be removed.
        """
        super().__init__(structure=structure)
        self.indices = indices

    @property
    def indices(self) -> ArrayLike[int]:
        """Get the indices of atoms to be removed."""
        return self._indices

    @indices.setter
    def indices(self, indices: ArrayLike[int]):
        """Set the indices of atoms to be removed."""
        if len(indices) > len(self.structure):
            raise ValueError("Indices cannot exceed the number of atoms in the structure.")
        self._indices = indices

    def execute(self) -> Structure:
        """Execute the action and return the modified structure."""
        new_structure = self.structure.copy()
        new_structure.remove_sites(self.indices)
        return new_structure