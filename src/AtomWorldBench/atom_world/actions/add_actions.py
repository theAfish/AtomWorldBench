"""Actions to add atoms."""
from typing import List

from .base import BaseAction

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core import Structure, get_el_sp
from pymatgen.util.typing import SpeciesLike


class AddMultiAtomsAction(BaseAction):
    """Action to add an atom at a random position around the center atom."""

    def __init__(
            self,
            structure: Structure,
            added_species: List[int | SpeciesLike] | SpeciesLike | int,
            added_positions: ArrayLike,
            cartesian_mode: bool = False,
            atomic_number_mode: bool = False,
    ):
        """Initialize the action with a structure and symbols to add.

        Args:
            structure (pymatgen.core.structure.Structure): The structure to be modified.
            added_species (List[int | SpeciesLike] | SpeciesLike | int): List of chemical symbols
             or atomic numbers (or a single symbol or atomic number) for the atoms to be added.
             Must be valid chemical elements, species or dummy symbols.
            added_positions (ArrayLike): Array of positions (or a single position) where
             the atoms will be added. Cartesian or fractional coordinates can be used.
            cartesian_mode (bool): If True, positions are interpreted as Cartesian coordinates.
            atomic_number_mode (bool): If True, added species are interpreted as atomic numbers.
        """
        super().__init__(structure=structure)
        # Jump to the setters to ensure species and positions are set correctly.
        self.added_species = added_species
        self.added_positions = added_positions
        self.cartesian_mode = cartesian_mode
        self.atomic_number_mode = atomic_number_mode

    @property
    def added_species(self) -> List[int | SpeciesLike]:
        """Get the species of atoms to be added."""
        return self._added_species

    @property
    def added_positions(self) -> ArrayLike:
        """Get the positions where the atoms will be added."""
        if self._added_positions is None:
            raise ValueError("Positions have not been set.")
        return self._added_positions

    @property
    def added_atomic_numbers(self) -> List[int]:
        """Get the atomic numbers of the species to be added."""
        return [sp.Z for sp in self.added_species]

    @added_species.setter
    def added_species(self, species: List[int | SpeciesLike] | SpeciesLike | int):
        """Set the species of atoms to be added."""
        if isinstance(species, SpeciesLike) or isinstance(species, int):
            species = [species]
        self._added_species = [get_el_sp(sp) for sp in species]
        self._added_positions = None  # Reset positions when species change.

    @added_positions.setter
    def added_positions(self, positions: ArrayLike):
        """Set the positions where the atoms will be added."""
        positions = np.array(positions)
        if positions.ndim > 2:
            raise ValueError("Positions must be a 1D or 2D array-like.")
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.shape[1] != 3:
            raise ValueError("Positions must have shape (n, 3) for 3D coordinates.")
        if len(positions) != len(self._added_species):
            raise ValueError("The number of species must match the number of positions.")
        self._added_positions = positions

    def _get_prompt(self) -> str:
        """Generate a prompt describing the action of adding atoms."""
        species_str = ', '.join([str(specie) for specie in self.added_species])
        # Keep positions to 4 decimal places to control context length.
        positions_str = ', '.join(
            [f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})" for pos in self.added_positions]
        )
        cartesian_word = "cartesian" if self.cartesian_mode else "fractional"
        species_word = "atomic numbers" if self.atomic_number_mode else "species"
        return (
            f"Add {len(self.added_species)} atom(s)"
            f" with {species_word} [{species_str}] at {cartesian_word}"
            f" positions [{positions_str}] to the structure."
        )

    def execute(self) -> Structure:
        """Execute the action to add atoms to the structure."""
        lattice = self.structure.lattice
        if self.cartesian_mode:
            added_frac_coords  = lattice.get_fractional_coords(self.added_positions)
        else:
            added_frac_coords = np.array(self.added_positions)
        frac_coords = np.concatenate(self.structure.frac_coords, added_frac_coords)
        species = self.structure.species + self.added_species
        return Structure(lattice, species, frac_coords, coords_are_cartesian=False)

class AddAtomAction(AddMultiAtomsAction):
    """Action to add a single atom at a random position around the center atom."""

    def __init__(
            self,
            structure: Structure,
            added_species: int | SpeciesLike,
            added_position: ArrayLike,
            cartesian_mode: bool = False,
            atomic_number_mode: bool = False,
    ):
        """Initialize the action with a structure and chemical symbol of atom to add.

        Args:
            structure (pymatgen.core.structure.Structure): The structure to be modified.
            added_species (int | SpeciesLike): Chemical symbol or atomic number of the atom to be added.
            added_position (ArrayLike): Position where the atom will be added.
            cartesian_mode (bool): If True, position is interpreted as Cartesian coordinates.
            atomic_number_mode (bool): If True, added species is interpreted as an atomic number.
        """
        super().__init__(
            structure=structure,
            added_species=[added_species],
            added_positions=[added_position],
            cartesian_mode=cartesian_mode,
            atomic_number_mode=atomic_number_mode
        )