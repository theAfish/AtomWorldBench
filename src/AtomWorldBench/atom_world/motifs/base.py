"""Define base structure motifs in the atom world."""
from typing import List, Tuple, Set, Optional, TypeAlias, Sequence
import numpy as np
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

from pymatgen.util.typing import SpeciesLike
from pymatgen.core import Lattice


LatticeLike: TypeAlias = Lattice | ArrayLike | float


class BaseMotif(ABC):
    """Base class for motifs in the atom world.

    A motif is defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.
    """
    # List of allowed actions that can be performed on this motif.
    allowed_actions = []
    # List of allowed detectors that can be used to find this motif in structures.
    allowed_detectors = []
    def __init__(
            self,
            name: str,
            species: List[SpeciesLike],
            frac_coords: ArrayLike,
            lattice: LatticeLike,
            indices: Optional[List[int]] = None,
    ):
        """Initialize the motif with a unique identifier.

        Typically, will not be used directly, but rather through subclass class methods
         that implement specific motifs.
        Args:
            name (str): The name of the motif. For example, "(PO4)3-"
            species (List[SpeciesLike]): List of species that make up the motif.
            frac_coords (ArrayLike): Fractional coordinates of the motif atoms.
            lattice (LatticeLike): Lattice vectors of the structure to which this motif
             belongs. Must be a 2D array with shape (3, 3).
            indices (Optional[List[int]]):
             Indices of the atoms in the structure that correspond to this motif.
        """
        self.name = name
        self.species = species
        self.frac_coords = frac_coords
        self.lattice = lattice
        self.indices = indices
        self.center = np.mean(self.frac_coords, axis=0)

    @property
    def species(self) -> List[SpeciesLike]:
        """Get the species of the motif."""
        return self._species

    @species.setter
    def species(self, species: List[SpeciesLike]):
        """Set the species of the motif."""
        if not isinstance(species, list) or not all(isinstance(sp, SpeciesLike) for sp in species):
            raise TypeError("Species must be a list of SpeciesLike objects.")
        self._species = species
        self._frac_coords = None  # Reset fractional coordinates when species change
        self._indices = None

    @property
    def frac_coords(self) -> ArrayLike:
        """Get the fractional coordinates of the motif."""
        if self._frac_coords is None:
            raise ValueError("Fractional coordinates have not been set.")
        return self._frac_coords

    @frac_coords.setter
    def frac_coords(self, frac_coords: ArrayLike):
        """Set the fractional coordinates of the motif."""
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim != 2 or frac_coords.shape[1] != 3:
            raise ValueError("Fractional coordinates must be a 2D array with shape (n, 3).")
        if self.species is None:
            raise ValueError("Species must be set before setting fractional coordinates.")
        if len(frac_coords) != len(self.species):
            raise ValueError("The number of fractional coordinates must match the number of species.")
        self._frac_coords = np.array(frac_coords)

    @property
    def lattice(self) -> ArrayLike:
        """Get the lattice vectors of the structure to which this motif belongs."""
        return self._lattice

    @lattice.setter
    def lattice(self, lattice: LatticeLike):
        """Set the lattice vectors of the structure to which this motif belongs.

        Args:
            lattice (LatticeLike): Lattice vectors of the structure.
             Can be a 2D array with shape (3, 3), a 1D array with shape (3,),
               a Lattice object, or a single float value representing a cubic lattice.
        """
        if not isinstance(lattice, Lattice):
            if isinstance(lattice, float):
                lattice = Lattice.cubic(lattice)
            else:
                lattice = Lattice(lattice)
        self._lattice = lattice

    @property
    def cart_coords(self) -> ArrayLike:
        """Get the Cartesian coordinates of the motif."""
        if self._frac_coords is None or self._lattice is None:
            raise ValueError(
                "Fractional coordinates and lattice must be set"
                " before getting Cartesian coordinates."
            )
        return self.lattice.get_cartesian_coords(self.frac_coords)

    @property
    def indices(self) -> List[int]:
        """Get the indices of the atoms in the structure that correspond to this motif."""
        return self._indices

    @indices.setter
    def indices(self, indices: Sequence[int]):
        """Set the indices of the atoms in the structure that correspond to this motif."""
        if self.species is None or self.frac_coords is None:
            raise ValueError("Species and fractional coordinates must be set before setting indices.")
        if isinstance(indices, (List, Tuple, Set)) and all(isinstance(i, int) for i in indices):
            self._indices = list(indices)
        else:
            raise TypeError("Indices must be a list of integers.")

    @abstractmethod
    def describe(self) -> str:
        """Return a description of the motif."""
        pass