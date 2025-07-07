"""Define base structure motifs in the atom world."""
from typing import List, Optional, TypeAlias, Dict, Tuple
import numpy as np
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

from pymatgen.util.typing import SpeciesLike
from pymatgen.core import Lattice

from ..motif_description_styles import description_style_factory


LatticeLike: TypeAlias = Lattice | ArrayLike | float

# TODO: the current implementation disregards periodicity of the lattice. Need to handle this!

class BaseMotif(ABC):
    """Base class for motifs in the atom world.

    A motif is defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.
    """
    # List of allowed actions that can be performed on this motif.
    allowed_actions = []
    # List of allowed description styles for this motif.
    allowed_description_styles = []

    def __init__(
            self,
            species: List[SpeciesLike],
            frac_coords: ArrayLike,
            lattice: LatticeLike,
            indices: Optional[List[int]] = None,
            name: Optional[str] = None,
    ):
        """Initialize the motif with a unique identifier.

        Args:
            species (List[SpeciesLike]): List of species that make up the motif.
            frac_coords (ArrayLike): Fractional coordinates of the motif atoms.
            lattice (LatticeLike): Lattice vectors of the structure to which this motif
             belongs. Must be a 2D array with shape (3, 3).
            indices (Optional[List[int]]):
             Indices of the atoms in the structure that correspond to this motif.
            name (Optional[str]): Optional name for the motif. Defaults to None.
             If provided, it will always overwrite automatically generated names
              based on species and coordinates. For example, if name = "a water molecule",
              then it won't be overwritten by the default name "a cluster of atoms/species H, O"
        """
        self.name = name
        self.species = species
        self.frac_coords = frac_coords
        self.lattice = lattice
        self.indices = indices

    @classmethod
    def from_structure_indices(
            cls,
            structure,
            indices: List[int],
            name: Optional[str] = None,
    ):
        """Create a BaseMotif from a structure and indices.

        Args:
            structure (Structure): The structure containing the atoms.
            indices (List[int]): Indices of the atoms in the structure that correspond to this motif.
            name (str): Optional name for the motif. Defaults to None.
              If provided, it will always overwrite automatically generated names
              based on species and coordinates. For example, if name = "a water molecule",
              then it won't be overwritten by the default name "a cluster of atoms/species H, O".

        Returns:
            ClusterMotif: An instance of ClusterMotif with the specified indices.
        """
        return cls(
            name=name,
            species=[structure[index].specie for index in indices],
            frac_coords=[structure[index].frac_coords for index in indices],
            lattice=structure.lattice,
            indices=indices,
        )

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
        self._cart_coords = None  # Reset Cartesian coordinates when fractional coordinates change.
        self._radius = None  # Reset radius when fractional coordinates change.
        self._edge_lengths = None  # Reset edge lengths when fractional coordinates change.

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
        # Compute and cache Cartesian coordinates if not already done.
        if self._cart_coords is None:
            self._cart_coords = self.lattice.get_cartesian_coords(self.frac_coords)
        return self._cart_coords

    @property
    def cart_centroid(self) -> ArrayLike:
        """Get the centroid of the motif in Cartesian coordinates."""
        return np.mean(self.cart_coords, axis=0)

    @property
    def frac_centroid(self) -> ArrayLike:
        """Get the centroid of the motif in Cartesian coordinates."""
        return self.lattice.get_fractional_coords(self.cart_centroid)

    @property
    def radius(self) -> float:
        """Calculate the radius of the motif based on its fractional coordinates.

        Radius is defined as the maximum distance from the centroid to any atom in the motif.
        """
        if self.frac_coords is None:
            raise ValueError("Fractional coordinates must be set before calculating radius.")
        if len(self.frac_coords) == 1:
            self._radius = 0.0  # Single atom motif has radius 0
        # Calculate the maximum distance from the centroid to any atom in the motif.
        if self._radius is not None:
            return self._radius
        distances = np.linalg.norm(self.cart_coords - self.cart_centroid, axis=1)
        self._radius = np.max(distances)
        return self._radius

    @property
    def edge_lengths(self) -> Dict[Tuple[int, int], float]:
        """Calculate the edge length of the motif based on its fractional coordinates.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary where keys are tuples of atom indices
            and values are the distances between those atoms.
        """
        if self.frac_coords is None:
            raise ValueError(
                "Fractional coordinates and lattice must be set before calculating"
                " edge lengths."
            )
        if self._edge_lengths is not None:
            return self._edge_lengths
        # Compute and cache edge lengths if not already done.
        edge_lengths = {}
        for i in range(len(self.cart_coords)):
            for j in range(i + 1, len(self.cart_coords)):
                dist = np.linalg.norm(
                    self.cart_coords[i] - self.cart_coords[j]
                )
                edge_lengths[(i, j)] = dist
        self._edge_lengths = edge_lengths
        return self._edge_lengths

    @property
    def indices(self) -> List[int]:
        """Get the indices of the atoms in the structure that correspond to this motif."""
        return self._indices

    @indices.setter
    def indices(self, indices: List[int]):
        """Set the indices of the atoms in the structure that correspond to this motif."""
        if self.species is None or self.frac_coords is None:
            raise ValueError("Species and fractional coordinates must be set before setting indices.")
        if isinstance(indices, (list, tuple, set)) and all(isinstance(i, int) for i in indices):
            self._indices = list(indices)
        else:
            raise TypeError("Indices must be a list of integers.")

    @property
    def name(self) -> str:
        """Get the name of the motif.

        Can be set to a custom name at initialization, but if not provided,
         will automatically generate a name based on motif type and species.
        For example, if name = "a water molecule", then it won't be overwritten
         by the automatically generated name "a cluster of atoms/species H, O".
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        """Set the name of the motif."""
        if name is None:
            # Automatically generate a name based on species and coordinates.
            self._name = self._get_default_name()
        else:
            self._name = name

    @abstractmethod
    def _get_default_name(self) -> str:
        """Generate a default name based on motif type, species and coordinates."""
        pass

    def describe(
            self,
            style: str = "coord",
            **kwargs
    ) -> str:
        """Return a string description of the cluster motif.

        Args:
            style (str): The style of description. Default is "coord".
            **kwargs: Additional keyword arguments for the description style.
                For example, `coord_style` and `precision` for coordinate descriptions.
                For other styles, refer to the specific description style documentation.
        Returns:
            str: A string description of the cluster motif.
        """
        style = style.lower()
        if style not in self.allowed_description_styles:
            raise ValueError(
                f"Description style '{style}' is not allowed for this motif. "
                f"Allowed styles: {self.allowed_description_styles}."
            )
        description_style = description_style_factory(
            style, **kwargs
        )
        return description_style.describe(self)

    def __len__(self):
        """Return the number of atoms in the motif."""
        return len(self.species)