"""Define base structure motifs in the atom world.

Motifs are inherited from ase.Atoms.
"""
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike
from ase import Atoms

from .utils import get_species_string
from ..motif_description_styles import description_style_factory


class BaseMotif(ABC, Atoms):
    """Base class for motifs in the atom world.

    A motif is defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.

    Notice: all fractional coordinates must be unwrapped, i.e., not confined to
     the range [0, 1).
    """
    # List of allowed actions that can be performed on this motif.
    allowed_actions = []
    # List of allowed description styles for this motif.
    allowed_description_styles = []

    def __init__(
            self,
            *args,
            name: Optional[str] = None,
            indices: Optional[List[int]] = None,
            **kwargs
    ):
        """A Motif is an ASE Atoms comprising a subset of atoms in original ase.Atoms.

        Args:
            *args, **kwargs: See `ase.Atoms.__init__`_.
             .. _ase.Atoms.__init__: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            indices (list of int, optional): Original indices from structure.
                Indices should always be provided, if the motif belongs to a specific structure.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.indices = indices

    @property
    def species_strings(self) -> List[str]:
        """Get the species of the motif."""
        return [
            get_species_string(el, c)
            for el, c in
            zip(
                self.get_chemical_symbols(),
                self.get_initial_charges()
            )
        ]

    @property
    def frac_coords(self):
        """Get the unwrapped fractional coordinates of the motif."""
        return self.get_scaled_positions(wrap=False)

    @property
    def cell_offsets(self) -> ArrayLike:
        """Get the cell offset for each atom in the cluster.

        cell offsets are the integer part of the fractional coordinates in the form of
         triplets (i, j, k).
        Returns:
            np.ndarray: A 2D array of shape (n, 3) representing the box positions.
        """
        return np.floor(self.frac_coords).astype(int)

    @property
    def cart_coords(self) -> ArrayLike:
        """Get the unwrapped Cartesian coordinates of the motif."""
        return self.get_positions(wrap=False)

    def get_centroid(self, fractional=False) -> ArrayLike:
        """Get the centroid of the motif.

        Args:
            fractional (bool): If True, return the centroid in fractional coordinates.
                If False, return in Cartesian coordinates. Default is False.
        """
        if fractional:
            return np.mean(self.frac_coords, axis=0)
        else:
            return np.mean(self.cart_coords, axis=0)

    @property
    def radius(self) -> float:
        """Calculate the radius of the motif based on its fractional coordinates.

        Radius is defined as the maximum distance from the centroid to any atom in the motif.
        """
        if len(self) == 1:
            return 0.0  # Single atom motif has radius 0
        # Calculate the maximum distance from the centroid to any atom in the motif.
        distances = np.linalg.norm(
            self.cart_coords - self.get_centroid(fractional=False),
            axis=1
        )
        return np.max(distances)

    @cached_property
    def edge_lengths(self) -> Dict[Tuple[int, int], float]:
        """Calculate the edge length of the motif based on its fractional coordinates.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary where keys are tuples of atom indices
            and values are the distances between those atoms.
        """
        edge_lengths = {}
        for i in range(len(self.cart_coords)):
            for j in range(i + 1, len(self.cart_coords)):
                dist = np.linalg.norm(
                    self.cart_coords[i] - self.cart_coords[j]
                )
                edge_lengths[(i, j)] = dist
        return edge_lengths

    @property
    def indices(self) -> ArrayLike:
        """Get the indices of the atoms in the structure that correspond to this motif."""
        if self.arrays.get("site_indices") is None:
            return None
        return self.get_array("site_indices")

    @indices.setter
    def indices(self, indices: Optional[ArrayLike] = None):
        """Set the indices of the atoms in the structure that correspond to this motif."""
        # If None, clear the existing indices. Already implemented in ASE.
        self.set_array("site_indices", indices, dtype=int)

    @property
    def name(self):
        """Set the name of the motif."""
        return self.info["motif_name"]

    @name.setter
    def name(self, name: Optional[str] = None):
        """Set the name of the motif.

        Args:
            name (str, optional): The name of the motif. If None, a default name will be generated.
        """
        self.info["motif_name"] = name if name is not None else self._get_default_name()

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

    @classmethod
    def from_atoms(
            cls,
            atoms: Atoms,
            name: Optional[str] = None,
            indices: Optional[List[int]] = None,
    ):
        """Create a BaseMotif from an ASE Atoms object.

        Args:
            atoms (Atoms): The ASE Atoms object to create the motif from.
            name (Optional[str]): Name of the motif. If None, a default name will be generated.
            indices (Optional[List[int]]): Indices of the atoms in the structure that correspond to this motif.
                Indices should always be provided, if the motif belongs to a specific structure.

        Returns:
            BaseMotif: An instance of BaseMotif with the specified atoms and indices.
        """
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(wrap=False),
            cell=atoms.get_cell(complete=True),
            pbc=atoms.get_cell(complete=True),
            charges=atoms.get_initial_charges(),
            name=name,
            indices=indices
        )

    def extend(self, other):
        """Extend the motif with another motif or ASE Atoms object.

        Args:
            other (BaseMotif or Atoms): The motif or ASE Atoms object to extend this motif with.
        Returns:
            BaseMotif: A new instance of BaseMotif that combines this motif and the other.
        """
        if (self.indices is None and other.indices is not None) or \
           (self.indices is not None and other.indices is None):
            raise ValueError("Both motifs must have indices set or not set to extend them.")
        super().extend(other)
        self.name = None  # Reset name to default to avoid conflicts with the original motif name.

    def __getitem__(self, i):
        """Return a subset of the motif."""
        atoms = super().__getitem__(i)
        atoms.name = None  # Reset name to default to avoid conflicts with the original motif name.'
        return atoms

    def __imul__(self, m):
        """Repeat the motif by a given factor."""
        _ = super().__imul__(m)
        self.name = None # Reset name to default to avoid conflicts with the original motif name.
        return self
