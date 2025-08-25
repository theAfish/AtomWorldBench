"""Define base structure motifs in the atom world.

Motifs are inherited from ase.Atoms.
"""
from typing import List, Optional
from abc import ABC
from collections import Counter
import copy

import numpy as np
from numpy.typing import ArrayLike
from ase import Atoms

from ....utils.description_utils import get_species_string, describe_arraylike
from ....utils.coord_utils import check_integer_translation

from ....common.globals import ALLOW_TRANSLATION_EQUIVALENCE, DEFAULT_FLOAT_TO_STRING_PRECISION
from ..base import BaseMotif


class BaseSiteCollectionMotif(BaseMotif, Atoms, ABC):
    """Base class for site collection motifs in the atom world.

    Defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.

    Notice: all fractional coordinates must be unwrapped, i.e., not confined to
     the range [0, 1).
    """
    # Do not modify this.
    _reserved_arrays = ["site_indices"]

    def __init__(
            self,
            in_atoms: Atoms,
            indices: List[int],
            offsets: Optional[ArrayLike] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """A Motif is an ASE Atoms comprising a subset of atoms in original ase.Atoms.

        Args:
            in_atoms (Atoms): The ASE Atoms object to create the motif from.
            indices (list of int): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k). If None, will assume all zeros.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        BaseMotif.__init__(self, in_atoms=in_atoms, name=name)
        subset = in_atoms[indices]
        # Apply offsets to fractional coordinates.
        positions_orig = subset.get_positions(wrap=False)
        positions_update = positions_orig + np.dot(
            offsets, in_atoms.get_cell(complete=True).array
        ) if offsets is not None else positions_orig
        subset.set_positions(positions_update)
        # Initialize internal attributes to inherit Atoms attributes.
        Atoms.__init__(self, subset)
        self.indices = indices
        if allow_translation_equivalence is None:
            allow_translation_equivalence = ALLOW_TRANSLATION_EQUIVALENCE
        self.allow_translation_equivalence = allow_translation_equivalence

        # Post init checks. Can be overridden by subclasses.
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization checks for the motif.

        Can be overridden by subclasses to perform additional checks
        """
        pass

    @property
    def species_strings(self) -> List[str]:
        """Get the species of the motif."""
        return [
            get_species_string(el, int(c))
            for el, c in
            zip(
                self.get_chemical_symbols(),
                self.get_initial_charges()
            )
        ]

    @property
    def species_groups(self):
        """Group motif site indices by species.

        Returns:
            dict: A dictionary where keys are species strings and
             values are lists of indices
        """
        species_groups = {}
        for idx, species in enumerate(self.species_strings):
            if species not in species_groups:
                species_groups[species] = []
            species_groups[species].append(idx)
        return {k: np.array(v, dtype=int) for k, v in species_groups.items()}

    @property
    def composition(self):
        """Get the composition of the motif."""
        return Counter(self.species_strings)

    @property
    def frac_coords(self) -> np.ndarray:
        """Get the unwrapped fractional coordinates of the motif."""
        return self.get_scaled_positions(wrap=False)

    @property
    def cell_offsets(self) -> np.ndarray:
        """Get the cell offset for each atom in the cluster.

        cell offsets are the integer part of the fractional coordinates in the form of
         triplets (i, j, k).
        Returns:
            np.ndarray: A 2D array of shape (n, 3) representing the box positions.
        """
        return np.floor(self.frac_coords).astype(int)

    @property
    def cart_coords(self) -> np.ndarray:
        """Get the unwrapped Cartesian coordinates of the motif."""
        return self.get_positions(wrap=False)

    def get_centroid(self, fractional=False) -> np.ndarray:
        """Get the centroid of the motif.

        Args:
            fractional (bool): If True, return the centroid in fractional coordinates.
                If False, return in Cartesian coordinates. Default is False.
        """
        cart_centroid = np.mean(self.cart_coords, axis=0)
        if fractional:
            return cart_centroid @ np.linalg.inv(self.cell.complete())
        else:
            return cart_centroid

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

    @property
    def indices(self) -> List | None:
        """Get the indices of the atoms in the structure that correspond to this motif."""
        return self.get_array("site_indices").tolist()

    @indices.setter
    def indices(self, indices: Optional[ArrayLike] = None):
        """Set the indices of the atoms in the structure that correspond to this motif."""
        # If None, clear the existing indices. Already implemented in ASE.
        self.set_array("site_indices", indices, dtype=int)

    def update_indices_offsets(
            self,
            new_indices: List[int],
            offsets: Optional[ArrayLike] = None
    ):
        """Update the indices of the atoms in the structure that correspond to this motif.


        Call this function to update indices after modifying the motif, otherwise
        other attributes might be inconsistent.

        Args:
            new_indices (List[int]): The new indices to set.
            offsets (ArrayLike, optional): The new cell offsets to set.
                If None, will use (0, 0, 0) for all sites.
                If provided, must be the same length as new_indices.
        """
        self.indices = new_indices
        subset = self.in_atoms[new_indices]
        # Apply offsets to fractional coordinates.
        positions_orig = subset.get_positions(wrap=False)
        positions_update = positions_orig + np.dot(
            offsets, self.in_atoms.cell.complete()
        ) if offsets is not None else positions_orig
        subset.set_positions(positions_update)
        # Initialize internal attributes to inherit Atoms attributes.
        Atoms.__init__(self, subset)


    def describe(
            self,
            style: str = "coord",
            is_addition: bool = False,
            coord_fractional: bool = False,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Return a string description of the cluster motif.

        Args:
            style (str): The style of description. Default is "coord".
            is_addition (bool): If True, the description is for an addition action.
                This affects how the description is formatted. Default is False.
            coord_fractional (bool): If True, use fractional coordinates in the description.
                If False, use Cartesian coordinates. Default is False.
            precision (int): The precision for floating-point numbers in the description.
        Returns:
            str: A string description of the cluster motif.
        """
        style = style.lower()
        if self.indices is None and style == "index":
            raise ValueError(
                f"Cannot describe {self.__class__.__name__} with style 'index' "
                f"because it has no indices set. Please set indices first."
                f"To find indices, use the `find_indices_in_atoms` method."
            )

        # addition of a single site motif, return the name directly.
        if len(self) == 1 and is_addition:
            return self.name
        else:
            if style == "coord":
                coord_word = "fractional" if coord_fractional else "cartesian"
                coords = self.frac_coords if coord_fractional else self.cart_coords
                coords_string = describe_arraylike(coords, precision=precision)
                return (
                    f"{self.name} with {coord_word} coordinates: "
                    f"{coords_string}"
                )
            elif style == "index":
                indices_string = describe_arraylike(self.indices, precision=0)
                if np.allclose(self.cell_offsets, 0):
                    # Notice: with this condition, you should always wrap atom coordinates
                    # to the unit cell before using detector to detect a motif.
                    offset_string = "."
                else:
                    offset_string = (
                        f" and offsets {describe_arraylike(self.cell_offsets, precision=0)}."
                        f" offsets represents how many unit cells the atoms"
                        f" are away from the origin unit cell"
                        f" (fractional coordinates between 0 and 1)"
                        f" in the direction"
                        f" of each lattice vector."
                    )
                return (
                    f"{self.name} at site indices: "
                    f"{indices_string}{offset_string}"
                )
            else:
                raise NotImplementedError(
                    f"Description style '{style}' is not implemented for site collection motifs."
                )

    def get_atoms(self) -> Atoms:
        """Get the ASE Atoms object corresponding to this motif.

        Returns:
            Atoms: An ASE Atoms object with the same properties as this motif.
        """
        atoms = Atoms(
            cell=self.cell, pbc=self.pbc, info=self.info,
            celldisp=self._celldisp.copy()
        )

        atoms.arrays = {}
        for name, a in self.arrays.items():
            # Do not copy reserved arrays (site_indices) as they are
            # not native to atoms object.
            if name not in self._reserved_arrays:
                atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        return atoms

    # For IDE linting only.
    # The actual implementation is the same as ASE Atoms class.
    def __add__(self, other: "BaseSiteCollectionMotif") -> "BaseSiteCollectionMotif":
        out = self.copy()
        out.extend(other)
        return out

    def __iadd__(self, other: "BaseSiteCollectionMotif") -> "BaseSiteCollectionMotif":
        self.extend(other)
        return self

    def extend(self, other: "BaseSiteCollectionMotif"):
        """Extend the motif with another motif or ASE Atoms object.

        Args:
            other (BaseSiteCollectionMotif or Atoms): The motif object to extend this motif with.
                Do not support pure ase.Atoms, only BaseSiteCollectionMotif.
        """
        if (self.indices is None and other.indices is not None) or \
           (self.indices is not None and other.indices is None):
            raise ValueError("Both motifs must have indices set or neither.")
        super().extend(other)
        self.name = None  # Reset name to default to avoid conflicts with the original motif name.

    def copy(self):
        """Return a copy of the motif."""
        return self.__class__(
            in_atoms=self.in_atoms.copy(),
            indices=copy.deepcopy(self.indices),
            offsets=copy.deepcopy(self.cell_offsets),
            name=self.name,
            allow_translation_equivalence=self.allow_translation_equivalence,
        )

    def __len__(self) -> int:
        """Return the number of atoms in the motif."""
        return Atoms.__len__(self)

    def __getitem__(self, i):
        """Return a subset of the motif."""
        if isinstance(i, int):
            idx = [i]  # Force list.
        else:
            idx = i
        indices = [self.indices[j] for j in idx] if self.indices is not None else None
        offsets = self.cell_offsets[idx]
        return self.__class__(
            self.in_atoms.copy(),
            name=None, # Reset name to default to avoid conflicts with the original motif name.
            indices=indices,
            offsets=offsets,
            allow_translation_equivalence=self.allow_translation_equivalence
        )

    def __imul__(self, m):
        """Repeat of motif not allowed!"""
        raise NotImplementedError("Repeating a motif is not allowed.")


    def __eq__(self, other):
        """Check for identity of two atoms objects.

        Identity means: same positions, atomic numbers, unit cell and
        periodic boundary conditions."""
        # Must be the same class to compare.
        if not isinstance(other, self.__class__):
            return False
        # Must be from the same original structure.
        if not self.in_atoms == other.in_atoms:
            return False
        if len(self) != len(other):
            return False
        if not (
                np.allclose(
                    self.cell.complete().array,
                    other.cell.complete().array,
                    atol=1e-6
                ) and
                np.array_equal(self.pbc, other.pbc)
        ):
            return False
        if self.composition != other.composition:
            return False

        sorted_indices1 = []
        sorted_indices2 = []
        prev_taus = None
        for sp in self.species_groups:
            group1 = self.species_groups[sp]
            group2 = other.species_groups[sp]
            frac1 = self.frac_coords[group1]
            frac2 = other.frac_coords[group2]
            # Allow permutation within species groups, but check for integer translation.
            results = check_integer_translation(frac1, frac2, atol=1e-6)
            if results is None:
                return False
            sorted_args1, sorted_args2, taus = results
            if prev_taus is None:
                prev_taus = taus
            else:
                if not np.array_equal(prev_taus, taus):
                    return False
            if not (self.allow_translation_equivalence and other.allow_translation_equivalence):
                if not np.all(taus == 0):
                    return False
            sorted_indices1.extend(group1[sorted_args1])
            sorted_indices2.extend(group2[sorted_args2])

        return (
                np.allclose(
                    self.get_initial_charges()[sorted_indices1],
                    other.get_initial_charges()[sorted_indices2],
                    atol=1e-6
                ) and
                (
                    np.array_equal(
                        np.array(self.indices, dtype=int)[sorted_indices1],
                        np.array(other.indices, dtype=int)[sorted_indices2],
                    ) if self.indices is not None and other.indices is not None
                    else (self.indices == other.indices)
                )
        )
