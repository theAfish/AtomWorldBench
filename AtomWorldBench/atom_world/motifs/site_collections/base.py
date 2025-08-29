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


class BaseSiteCollectionMotif(BaseMotif, ABC):
    """Base class for site collection motifs in the atom world.

    Defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.

    Notice: all fractional coordinates must be unwrapped, i.e., not confined to
     the range [0, 1).
    """

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
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            indices (list of int): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k) representing their unwrapped location in periodic images.
                If None, will assume all zeros.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        # Wraps atom after super init.
        super().__init__(self, in_atoms=in_atoms, name=name)
        self._atoms = None
        self._indices = None
        self._offsets = None
        self.indices = indices
        self.offsets = (
            np.array(offsets, dtype=int) if offsets is not None
            else np.zeros((len(indices), 3), dtype=int)
        )

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
    def indices(self) -> List[int]:
        """Get the indices of the atoms in the original structure that correspond to this motif."""
        return self._indices

    @property
    def cell_offsets(self) -> np.ndarray:
        """Get the cell offset for each atom in the cluster.

        cell offsets are the integer part of the fractional coordinates in the form of
         triplets (i, j, k).
        Returns:
            np.ndarray: A 2D array of shape (n, 3) representing the box positions.
        """
        return self._offsets

    @indices.setter
    def indices(self, value: List[int]):
        """Set the indices of the atoms in the `in_atoms` that correspond to this motif."""
        if not all(isinstance(i, int) for i in value):
            raise ValueError("Indices must be a list of integers.")
        self._indices = value
        # Resetting indices should update the internal atoms attribute.
        if self._indices is not None:
            self._atoms = self.get_atoms()

    @cell_offsets.setter
    def cell_offsets(self, value: ArrayLike):
        """Set the cell offsets for each atom in the motif.

        Cell offsets are the integer part of the fractional coordinates in the form of
         triplets (i, j, k) representing their unwrapped location in periodic images.
        Args:
            value (ArrayLike): A 2D array of shape (n, 3) representing the cell offsets.
                Must be the same length as indices.
        """
        value = np.asarray(value, dtype=int)
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(
                f"Cell offsets must be a 2D array with shape (n, 3), got {value.shape}."
            )
        if len(value) != len(self.indices):
            raise ValueError(
                f"Cell offsets must have the same length as indices. "
                f"Got {len(value)} and {len(self.indices)}."
            )
        self._offsets = value
        # Resetting offsets should update the internal atoms attribute.
        self._atoms = self.get_atoms()

    def get_atoms(self) -> Atoms:
        """Get the ASE Atoms object corresponding to this motif.

        Returns:
            Atoms: An ASE Atoms object with the same properties as this motif.
        """
        if self._atoms is not None:
            return self._atoms
        subset = self.in_atoms[self.indices]
        cell = self.in_atoms.cell.complete()
        positions_orig = subset.get_positions(wrap=False)
        positions_update = positions_orig + np.dot(
            self.cell_offsets, cell
        ) if self.cell_offsets is not None else positions_orig
        subset.set_positions(positions_update)
        return subset

    @property
    def species_strings(self) -> List[str]:
        """Get the species of the motif."""
        self_atoms = self.get_atoms()
        return [
            get_species_string(el, int(c))
            for el, c in
            zip(
                self_atoms.get_chemical_symbols(),
                self_atoms.get_initial_charges()
            )
        ]

    @property
    def composition(self):
        """Get the composition of the motif."""
        return Counter(self.species_strings)

    @property
    def frac_coords(self) -> np.ndarray:
        """Get the unwrapped fractional coordinates of the motif."""
        return self.get_atoms().get_scaled_positions(wrap=False)

    @property
    def cart_coords(self) -> np.ndarray:
        """Get the unwrapped Cartesian coordinates of the motif."""
        return self.get_atoms().get_positions(wrap=False)

    def get_centroid(self, fractional=False) -> np.ndarray:
        """Get the centroid of the motif.

        Args:
            fractional (bool): If True, return the centroid in fractional coordinates.
                If False, return in Cartesian coordinates. Default is False.
        """
        cart_centroid = np.mean(self.cart_coords, axis=0)
        cell = self.get_atoms().cell.complete()
        if fractional:
            return cart_centroid @ np.linalg.inv(cell)
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
        Returns:
            BaseSiteCollectionMotif: The extended motif. Type of the motif follows
                the type of self. For example, ClusterMotif + SiteMotif = ClusterMotif.
        """
        if other.in_atoms != self.in_atoms:
            raise ValueError("Can only extend motifs from the same original structure.")
        # Reset name to default to avoid conflicts with the original motif name.
        return self.__class__(
            in_atoms=self.in_atoms.copy(),
            indices=self.indices + other.indices,
            offsets=np.vstack((self.cell_offsets, other.cell_offsets)),
            name=None,
            allow_translation_equivalence=self.allow_translation_equivalence
        )


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
        return len(self.indices)

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
        self_indices = sorted(self.indices)
        other_indices = sorted(other.indices)
        if self_indices != other_indices:
            return False
        # Subset atoms must be identical.
        if not self.allow_translation_equivalence:
            return (
                    self.in_atoms[self_indices] ==
                    other.in_atoms[other_indices]
            )
        else:
            # Check if two motifs are identical up to an integer translation.
            frac_self = self.in_atoms[self_indices].get_scaled_positions(wrap=False)
            frac_other = other.in_atoms[other_indices].get_scaled_positions(wrap=False)
            res = check_integer_translation(
                frac_self,
                frac_other,
                atol=1e-5
            )
            if res is None:
                return False
            else:
                _, _, tau = res
                self_atoms = self.in_atoms[self_indices].copy()
                self_atoms.set_scaled_positions(frac_self + tau)
                return self_atoms == other.in_atoms[other_indices]
