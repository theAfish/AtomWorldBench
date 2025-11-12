"""Define base structure motifs in the atom world.

Motifs are inherited from ase.Atoms.
"""
from typing import List, Optional
from abc import ABC
from collections import Counter
import copy

import numpy as np
from numpy import ndarray
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
            in_atoms: Optional[Atoms] = None,
            indices: Optional[List[int]] = None,
            offsets: Optional[ArrayLike] = None,
            atoms: Optional[Atoms] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """A Motif is an ASE Atoms comprising a subset of atoms in original ase.Atoms.

        Args:
            in_atoms (Atoms, optional): The ASE Atoms object to create the motif from.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            indices (list of int, optional): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k) representing their unwrapped location in periodic images.
                If None, will assume all zeros.
            atoms (Atoms, optional): An ASE Atoms object representing the motif.
                When none of in_atoms, indices, offsets are provided, and atoms is provided,
                will create a motif directly from atoms. In this case, the motif can only be
                added in the AddMotifAction (additive mode).
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
                Does not matter for additive motifs.
        """
        # Wraps atom after super init.
        BaseMotif.__init__(self, in_atoms, name=name)

        if in_atoms is None and indices is None and offsets is None and atoms is not None:
            self.is_additive = True
        elif in_atoms is not None and indices is not None and atoms is None:
            self.is_additive = False
        else:
            raise ValueError(
                "Must provide either (in_atoms and indices) to define a motif from"
                " an existing structure, or provide atoms only to define an additive motif."
            )

        self._indices = indices
        if not self.is_additive:
            self._offsets = (
                np.array(offsets, dtype=int) if offsets is not None
                else np.zeros((len(indices), 3), dtype=int)
            )
        else:
            self._offsets = None

        if allow_translation_equivalence is None:
            allow_translation_equivalence = ALLOW_TRANSLATION_EQUIVALENCE
        self.allow_translation_equivalence = allow_translation_equivalence

        if atoms is None:
            self._atoms = self.get_atoms()
        else:
            self._atoms = atoms

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
        """Set the indices of the atoms in the `in_atoms` that correspond to this motif.

        Not to be called at init, only for modifying existing motif.
        Args:
            value (List[int]): A list of integers representing the indices.
        """
        if not all(isinstance(i, int) for i in value):
            raise ValueError("Indices must be a list of integers.")
        self._indices = value
        # Resetting indices should update the internal atoms attribute.
        self._atoms = None

    @cell_offsets.setter
    def cell_offsets(self, value: ArrayLike):
        """Set the cell offsets for each atom in the motif.

        Cell offsets are the integer part of the fractional coordinates in the form of
         triplets (i, j, k) representing their unwrapped location in periodic images.

        Not to be called at init, only for modifying existing motif.
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
        self._atoms = None

    def _get_site_indices_offsets_in_atoms(self) -> tuple[list[int], ndarray]:
        """Get the indices and periodic offsets of sites in this motif within the parent atoms.

        Returns:
            indices (list[int]): A list of indices of sites in this motif within the parent atoms.
            offsets (ndarray): A 2D array of shape (n, 3) representing the cell offsets
                for each atom in the motif.
        """
        return self.indices, self.cell_offsets


    @property
    def species_strings(self) -> List[str]:
        """Get the species of the motif."""
        return [
            get_species_string(el, int(c))
            for el, c in
            zip(
                self.get_atoms().get_chemical_symbols(),
                self.get_atoms().get_initial_charges()
            )
        ]

    @property
    def composition(self):
        """Get the composition of the motif."""
        return Counter(self.species_strings)

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
            style: str = "index",  # Use index as default style for site collection.
            coord_fractional: bool = False,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Return a string description of the cluster motif.

        Args:
            style (str): The style of description. Default is "coord".
            coord_fractional (bool): If True, use fractional coordinates in the description.
                If False, use Cartesian coordinates. Default is False.
            precision (int): The precision for floating-point numbers in the description.
        Returns:
            str: A string description of the cluster motif.
        """
        style = style.lower()

        if self.is_additive:
            style = "coord"  # Force coord style for additive motifs.

        # addition of a single site motif, return the name directly.
        if len(self) == 1 and self.is_additive:
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
        # Do not call setters as get_atoms() will be called at the end.
         # This is to avoid multiple calls to get_atoms().
        if self.is_additive and other.is_additive:
            self._indices = None
            self._offsets = None
            self._atoms = self._atoms + other._atoms
        else:
            self._indices = self.indices + other.indices
            self._offsets = np.vstack((self.cell_offsets, other.cell_offsets))
            self._atoms = None  # Reset atoms to force re-computation.
        # Post init checks to avoid invalid addition, for example, adding cluster to site.
        self.__post_init__()
        # Reset name to default to avoid conflicts with the original motif name.
        self.name = None


    def copy(self) -> "BaseSiteCollectionMotif":
        """Return a copy of the motif."""
        return self.__class__(
            in_atoms=copy.deepcopy(self.in_atoms),
            indices=copy.deepcopy(self.indices),
            offsets=copy.deepcopy(self.cell_offsets),
            atoms=(copy.deepcopy(self._atoms) if self.is_additive else None),
            name=self.name,
            allow_translation_equivalence=self.allow_translation_equivalence,
        )

    def __len__(self) -> int:
        """Return the number of atoms in the motif."""
        return len(self.get_atoms())

    def __getitem__(self, i):
        """Return a subset of the motif."""
        if isinstance(i, int):
            idx = [i]  # Force list.
        else:
            idx = i
        if self.is_additive:
            indices = None
            offsets = None
            atoms = self.get_atoms()[idx]
        else:
            indices = np.array(self.indices, dtype=int)[idx].tolist()
            offsets = self.cell_offsets[idx]
            atoms = None  # Force re-computation in the new motif.

        return self.__class__(
            in_atoms=copy.deepcopy(self.in_atoms),
            name=None, # Reset name to default to avoid conflicts with the original motif name.
            indices=indices,
            offsets=offsets,
            atoms=atoms,
            allow_translation_equivalence=self.allow_translation_equivalence
        )


    def __eq__(self, other):
        """Check for identity of two atoms objects.

        Identity means: same positions, atomic numbers, unit cell and
        periodic boundary conditions."""
        # Must be the same class to compare.
        if not isinstance(other, self.__class__):
            return False
        if self.is_additive and other.is_additive:
            return self.get_atoms() == other.get_atoms()
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
