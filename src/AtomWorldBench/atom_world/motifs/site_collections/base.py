"""Define base structure motifs in the atom world.

Motifs are inherited from ase.Atoms.
"""
from typing import List, Optional, Dict, Tuple
from abc import ABC
from functools import cached_property
from collections import Counter

import numpy as np
from numpy.typing import ArrayLike
from ase import Atoms

from ....utils.description_utils import get_species_string, describe_arraylike
from ....utils.coord_utils import check_integer_translation, find_coordinate_subset_indices

from ....globals import ALLOW_TRANSLATION_EQUIVALENCE, DEFAULT_FLOAT_TO_STRING_PRECISION
from ..base import BaseMotif


class BaseSiteCollectionMotif(ABC, BaseMotif, Atoms):
    """Base class for site collection motifs in the atom world.

    Defined as a specific collection of atoms that can be
    recognized and manipulated within a structure. This class provides an interface
    for defining and applying motifs to structures.

    Notice: all fractional coordinates must be unwrapped, i.e., not confined to
     the range [0, 1).
    """

    def __init__(
            self,
            *args,
            name: Optional[str] = None,
            indices: Optional[List[int]] = None,
            allow_translation_equivalence: bool = None,
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
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        BaseMotif.__init__(self, name=name)
        Atoms.__init__(self, *args, **kwargs)
        self.indices = indices
        if allow_translation_equivalence is None:
            allow_translation_equivalence = ALLOW_TRANSLATION_EQUIVALENCE
        self.allow_translation_equivalence = allow_translation_equivalence

        # Post init checks. Can be overridden by subclasses.
        self.__post_init__()

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
            return cart_centroid @ np.linalg.inv(self.cell.complete)
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
    def indices(self) -> List | None:
        """Get the indices of the atoms in the structure that correspond to this motif."""
        if self.arrays.get("site_indices") is None:
            return None
        return self.get_array("site_indices").tolist()

    @indices.setter
    def indices(self, indices: Optional[ArrayLike] = None):
        """Set the indices of the atoms in the structure that correspond to this motif."""
        # If None, clear the existing indices. Already implemented in ASE.
        self.set_array("site_indices", indices, dtype=int)

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

        if style not in self.allowed_description_styles:
            raise ValueError(
                f"Description style '{style}' is not allowed for {self.__class__.__name__}. "
                f"Allowed styles: {self.allowed_description_styles}."
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
                return (
                    f"{self.name} at site indices: "
                    f"{indices_string} in the structure."
                )
            else:
                raise NotImplementedError(
                    f"Description style '{style}' is not implemented for site collection motifs."
                )

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
            BaseSiteCollectionMotif: An instance of BaseMotif with the specified atoms and indices.
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

    def find_indices_in_atoms(
            self,
            atoms: Atoms,
            modify_indices_in_place: bool = False,
    ) -> Tuple[None | List[int], str]:
        """Find the indices of this motif in the given ASE Atoms object.

        Check with wrapped fractional coordinates of the motif.
        Args:
            atoms (Atoms): The ASE Atoms object to search in.
            modify_indices_in_place (bool):
                If True, modify the indices of this motif in place, according to indices in atoms.
                Will always overwrite the indices of this motif with newly found indices.

        Returns:
            Tuple[None | List[int], str]:
             A tuple containing the indices of the motif in the atoms or None if failed to find
             motif in atoms. Also returns a message indicating the reason for failure, if any.
        """
        # Check if the motif's periodic boundary conditions and cell match the atoms.
        if (
                not np.array_equal(self.pbc, atoms.pbc) or
                not np.allclose(
                    self.get_cell(complete=True).array,
                    atoms.get_cell(complete=True).array,
                    atol=1e-6
                )
        ):
            return None, "Motif's cell or pbc does not match the atoms."
        indices = find_coordinate_subset_indices(
            self.frac_coords, atoms.get_scaled_positions(), wrap=True
        )
        if indices is not None:
            symbols = [
                get_species_string(el, c)
                for el, c in zip(
                    atoms[indices].get_chemical_symbols(),
                    atoms[indices].get_initial_charges()
                )
            ]
            self_symbols = self.species_strings
            if symbols == self_symbols:
                if modify_indices_in_place:
                    self.indices = indices
                else:
                    indices = indices
                return indices, ""
            else:
                return None, "Motif's species do not match the atoms' species."
        return None, "Motif's fractional coordinates not found in the atoms."

    def get_site_indices_in_atoms(self, atoms: Atoms) -> List[int]:
        """Return the indices of sites included in the motif.

        This method will be the interface for the action to determine the
        sites to operate on.

        Args:
            atoms (Atoms): An ASE Atoms object containing all atoms in the system.

        Returns:
            List[int]: A list of indices of sites that are included in the motif.
        """
        indices, _ = self.find_indices_in_atoms(atoms, modify_indices_in_place=True)
        if indices is None:
            raise ValueError(
                f"Motif [{self.name}] not found in the provided Atoms object."
            )
        return indices

    def get_atoms(self) -> Atoms:
        """Get the ASE Atoms object corresponding to this motif.

        Returns:
            Atoms: An ASE Atoms object with the same properties as this motif.
        """
        return Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.cart_coords.tolist(),
            cell=self.cell.array,
            pbc=self.pbc,
            charges=self.get_initial_charges(),
        )

    def extend(self, other):
        """Extend the motif with another motif or ASE Atoms object.

        Args:
            other (BaseSiteCollectionMotif or Atoms): The motif or ASE Atoms object to extend this motif with.
        """
        if (self.indices is None and other.indices is not None) or \
           (self.indices is not None and other.indices is None):
            raise ValueError("Both motifs must have indices set or neither.")
        super().extend(other)
        self.name = None  # Reset name to default to avoid conflicts with the original motif name.

    def copy(self):
        """Return a copy of the motif."""
        atoms_copy = Atoms.copy(self)
        return self.__class__.from_atoms(
            atoms_copy,
            name=self.name,  # Keep the name of the motif.
            indices=self.indices
        )

    def __len__(self) -> int:
        """Return the number of atoms in the motif."""
        return Atoms.__len__(self)

    def __getitem__(self, i):
        """Return a subset of the motif."""
        atoms = super().__getitem__(i)
        indices = self.indices[i] if self.indices is not None else None
        return self.__class__.from_atoms(
            atoms,
            name=None, # Reset name to default to avoid conflicts with the original motif name.
            indices=indices
        )

    def __imul__(self, m):
        """Repeat the motif by a given factor."""
        _ = Atoms.__imul__(self, m)
        self.name = None # Reset name to default to avoid conflicts with the original motif name.
        return self

    def __eq__(self, other):
        """Check for identity of two atoms objects.

        Identity means: same positions, atomic numbers, unit cell and
        periodic boundary conditions."""
        # Must be the same class to compare.
        if not isinstance(other, self.__class__):
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
            sorted_args1, sorted_args2, taus = check_integer_translation(frac1, frac2, atol=1e-6)
            if taus is None:
                return False
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
                    self.arrays['initial_charges'][sorted_indices1],
                    other.arrays['initial_charges'][sorted_indices2],
                    atol=1e-6
                ) and
                np.array_equal(
                    np.array(self.indices, dtype=int)[sorted_indices1],
                    np.array(other.indices, dtype=int)[sorted_indices2],
                )
        )
