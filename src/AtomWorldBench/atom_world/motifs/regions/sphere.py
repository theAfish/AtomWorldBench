"""Defines spherical region motif."""
from typing import Optional

from ase import Atoms
from numpy import ndarray
from numpy.typing import ArrayLike

from .base import BaseRegionMotif
from ....utils.coord_utils import check_coordinates_shape
from ....utils.neighbor_utils import detect_indices_offests_around_frac_coords
from ....utils.description_utils import describe_arraylike
from ....globals import DEFAULT_FLOAT_TO_STRING_PRECISION


# Design philosophy: Inherit from multimode object when initialization has multiple modes.
# This allows us to have a single class that can be used in different ways,
# such as by specifying a center coordinate or an index of an atom.
# Don't do multiple classes.
class SphereAroundCoordRegionMotif(BaseRegionMotif):
    """A spherical region motif that defines a spherical operable region in space."""

    def __init__(
            self,
            center: ArrayLike,
            radius: float,
            center_is_fractional: bool = False,
            symbols: Optional[list[str]] = None
    ):
        """Initialize the spherical region motif.

        Args:
            center (ArrayLike): The center of the sphere as a list of three coordinates [x, y, z].
            radius (float): The radius of the sphere.
            center_is_fractional (bool): Whether the center coordinates are fractional or Cartesian.
                Defaults to False (Cartesian coordinates).
            symbols (Optional[list[str]]): An optional list of symbols to filter atoms by.
                If provided, only atoms with these symbols will be included in the motif.
        """
        # Always use default name.
        super().__init__(name=None, symbols=symbols)
        self.center = check_coordinates_shape(
            center, "center",
            expected_1d=True, allow_none=False
        )
        self.radius = radius
        self.center_is_fractional = center_is_fractional

    def _get_center(self, atoms:Atoms) -> ndarray:
        """Get the center of the spherical region motif."""
        if self.center_is_fractional:
            return self.center @ atoms.cell.complete()
        else:
            return self.center

    def get_site_indices_in_atoms(self, atoms: Atoms) -> Atoms:
        """Return the subset of atoms included in the spherical region motif.

        If one of the periodic images lies within the region,
        it will include the corresponding atom.
        Args:
            atoms (Atoms): The ASE Atoms object containing all atoms in the system.
        Returns:
            ndarray[int]: An array of indices of atoms that are within the region motif.
        """
        indices, _ = detect_indices_offests_around_frac_coords(
            atoms, self._get_center(atoms), self.radius, self.symbols
        )
        return indices

    def _get_default_name(self) -> str:
        # Not used, just to satisfy the abstract method requirement.
        return self.__class__.__name__

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Return a string description of the spherical region motif."""
        coord_word = "fractional" if self.center_is_fractional else "Cartesian"
        center_string = describe_arraylike(self.center, precision=precision)
        symbol_word = "all atoms" if self.symbols is None else f"all atoms with symbols {self.symbols}"
        return (
            f"{symbol_word} in the spherical region centered at {coord_word}"
            f" {center_string} with radius {self.radius:.{precision}f} angstroms"
        )


class SphereAroundIndexRegionMotif(BaseRegionMotif):
    """A spherical region motif that defines a spherical operable region in space."""

    def __init__(
            self,
            center_index: int,
            radius: float,
            symbols: Optional[list[str]] = None
    ):
        """Initialize the spherical region motif.

        Args:
            center_index (int): The index of the atom that serves as the center of the sphere.
            radius (float): The radius of the sphere.
            symbols (Optional[list[str]]): An optional list of symbols to filter atoms by.
                If provided, only atoms with these symbols will be included in the motif.
        """
        super().__init__(name=None, symbols=symbols)
        self.center_index = center_index
        self.radius = radius

    def _get_center(self, atoms: Atoms) -> ndarray:
        """Get the center of the spherical region motif."""
        return atoms.get_positions(wrap=False)[self.center_index]

    def get_site_indices_in_atoms(self, atoms: Atoms) -> Atoms:
        """Return the subset of atoms included in the spherical region motif.

        If one of the periodic images lies within the region,
        it will include the corresponding atom.
        Args:
            atoms (Atoms): The ASE Atoms object containing all atoms in the system.
        Returns:
            ndarray[int]: An array of indices of atoms that are within the region motif.
        """
        indices, _ = detect_indices_offests_around_frac_coords(
            atoms, self._get_center(atoms), self.radius, self.symbols
        )
        return indices

    def _get_default_name(self) -> str:
        # Not used, just to satisfy the abstract method requirement.
        return self.__class__.__name__

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Return a string description of the spherical region motif."""
        symbol_word = "all atoms" if self.symbols is None else f"all atoms with symbols {self.symbols}"
        return (
            f"{symbol_word} in the spherical region centered at atom index {self.center_index}"
            f" with radius {self.radius:.{precision}f} angstroms"
        )
