"""Defines spherical region motif."""
from typing import Optional, Union

from ase import Atoms
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike

from .base import BaseRegionMotif
from ....utils.coord_utils import check_coordinates_shape
from ....utils.neighbor_utils import detect_indices_offests_around_frac_coords
from ....utils.description_utils import describe_arraylike
from ....globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....mixin_classes import MultiModeInitMixin


def _check_radius(x: Union[int, float]) -> float:
    """Check if the radius is a positive number."""
    if not isinstance(x, (int, float)) or x <= 0:
        raise ValueError("The radius must be a positive number.")
    return float(x)

# Design philosophy: Inherit from multimode object when initialization has multiple modes.
# This allows us to have a single class that can be used in different ways,
# such as by specifying a center coordinate or an index of an atom.
# Don't do multiple classes.
class SphereRegionMotif(BaseRegionMotif, MultiModeInitMixin):
    """A spherical region motif that defines a spherical operable region in space."""
    kwargs_formatting_functions = {
        "center": lambda x: check_coordinates_shape(x, "center", expected_1d=True, allow_none=True),
        "radius": _check_radius,
    }

    mode_definitions = {
        # in_atoms and radius is always required, so it is not included in the modes.
        # center_is_fractional and symbols are not needed to be checked.
        "_excluded": ["in_atoms", "radius", "center_is_fractional", "symbols"],
        "center_around_coordinates": {
            "center": None,
        },
        "center_around_atom_index": {
            "center_id": (
                lambda x: isinstance(x, int) and x >= 0,
                "The center_id must be a non-negative integer representing the index of an atom."
            ),
        },
    }

    # Can perform resize.
    forbidden_actions = ["replace", "add", "translate"]
    def __init__(
            self,
            in_atoms: Atoms,
            radius: float,
            center: Optional[ArrayLike] = None,
            center_id: Optional[int] = None,
            center_is_fractional: bool = False,
            symbols: Optional[list[str]] = None
    ):
        """Initialize the spherical region motif.


        Currently, allows 2 modes of operation:
            1. Centered around specified coordinates (fractional or Cartesian).
                In this mode, `center` and `radius` must be provided. No other
                arguments other than `in_atoms`, `center_is_fractional` and `symbols`
                can be provided.
            2. Centered around a specified atom index in the provided Atoms object.
                In this mode, `center_id` and `radius` must be provided. No other
                arguments other than `in_atoms`, `symbols` can be provided.
        Args:
            in_atoms (Atoms): The atoms object that this region motif is in. Required.
            radius (float): The radius of the sphere. Required.
            center (ArrayLike): The center of the sphere as a list of three coordinates [x, y, z].
            center_id (Optional[int]): The index of the atom that serves as the center of the sphere.
            center_is_fractional (bool): Whether the center coordinates are fractional or Cartesian.
                Defaults to False (Cartesian coordinates).
            symbols (Optional[list[str]]): An optional list of symbols to filter atoms by.
                If provided, only atoms with these symbols will be included in the motif.
        """
        # Always use default name.
        BaseRegionMotif.__init__(
            self, in_atoms=in_atoms, name=None, symbols=symbols
        )
        MultiModeInitMixin.__init__(
            self,
            center=center,
            center_id=center_id,
            radius=radius,
            center_is_fractional=center_is_fractional,
            symbols=symbols
        )

    def __post_init__(self):
        """Post-initialization to ensure the spherical region motif is valid."""
        # Has to be checked in post_init as it requires self.in_atoms to be set before checking.
        if self.mode_flag == "center_around_atom_index":
            if self.center_id >= len(self.in_atoms):
                raise ValueError(
                    f"center_id {self.center_id} is out of bounds for the provided"
                    f" Atoms object with {len(self.in_atoms)} atoms."
                )
            elif self.center_id < 0:
                raise ValueError(
                    f"center_id {self.center_id} must be a non-negative integer."
                )

    def _get_center(self) -> ndarray:
        """Get the center of the spherical region motif."""
        if self.mode_flag == "center_around_coordinates":
            if self.center_is_fractional:
                return self.center @ self.in_atoms.cell.complete()
            else:
                return self.center
        elif self.mode_flag == "center_around_atom_index":
            return self.in_atoms.get_positions()[self.center_id]
        else:
            raise NotImplementedError(
                f"operation mode {self.mode_flag} for {self.__class__.__name__} not implemented."
            )

    def get_centroid(self, fractional=False) -> ndarray:
        """Get the centroid of the motif.

        Args:
            fractional (bool): If True, return the centroid in fractional coordinates.
                If False, return in Cartesian coordinates. Default is False.
        Returns:
            np.ndarray: The centroid of the spherical region motif.
        """
        cart_centroid = self._get_center()
        if fractional:
            return cart_centroid @ np.linalg.inv(self.cell.complete())
        else:
            return cart_centroid

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
            atoms, self._get_center(), self.radius, self.symbols
        )
        return indices

    def _get_default_name(self) -> str:
        # Not used, just to satisfy the abstract method requirement.
        return self.__class__.__name__

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Return a string description of the spherical region motif.

        Args:
            precision (int): The number of decimal places to use for floating-point numbers.
                Defaults to DEFAULT_FLOAT_TO_STRING_PRECISION.
        Returns:
            str: A string description of the spherical region motif.
        """
        if self.mode_flag == "center_around_coordinates":
            coord_word = "fractional" if self.center_is_fractional else "Cartesian"
            coord_string = describe_arraylike(self.center, precision=precision)
            center_string = f"{coord_word} coordinates {coord_string}"
        elif self.mode_flag == "center_around_atom_index":
            center_string = f"atom with index {self.center_id} in structure"
        else:
            raise NotImplementedError(
                f"operation mode {self.mode_flag} for {self.__class__.__name__} not implemented."
            )

        symbol_word = (
            "all atoms" if self.symbols is None
            else f"all atoms with element symbols {self.symbols}"
        )
        return (
            f"{symbol_word} in the spherical region centered at"
            f" {center_string} with radius {self.radius:.{precision}f} angstroms"
        )
