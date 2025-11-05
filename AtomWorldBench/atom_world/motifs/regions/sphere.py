"""Defines spherical region motif."""
from typing import Optional, Union

from ase import Atoms
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike

from .base import BaseRegionMotif
from ..base import BaseMotif
from ....utils.coord_utils import check_coordinates_shape
from ....utils.neighbor_utils import detect_indices_offsets_around_frac_coords
from ....utils.description_utils import describe_arraylike
from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....common.mixin_classes import MultiModeInitMixin

from ....common.registry import register


def _check_radius(x: Union[int, float]) -> float:
    """Check if the radius is a positive number."""
    if not isinstance(x, (int, float)) or x <= 0:
        raise ValueError("The radius must be a positive number.")
    return float(x)

# Design philosophy: Inherit from multimode object when initialization has multiple modes.
# This allows us to have a single class that can be used in different ways,
# such as by specifying a center coordinate or an index of an atom.
# Don't do multiple classes.
@register(BaseMotif, ["sphere", "sphere-region"])
@register(BaseRegionMotif,["sphere", "sphere-region"])
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
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            radius (float): The radius of the sphere. Required.
            center (ArrayLike): The center of the sphere as a list of three coordinates [x, y, z].
            center_id (Optional[int]): The index of the atom that serves as the center of the sphere.
            center_is_fractional (bool): Whether the center coordinates are fractional or Cartesian.
                Defaults to False (Cartesian coordinates).
            symbols (Optional[list[str]]): An optional list of symbols to filter atoms by.
                If provided, only atoms with these symbols will be included in the motif.
        """
        # Added to supress mypy warnings about uninitialized attributes.
        self.radius = None
        self.center = None
        self.center_id = None
        self.center_is_fractional = None
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
        """Get the center of the spherical region motif in cartesian."""
        if self.mode_flag == "center_around_coordinates":
            if self.center_is_fractional:
                return self.center @ self.in_atoms.cell.complete()
            else:
                return self.center
        elif self.mode_flag == "center_around_atom_index":
            return self.in_atoms.get_positions(wrap=False)[self.center_id]
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
            return cart_centroid @ np.linalg.inv(self.in_atoms.cell.complete())
        else:
            return cart_centroid

    def _get_site_indices_offsets_in_atoms(self) -> tuple[list[int], ndarray]:
        """Return the subset of atoms included in the spherical region motif.

        If one of the periodic images lies within the region,
        it will include the corresponding atom.

        Returns:
            ndarray[int]: An array of indices of atoms that are within the region motif.
        """
        return detect_indices_offsets_around_frac_coords(
            self.in_atoms, self.get_centroid(fractional=True), self.radius, self.symbols
        )

    def _get_default_name(self) -> str:
        # Not used, just to satisfy the abstract method requirement.
        return "a sphere region"

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
            coord_word = "fractional" if self.center_is_fractional else "cartesian"
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

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            radius: Optional[float] = None,
            randomize_symbols: bool = False,
            style="center_around_atom_index",
            seed: Optional[int] = None
    ) -> 'SphereRegionMotif':
        """Detect a random spherical region motif from the given atoms.

        The center will be randomly chosen from the atoms in the provided Atoms object.
        The radius will be randomly chosen between 1 and half of the shortest cell vector length.
        Args:
            atoms (Atoms): The ASE Atoms object containing all atoms in the system.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            radius (float, optional): The radius of the spherical region motif.
                Unit is angstroms.
                If None, a random radius will be chosen between 2 and half of the shortest
                cell vector length. Defaults to None.
            randomize_symbols (bool): If True, the symbols of the atoms in the motif will be
                randomly chosen from the symbols of the atoms in the provided Atoms object.
                If False, the symbols will be set to None, meaning all atoms in the region
                will be included regardless of their symbols.
                Defaults to False.
            style (str): The style of the spherical region motif to create.
                Can be "center_around_atom_index" or "center_around_coordinates".
                Defaults to "center_around_atom_index".
            seed (Optional[int]): Random seed for reproducibility.
                If None, a random seed will be used.
        Returns:
            SphereRegionMotif: A randomly generated spherical region motif.
        """
        atoms.wrap()
        rng = np.random.default_rng(seed)
        if len(atoms) == 0:
            raise ValueError("The provided Atoms object is empty.")
        if radius is None:
            # Generate a random radius from 2 angstroms to half of
            # the shortest cell vector length.
            cell_lengths = atoms.cell.lengths()
            min_cell_length = np.min(cell_lengths)
            if min_cell_length / 2.0 <= 3.0:
                radius = 3.0
            else:
                radius = rng.uniform(3.0, min_cell_length / 2.0)

        if randomize_symbols:
            all_symbols = list(set(atoms.get_chemical_symbols()))
            num_symbols = int(rng.integers(1, len(all_symbols) + 1))
            symbols = rng.choice(
                all_symbols, size=num_symbols, replace=False
            ).tolist()
        else:
            symbols = None

        if style == "center_around_atom_index":
            center_id = int(rng.integers(0, len(atoms)))

            return cls(
                in_atoms=atoms,
                center_id=center_id,
                radius=radius,
                symbols=symbols
            )
        elif style == "center_around_coordinates":
            # Generate a random cartesian position within the cell box.
            rand_frac = rng.random(3)
            center = rand_frac @ atoms.cell.complete()
            return cls(
                in_atoms=atoms,
                center=center,
                radius=radius,
                center_is_fractional=False,
                symbols=symbols
            )
        else:
            raise NotImplementedError(
                f"Invalid style '{style}' for detecting random spherical region motif."
            )
