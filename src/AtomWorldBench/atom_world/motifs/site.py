from typing import List, Optional

from numpy.typing import ArrayLike
from pymatgen.util.typing import SpeciesLike
from pymatgen.core import get_el_sp, Structure, Element

from .base import BaseMotif, LatticeLike
from ...utils.format_utils import format_arraylike

class SiteMotif(BaseMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """
    allowed_actions = [
        "AddAction",
        "RemoveAction",
        "ReplaceAction",
        "TranslateAction",
        "ChangeAction"
    ]

    def __init__(
            self,
            name: str,
            species: List[SpeciesLike],
            frac_coords: ArrayLike,
            lattice: LatticeLike,
            indices: Optional[List[int]] = None,
    ):
        """Initialize the SiteMotif with a unique identifier."""
        if len(species) != 1 or len(frac_coords) != 1 or len(indices) != 1:
            raise ValueError(
                "SiteMotif must be initialized with exactly one species,"
                " one fractional coordinate, and one index."
            )
        super().__init__(
            name=name,
            species=species,
            frac_coords=frac_coords,
            lattice=lattice,
            indices=indices,
        )

    @classmethod
    def from_fractional_coordinates(
            cls,
            specie: SpeciesLike,
            frac_coords: ArrayLike,
            lattice: LatticeLike,
            index: int = None,
    ):
        """Create a AtomMotif from fractional coordinates.

        Args:
            specie (SpeciesLike): The species of the atom.
            frac_coords (ArrayLike): Fractional coordinates of the motif.
            lattice (LatticeLike): The lattice of the structure to which this motif belongs.
            index (int, optional): Index of the atom in the structure that correspond to this motif.
             If not yet added, these indices can be empty or None and will be set by the AddAction.

        Returns:
            SiteMotif: An instance of AtomMotif with the specified coordinates.
        """
        specie = get_el_sp(specie)
        return cls(
            name=str(specie),
            species=[specie],
            frac_coords=[frac_coords],
            lattice=lattice,
            indices=[index] if index is not None else None,
        )

    @classmethod
    def from_structure_indices(
            cls,
            structure,
            indices: List[int],
    ):
        raise NotImplementedError(
            "SiteMotif does not support from_structure_indices method. Use from_structure_index instead."
        )

    @classmethod
    def from_structure_index(cls, structure: Structure, index: int):
        """Create a PointMotif from a structure and its indices.

        Args:
            structure (Structure): The structure containing the atom.
            index (int): Index of the atom in the structure that correspond to this motif.

        Returns:
            PointMotif: An instance of AtomMotif with the specified coordinates.
        """
        return cls(
            name=str(structure[index].specie),
            species=[structure[index].specie],
            frac_coords=structure[index].frac_coords,
            lattice=structure.lattice,
            indices=[index],
        )

    def describe(
            self,
            style: str = "coordinates",
            coord_style: str ="fractional",
            precision: int =4
    ) -> str:
        """Return a description of the motif.
        
        Args:
            style (str): The style of description. Can be "coordinates", or "index".
                Defaults to "coordinates".
            coord_style (str): The coordinate style to use for the description.
             Can be "fractional" or "cartesian". Defaults to "fractional".
            precision (int): Number of decimal places for coordinates. Defaults to 4.
        Returns:
            str: A string description of the motif to be concatenated to the prompt.
        """
        if isinstance(self.species[0], Element):
            prefix = "an atom"
        else:
            prefix = "a species"
        if style == "coordinates":
            if coord_style=="fractional":
                coord_word = "fractional coordinates:"
                coord = self.frac_coords[0]
            elif coord_style == "cartesian":
                coord_word = "cartesian coordinates (unit in Angstroms):"
                coord = self.cart_coords[0]
            else:
                raise ValueError(f"Unknown coordinate style: {style}.")
            coords_str =  format_arraylike(coord, precision=precision)
            return f"{prefix} with {coord_word} {coords_str}"
        elif style == "index":
            if self.indices is None or len(self.indices) == 0:
                raise ValueError("Indices are not set for this motif.")
            return f"{prefix} with site index {self.indices[0]}"
        else:
            raise ValueError(f"Unknown representation style: {style}.")