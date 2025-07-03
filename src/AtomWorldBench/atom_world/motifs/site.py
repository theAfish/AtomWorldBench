from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.util.typing import SpeciesLike
from pymatgen.core import Structure, Element

from .base import BaseMotif, LatticeLike
from ..motif_description_styles import description_style_factory

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
    allowed_description_styles = [
        "CoordDescriptionStyle",
        "IndexDescriptionStyle",
    ]

    def __init__(
            self,
            species: SpeciesLike | List[SpeciesLike],
            frac_coords: ArrayLike,
            lattice: LatticeLike,
            indices: Optional[int | List[int]] = None,
            name: Optional[str] = None
    ):
        """Initialize the SiteMotif with a unique identifier.

        Args:
            species (SpeciesLike | List[SpeciesLike]): the specie that make up the motif.
            frac_coords (ArrayLike): Fractional coordinates of the motif.
            lattice (LatticeLike): Lattice of the structure.
            indices (Optional[int | List[int]]): Indices of the site in the structure.
            name (Optional[str]): Name of the motif, defaults to None.
              If provided, it will always overwrite automatically generated names
              based on species and coordinates.
        """
        # Unify format.
        if not isinstance(species, (list, tuple, set)):
            species = [species]
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim == 1:
            frac_coords = frac_coords.reshape(1, -1)
        if len(species) != 1 or len(frac_coords) != 1 or len(indices) != 1:
            raise ValueError(
                "SiteMotif must be initialized with exactly one species,"
                " one fractional coordinate, and one index."
            )
        super().__init__(
            species=species,
            frac_coords=frac_coords,
            lattice=lattice,
            indices=indices,
            name=name,
        )

    @classmethod
    def from_structure_indices(
            cls,
            structure,
            indices: List[int],
            name: Optional[str] = None,
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

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if isinstance(self.species[0], Element):
            return f"an atom {self.species[0].symbol}"
        else:
            return f"a species {str(self.species[0])}"

    def describe(
            self,
            style: str = "coord",
            coord_style: str ="fractional",
            precision: int = 4,
    ) -> str:
        """Return a description of the motif.
        
        Args:
            style (str): The style of description. Can be "coordi", or "index".
                Defaults to "coord".
            coord_style (str): The coordinate style to use for the description.
             Can be "fractional" or "cartesian". Defaults to "fractional".
            precision (int): Number of decimal places for coordinates. Defaults to 4.
        Returns:
            str: A string description of the motif to be concatenated to the prompt.
        """
        description_style = description_style_factory(
            style,
            flavor=coord_style,
            precision=precision
        )