from numpy.typing import ArrayLike
from pymatgen.util.typing import SpeciesLike
from pymatgen.core import get_el_sp, Structure, Element

from .base import BaseMotif, LatticeLike

class SiteMotif(BaseMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """
    allowed_actions = [
        "AddAction",
        "RemoveAction",
        "TranslateAction",
        "ChangeAction"
    ]
    allowed_detectors = ["SiteDetector"]

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
    def from_structure_indices(cls, structure: Structure, index: int):
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

    def describe(self, style="frac_coord") -> str:
        """Return a description of the motif."""
        if isinstance(self.species[0], Element):
            prefix = "an atom"
        else:
            prefix = "a species"
        if "coord" in style:
            if style=="frac_coord":
                coord_word = "fractional coordinates"
                coord = self.frac_coords[0]
            elif style=="cart_coord":
                coord_word = "cartesian coordinates"
                coord = self.cart_coords[0]
            else:
                raise ValueError(f"Unknown style: {style}")
            coords_str =  f"({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})"
            return f"{prefix} with {coord_word} {coords_str}."
        elif "index" in style:
            return f"{prefix} with site index {self.indices[0]}."
        else:
            raise ValueError(f"Unknown style: {style}")