"""Motif comprising multiple atoms."""

from .base import BaseMotif
from src.AtomWorldBench.atom_world.motif_description_styles.utils import format_arraylike


class ClusterMotif(BaseMotif):
    """Motif representing a cluster of atoms.

    This motif is defined by a list of species and their fractional coordinates.
    It can be used to represent clusters of atoms in a structure.
    """
    allowed_actions = [
        "AddAction",
        "RemoveAction",
        "ReplaceAction",
        "TranslateAction",
        "RotateAction",
        "E3Action",
        "ResizeAction",
        "EdgeAction"
    ]

    def _get_default_name(self) -> str:
        """Generate a default name for the cluster motif based on species and coordinates."""
        return f"a cluster of atoms/species {', '.join(map(str, self.species))}"

    def describe(
            self,
            style: str = "coordinates",
            coord_style: str = "fractional",
    ) -> str:
        """Return a string description of the cluster motif.

        Args:
            style (str): The style of description. Default is "coordinates".
            coord_style (str): The coordinate style to use. Default is "fractional".
        Returns:
            str: A string description of the cluster motif.
        """
        prefix = (f"a cluster of {len(self.species)} atoms with"
                  f" species/elements: {', '.join(map(str, self.species))}")
        if style == "coordinates":
            if coord_style == "fractional":
                coord_word = "fractional coordinates"
                coords = self.frac_coords
            elif coord_style == "cartesian":
                coord_word = "cartesian coordinates (unit in Angstroms):"
                coords = self.cart_coords
            else:
                raise ValueError(f"Unknown coordinate style: {coord_style}.")
            return f"{prefix} corresponding to {coord_word}: {format_arraylike(coords)}"
        elif style == "index":
            if self.indices is None:
                raise ValueError("Indices are not set for this motif.")
            return f"{prefix} corresponding to site indices: {self.indices}"
        elif style == "centroid_relative_coordinates":
            centroid = self.frac_coords.mean(axis=0)
            relative_coords = self.frac_coords - centroid
            return (f"{prefix} with centroid-relative fractional coordinates: "
                    f"{format_arraylike(relative_coords)}")


