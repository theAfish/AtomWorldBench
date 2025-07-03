"""Describe a motif in its atomic coordinates."""

from .base import BaseDescriptionStyle

from ..motifs.base import BaseMotif
from .utils import format_arraylike

class CoordDescriptionStyle(BaseDescriptionStyle):
    """Description style for motifs in atomic coordinates.

    This style describes a motif using its atomic coordinates, either in fractional
    or Cartesian format.
    """

    def __init__(self, flavor: str = "fractional", precision: int = 4):
        """Initialize the description style with a specific flavor.

        Args:
            flavor (str): The flavor of the description style, e.g., "fractional", "cartesian".
        """

        self.flavor = flavor
        self.precision = precision

    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif.

        Args:
            motif: The motif to describe.

        Returns:
            str: A string description of the motif's coordinates.
        """
        if self.flavor == "fractional":
            coords = motif.frac_coords
            coord_word = "fractional coordinates"
        elif self.flavor == "cartesian":
            coords = motif.cart_coords
            coord_word = "cartesian coordinates (unit in Angstroms)"
        else:
            raise ValueError(f"Unknown coordinate flavor: {self.flavor}.")
        coords_str = format_arraylike(coords, precision=self.precision)

        return f"{motif.name} with {coord_word} {coords_str}"