"""Describe a motif in its atomic coordinates."""

from .base import BaseDescriptionStyle

from ..motifs.base import BaseMotif
from src.AtomWorldBench.utils.description_utils import format_arraylike

class CoordDescriptionStyle(BaseDescriptionStyle):
    """Description style for motifs in atomic coordinates.

    This style describes a motif using its atomic coordinates, either in fractional
    or Cartesian format.
    """

    introduction = "" # No specific introduction for this style.

    def __init__(
            self,
            flavor: str = "fractional",
            precision: int = 4,
            center: bool = False,
    ):
        """Initialize the description style with a specific flavor.

        Args:
            flavor (str): The flavor of the description style, e.g., "fractional", "cartesian".
            precision (int): The number of decimal places to format the coordinates.
            center (bool): Whether to center the motif cartesian positions around the centroid,
             and describe the coordinates relative to the centroid.
        """

        self.flavor = flavor
        self.precision = precision
        self.center = center

    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif.

        Args:
            motif: The motif to describe.

        Returns:
            str: A string description of the motif's coordinates.
        """
        if not self.center:
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
        else:
            if self.flavor == "fractional":
                centroid = motif.get_centroid(fractional=True)
                coords = motif.frac_coords - centroid
                coord_word = "fractional coordinates"
            elif self.flavor == "cartesian":
                centroid = motif.get_centroid(fractional=False)
                coords = motif.cart_coords - centroid
                coord_word = "cartesian coordinates"
            else:
                raise ValueError(f"Unknown coordinate flavor: {self.flavor}.")
            centroid_str = format_arraylike(centroid, precision=self.precision)
            coords_str = format_arraylike(coords, precision=self.precision)

            return (f"{motif.name} with {coord_word} {coords_str}"
                    f" relative to a centroid {coord_word} {centroid_str}")