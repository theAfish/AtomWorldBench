"""Detector classes to find atoms."""
from typing import List

from numpy.typing import ArrayLike
from pymatgen.core import Structure

from .base import BaseDetector
from ..motifs import SiteMotif

class SiteDetector(BaseDetector):
    """Class for detecting atoms."""

    def __init__(self, radius: float = 3.0):
        """Initialize the SiteDetector with a default radius.

        Args:
            radius (float): The default radius for detecting atoms around fractional coordinates.
                Default is 3.0 Angstroms.
        """
        self.radius = radius

    def detect_around_frac_coords(
            self,
            structure: Structure,
            frac_coords: ArrayLike,
    ) -> List[SiteMotif]:
        """Detect atoms in the given structure around fractional coordinates.

        Notice: all site motifs detected by this method will be set to the default name.
        Args:
            structure (Structure): The structure to analyze.
            frac_coords (ArrayLike): Fractional coordinates for the atom detection center.
                Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected atoms within the specified radius.
        """
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = structure.get_sites_in_sphere(cart_coords, self.radius)
        return [
            SiteMotif(
                species=site.specie,
                frac_coords=site.frac_coords,
                lattice=structure.lattice,
                indices=site.index
            )
            for site in neighbors
        ]

    def detect_all(
            self,
            structure: Structure,
    ) -> List[SiteMotif]:
        """Detect all atoms in the given structure.

        Notice: all site motifs detected by this method will be set to the default name.
        Args:
            structure (Structure): The structure to analyze.

        Returns:
            List of detected atoms.
        """
        return [
            SiteMotif.from_structure_index(structure, index)
            for index in range(len(structure))
        ]
