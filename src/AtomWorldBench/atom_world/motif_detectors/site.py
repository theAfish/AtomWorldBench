"""Detector classes to find atoms."""
from numpy._typing import ArrayLike
from pymatgen.core import Structure

from .base import BaseDetector
from ..motifs import SiteMotif

class SiteDetector(BaseDetector):
    """Class for detecting atoms."""

    def detect_around_frac_coords(
            self,
            structure: Structure,
            frac_coords: ArrayLike,
            radius: float = 3.0,
    ):
        """Detect atoms in the given structure around fractional coordinates.

        Args:
            structure (Structure): The structure to analyze.
            frac_coords (ArrayLike): Fractional coordinates for the atom detection center.
                Must be a one-dimensional array of shape (3,).
            radius (float): The radius around the fractional coordinates to consider for detection.

        Returns:
            List of detected atoms.
        """
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = structure.get_sites_in_sphere(cart_coords, radius)
        return [
            SiteMotif.from_fractional_coordinates(
                specie=site.specie,
                frac_coords=site.frac_coords,
                lattice=structure.lattice,
                index=site.index
            )
            for site in neighbors
        ]

    def detect_all(
            self,
            structure: Structure,
    ):
        """Detect all atoms in the given structure.

        Args:
            structure (Structure): The structure to analyze.

        Returns:
            List of detected atoms.
        """
        return [
            SiteMotif.from_structure_indices(structure, index)
            for index in range(len(structure))
        ]
