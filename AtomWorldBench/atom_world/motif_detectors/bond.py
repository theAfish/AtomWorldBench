"""Bond motif detector."""
from typing import List, Optional

from ase import Atoms
from numpy.typing import ArrayLike

from .cluster import ClusterDetector

from ..motifs.site_collections.bond import BondMotif
from .base import BaseDetector

from ...common.registry import register


@register(BaseDetector, ["bond"])
class BondDetector(ClusterDetector):
    """Detects bonds between atoms in a structure.

    This detector identifies pairs of atoms that are within a specified distance (cutoff)
    and can be used to find chemical bonds or interactions between atoms.
    """
    def __init__(self, cutoff: float | dict = 3.0, symbols: list[str] = None):
        """Initialize the BondDetector with a default radius.

        Args:
            cutoff (float | dict): The default radius for detecting bonds.
             See `ase.neighbor_list` for details.
            symbols (list[str]): If provided, only bonds between these symbols will be detected.
             Default is None, which means no filtering.
        """
        super().__init__(
            cutoff=cutoff,
            max_cluster_size=2,  # Bonds are always pairs of atoms.
            max_cluster_radius=cutoff if isinstance(cutoff, (float, int)) else max(cutoff.values()),
            must_include_center=True,
            symbols=symbols
        )

    def detect_around_frac_coords(
            self,
            atoms: Atoms,
            frac_coords: ArrayLike,
    ) -> List[BondMotif]:
        """Detect bonds in the given structure around fractional coordinates.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
            frac_coords (ArrayLike): Fractional coordinates for the bond detection center.
             Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected bond motifs within the specified radius.
        """
        return [
            BondMotif.from_cluster_motif(motif)
            for motif in super().detect_around_frac_coords(atoms, frac_coords)
        ]

    def detect_one(
            self,
            atoms: Atoms,
            size: Optional[int] = None,
            n_attempts: Optional[int] = 10,
    ) -> BondMotif | None:
        """Detect a single bond motif in the given structure.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
            size (int, optional): Unused parameter just for linting. Will always be 2.
            n_attempts (Optional[int]): Number of attempts to find a bond motif. Default is 10.

        Returns:
            A BondMotif object if a bond is detected, otherwise None.
        """
        _ = size  # Size is ignored for bonds, always 2.
        motif = ClusterDetector.detect_one(self, atoms, size=2, n_attempts=n_attempts)
        if motif is None:
            return None
        return BondMotif.from_cluster_motif(motif)
