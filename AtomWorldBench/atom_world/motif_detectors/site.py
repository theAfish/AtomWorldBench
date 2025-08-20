"""Detector classes to find atoms."""
from typing import List, Optional

from numpy.typing import ArrayLike
from ase import Atoms

from .base import BaseDetector
from ..motifs.site_collections.site import SiteMotif
from ...utils.neighbor_utils import detect_indices_offests_around_frac_coords
from ...common.registry import register


@register(BaseDetector, ["site", "single-site"])
class SiteDetector(BaseDetector):
    """Class for detecting atoms."""

    def __init__(
            self,
            cutoff: float,
            symbols: List[str] = None,
            wrap: bool = True,
            seed: Optional[int] = None
    ):
        """Initialize the SiteDetector with a default radius.

        Args:
            cutoff (float): The default radius for detecting atoms around fractional coordinates.
            symbols (List[str]): If provided, only atoms with these symbols will be detected.
             Default is None, which means no filtering.
            wrap (bool): If True, the coordinates of the atoms will be wrapped to the unit cell.
             Does not apply to the `detect_around_frac_coords` method.
            seed (Optional[int]):
             Random seed for reproducibility of the `detect_one` method.
             Default is None, will generate with a random seed.
        """
        super().__init__(seed=seed)
        self.cutoff = cutoff
        self.symbols = symbols
        self.wrap = wrap

    def detect_around_frac_coords(
            self,
            atoms: Atoms,
            frac_coords: ArrayLike,
    ) -> List[SiteMotif]:
        """Detect atoms in the given structure around fractional coordinates.

        Notice: all site motifs detected by this method will be set to the default name.
        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
            frac_coords (ArrayLike): Fractional coordinates for the atom detection center.
                Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected site motifs within the specified radius.
        """
        if len(frac_coords) != 3:
            raise ValueError("frac_coords must be a one-dimensional array of shape (3,).")
        if "X" in atoms.get_chemical_symbols():
            raise ValueError("The structure already contains a dummy atom with symbol 'X'. "
                             "Please use a different symbol for the dummy atom.")


        indices_j_valid, offsets_valid = detect_indices_offests_around_frac_coords(
            atoms,
            frac_coords,
            self.cutoff,
            symbols=self.symbols,
        )
        positions_valid = (
                atoms.get_positions(wrap=False)[indices_j_valid] +
                offsets_valid @ atoms.cell.complete()
        )
        symbols_valid = atoms.get_chemical_symbols()[indices_j_valid]
        charges_valid = atoms.get_initial_charges()[indices_j_valid]

        deduplicated_motifs = []
        for ii in range(len(symbols_valid)):
            motif = SiteMotif(
                symbols=[symbols_valid[ii]],
                positions=[positions_valid[ii]],
                cell=atoms.cell,
                pbc=atoms.pbc,
                charges=charges_valid[ii],
                name=None,  # Default name will be generated in the SiteMotif class.
                indices=[indices_j_valid[ii]],
            )
            deduplicated_motifs.append(motif)
        return deduplicated_motifs

    def _get_symbol_valid_indices(self, atoms: Atoms) -> List[int]:
        """Get indices of atoms that match the specified symbols."""
        if self.symbols is None:
            return list(range(len(atoms)))
        else:
            return [
                i for i, symbol in enumerate(atoms.get_chemical_symbols())
                if symbol in self.symbols
            ]

    def detect_all(
            self,
            atoms: Atoms,
    ) -> List[SiteMotif]:
        """Detect all atoms in the given structure.

        Here, we return all site motifs with wrapped coordinates.
        Notice: all site motifs detected by this method will be set to the default name.
        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.

        Returns:
            List of detected site motifs. Whether the coordinates are wrapped or not
            depends on the `wrap` parameter at initialization.
        """
        if self.wrap:
            atoms_cp = atoms.copy()
            atoms_cp.wrap()
        else:
            atoms_cp = atoms
        valid_indices = self._get_symbol_valid_indices(atoms_cp)
        return [
            SiteMotif.from_atoms(atoms_cp[[i]], indices=[i])
            for i in valid_indices
        ]

    def detect_one(
            self,
            atoms: Atoms,
            **kwargs
    ) -> SiteMotif:
        """Detect a single atom in the given structure.

        This method detects a single atom at random in the structure.
        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.

        Returns:
            Detected site motif. Whether the coordinates are wrapped or not
            depends on the `wrap` parameter at initialization.
        """
        if self.wrap:
            atoms_cp = atoms.copy()
            atoms_cp.wrap()
        else:
            atoms_cp = atoms
        valid_indices = self._get_symbol_valid_indices(atoms_cp)
        rand_idx = self.rng.choice(valid_indices)
        return SiteMotif.from_atoms(atoms_cp[[rand_idx]], indices=[rand_idx])
