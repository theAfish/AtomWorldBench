"""Detector classes to find atoms."""
from typing import List, Dict

import numpy as np
from numpy.typing import ArrayLike
from ase import Atoms
from ase.neighborlist import neighbor_list

from .base import BaseDetector
from ..motifs.site import SiteMotif


class SiteDetector(BaseDetector):
    """Class for detecting atoms."""

    def __init__(self, cutoff: float, symbols: List[str] = None):
        """Initialize the SiteDetector with a default radius.

        Args:
            cutoff (float): The default radius for detecting atoms around fractional coordinates.
            symbols (List[str]): If provided, only atoms with these symbols will be detected.
             Default is None, which means no filtering.
        """
        self.cutoff = cutoff
        self.symbols = symbols

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

        # Add a dummy atom to the structure at the given fractional coordinates to mark.
        atoms_modified = atoms.copy()
        atoms_modified += Atoms("X", positions=[frac_coords], cell=atoms.cell, pbc=atoms.pbc)
        # Dummy atom is added to the end of the atoms list.
        dummy_index = len(atoms_modified) - 1

        # Get the indices of atoms within the cutoff distance from the given fractional coordinates
        # No need to use NeighborList class, as it also checks for the whole structure.
        indices_i, indices_j, offsets = neighbor_list(
            "ijS",
            atoms_modified,
            cutoff=self.cutoff,
            self_interaction=False,
        )

        symbols = np.array(atoms_modified.get_chemical_symbols())
        # Filter indices to only include those that are within the cutoff distance from the dummy atom,
        # and that match the specified symbols if provided.
        indices_valid = (indices_i == dummy_index) & np.vectorize(lambda x: x in self.symbols)(symbols[indices_j])
        indices_j_valid = indices_j[indices_valid]
        offsets_valid = offsets[indices_valid, :]
        positions_valid = (
                atoms_modified.get_positions(wrap=False)[indices_j_valid] +
                offsets_valid @ atoms_modified.cell
        )
        symbols_valid = symbols[indices_j_valid]
        charges_valid = atoms_modified.get_initial_charges()[indices_j_valid]

        deduplicated_motifs = []
        for ii in range(len(symbols_valid)):
            motif = SiteMotif(
                symbols=[symbols_valid[ii]],
                positions=[positions_valid[ii]],
                cell=atoms_modified.cell,
                pbc=atoms_modified.pbc,
                charges=charges_valid[ii],
                name=None,  # Default name will be generated in the SiteMotif class.
                indices=[indices_j_valid[ii]],
            )
            deduplicated_motifs.append(motif)
        return deduplicated_motifs

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
            List of detected site motifs.
        """
        atoms_cp = atoms.copy()
        atoms_cp.wrap()
        return [
            SiteMotif.from_atoms(atoms_cp[[i]], indices=[i])
            for i in range(len(atoms_cp))
        ]
