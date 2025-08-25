"""Defines a base region motif class."""

from abc import ABC, abstractmethod
from typing import Optional

from ase import Atoms
from numpy import ndarray

from ..base import BaseMotif


class BaseRegionMotif(BaseMotif, ABC):
    """A base class for region motifs.

    A region motif is a geometric element that defines an operable region in space.
    It can be used to include all sites in a region for actions such as moving atoms.

    Regions do not support replace or add actions, as you can only add or replace specific
    atoms not regions.
    Also translate actions are not supported, as translating a region is equivalent to
    translating all atoms in the region, which can be done with a translation action on the atoms
    directly.
    Resize actions are also not supported except for sphere region, as defining a radius for
    the is not meaningful for most regions.
    """
    forbidden_actions = ["replace", "add", "translate", "resize"]

    def __init__(
            self,
            in_atoms: Atoms,
            name: Optional[str] = None,
            symbols: Optional[list[str]] = None,
    ):
        """Initialize the base region motif.

        Args:
            in_atoms (Atoms): The atoms that this region motif is in.
            name (str, optional): The name of the motif. If None, a default name will be generated.
            symbols (list[str], optional): A list of chemical symbols that this motif includes.
                Other elements will not be selected as part of this motif.
        """
        BaseMotif.__init__(self, in_atoms, name=name)
        self.symbols = symbols
        self._atoms_subset = None

    @abstractmethod
    def _get_site_indices_offsets_in_atoms(self) -> tuple[list[int], ndarray]:
        """Get the indices and periodic offsets of sites in this region within the parent atoms.

        Must be implemented by subclasses.
        Returns:
            indices (list[int]): A list of indices of sites in this region within the parent atoms.
            offsets (list[ndarray]): A list of periodic offsets for each site in this region relative
                to the parent atoms.get_positions(wrap=False).
                Each offset is a numpy array of shape (3,).
        """
        pass

    def get_atoms(self) -> Atoms:
        """Return an atoms object including all sites in this region."""
        if self._atoms_subset is None:
            indices, offsets = self._get_site_indices_offsets_in_atoms()
            self._atoms_subset = self.in_atoms[indices]
            # Apply offsets to positions.
            positions_orig = self._atoms_subset.get_positions(wrap=False)
            positions_new = positions_orig + offsets @ self.in_atoms.cell.complete()
            self._atoms_subset.set_positions(positions_new)
        return self._atoms_subset

    @property
    def cart_coords(self) -> ndarray:
        return self.get_atoms().get_positions(wrap=False)

    @property
    def frac_coords(self) -> ndarray:
        return self.get_atoms().get_scaled_positions(wrap=False)
