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
    forbidden_actions = [
        "replace-motif",
        "add-motif",
        "translate-motif",
        "rotate-motif",
        "resize-motif"
    ]  # For most regions, these actions are not meaningful except for removal.

    def __init__(
            self,
            in_atoms: Atoms,
            name: Optional[str] = None,
            symbols: Optional[list[str]] = None,
    ):
        """Initialize the base region motif.

        Args:
            in_atoms (Atoms): The atoms that this region motif is in.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            name (str, optional): The name of the motif. If None, a default name will be generated.
            symbols (list[str], optional): A list of chemical symbols that this motif includes.
                Other elements will not be selected as part of this motif.
        """
        BaseMotif.__init__(self, in_atoms, name=name)
        self.symbols = symbols
        self._indices = None
        self._offsets = None
        # In regions, indices and offsets are not directly settable.
        # In principle, the region boundaries should never be changed directly.

    @property
    def indices(self) -> list[int]:
        """Return the indices of sites in this region within the parent atoms."""
        if self._indices is None:
            indices, _ = self._get_site_indices_offsets_in_atoms()
            self._indices = indices
        return self._indices

    @property
    def cell_offsets(self) -> ndarray:
        """Return the cell offsets of sites in this region within the parent atoms."""
        if self._offsets is None:
            _, cell_offsets = self._get_site_indices_offsets_in_atoms()
            self._offsets = cell_offsets
        return self._offsets

