"""Defines a base region motif class."""

from abc import ABC, abstractmethod
from typing import Optional

from ase import Atoms
from numpy import ndarray
from numpy.typing import ArrayLike

from ..base import BaseMotif


class BaseRegionMotif(ABC, BaseMotif):
    """A base class for region motifs.

    A region motif is a geometric element that defines an operable region in space.
    It can be used to include all sites in a region for actions such as moving atoms.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            symbols: Optional[list[str]] = None,
    ):
        """Initialize the base region motif."""
        BaseMotif.__init__(self, name=name)
        self.symbols = symbols if symbols is not None else []

    @abstractmethod
    def get_included_atom_indices(self, atoms: Atoms) -> ndarray[int]:
        """Return the subset of atoms included in the region motif.

        If one of the periodic images lies within the region,
        it will include the corresponding atom.
        Args:
            atoms (Atoms): The ASE Atoms object containing all atoms in the system.
        Returns:
            ndarray[int]: An array of indices of atoms that are within the region motif.
        """
        pass
