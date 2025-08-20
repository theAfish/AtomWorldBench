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
        BaseMotif.__init__(self, name=name)
        self.in_atoms = in_atoms
        self.symbols = symbols if symbols is not None else []

    def get_atoms(self) -> Atoms:
        """Return an atoms object including all sites in this region."""
        indices = self.get_site_indices_in_atoms(self.in_atoms)
        return self.in_atoms[indices]

    @property
    def cart_coords(self) -> ndarray:
        return self.get_atoms().get_positions(wrap=False)

    @property
    def frac_coords(self) -> ndarray:
        return self.get_atoms().get_scaled_positions(wrap=False)
