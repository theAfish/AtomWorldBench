"""Defines a base region motif class."""

from abc import ABC, abstractmethod
from typing import Optional

from ase import Atoms
from numpy import ndarray

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
