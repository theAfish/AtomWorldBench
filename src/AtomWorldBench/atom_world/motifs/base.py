"""Define base motif class."""
from abc import ABC, abstractmethod
from typing import Optional

from ase import Atoms


class BaseMotif(ABC):
    """A motif is a geometric element.

    Used to define an operable region, geometric element, or sub-collection
    of sites in a system.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the motif with an optional name."""
        self.name = name

    def __post_init__(self):
        """Post-initialization to set the motif mets criterion of its type.

        This method can be overridden by subclasses to perform additional initialization.
        """
        pass

    @abstractmethod
    def _get_default_name(self) -> str:
        """Generate a default name based on motif type, species and coordinates."""
        pass

    @property
    def name(self) -> str:
        """Set the name of the motif."""
        return self._name

    @name.setter
    def name(self, name: Optional[str] = None):
        """Set the name of the motif.

        Args:
            name (str, optional): The name of the motif. If None, a default name will be generated.
        """
        self._name = name if name is not None else self._get_default_name()

    @abstractmethod
    def describe(self) -> str:
        """Return a string description of the motif."""
        pass

    @abstractmethod
    def get_site_indices_in_atoms(self, atoms: Atoms) -> list[int]:
        """Return the indices of sites included in the motif.

        This method will be the interface for the action to determine the
        sites to operate on.
        Args:
            atoms: An ASE Atoms object containing all atoms in the system.
        Returns:
            list[int]: A list of indices of sites that are included in the motif.
        """
        pass