"""Define base motif class."""
from abc import ABC, abstractmethod
from typing import Optional
from numpy import ndarray

from ase import Atoms


class BaseMotif(ABC):
    """A motif is a geometric element.

    Used to define an operable region, geometric element, or sub-collection
    of sites in a system.

    Class Attributes:
        forbidden_actions (list): A list of action class names that are not
            allowed to operate on this motif. We only use black-listing for now.
    """
    forbidden_actions = []

    def __init__(
            self,
            in_atoms: Atoms,
            name: Optional[str] = None
    ):
        """Initialize the motif with an optional name.

        Args:
            in_atoms (Atoms): An ASE Atoms object that this motif belongs to.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            name (str, optional): The name of the motif.
                If None, a default name will be generated and set.
        """
        # Prevent calling the setter directly in the constructor
        # As at the initialization, the attributes required to compute
        # the default name are not set yet.
        # Wrap the atoms if not already.
        in_atoms_cp = in_atoms.copy()
        in_atoms_cp.wrap()
        self.in_atoms = in_atoms_cp  # Avoid affecting the original atoms.
        self._name = name
        self._atoms = None  # Subset of in_atoms that belongs to this motif.

    @abstractmethod
    def _get_default_name(self) -> str:
        """Generate a default name based on motif type, species and coordinates."""
        pass

    @property
    def name(self) -> str:
        """Set the name of the motif."""
        if self._name is None:
            return self._get_default_name()
        return self._name

    @name.setter
    def name(self, name: Optional[str] = None):
        """Set the name of the motif.

        Args:
            name (str, optional): The name of the motif.
                If None, a default name will be generated and set.
        """
        self._name = name if name is not None else self._get_default_name()

    @abstractmethod
    def describe(self) -> str:
        """Return a string description of the motif."""
        pass

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
        if self._atoms is None:
            indices, offsets = self._get_site_indices_offsets_in_atoms()
            self._atoms = self.in_atoms[indices]
            # Apply offsets to positions.
            positions_orig = self._atoms.get_positions(wrap=False)
            positions_new = positions_orig + offsets @ self.in_atoms.cell.complete()
            self._atoms.set_positions(positions_new)
        return self._atoms

    @property
    def cart_coords(self) -> ndarray:
        return self.get_atoms().get_positions(wrap=False)

    @property
    def frac_coords(self) -> ndarray:
        return self.get_atoms().get_scaled_positions(wrap=False)

    @classmethod
    @abstractmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            seed: Optional[int] = None,
            **kwargs,
    ) -> 'BaseMotif':
        """Detect a random motif from the given atoms.

        This method should be implemented by every non-abstract subclass
        to detect a random motif from the provided Atoms object.

        Used for generating actions and prompts from a given atoms object.

        Args:
            atoms (Atoms): An ASE Atoms object containing all atoms in the system.
            seed (Optional[int]): An optional seed for random number generation.
                If provided, it ensures reproducibility of the random motif detection.

        Returns:
            BaseMotif: An instance of a subclass of BaseMotif representing the detected motif.
        """
        pass
