"""BaseDetector class for AtomWorldBench."""
from typing import List
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from ase import Atoms

class BaseDetector(ABC):
    """Base class for motif detectors in the atom world.

    A detector is responsible for identifying motifs within a structure.
    This class provides an interface for defining and applying detectors to structures.
    """

    @abstractmethod
    def detect_around_frac_coords(
            self,
            atoms: Atoms,
            frac_coords: ArrayLike,
    ):
        """Detect motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
            frac_coords(ArrayLike): Fractional coordinates for the motif detection center.
             Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected motifs.
        """
        pass

    def detect_around_site_indices(
            self,
            atoms: Atoms,
            indices: List[int],
    ):
        """Detect motifs in the given structure based on indices.

        This method can be overridden by subclasses if they need to implement
        detection based on specific indices rather than fractional coordinates.
        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
            indices(List[int]): List of atomic site indices in structure to consider
             for detecting around.

        Returns:
            List of detected motifs.
        """
        motifs = []
        for index in indices:
            frac_coords = atoms.get_scaled_positions(wrap=False)[index]
            motifs.extend(self.detect_around_frac_coords(atoms, frac_coords))
        return motifs


    def detect_all(
            self,
            atoms: Atoms,
    ):
        """Detect all motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Separated from detect_around_frac_coords to allow more efficient detection strategies.
        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.

        Returns:
            List of detected motifs.
        """
        return self.detect_around_site_indices(
            atoms,
            list(range(len(atoms)))
        )