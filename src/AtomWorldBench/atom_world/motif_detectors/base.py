"""BaseDetector class for AtomWorldBench."""
from typing import List
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from pymatgen.core import Structure

class BaseDetector(ABC):
    """Base class for motif detectors in the atom world.

    A detector is responsible for identifying motifs within a structure.
    This class provides an interface for defining and applying detectors to structures.
    """

    @abstractmethod
    def detect_around_frac_coords(
            self,
            structure: Structure,
            frac_coords: ArrayLike,
    ):
        """Detect motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Args:
            structure(Structure): The structure to analyze.
            frac_coords(ArrayLike): Fractional coordinates for the motif detection center.
             Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected motifs.
        """
        pass

    def detect_around_site_indices(
            self,
            structure: Structure,
            indices: List[int],
    ):
        """Detect motifs in the given structure based on indices.

        This method can be overridden by subclasses if they need to implement
        detection based on specific indices rather than fractional coordinates.
        Args:
            structure(Structure): The structure to analyze.
            indices(List[int]): List of atomic site indices in structure to consider
             for detecting around.

        Returns:
            List of detected motifs.
        """
        motifs = []
        for index in indices:
            frac_coords = structure[index].frac_coords
            motifs.extend(self.detect_around_frac_coords(structure, frac_coords))
        return motifs

    @abstractmethod
    def detect_all(
            self,
            structure: Structure,
    ):
        """Detect all motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Separated from detect_around_frac_coords to allow more efficient detection strategies.
        Args:
            structure(Structure): The structure to analyze.

        Returns:
            List of detected motifs.
        """
        pass