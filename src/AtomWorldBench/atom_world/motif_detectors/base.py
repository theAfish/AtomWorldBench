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
            radius: float = 3.0,
    ):
        """Detect motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Args:
            structure(Structure): The structure to analyze.
            frac_coords(ArrayLike): Fractional coordinates for the motif detection center.
             Must be a one-dimensional array of shape (3,).
            radius(float): The radius around the fractional coordinates to consider for detection.

        Returns:
            List of detected motifs.
        """
        pass

    def detect_around_site_indices(
            self,
            structure: Structure,
            indices: List[int],
            radius: float | List[float] = 3.0,
    ):
        """Detect motifs in the given structure based on indices.

        This method can be overridden by subclasses if they need to implement
        detection based on specific indices rather than fractional coordinates.
        Args:
            structure(Structure): The structure to analyze.
            indices(List[int]): List of atomic site indices in structure to consider
             for detecting around.
            radius(float | List[float]): The radius around the atomic sites to consider
             for detection. If specified as a single float, it applies to all indices.
             If specified as a list, it should match the length of indices.

        Returns:
            List of detected motifs.
        """
        if isinstance(radius, float):
            radius = [radius] * len(indices)
        if len(radius) != len(indices):
            raise ValueError("Length of radius must match length of indices.")
        motifs = []
        for index, r in zip(indices, radius):
            frac_coords = structure[index].frac_coords
            motifs.extend(self.detect_around_frac_coords(structure, frac_coords, r))
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