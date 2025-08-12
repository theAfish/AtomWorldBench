"""BaseDetector class for AtomWorldBench."""
from typing import List, Optional
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from numpy.random import default_rng
from ase import Atoms

from ..motifs.site_collections.base import BaseSiteCollectionMotif

class BaseDetector(ABC):
    """Base class for motif detectors in the atom world.

    A detector is responsible for identifying motifs within a structure.
    This class provides an interface for defining and applying detectors to structures.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            **kwargs
    ):
        """Initialize the detector with an optional random seed.

        Args:
            seed (int, optional): Random seed for reproducibility in method `detect_one`.
                Defaults to None, will use a random seed if not provided.
            **kwargs: Additional keyword arguments that may be used by specific detectors.
        """
        self._rng = default_rng(seed)

    @property
    def rng(self):
        """Random number generator for the detector.

        Uses numpy's default_rng for random number generation.
        Cannot be set directly, set seed at initialization.
        """
        return self._rng

    @abstractmethod
    def detect_around_frac_coords(
            self,
            atoms: Atoms,
            frac_coords: ArrayLike,
    ) -> List[BaseSiteCollectionMotif]:
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
    ) -> List[BaseSiteCollectionMotif]:
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

    @abstractmethod
    def detect_one(
            self,
            atoms: Atoms,
            **kwargs
    ) -> BaseSiteCollectionMotif:
        """Detect a single motif at random in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        It is intended for use when only one motif is expected to be detected.

        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
            **kwargs: Additional keyword arguments that may be used by specific detectors.
        Returns:
            Detected motif or None if no motif is found.
        """
        pass


    def detect_all(
            self,
            atoms: Atoms,
    ) -> List[BaseSiteCollectionMotif]:
        """Detect all motifs in the given structure.

        This method should be implemented by every subclass to analyze the structure.
        Separated from detect_around_frac_coords to allow more efficient detection strategies.
        Developers are encouraged to override this method with more efficient implementations
         if they can.
        Args:
            atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.

        Returns:
            List of detected motifs.
        """
        return self.detect_around_site_indices(
            atoms,
            list(range(len(atoms)))
        )