from abc import ABC, abstractmethod

from ..motifs.base import BaseMotif


class BaseDescriptionStyle(ABC):
    """Base class for description styles in AtomWorldBench.

    A description style defines how to represent motifs in a specific format.
    This class provides an interface for defining and applying description styles
    to motifs.
    """

    @abstractmethod
    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif."""
        pass