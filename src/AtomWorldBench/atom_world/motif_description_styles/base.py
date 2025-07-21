from abc import ABC, abstractmethod

from ..motifs.base import BaseMotif


class BaseDescriptionStyle(ABC):
    """Base class for description styles in AtomWorldBench.

    A description style defines how to represent motifs in a specific format.
    This class provides an interface for defining and applying description styles
    to motifs.
    """
    # Some description styles need to append an introduction to the prompt.
    introduction = ""

    def __init__(self, is_addition: bool = False):
        """Initialize the description style.

        Args:
            is_addition (bool): whether this style is used for describing an add motif action.
                Controls generated description. For example, add motif action typically does
                not require to describe the motif's centroid coordinates or its indices in
                structure, as the action is about adding a motif to the structure.
                Default is False.
        """
        self.is_addition = is_addition

    @abstractmethod
    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif."""
        pass