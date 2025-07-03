from abc import ABC, abstractmethod


class BaseDescriptionStyle(ABC):
    """Base class for description styles in AtomWorldBench.

    A description style defines how to represent motifs in a specific format.
    This class provides an interface for defining and applying description styles
    to motifs.
    """
    def __init__(self, flavor: str):
        """Initialize the description style with a specific flavor.

        Args:
            flavor (str): The flavor of the description style, e.g., "fractional", "cartesian".
        """
        self.flavor = flavor

    @abstractmethod
    def describe(self, motif):
        """Generate a description for the given motif."""
        pass

    @abstractmethod
    def format_description(self, description):
        """Format the generated description into a string."""
        pass