"""Define base motif class."""


from abc import ABC, abstractmethod
from typing import Optional


class BaseMotif(ABC):
    """A motif is a geometric element.

    Used to define an operable region, geometric element, or sub-collection
    of sites in a system.
    """
    # TODO: in the future, change these to read from registry.
    # List of allowed actions that can be performed on this motif.
    allowed_actions = []
    # List of allowed description styles for this motif.
    allowed_description_styles = []
    # Dict of styles stating how this motif can be used as a reference to the action on other motifs.
    # Keys are allowed styles, values are condition checking functions.
    allowed_relative_styles = {}

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