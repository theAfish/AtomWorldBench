"""Defines the base class for actions in the atom world."""
from abc import ABC, abstractmethod

from pymatgen.core import Structure

# Create two more abstract subclasses of BaseAction: relative (operation of other sites with respect to a reference motif)
# and motif-action (action that operates on a motif, e.g., adding, removing, or replacing a motif).

class BaseAction(ABC):
    """Base class for actions that can be performed on crystal structures.

    An action is defined as an interface to change a pymatgen.structure object,
    yielding a new pymatgen structure object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.
    """
    def __init__(self, structure: Structure):
        """Initialize the action with a structure.

        Args:
            structure (pymatgen.core.structure.Structure): The structure to be modified by this action.
        """
        self._structure = structure
        self._prompt = None

    @property
    def structure(self):
        """Get the structure associated with this action."""
        return self._structure

    @structure.setter
    def structure(self, structure: Structure):
        """Set the structure associated with this action."""
        self._structure = structure
        self._prompt = None  # Reset prompt when structure changes.

    @property
    def prompt(self) -> str:
        if self._prompt is None:
            self._prompt = self._get_prompt()
        return self._prompt

    @abstractmethod
    def _get_prompt(self) -> str:
        """Generate a prompt describing the action performed on the structure."""
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> Structure:
        """Execute the action on the structure to generate the ground truth structure."""
        raise NotImplementedError