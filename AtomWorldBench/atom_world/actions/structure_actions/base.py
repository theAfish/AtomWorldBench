"""Defines the base class for structure-wise operations in the atom world."""
from abc import ABC, abstractmethod
import re
from typing import Optional

from ase import Atoms

from AtomWorldBench.common.mixin_classes import MultiModeInitMixin


class BaseStructureAction(MultiModeInitMixin, ABC):
    """Base class for actions that can be performed on a crystal structure.

    Currently, support actions that replaces all atoms of a certain species
    with another species, or resizes of changes the cell lattice parameters.

    An action is defined as an interface to change a ase.Atoms object,
    yielding a new ase atoms object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.

    A specific action should be a subclass of BaseStructureAction and MultiModeInitMixin.
    See documentation for AtomWorldBench.mixin_classes for more details on how to
    implement actions.
    """
    def __init__(
            self,
            operated_atoms: Atoms,
            **kwargs,
    ):
        """Initialize the action with the operated motif and atoms.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on. Required.
        """
        # Only for suppressing linter warnings
        self.operated_atoms = None
        MultiModeInitMixin.__init__(
            self,
            operated_atoms=operated_atoms,
            **kwargs
        )

    @abstractmethod
    def __post_init__(self):
        """Check compatibility of the action with operated motif, relative_motif and atoms.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        pass

    @abstractmethod
    def describe(self, **kwargs) -> str:
        """Generate a description of the action to be performed on the structure."""
        pass

    @classmethod
    @abstractmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Generate a random mode for the action.

        Must be implemented in subclasses.
        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on.
            seed (Optional[int], optional):
                Random seed for reproducibility. Defaults to None.
        """
        pass
