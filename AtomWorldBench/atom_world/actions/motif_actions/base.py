"""Defines the base class for actions in the atom world."""
from abc import ABC, abstractmethod
from typing import Optional
import re

from ase import Atoms

from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.common.mixin_classes import MultiModeInitMixin


class BaseMotifAction(MultiModeInitMixin, ABC):
    """Base class for actions that can be performed on a motif of crystal structures.

    An action is defined as an interface to change a pymatgen.structure object,
    yielding a new pymatgen structure object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.

    A specific action should be a subclass of BaseMotifAction and MultiModeInitMixin.
    See documentation for AtomWorldBench.mixin_classes for more details on how to
    implement actions.
    """

    def __init__(
            self,
            operated_motif: BaseMotif,
            relative_to_motif: Optional[BaseMotif] = None,
            **kwargs,
    ):
        """Initialize the action with the operated motif and atoms.

        Args:
            operated_motif (BaseMotif): The motif that this action operates on. Required.
            relative_to_motif (Optional[BaseMotif]): An optional motif that the action
                may use as a reference. Default is None.
            **kwargs: Additional keyword arguments for initialization.
        """
        # Only for suppressing linter warnings
        self.operated_motif = None
        self.relative_to_motif = None
        MultiModeInitMixin.__init__(
            self,
            operated_motif=operated_motif,
            relative_to_motif=relative_to_motif,
            **kwargs
        )

    @property
    def operated_atoms(self) -> Atoms:
        """Return the Atoms object associated with the operated motif."""
        return self.operated_motif.in_atoms

    @property
    def action_name(self) -> str:
        """Return the name of the action."""
        cls_name = self.__class__.__name__
        # Extract the action name from the class name
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)Action$", cls_name)
        if not match:
            raise ValueError(f"Cannot infer action name from class name '{cls_name}'")
        return match.group(1).lower()

    @abstractmethod
    def __post_init__(self):
        """Check compatibility of the action with operated motif, relative_motif and atoms.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def __check_operated_motif_in_atoms(self):
        """Check if the operated motif is present in the provided atoms.

        Use in __post_init__ when needed.
        """
        if self.operated_motif.in_atoms is None:
            raise ValueError(
                f"Operated motif {self.operated_motif.name} must be"
                f" attached to an Atoms object!"
            )

    def __check_relative_motif_in_atoms(self):
        """Check if the relative motif is present in the provided atoms.

        Use in __post_init__ when needed.
        """
        if self.relative_to_motif is not None:
            if self.relative_to_motif.in_atoms != self.operated_atoms:
                raise ValueError(
                    f"Relative motif {self.relative_to_motif.name} must be"
                    f" attached to the same Atoms object as the operated motif!"
                )

    def __check_operated_motif_compatibility(self):
        """Check if the operated motif is compatible with the action.

        Should be used in __post__init__.
        """
        if self.action_name in self.operated_motif.forbidden_actions:
            raise ValueError(
                f"Action '{self.action_name}' is not allowed for the operated motif"
                f" {self.operated_motif.__class__.__name__}."
            )

    def __check_relative_motif_compatibility(self):
        """Check if the relative motif is compatible with the action.

        Should be used in __post__init__.
        """
        if self.relative_to_motif is not None:
            if self.action_name in self.relative_to_motif.forbidden_actions:
                raise ValueError(
                    f"Action '{self.action_name}' is not allowed for the relative motif"
                    f" {self.relative_to_motif.__class__.__name__}."
                )

    @abstractmethod
    def execute(self) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        pass

    @abstractmethod
    def describe(self, **kwargs) -> str:
        """Generate a description of the action to be performed on the structure."""
        pass
