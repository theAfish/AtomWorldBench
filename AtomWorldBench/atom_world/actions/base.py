"""Defines the base class for actions in the atom world."""
from abc import ABC, abstractmethod
from typing import Optional
import re

from ase import Atoms

from ..motifs.base import BaseMotif
from ...common.mixin_classes import MultiModeInitMixin


class BaseAction(MultiModeInitMixin, ABC):
    """Base class for actions that can be performed on crystal structures.

    An action is defined as an interface to change a pymatgen.structure object,
    yielding a new pymatgen structure object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.

    A specific action should be a subclass of BaseAction and MultiModeInitMixin.
    See documentation for AtomWorldBench.mixin_classes for more details on how to
    implement actions.
    """

    def __init__(
            self,
            operated_motif: BaseMotif,
            operated_atoms: Atoms,
            relative_to_motif: Optional[BaseMotif] = None,
            **kwargs,
    ):
        """Initialize the action with the operated motif and atoms.

        Args:
            operated_motif (BaseMotif): The motif that this action operates on. Required.
            operated_atoms (Atoms): The atoms that this action operates on. Required.
            relative_to_motif (Optional[BaseMotif]): An optional motif that the action
                may use as a reference. Default is None.
            **kwargs: Additional keyword arguments for initialization.
        """
        MultiModeInitMixin.__init__(
            self,
            operated_motif=operated_motif,
            operated_atoms=operated_atoms,
            relative_to_motif=relative_to_motif,
            **kwargs
        )

    @property
    def action_name(self) -> str:
        """Return the name of the action."""
        cls_name = self.__class__.__name__
        # Extract the action name from the class name
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)Action$", cls_name)
        if not match:
            raise ValueError(f"Cannot infer action name from class name '{cls_name}'")
        return match.group(1).lower()

    def __post_init__(self):
        """Check compatibility of the action with operated motif, relative_motif and atoms.

        It is strongly recommended to overwrite this method in subclasses.
        """
        self.__check_operated_motif_compatibility()

    def __check_operated_motif_in_atoms(self):
        """Check if the operated motif is present in the provided atoms.

        Use in __post_init__ when needed.
        """
        operated_motif_indices = self.operated_motif.get_site_indices_in_atoms(
            self.operated_atoms, modify_indices_in_place=True
        )
        if operated_motif_indices is None:
            raise ValueError(
                f"Operated motif {self.operated_motif.name} not found in the provided Atoms."
            )

    def __check_relative_motif_in_atoms(self):
        """Check if the relative motif is present in the provided atoms.

        Use in __post_init__ when needed.
        """
        if self.relative_to_motif is not None:
            relative_motif_indices = self.relative_to_motif.get_site_indices_in_atoms(
                self.operated_atoms, modify_indices_in_place=True
            )
            if relative_motif_indices is None:
                raise ValueError(
                    f"Relative motif {self.relative_to_motif.name} not found in the"
                    f" provided Atoms."
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

    @abstractmethod
    def execute(self) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        pass

    @abstractmethod
    def describe(self, **kwargs) -> str:
        """Generate a description of the action to be performed on the structure."""
        pass
