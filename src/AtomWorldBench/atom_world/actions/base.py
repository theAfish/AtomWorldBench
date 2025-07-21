"""Defines the base class for actions in the atom world."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ase import Atoms
import numpy as np

from ..motifs.base import BaseMotif


class BaseAction(ABC):
    """Base class for actions that can be performed on crystal structures.

    An action is defined as an interface to change a pymatgen.structure object,
    yielding a new pymatgen structure object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.
    """
    allowed_relative_styles = []
    def __init__(
            self,
            relative_to_motif: Optional[BaseMotif] = None,
            relative_style: Optional[str] = None,
    ):
        """Initialize the BaseAction with an optional relative motif.

        Args:
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
             relative to. This can be used to define the context of the action.
            relative_style (str, optional): The style to determine relative action.
             For example, an action can be relative to a motif's centroid in distance.
             See `allowed_relative_styles` for the list of allowed styles. If None,
             will use the first style in `allowed_relative_styles`.
             Also, the type of motif should also allow the relative style.
        """
        self.relative_to_motif = relative_to_motif
        self.relative_style = relative_style
        self.check_relative_motif_compatibility()

    def check_relative_motif_compatibility(self):
        """Check if the relative style is compatible with the given relative motif and action."""
        if self.relative_style is None:
            self.relative_style = (
                self.allowed_relative_styles[0]
                if len(self.allowed_relative_styles) > 0
                else None
            )
        elif self.relative_style not in self.allowed_relative_styles:
            raise ValueError(
                f"Relative style {self.relative_style} is not allowed. "
                f"Allowed styles are: {self.allowed_relative_styles}"
            )
        if self.relative_to_motif is not None:
            if self.relative_style not in self.relative_to_motif.allowed_relative_styles:
                raise ValueError(
                    f"Relative style {self.relative_style} is not allowed for the "
                    f"motif {self.relative_to_motif.__class__.__name__}. "
                    f"Allowed styles are: {self.relative_to_motif.allowed_relative_styles}"
                )
            elif self.relative_to_motif.allowed_relative_styles[self.relative_style] is not None:
                condition, message = self.relative_to_motif.allowed_relative_styles[self.relative_style]
                if not condition(self.relative_to_motif):
                    raise ValueError(
                        f"Relative style {self.relative_style} is not compatible with the "
                        f"motif {self.relative_to_motif.__class__.__name__}. Reason: {message}."
                    )

    def execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        passed, message = self.check_compatibility(atoms, operated_motif)
        if passed:
            return self._execute(atoms, operated_motif)
        raise ValueError(
            f"Action {self.__class__.__name__} cannot be performed with the given"
            f" Atoms and motif. Reason: {message}."
        )

    @abstractmethod
    def _execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure.

        Must be overridden by subclasses to implement specific actions.
        """
        pass

    @classmethod
    def class_compatibility(cls, motif: BaseMotif) -> bool:
        """Check if the action is compatible with the given Atoms and motif object.

        Args:
            motif: An instance of BaseMotif to check compatibility with.
        Returns:
            bool: True if the action is compatible with the motif, False otherwise.
        """
        return cls.__name__ in motif.allowed_actions

    def check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms object.

        Args:
            atoms(Atoms): An instance of Atoms to check compatibility with.
            motif(BaseMotif): An instance of BaseMotif to check compatibility with.
        Returns:
            Tuple[bool, str]:
              True if the action is compatible with the Atoms and motif, False otherwise.
        """
        if not self.__class__.class_compatibility(motif):
            return False, "motif does not allow this action"
        if self.relative_to_motif is not None:
            # Check if the motif is in the structure.
            indices = self.relative_to_motif.find_indices_in_atoms(
                atoms,
                modify_indices_in_place=True
            )
            if indices is not None:
                return True, ""
            return False, "relative_to_motif not found in the structure."
        if not np.allclose(
                atoms.cell.complete().array,
                motif.cell.complete().array,
                atol=1e-6
        ) or not np.all(
            atoms.pbc == motif.pbc
        ):
            return False, "The operated motif's cell/pbc does not match the atoms cell/pbc."
        return self._check_compatibility(atoms, motif)

    @abstractmethod
    def _check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms object.

        Must be overridden by subclasses to implement specific compatibility checks.
        Args:
            atoms(Atoms): An instance of Atoms to check compatibility with.
            motif(BaseMotif): An instance of BaseMotif to check compatibility with.
        Returns:
            Tuple[bool,str]:
            True if the action is compatible with the Atoms and motif, False otherwise.
        """
        pass

    @abstractmethod
    def describe(self, motif: BaseMotif, **kwargs) -> str:
        """Generate a description of the action to be performed on the structure."""
        pass
