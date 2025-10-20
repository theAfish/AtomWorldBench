from typing import Optional

from ase import Atoms

from .base import BaseStructureAction
from ....common.registry import register


@register(BaseStructureAction, ["change-element"])
class ChangeElementAction(BaseStructureAction):
    """An action that changes all atoms of a certain element to another element.

    This action replaces all occurrences of a specified element in the input
    structure with another specified element.
    """

    mode_definitions = {
        "_excluded": ["operated_atoms"],
        "replace_element": {"from_element": None, "to_element": None},
        "remove_element": {"from_element": None},
    }

    def __init__(
            self,
            operated_atoms: Atoms,
            from_element: str,
            to_element: Optional[str]=None,
    ):
        """Initialize the ChangeElementAction.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on. Required.
            from_element (str):
                The chemical symbol of the element to be replaced. Required.
            to_element (str, optional):
                The chemical symbol of the element to replace with. If None, the action
                will remove all atoms of from_element. Optional.
        """
        self.operated_atoms = None
        self.from_element = None
        self.to_element = None
        super().__init__(
            operated_atoms=operated_atoms,
            from_element=from_element,
            to_element=to_element,
        )

    def __post_init__(self):
        if self.from_element not in self.operated_atoms.get_chemical_symbols():
            raise ValueError(
                f"from_element '{self.from_element}' not found in operated_atoms."
            )
        if self.to_element is not None:
            if self.to_element == self.from_element:
                raise ValueError(
                    "to_element must be different from from_element."
                )

    def execute(self) -> Atoms:
        """Execute the ChangeElementAction.
        Returns:
            Atoms:
                A new Atoms object with the specified element changes applied.
        """
        new_atoms = self.operated_atoms.copy()
        symbols = new_atoms.get_chemical_symbols()
        if self.mode_flag == "replace_element":
            # Replace from_element with to_element
            new_symbols = [
                self.to_element if symbol == self.from_element else symbol
                for symbol in symbols
            ]
            new_atoms.set_chemical_symbols(new_symbols)
        else:
            # Remove all atoms of from_element
            new_atoms = new_atoms[[i for i, symbol in enumerate(symbols) if symbol != self.from_element]]
        return new_atoms

    def describe(
            self) -> str:
        """Generate a description of the action performed.
        Returns:
            str:
                A textual description of the element change action.
        """
        if self.mode_flag == "replace_element":
            return (
                f"replace all atoms of element '{self.from_element}' "
                f"with element '{self.to_element}'."
            )
        else:
            return f"remove all atoms of element '{self.from_element}'."
