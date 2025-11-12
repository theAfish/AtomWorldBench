from typing import Optional

from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

from .base import BaseStructureAction
from ....common.registry import register


@register(BaseStructureAction, ["change-element"])
class ChangeElementAction(BaseStructureAction):
    """An action that changes all atoms of a certain element to another element.

    This action replaces all occurrences of a specified element in the input
    structure with another specified element.

    Allows two modes of operation:
    1. "replace_element": Replace all atoms of 'from_element' with 'to_element'.
        Provide both 'from_element' and 'to_element'.
    2. "remove_element": Remove all atoms of 'from_element' from the structure.
        Provide only 'from_element'; 'to_element' should be omitted.
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
                f"replace all atoms of element {self.from_element} "
                f"with {self.to_element}."
            )
        else:
            return (f"remove all atoms of element {self.from_element}"
                    f" without affecting the order of other atoms.")

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ) -> "ChangeElementAction":
        """Generate a random ChangeElementAction instance.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on.
            seed (int, optional):
                Seed for random number generator for reproducibility. Optional.
                Will also influence the choice of mode.

        Returns:
            ChangeElementAction:
                A randomly generated ChangeElementAction instance.
        """
        rng = np.random.default_rng(seed)

        unique_elements = set(operated_atoms.get_chemical_symbols())
        all_elements = set(chemical_symbols[1:])  # Exclude the first entry which is ''
        from_element = str(rng.choice(list(unique_elements)))

        # Choose a mode based on mode_probabilities.
        chosen_mode = cls.get_random_mode(seed)
        if chosen_mode == "replace_element":
            # Replace element
            possible_to_elements = all_elements - {from_element}
            to_element = str(rng.choice(list(possible_to_elements)))
            return cls(
                operated_atoms=operated_atoms,
                from_element=from_element,
                to_element=to_element,
            )
        elif chosen_mode == "remove_element":
            # Remove element
            return cls(
                operated_atoms=operated_atoms,
                from_element=from_element,
            )
        else:
            raise ValueError(f"Invalid mode: {chosen_mode}")
