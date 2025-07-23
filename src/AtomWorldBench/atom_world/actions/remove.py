"""Actions to remove atoms."""
from typing import Tuple, Optional

import numpy as np
from ase import Atoms

from .base import BaseAction
from ..motifs import BaseMotif


class RemoveMotifAction(BaseAction):
    """Action to remove a motif from the structure.

    This action removes motifs from the structure based on their fractional coordinates.
    """
    allowed_relative_styles = []
    # Only absolute allowed (specify the motif to remove directly). No parameters needed
    # for init.
    mode_definitions = {"default": {}}

    def __init__(self):
        """Initialize the RemoveMotifAction with fractional coordinates and cutoff.

        Does not take relative motif or style, as it operates independently.
        Specify the motif to remove directly in the `execute` method.
        No need for additional parameters in the constructor.
        """
        super().__init__()

    def _check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the motif can be removed from the structure.

        Will override the motif's original indices attribute, if the motif is found in the structure.
        Args:
            atoms (Atoms): The structure from which the motif is to be removed.
            motif (BaseMotif): The motif to be removed.

        Returns:
            Tuple[bool, str]: A tuple indicating compatibility and a message.
        """
        # Check if the motif is in the structure.
        indices = motif.find_indices_in_atoms(atoms, modify_indices_in_place=True)
        if indices is not None:
            return True, ""
        return False, "Motif not found in the structure."

    def _execute(self, atoms: Atoms, motif: BaseMotif) -> Atoms:
        """Execute the action to remove the motif from the structure.

        Removes the motif from the structure by its indices, does not change the
        order of remaining atoms in structure.
        Args:
            atoms (Atoms): The structure from which the motif is to be removed.
            motif (BaseMotif): The motif to be removed.

        Returns:
            Atoms: The modified structure with the motif removed.
        """
        # Remove the motif by its indices.
        indices = motif.find_indices_in_atoms(atoms, modify_indices_in_place=False)
        remaining_indices = np.setdiff1d(np.arange(len(atoms), dtype=int), indices, assume_unique=True).tolist()
        return atoms[remaining_indices]

    def describe(
            self,
            motif: BaseMotif,
            motif_kwargs: Optional[dict] = None,
            **kwargs
    ) -> str:
        """Describe the action to remove a motif.

        Args:
            motif (BaseMotif): The motif to be removed.
            motif_kwargs (Optional[dict]): Additional keyword arguments for the motif.describe method.
                Not used, just to match the interface.
            **kwargs: Additional keyword arguments. Not used, just to match the interface.

        Returns:
            str: A description of the action.
        """
        return (f"Remove [{motif.describe(**motif_kwargs)}] from the structure."
                f" Do not change the order of remaining atoms in structure.")
