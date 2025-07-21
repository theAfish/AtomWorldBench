"""Implements replace action."""
from typing import Optional

from ase import Atoms
import numpy as np

from .base import BaseAction
from ..motifs.base import BaseMotif


class ReplaceMotifAction(BaseAction):
    """Action to replace a motif in the structure.

    This action replaces motifs in the structure based on their fractional coordinates.
    """
    allowed_relative_styles = []

    def __init__(
            self,
            relative_to_motif: BaseMotif,
    ):
        """Initialize the ReplaceMotifAction with fractional coordinates and cutoff.

        Args:
            relative_to_motif (BaseMotif):
                A motif that the action will replace in the structure.
        """
        super().__init__(relative_to_motif=relative_to_motif, relative_style=None)

    def _check_compatibility(self, atoms, motif):
        """Check if the motif can be replaced in the structure."""
        # Type, style and inclusion checks already done in the base class.
        # No additional checks needed for this action.
        pass

    def _execute(
            self,
            atoms: Atoms,
            motif: BaseMotif,
    ) -> Atoms:
        """Execute the action to replace the motif in the structure.

        In this action, self.relative_to_motif is the motif to be replaced in the structure.
        Args:
            atoms (Atoms): The structure to operate on.
            motif (BaseMotif): The motif to put in the structure.
        Returns:
            Atoms: The modified structure with the motif replaced.
        """
        remove_indices = self.relative_to_motif.find_indices_in_atoms(
            atoms,
            modify_indices_in_place=False
        )
        remaining_indices = np.setdiff1d(
            np.arange(len(atoms), dtype=int),
            remove_indices,
            assume_unique=True
        ).tolist()
        atoms_cp = atoms[remaining_indices]
        atoms_cp += motif.get_atoms()

        return atoms_cp

    def describe(
            self,
            motif: BaseMotif,
            motif_kwargs: Optional[dict] = None,
            relative_motif_kwargs: Optional[dict] = None,
            **kwargs
    ) -> str:
        """Describe the action to replace a motif.

        Args:
            motif (BaseMotif): The motif to put in the structure.
            motif_kwargs (Optional[dict]):
                Additional keyword arguments for describing the motif.
            relative_motif_kwargs (Optional[dict]):
                Additional keyword arguments for describing the relative motif.
            **kwargs: Additional keyword arguments.
        Returns:
            str: Description of the action.
        """
        motif_kwargs = motif_kwargs or {}
        relative_motif_kwargs = relative_motif_kwargs or {}
        motif_kwargs.update({"is_addition": True})
        relative_motif_kwargs.update({"is_addition": False})

        return (
            f"Replace {self.relative_to_motif.describe(**relative_motif_kwargs)}"
            f" with {motif.describe(**motif_kwargs)}."
        )
