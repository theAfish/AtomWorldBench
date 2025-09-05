"""Implements replace action."""
from typing import Optional
import inspect

from ase import Atoms
import numpy as np

from .base import BaseMotifAction
from ...motifs.base import BaseMotif

from ....common.registry import register


@register(BaseMotif, ["replace", "replace-motif"])
class ReplaceMotifAction(BaseMotifAction):
    """Action to replace a motif in the structure.

    This action replaces motifs in the structure based on their fractional coordinates.
    """
    mode_definitions = {
        "_excluded": ["operated_motif", "relative_to_motif"],
        "default": {},
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            relative_to_motif: BaseMotif,
    ):
        """Initialize the ReplaceMotifAction with fractional coordinates and cutoff.

        Args:
            operated_motif (BaseMotif):
                A motif that will be added to the structure.
            relative_to_motif (BaseMotif):
                A motif that the action will replace in the structure.
                Must be in the structure (will check at `execute` call).
        """
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=relative_to_motif,
        )
        self.replaced_motif = self.relative_to_motif  # Make an alias for clarity.

    def __post_init__(self):
        """Post-initialization to ensure the action is valid."""
        self.__check_operated_motif_compatibility()
        self.__check_relative_motif_in_atoms()

    def execute(self) -> Atoms:
        """Execute the action to replace the motif in the structure.

        Removes the motif defined by `self.relative_to_motif` from the structure, and
        append `motif` to the remaining atoms. The order of the remaining atoms is preserved.

        Returns:
            Atoms: The modified structure with the motif replaced.
        """
        # __check_relative_motif_in_atoms() in __post_init__ guarantees that
        # the motif to be replaced is in the atoms and has indices.
        remove_indices = self.replaced_motif.indices
        remaining_indices = np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int),
            remove_indices,
            assume_unique=True
        ).tolist()
        # Preserve the order of remaining atoms.
        atoms_cp = self.operated_atoms[np.sort(remaining_indices)]
        atoms_cp += self.operated_motif.get_atoms()

        return atoms_cp

    def describe(
            self,
            motif_desc_kwargs: Optional[dict] = None,
            relative_motif_desc_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to replace a motif.

        Args:
            motif_desc_kwargs (Optional[dict]):
                Additional keyword arguments for describing the motif.
            relative_motif_desc_kwargs (Optional[dict]):
                Additional keyword arguments for describing the relative motif.
        Returns:
            str: Description of the action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}
        relative_motif_desc_kwargs = relative_motif_desc_kwargs or {}

        # Update motif description kwargs.
        motif_desc_params = inspect.signature(self.operated_motif.describe).parameters
        relative_motif_desc_params = inspect.signature(
            self.relative_to_motif.describe
        ).parameters if self.relative_to_motif is not None else {}
        # Use addition mode as site motif needs different short description.
        # when being added.
        if "is_addition" in motif_desc_params:
            motif_desc_kwargs["is_addition"] = True
        if "is_addition" in relative_motif_desc_params:
            relative_motif_desc_kwargs["is_addition"] = False

        return (
            f"replace [{self.replaced_motif.describe(**relative_motif_desc_kwargs)}]"
            f" with [{self.operated_motif.describe(**motif_desc_kwargs)}]."
            f" do not change the order of other unaffected atoms,"
            f" and the newly added atoms should be appended to the end of"
            f" the structure in the order as described."
        )
