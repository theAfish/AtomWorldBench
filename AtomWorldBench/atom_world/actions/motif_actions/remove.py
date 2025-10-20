"""Actions to remove atoms."""
from typing import Optional
import inspect

import numpy as np
from ase import Atoms

from .base import BaseMotifAction
from ...motifs.base import BaseMotif

from ....common.registry import register


@register(BaseMotifAction, ["remove", "remove-motif"])
class RemoveMotifAction(BaseMotifAction):
    """Action to remove a motif from the structure.

    This action removes motifs from the structure based on their fractional coordinates.
    """
    # Only absolute allowed (specify the motif to remove directly). No parameters needed
    # for init.
    mode_definitions = {
        "_excluded": ["operated_motif"],
        "default": {},
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
    ):
        """Initialize the RemoveMotifAction with fractional coordinates and cutoff.

        Only has "default" mode, which does not require any additional parameters
        than operated_motif.
        Args:
            operated_motif (BaseMotif): The motif to be removed.
        """
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=None,
        )

    def __post_init__(self):
        """Post-initialization to ensure the action is valid."""
        self.__check_operated_motif_compatibility()

    def execute(self) -> Atoms:
        """Execute the action to remove the motif from the structure.

        Removes the motif from the structure by its indices, does not change the
        order of remaining atoms in structure.
        Returns:
            Atoms: The modified structure with the motif removed.
        """
        # Remove the motif by its indices.
        # __check_operated_motif_in_atoms() in __post_init()
        # ensures that the motif is in the atoms and
        # has indices.
        indices = self.operated_motif.indices
        remaining_indices = np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int),
            indices, assume_unique=True
        ).tolist()
        # Guarantee the order of remaining atoms is unchanged.
        return self.operated_atoms[np.sort(remaining_indices)]

    def describe(
            self,
            motif_desc_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to remove a motif.

        Args:
            motif_desc_kwargs (Optional[dict]): Additional keyword arguments for the motif.describe method.
                Not used, just to match the interface.
        Returns:
            str: A description of the action.
        """
        # Update motif description kwargs. Prevent using addition mode.
        motif_desc_params = inspect.signature(self.operated_motif.describe).parameters
        if "is_addition" in motif_desc_params:
            motif_desc_kwargs["is_addition"] = False

        return (
            f"remove [{self.operated_motif.describe(**motif_desc_kwargs)}] from the structure."
            f" Do not change the order of remaining atoms in structure."
        )
