"""Actions to remove atoms."""
from typing import Optional

import numpy as np
from ase import Atoms

from .base import BaseMotifAction
from .utils import get_random_motif

from ...motifs.base import BaseMotif
from ...motifs.site_collections.bond import BondMotif

from ....common.registry import register


def _check_operated_motif_compatibility(m):
    """Check if the operated motif is compatible with the action.

    Raises:
        ValueError: If the operated motif does not have indices.
    """
    if m.indices is None or len(m.indices) == 0:
        raise ValueError(
            "The operated motif must have indices to be removed from the structure."
        )
    if isinstance(m, BondMotif):
        raise ValueError(
            "BondMotif cannot be removed directly."
            " Please remove the pair cluster forming the bond instead."
        )
    return m


@register(BaseMotifAction, ["remove", "remove-motif"])
class RemoveMotifAction(BaseMotifAction):
    """Action to remove a motif from the structure.

    This action removes motifs from the structure based on their fractional coordinates.
    """
    # Only absolute allowed (specify the motif to remove directly). No parameters needed
    # for init.
    kwargs_formatting_functions = {
        "operated_motif": _check_operated_motif_compatibility,
    }
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
        self._check_operated_motif_in_atoms()

    def execute(self) -> Atoms:
        """Execute the action to remove the motif from the structure.

        Removes the motif from the structure by its indices, does not change the
        order of remaining atoms in structure.
        Returns:
            Atoms: The modified structure with the motif removed.
        """
        # Remove the motif by its indices.
        # _check_operated_motif_in_atoms() in __post_init()
        # ensures that the motif is in the atoms and
        # has indices.
        # Notice: region indices can duplicate, so must deduplicate first.
        indices = np.unique(self.operated_motif.indices)
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
        motif_desc_kwargs = motif_desc_kwargs or {}
        return (
            f"remove [{self.operated_motif.describe(**motif_desc_kwargs)}] from the structure."
            f" do not change the order of remaining atoms in structure."
        )

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Get a random instance of RemoveMotifAction.

        Args:
            operated_atoms (Atoms): The atoms to operate on.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            RemoveMotifAction: A random instance of RemoveMotifAction.
        """
        rng = np.random.default_rng(seed)

        max_cluster_size = max(4, len(operated_atoms) - 1)

        class_alias = rng.choice(
            ["site", "cluster", "sphere", "box"]
        )
        operated_motif_kwargs = {
            "class_alias": class_alias,
            "atoms": operated_atoms,
            "seed": seed,
        }
        if class_alias == "sphere":
            motif_style = rng.choice(
                ["center_around_atom_index", "center_around_coordinates"],
                p = [0.3, 0.7], # Prefer coordinates to avoid always picking existing atoms.
            )
            operated_motif_kwargs["style"] = motif_style
        elif class_alias == "cluster":
            cluster_size = rng.integers(2, max_cluster_size + 1)
            operated_motif_kwargs["cluster_size"] = cluster_size
            operated_motif_kwargs["max_cluster_radius"] = 4.0

        operated_motif = get_random_motif(**operated_motif_kwargs)

        return cls(
            operated_motif=operated_motif,
        )
