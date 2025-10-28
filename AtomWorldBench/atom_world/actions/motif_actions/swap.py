"""Swap two motifs."""
from typing import Optional
from ase import Atoms

from .base import BaseMotifAction
from ...motifs.base import BaseMotif

from ....common.registry import register

from .utils import _must_be_non_bond_site_collection_motif


@register(BaseMotifAction, ["swap", "swap-motif"])
class SwapMotifAction(BaseMotifAction):
    """Action to swap two motifs in the structure.

    This action swaps motifs in the structure based on their fractional coordinates.
    """
    kwargs_formatting_functions = {
        "operated_motif": _must_be_non_bond_site_collection_motif,
        "relative_to_motif": _must_be_non_bond_site_collection_motif,
    }
    mode_definitions = {
        "_excluded": ["operated_motif", "relative_to_motif"],
        "default": {},
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            relative_to_motif: BaseMotif,
    ):
        """Initialize the SwapMotifAction with two motifs to swap.

        This action simply swaps the centroid positions of two motifs in the structure without
        altering their internal configurations and relative coordinates with respect to their
        centroids.
        In principle, only site collection motifs are allowed to swap.
        Args:
            operated_motif (BaseMotif):
                The first motif to be swapped in the structure.
            relative_to_motif (BaseMotif):
                The second motif to be swapped in the structure.
        """
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=relative_to_motif,
        )

    def __post_init__(self):
        """Post-initialization to ensure the action is valid."""
        self._check_operated_motif_in_atoms()
        self._check_relative_motif_in_atoms()
        # Disjoint check.
        if len(set(self.operated_motif.indices) & set(self.relative_to_motif.indices)) > 0:
            raise ValueError("The two motifs to swap must not share any atoms.")

    def execute(self) -> Atoms:
        """Execute the action to swap the two motifs in the structure.

        Swaps the motifs defined by `self.motif_a` and `self.motif_b` in the structure.
        The order of the remaining atoms is preserved.

        Returns:
            Atoms: The modified structure with the motifs swapped.
        """
        atoms = self.operated_atoms.copy()

        motif_a_positions = self.operated_motif.cart_coords
        motif_b_positions = self.relative_to_motif.cart_coords

        # Calculate centroids
        centroid_a = self.operated_motif.get_centroid(fractional=False)
        centroid_b = self.relative_to_motif.get_centroid(fractional=False)

        # Calculate relative positions
        relative_a = motif_a_positions - centroid_a
        relative_b = motif_b_positions - centroid_b

        # Swap positions
        new_motif_a_positions = relative_a + centroid_b
        new_motif_b_positions = relative_b + centroid_a

        atom_positions = atoms.get_positions(wrap=False).copy()
        motif_a_indices = self.operated_motif.indices
        motif_b_indices = self.relative_to_motif.indices
        atom_positions[motif_a_indices] = new_motif_a_positions
        atom_positions[motif_b_indices] = new_motif_b_positions

        # Update atom positions. NOT wrapped!
        atoms.set_positions(atom_positions)

        return atoms

    def describe(
            self,
            describe_motif_a_kwargs: Optional[dict] = None,
            describe_motif_b_kwargs: Optional[dict] = None,
         ) -> str:
        """Describe the action to swap two motifs.

        Args:
            describe_motif_a_kwargs (Optional[dict]):
                Additional keyword arguments for describing motif A.
                See corresponding `describe` method in motif class for details.
            describe_motif_b_kwargs (Optional[dict]):
                Additional keyword arguments for describing motif B.
                See corresponding `describe` method in motif class for details.
        Returns:
            str: Description of the action.
        """
        motif_a_kwargs = describe_motif_a_kwargs or {}
        motif_b_kwargs = describe_motif_b_kwargs or {}
        desc_a = self.operated_motif.describe(**motif_a_kwargs)
        desc_b = self.relative_to_motif.describe(**motif_b_kwargs)

        description = (
            f"swap the position of [{desc_a}] with the position of [{desc_b}] "
            f"in the structure."
        )
        if len(self.operated_motif) > 1 or len(self.relative_to_motif) > 1:
            description += (
                " swap by translating to each other's centroid, such that"
                " the internal configuration and relative position to centroid"
                " within each motif should remain unchanged."
            )
        return description