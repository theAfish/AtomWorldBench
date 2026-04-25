"""Swap two motifs."""
from typing import Optional
from ase import Atoms
import numpy as np

from .base import BaseMotifAction
from .utils import get_random_motif
from ..motifs.base import BaseMotif

from ..common.registry import register

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
        desc_a, info_a = self.operated_motif.describe(**motif_a_kwargs)
        desc_b, info_b = self.relative_to_motif.describe(**motif_b_kwargs)
        if info_b == info_a:
            info_b = ''

        description = (
            f"swap the position of {desc_a} with the position of {desc_b} "
            f"in the structure."
            f" {info_a} {info_b}"
        )
        if len(self.operated_motif) > 1 or len(self.relative_to_motif) > 1:
            description += (
                " swap by translating to each other's centroid, such that"
                " the internal configuration and relative position to centroid"
                " within each motif should remain unchanged."
            )
        return description

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Get a random instance of SwapMotifAction.

        Args:
            operated_atoms (Atoms): The structure in which to swap motifs.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            SwapMotifAction: A random instance of SwapMotifAction.
        """

        rng = np.random.default_rng(seed)

        max_cluster_size = min(4, len(operated_atoms) - 1)

        # Detect two random non-bond site collection motifs
        class_alias1 = rng.choice(
            ["site", "cluster"]
        )
        motif_1_kwargs = {
            "class_alias": class_alias1,
            "atoms": operated_atoms,
            "seed": seed,
        }
        if class_alias1 == "cluster":
            # Use smaller cluster size to prevent overlap issues.
            motif_1_kwargs["cluster_size"] = rng.integers(2, max_cluster_size + 1)
            motif_1_kwargs["max_cluster_radius"] = 4.0
        motif_a = get_random_motif(**motif_1_kwargs)
        # check the max cluster size for motif 2, if is 1, then must be site motif
        max_cluster_size_2 = min(4, len(operated_atoms) - len(motif_a))
        if max_cluster_size_2 < 2:
            class_alias2 = "site"
        else:
            class_alias2 = rng.choice(
                ["site", "cluster"]
            )
        motif_2_kwargs = {
            "class_alias": class_alias2,
            "atoms": operated_atoms,
            "seed": seed + 1 if seed is not None else None,
            "excluded_site_indices": motif_a.indices,
        }
        if class_alias2 == "cluster":
            motif_2_kwargs["cluster_size"] = rng.integers(2, max_cluster_size_2 + 1)
            motif_2_kwargs["max_cluster_radius"] = 4.0
        motif_b = get_random_motif(**motif_2_kwargs)

        return cls(
            operated_motif=motif_a,
            relative_to_motif=motif_b,
        )
