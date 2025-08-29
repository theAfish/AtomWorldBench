from typing import Optional, List

from ase import Atoms
import numpy as np
from numpy.typing import ArrayLike

from .base import BaseSiteCollectionMotif
from ..base import BaseMotif

from ....common.registry import register

@register(BaseMotif, aliases=["site", "single-site"])
@register(BaseSiteCollectionMotif, aliases=["site", "single-site"])
class SiteMotif(BaseSiteCollectionMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """
    forbidden_actions = ["resize"]

    def __init__(
            self,
            in_atoms: Atoms,
            indices: List[int],
            offsets: Optional[ArrayLike] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """SiteMotif constructor.

        Args:
            in_atoms (Atoms): The ASE Atoms object to create the motif from.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            indices (list of int): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k) representing their unwrapped location in periodic images.
                If None, will assume all zeros.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        super().__init__(
            in_atoms,
            indices,
            offsets,
            name,
            allow_translation_equivalence,
        )

    def __post_init__(self):
        """Post-initialization to check whether motif size is 1."""
        if len(self) != 1:
            raise ValueError(f"SiteMotif must contain exactly one site, but got {len(self)} sites.")

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if self.in_atoms.get_initial_charges()[self.indices[0]] == 0:
            return f"an atom {self.species_strings[0]}"
        else:
            return f"a species {self.species_strings[0]}"

    @classmethod
    def detect_random_one(cls, atoms: Atoms, seed: Optional[int] = None) -> 'SiteMotif':
        """Detect a random site motif from the given atoms.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            seed (Optional[int]): Random seed for reproducibility. Default is None.
        Returns:
            SiteMotif: A single site motif detected from the atoms.
        """
        rng = np.random.default_rng(seed)
        rand_idx = int(rng.integers(0, len(atoms)))
        return SiteMotif(
            atoms,
            indices=[rand_idx]
        )
