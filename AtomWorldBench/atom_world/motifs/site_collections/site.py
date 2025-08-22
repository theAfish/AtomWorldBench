from typing import Optional

from ase import Atoms
import numpy as np

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

    def __post_init__(self):
        """Post-initialization to check whether motif size is 1."""
        if len(self) != 1:
            raise ValueError(f"SiteMotif must contain exactly one site, but got {len(self)} sites.")

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if self.get_initial_charges()[0] == 0:
            return f"an atom {self.species_strings[0]}"
        else:
            return f"a species {self.species_strings[0]}"

    @classmethod
    def detect_random_one(cls, atoms: Atoms, seed: Optional[int] = None) -> 'SiteMotif':
        """Detect a random site motif from the given atoms.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
            seed (Optional[int]): Random seed for reproducibility. Default is None.
        Returns:
            SiteMotif: A single site motif detected from the atoms.
        """
        rng = np.random.default_rng(seed)
        rand_idx = int(rng.integers(0, len(atoms)))
        return SiteMotif.from_atoms(
            atoms[[rand_idx]],
            indices=[rand_idx]
        )
