from typing import Optional, List

from ase import Atoms
from ase.data import chemical_symbols
import numpy as np
from numpy.typing import ArrayLike

from .base import BaseSiteCollectionMotif
from ..base import BaseMotif

from ...common.registry import register

@register(BaseMotif, aliases=["site", "single-site"])
@register(BaseSiteCollectionMotif, aliases=["site", "single-site"])
class SiteMotif(BaseSiteCollectionMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """

    def __init__(
            self,
            in_atoms: Optional[Atoms] = None,
            indices: Optional[List[int]] = None,
            offsets: Optional[ArrayLike] = None,
            atoms: Optional[Atoms] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """SiteMotif constructor.

        Args:
            in_atoms (Atoms, optional): The ASE Atoms object to create the motif from.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            indices (list of int, optional): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
                Must be of length 1.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k) representing their unwrapped location in periodic images.
                If None, will assume all zeros. Must be of length 1.
            atoms (Atoms, optional): An ASE Atoms object representing the motif.
                When none of in_atoms, indices, offsets are provided, and atoms is provided,
                will create a motif directly from atoms. In this case, the motif can only be
                added in the AddMotifAction. Must be of length 1.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        BaseSiteCollectionMotif.__init__(
            self,
            in_atoms=in_atoms,
            indices=indices,
            offsets=offsets,
            atoms=atoms,
            name=name,
            allow_translation_equivalence=allow_translation_equivalence,
        )

    def __post_init__(self):
        """Post-initialization to check whether motif size is 1."""
        if len(self) != 1:
            raise ValueError(f"SiteMotif must contain exactly one site, but got {len(self)} sites.")

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if self.get_atoms().get_initial_charges()[0] == 0:
            return f"an atom {self.species_strings[0]}"
        else:
            return f"a species {self.species_strings[0]}"

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            seed: Optional[int] = None,
            additive_mode: bool = False,
            additive_mode_allowed_symbols: Optional[List[str]] = None,
            excluded_site_indices: Optional[List[int]] = None,
    ) -> 'SiteMotif':
        """Detect a random site motif from the given atoms.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            seed (Optional[int]): Random seed for reproducibility. Default is None.
            additive_mode (bool): If True, only consider atoms with symbols in
                additive_mode_allowed_symbols. Default is False.
            additive_mode_allowed_symbols (Optional[List[str]]):
                List of allowed atomic symbols when additive_mode is True.
                If None, all symbols are allowed.
            excluded_site_indices (Optional[List[int]]):
                List of site indices to exclude from selection. Used to prevent
                overlap, if already generated motifs, should not be selected again.
                Default is None.
        Returns:
            SiteMotif: A single site motif detected from the atoms.
        """
        rng = np.random.default_rng(seed)
        if not additive_mode:
            allowed_indices = (
                list(set(range(len(atoms))) - set(excluded_site_indices))
                if excluded_site_indices is not None else
                list(range(len(atoms)))
            )
            rand_idx = int(rng.choice(allowed_indices))
            return SiteMotif(
                atoms,
                indices=[rand_idx]
            )
        else:
            if additive_mode_allowed_symbols is None:
                allowed_symbols = chemical_symbols[1:]  # Any valid element now allowed.
            else:
                allowed_symbols = set(additive_mode_allowed_symbols)
            symbol = str(rng.choice(list(allowed_symbols)))
            return SiteMotif(
                atoms=Atoms(
                    symbols=[symbol],
                    positions=[(0.0, 0.0, 0.0)],
                    cell=atoms.cell.complete(),
                )
            )
