"""Bond motif class."""
from typing import Optional, List

from ase import Atoms
from numpy.typing import ArrayLike

from .cluster import ClusterMotif
from ....common.registry import register
from ..base import BaseMotif
from .base import BaseSiteCollectionMotif
from ....common.globals import ALLOW_TRANSLATION_EQUIVALENCE


@register(BaseMotif, ["bond"])
@register(BaseSiteCollectionMotif, ["bond"])
class BondMotif(ClusterMotif):
    """Motif representing a bond between two sites.

    This motif is defined by the species of the two sites and their fractional coordinates.
    It can be used to represent bonds in a structure.

    Only resize allowed.
    Add, remove and replace "bond" may cause confusion, as they imply
    a change in the bond length, etc., which is not the case for a bond motif.
    Rotate and translate are not applicable to bonds.
    If you wish to do so, you should use a cluster motif of size 2 instead.
    Therefore, we restrict the actions to resizing only to allow changing bond
    length.
    """

    def __init__(
            self,
            in_atoms: Atoms,
            indices: List[int],
            offsets: Optional[ArrayLike] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """BondMotif constructor.

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
            in_atoms=in_atoms,
            indices=indices,
            offsets=offsets,
            atoms=None,
            name=name,
            allow_translation_equivalence=allow_translation_equivalence,
        )

    def __post_init__(self):
        """Post-initialization to check whether motif size is 1."""
        if len(self) != 2:
            raise ValueError(f"BondMotif must contain exactly two sites, but got {len(self)} sites.")

    def _get_default_name(self) -> str:
        """Generate a default name for the bond motif based on species and coordinates."""
        return (
            f"a bond between {self.species_strings[0]}"
            f" and {self.species_strings[1]}"
        )

    @classmethod
    def from_cluster_motif(cls, cluster_motif: ClusterMotif) -> "BondMotif":
        """Create a BondMotif from a ClusterMotif."""
        if len(cluster_motif) != 2:
            raise ValueError("ClusterMotif must contain exactly two sites to create a BondMotif.")
        return cls(
                in_atoms=cluster_motif.in_atoms,
                indices=cluster_motif.indices,
                offsets=cluster_motif.cell_offsets,
                allow_translation_equivalence=cluster_motif.allow_translation_equivalence,
            )

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            max_cluster_radius: float = 3.0,
            n_attempts: int = 10,
            randomize_symbols: bool = False,
            seed: Optional[int] = None,
            cluster_size: int = 2,
            allow_translation_equivalence: bool = ALLOW_TRANSLATION_EQUIVALENCE,
    ) -> "BondMotif":
        """Detect a random bond motif from the given atoms.

        Args:
            atoms (Atoms): The Atoms object from which to detect the cluster.
            max_cluster_radius (float): Maximum allowable radius of the cluster.
                Defaults to 3.0.
            n_attempts (int): Number of attempts to find a valid cluster. Defaults to 10.
            randomize_symbols (bool): If True, the symbols of the atoms in the motif will be
                randomly chosen from the symbols of the atoms in the provided Atoms object.
                If False, the symbols will be set to None, meaning all atoms in the region
                will be included regardless of their symbols.
                Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            cluster_size (int): The desired size of the cluster.
                No effect for BondMotif, as it is always 2.
            allow_translation_equivalence (bool): Whether to consider translation equivalence
                when detecting the cluster. Defaults to the global setting.

        Returns:
            BondMotif: A single bond motif detected from the atoms.
        """
        cluster = ClusterMotif.detect_random_one(
            atoms,
            cluster_size=2,
            max_cluster_radius=max_cluster_radius,
            n_attempts=n_attempts,
            randomize_symbols=randomize_symbols,
            seed=seed
        )
        return cls.from_cluster_motif(cluster)
