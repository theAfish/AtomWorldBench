"""Motif comprising multiple atoms."""
from typing import List

from .base import BaseSiteCollectionMotif
from ..base import BaseMotif
from .site import SiteMotif

from ....common.registry import register

@register(BaseMotif, aliases=["cluster", "atom-cluster"])
@register(BaseSiteCollectionMotif, aliases=["cluster", "atom-cluster"])
class ClusterMotif(BaseSiteCollectionMotif):
    """Motif representing a cluster of atoms.

    This motif is defined by a list of species and their fractional coordinates.
    It can be used to represent clusters of atoms in a structure.
    """

    def __post_init__(self):
        # Widen restriction to allow single point cluster, such that cluster detector can grow from it.
        if len(self) < 1:
            raise ValueError("ClusterMotif must contain at least one atom.")

    def _get_default_name(self) -> str:
        """Generate a default name for the cluster motif based on species and coordinates."""
        if len(self) == 1:
            prefix = "point"
        elif len(self) == 2:
            prefix = "pair"
        elif len(self) == 3:
            prefix = "triplet"
        elif len(self) == 4:
            prefix = "quadruplet"
        elif len(self) == 5:
            prefix = "quintuplet"
        elif len(self) == 6:
            prefix = "sextuplet"
        else:
            prefix = f"{len(self)}-sites cluster"

        return f"a {prefix} of atoms/species {', '.join(map(str, self.species_strings))}"

    @property
    def site_motifs(self) -> List[SiteMotif]:
        """Return a list of SiteMotif objects representing the sites in the cluster."""
        return [
            SiteMotif.from_atoms(
                self[[i]], indices=[int(self.indices[i])]
            )
            for i in range(len(self))
        ]
