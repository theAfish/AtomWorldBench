from ..motifs.site_collections.base import BaseSiteCollectionMotif
from ..motifs.base import BaseMotif
from ..motifs.site_collections.bond import BondMotif
from ..common.registry import get_registered


def _must_be_non_bond_site_collection_motif(m):
    """Check if the motif is a site collection motif."""
    # By default, let None pass.
    if m is None:
        return m
    if (not isinstance(m, BaseSiteCollectionMotif)) or isinstance(m, BondMotif):
        raise ValueError(
            f"The motif {m} must be a non-bond site collection motif"
            f" to perform action."
        )
    return m


def get_random_motif(class_alias, atoms, seed=42, **kwargs):
    """Helper function to get a random motif of a given class alias."""
    motif_class = get_registered(BaseMotif)[class_alias]
    assert issubclass(motif_class, BaseMotif)
    return motif_class.detect_random_one(atoms, seed=seed, **kwargs)
