from ...motifs.site_collections.base import BaseSiteCollectionMotif
from ...motifs.site_collections.bond import BondMotif


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