from ...motifs.site_collections.base import BaseSiteCollectionMotif
from ...motifs.site_collections.bond import BondMotif


def _must_be_non_bond_site_collection_motif(m):
    """Check if the motif is a site collection motif."""
    if not isinstance(m, BaseSiteCollectionMotif) or isinstance(m, BondMotif):
        raise ValueError(
            f"The motif {m} must be a non-bond site collection motif"
            f" to perform swap action."
        )
    return m