"""Bond motif class."""

from .cluster import ClusterMotif

class BondMotif(ClusterMotif):
    """Motif representing a bond between two sites.

    This motif is defined by the species of the two sites and their fractional coordinates.
    It can be used to represent bonds in a structure.
    """
    allowed_actions = [
        "ResizeMotifAction", # Resize the bond motif's length wrt centroid or a node in cluster.
    ]
    allowed_description_styles = [
        "index",
    ]

    def _get_default_name(self) -> str:
        """Generate a default name for the bond motif based on species and coordinates."""
        return f"a bond between {self.species[0]} and {self.species[1]}"