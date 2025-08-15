"""Bond motif class."""

from .cluster import ClusterMotif

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

    forbidden_actions = ["add", "remove", "replace", "rotate", "translate"]

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
                symbols=cluster_motif.get_chemical_symbols(),
                positions=cluster_motif.get_positions(wrap=False),
                cell=cluster_motif.get_cell(complete=True),
                pbc=cluster_motif.get_pbc(),
                charges=cluster_motif.get_initial_charges(),
                name=None, # Bonds do not have a specific name.
                indices=cluster_motif.indices
            )
