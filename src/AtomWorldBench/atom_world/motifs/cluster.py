"""Motif comprising multiple atoms."""

from .base import BaseMotif


class ClusterMotif(BaseMotif):
    """Motif representing a cluster of atoms.

    This motif is defined by a list of species and their fractional coordinates.
    It can be used to represent clusters of atoms in a structure.
    """
    allowed_actions = [
        "AddMotifAction",
        "RemoveMotifAction",
        "ReplaceMotifAction",
        "TranslateMotifAction",
        "RotateMotifAction",
        "E3MotifAction",  # E3 operation relative to other motifs or some coordinates.
        "ResizeMotifAction",  # Resize the cluster motif's radius wrt centroid or a node in cluster.
        "EdgeResizeMotifAction",  # Resize selected edges of the cluster motif.
    ]
    allowed_description_styles = [
        "coord",
        "index",
    ]

    def _get_default_name(self) -> str:
        """Generate a default name for the cluster motif based on species and coordinates."""
        if len(self) == 2:
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
