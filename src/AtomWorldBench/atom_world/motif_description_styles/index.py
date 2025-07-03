from .base import BaseDescriptionStyle
from ..motifs.base import BaseMotif


class IndexDescriptionStyle(BaseDescriptionStyle):
    """Description style for motifs using site indices.

    This style describes a motif using its site indices in the structure.
    """

    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif.

        Args:
            motif: The motif to describe.

        Returns:
            str: A string description of the motif's site indices.
        """
        if motif.indices is None:
            raise ValueError("Indices are not set for this motif.")
        return f"{motif.name} with site indices: {', '.join(map(str, motif.indices))}"