import numpy as np

from .base import BaseDescriptionStyle
from ..motifs.base import BaseMotif

from .utils import format_arraylike


class IndexDescriptionStyle(BaseDescriptionStyle):
    """Description style for motifs using site indices.

    This style describes a motif using its site indices in the structure.
    """
    introduction = (
        "Cell offsets: triplets of integers indicating how many unit"
        " cells the atom is displaced along the lattice vectors from"
        " the central reference cell. For example, an offset of (-1, 0, 1)"
        " means the atom is located one unit cell back along the a-axis,"
        " remains in the same position along the b-axis,"
        " and is shifted forward by one unit cell along the c-axis."
    )

    def describe(self, motif: BaseMotif) -> str:
        """Generate a description for the given motif.

        Args:
            motif: The motif to describe.

        Returns:
            str: A string description of the motif's site indices.
        """
        if motif.indices is None:
            raise ValueError("Indices are not set for this motif.")
        # All sites in motif is in the central reference cell.
        if np.all(motif.cell_offsets == 0):
            return (f"{motif.name} with site indices: {', '.join(map(str, motif.indices))}"
                    f" in the central reference cell.")

        return (f"{motif.name} with site indices: {', '.join(map(str, motif.indices))}"
                f" and cell offsets: {format_arraylike(motif.cell_offsets, precision=0)}.")