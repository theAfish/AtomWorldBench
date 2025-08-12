"""Implement Resize action."""
from typing import Optional, Tuple

from ase import Atoms
import numpy as np

from .base import BaseAction
from ..motifs.site_collections.base import BaseSiteCollectionMotif

from ...utils.atoms_utils import merge_atoms
from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


class ResizeMotifAction(BaseAction):
    """Resize a motif by changing the motif's radius with respect to its centroid or a node."""
    mode_definitions = {
        "relative_to_centroid_scale_by": {
            "relative_to_centroid": (
                lambda x: x is True,
                "relative_to_centroid must be True for relative_to_centroid mode."
            ),
            "scale_by": None,
        },
        "relative_to_node_index_scale_by": {
            "relative_to_node_index": (
                lambda x: isinstance(x, int) and x >= 0,
                "relative_to_node_index must be a non-negative integer for relative_to_node_index mode."
            ),
            "scale_by": None,
        },
        "relative_to_centroid_to_radius": {
            "relative_to_centroid": (
                lambda x: x is True,
                "relative_to_centroid must be True for relative_to_centroid mode."
            ),
            "to_radius": None,
        },
        "relative_to_node_index_to_radius": {
            "relative_to_node_index": (
                lambda x: isinstance(x, int) and x >= 0,
                "relative_to_node_index must be a non-negative integer for relative_to_node_index mode."
            ),
            "to_radius": None,
        },
    }

    def __init__(
            self,
            relative_to_centroid: Optional[bool] = None,
            relative_to_node_index: Optional[int] = None,
            scale_by: Optional[float] = None,
            to_radius: Optional[float] = None,
    ):
        """Initialize the ResizeMotifAction.

        Currently, allows 4 modes of operation:
            1, "relative_to_centroid_scale_by":
                resize the motif relative to its centroid by a scale factor.
                In this mode, `relative_to_centroid` and `scale_by` are required.
                No other parameters are allowed.
            2, "relative_to_node_index_scale_by":
                resize the motif relative to a node by a scale factor.
                In this mode, `relative_to_node_index` and `scale_by` are required.
                No other parameters are allowed.
            3, "relative_to_centroid_to_radius":
                resize the motif relative to its centroid to a specific radius.
                In this mode, `relative_to_centroid` and `to_radius` are required.
                No other parameters are allowed.
            4, "relative_to_node_index_to_radius":
                resize the motif relative to a node to a specific radius.
                In this mode, `relative_to_node_index` and `to_radius` are required.
                No other parameters are allowed.

        Args:
            relative_to_centroid (Optional[bool]): If True, resize the motif relative to its centroid.
            relative_to_node_index (Optional[int]): The index of the node to resize relative to.
            scale_by (Optional[float]): Scale factor to apply to the motif's radius.
            to_radius (Optional[float]): The new radius for the motif. Unit is Angstroms.
        """
        # Static declaration for IDE linting.
        self.relative_to_centroid = None
        self.relative_to_node_index = None
        self.scale_by = None
        self.to_radius = None

        super().__init__(
            relative_to_centroid = relative_to_centroid,
            relative_to_node_index = relative_to_node_index,
            scale_by = scale_by,
            to_radius = to_radius,
        )

    def _check_compatibility(self, atoms: Atoms, motif: BaseSiteCollectionMotif) -> Tuple[bool, str]:
        """Check if the motif can be resized in the structure.

        Args:
            atoms (Atoms): The structure containing the motif.
            motif (BaseSiteCollectionMotif): The motif to be resized.

        Returns:
            Tuple[bool, str]: A tuple indicating compatibility and a message.
        """
        # Check if the motif is in the structure.
        indices, message = motif.find_indices_in_atoms(atoms, modify_indices_in_place=True)
        return indices is not None, f"operated motif not found in structure: {message}"

    def _get_resized_positions(self, motif):
        """Get position of the resized motif."""
        if "relative_to_centroid" in self.mode_flag:
            center = motif.get_centroid(fractional=False)
        elif "relative_to_node_index" in self.mode_flag:
            center = motif.cart_coords[self.relative_to_node_index]
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        if "scale_by" in self.mode_flag:
            scale = self.scale_by
        elif "to_radius" in self.mode_flag:
            scale = self.to_radius / motif.get_radius(fractional=False)
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        return (motif.cart_coords - center) * scale + center


    def _execute(self, atoms: Atoms, motif: BaseSiteCollectionMotif) -> Atoms:
        """Execute the action to resize the motif in the structure.

        Resizes the motif in the structure based on the action parameters.
        Order of atoms in the structure is preserved, but the motif is resized.
        Args:
            atoms (Atoms): The structure containing the motif.
            motif (BaseSiteCollectionMotif): The motif to be resized.

        Returns:
            Atoms: The modified structure with the resized motif.
        """
        # Get motif indices in the structure.
        indices, _ = motif.find_indices_in_atoms(atoms, modify_indices_in_place=False)

        other_indices = np.setdiff1d(
            np.arange(len(atoms), dtype=int), indices, assume_unique=True
        ).tolist()
        motif_atoms = motif.get_atoms()
        motif_atoms.set_positions(
            self._get_resized_positions(motif)
        )
        # Merge with the original atoms to maintain other properties.
        return merge_atoms(
            [atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            motif: BaseSiteCollectionMotif,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_kwargs: Optional[dict] = None
    ) -> str:
        """Generate a description for the resize action.

        Args:
            motif (BaseSiteCollectionMotif): The motif being resized.
            precision (int): The number of decimal places to format the coordinates.
            motif_kwargs (Optional[dict]): Additional keyword arguments for the motif.describe method.

        Returns:
            str: A string description of the resize action.
        """
        motif_kwargs = motif_kwargs or {}

        motif_kwargs.update({"precision": precision, "is_addition": False})

        size_word = "length" if len(motif) == 1 else "radius"
        if "relative_to_centroid" in self.mode_flag:
            relative_word = "its centroid"
        elif "relative_to_node_index" in self.mode_flag:
            relative_word = f"the atom at index {motif.indices[self.relative_to_node_index]}"
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        if "scale_by" in self.mode_flag:
            scale_word = f"by a scale factor of {self.scale_by}"
        elif "to_radius" in self.mode_flag:
            scale_word = f"to a {size_word} of {self.to_radius} angstroms"
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")


        # A common instruction to prevent shuffling indices.
        common_instruction = "modify coordinates only, do not change the order of atoms in structure."


        return (
            f"resize [{motif.describe(**motif_kwargs)}] relative to {relative_word}, {scale_word}."
            + common_instruction
        )
