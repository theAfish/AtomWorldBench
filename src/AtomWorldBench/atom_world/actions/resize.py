"""Implement Resize action."""
from typing import Optional, Tuple
from numbers import Number
import inspect

from ase import Atoms
import numpy as np

from .base import BaseAction
from ..motifs.base import BaseMotif

from ...utils.atoms_utils import merge_atoms
from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


class ResizeAction(BaseAction):
    """Resize a motif by changing the motif's radius with respect to its centroid or a node."""
    mode_definitions = {
        # Parameters that are always required.
        "_excluded": ["operated_motif", "operated_atoms"],
        # Four valid modes of operation from combinations of 2x2 options.
        "_combinations": [
            {
                "name_template": "relative_to_{relative_to}_{size_mode}",
                "relative_to":{
                    "centroid": {
                        "relative_to_centroid": (
                            lambda x: x is True,
                            "relative_to_centroid must be True for relative_to_centroid mode."
                        ),
                    },
                    "node_index": {
                        "relative_to_node_index": (
                            lambda x: isinstance(x, int) and x >= 0,
                            "relative_to_node_index must be a non-negative integer for"
                            " relative_to_node_index mode."
                        ),
                    },
                },
                "size_mode": {
                    "scale_by": {
                        "scale_by": (
                            lambda x: isinstance(x, Number) and x > 0 and x != 1,
                            "scale_by must be a positive number not equal to 1 for scale_by mode."
                        ),
                    },
                    "to_radius": {
                        "to_radius": (
                            lambda x: isinstance(x, Number) and x > 0,
                            "to_radius must be a positive number for to_radius mode."
                        ),
                    },
                },
            },
        ],
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            operated_atoms: Atoms,
            relative_to_centroid: Optional[bool] = None,
            relative_to_node_index: Optional[int] = None,
            scale_by: Optional[float] = None,
            to_radius: Optional[float] = None,
    ):
        """Initialize the ResizeMotifAction.

        `operated_motif` and `operated_atoms` are always required.
        For the rest of parameters, currently, allows 4 modes of operation:
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
            operated_motif (BaseMotif): The motif to be resized.
            operated_atoms (Atoms): The structure containing the motif.
            relative_to_centroid (Optional[bool]): If True, resize the motif relative to its
                centroid.
            relative_to_node_index (Optional[int]): The index of the node to resize relative to.
            scale_by (Optional[float]): Scale factor to apply to the motif's radius.
            to_radius (Optional[float]): The new radius for the motif. Unit is Angstroms.
        """
        super().__init__(
            operated_motif=operated_motif,
            operated_atoms=operated_atoms,
            relative_to_motif=None, # Does not need a relative motif.
            relative_to_centroid=relative_to_centroid,
            relative_to_node_index=relative_to_node_index,
            scale_by=scale_by,
            to_radius=to_radius,
        )

    def __post_init__(self):
        """Post-initialization to validate parameters."""
        self.__check_operated_motif_compatibility()
        self.__check_operated_motif_in_atoms()

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

    def execute(self) -> Atoms:
        """Execute the action to resize the motif in the structure.

        Resizes the motif in the structure based on the action parameters.
        Order of atoms in the structure is preserved, but the motif is resized.

        Returns:
            Atoms: The modified structure with the resized motif.
        """
        # Get motif indices in the structure.
        # __check_operated_motif_in_atoms has been called in __post_init__,
        # so the indices are guaranteed to be valid.
        indices = self.operated_motif.indices

        other_indices = np.sort(np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int),
            indices, assume_unique=True
        )).tolist()
        motif_atoms = self.operated_motif.get_atoms()
        motif_atoms.set_positions(
            self._get_resized_positions(self.operated_motif)
        )
        # Merge with the original atoms to maintain other properties.
        return merge_atoms(
            [self.operated_atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_desc_kwargs: Optional[dict] = None
    ) -> str:
        """Generate a description for the resize action.

        Args:
            precision (int): The number of decimal places to format numerical values in
                the action.
                Note that the precision in the description of the operated motif
                is controlled by `motif_desc_kwargs`, not by this parameter!
            motif_desc_kwargs (Optional[dict]): Additional keyword arguments for the
                motif.describe method.
        Returns:
            str: A string description of the resize action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}

        # Update motif description kwargs. Prevent using addition mode.
        motif_desc_params = inspect.signature(self.motif.describe).parameters
        if "is_addition" in motif_desc_params:
            motif_desc_kwargs["is_addition"] = False

        is_pair = len(self.operated_motif) == 2
        size_word = "length" if is_pair == 2 else "radius"
        if "relative_to_centroid" in self.mode_flag:
            relative_word = "its centroid"
        elif "relative_to_node_index" in self.mode_flag:
            relative_word = (
                f"the atom at index {self.operated_motif.indices[self.relative_to_node_index]}"
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        if "scale_by" in self.mode_flag:
            scale_word = f"by a scale factor of {self.scale_by}"
            if is_pair:
                if self.scale_by > 1:
                    op_word = "elongate"
                else:
                    op_word = "shorten"
            else:
                if self.scale_by > 1:
                    op_word = "enlarge"
                else:
                    op_word = "shrink"
        elif "to_radius" in self.mode_flag:
            scale_word = f"to a {size_word} of {self.to_radius} angstroms"
            is_enlarge = self.to_radius > self.operated_motif.radius()
            if is_pair:
                if is_enlarge:
                    op_word = "elongate"
                else:
                    op_word = "shorten"
            else:
                if is_enlarge:
                    op_word = "enlarge"
                else:
                    op_word = "shrink"
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        return (
            f"{op_word} [{self.operated_motif.describe(**motif_desc_kwargs)}]"
            f" with {relative_word} fixed, {scale_word}."
            f" update atom coordinates only, do not change their order in structure."
        )
