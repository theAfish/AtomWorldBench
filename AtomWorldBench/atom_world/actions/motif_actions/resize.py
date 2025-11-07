"""Implement Resize action."""
from typing import Optional
import inspect

from ase import Atoms
import numpy as np

from .base import BaseMotifAction
from .utils import get_random_motif
from ...motifs.base import BaseMotif
from ...motifs.site_collections.base import BaseSiteCollectionMotif
from ...motifs.regions.base import BaseRegionMotif

from ....utils.atoms_utils import merge_atoms
from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION

from ....common.registry import register


def _check_operated_motif_compatibility(m, mode_flag):
    """Check if the operated motif is compatible with ResizeMotifAction."""

    if "relative_to_centroid" in mode_flag:
        if not hasattr(m, "get_centroid"):
            raise ValueError(
                "operated_motif must have get_centroid method for"
                " relative_to_centroid mode."
            )
    if hasattr(m, "__len__") and len(m) < 2:
        raise ValueError("operated_motif must have at least 2 atoms"
                         " to be resizable.")
    if "to_radius" in mode_flag:
        if not hasattr(m, "radius"):
            raise ValueError(
                "operated_motif must have radius attribute for to_radius mode."
            )
    # Node index mode does not support region motifs.
    if "relative_to_node_index" in mode_flag:
        if isinstance(m, BaseRegionMotif):
            raise ValueError(
                "operated_motif cannot be a region motif for"
                " relative_to_node_index mode."
            )
    return m


# Can only be called "resize-motif" as "resize" may conflict with ResizeStructureAction.
@register(BaseMotifAction, ["resize", "resize-motif"])
class ResizeMotifAction(BaseMotifAction):
    """Resize a motif by changing the motif's radius with respect to its centroid or a node."""
    kwargs_formatting_functions = {
        "operated_motif": _check_operated_motif_compatibility,
    }
    mode_definitions = {
        # Parameters that are always required.
        "_excluded": ["operated_motif"],
        # Four valid modes of operation from combinations of 2x2 options.
        "_combinations": [
            {
                "name_template": "relative_to_{relative_to}_{size_mode}",
                "relative_to":{
                    "centroid": {},  # Does not need extra parameters.
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
                            lambda x: isinstance(x, (int, float)) and x > 0 and x != 1,
                            "scale_by must be a positive number not equal to 1 for scale_by mode."
                        ),
                    },
                    "to_radius": {
                        "to_radius": (
                            lambda x: isinstance(x, (int, float)) and x > 0,
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
            relative_to_node_index: Optional[int] = None,
            scale_by: Optional[float] = None,
            to_radius: Optional[float] = None,
    ):
        """Initialize the ResizeMotifAction.

        `operated_motif` and `operated_atoms` are always required.
        For the rest of parameters, currently, allows 4 modes of operation:
            1, "relative_to_centroid_scale_by":
                resize the motif relative to its centroid by a scale factor.
                In this mode, only `scale_by` is required.
                No other parameters are allowed.
            2, "relative_to_node_index_scale_by":
                resize the motif relative to a node by a scale factor.
                In this mode, `relative_to_node_index` and `scale_by` are required.
                No other parameters are allowed.
            3, "relative_to_centroid_to_radius":
                resize the motif relative to its centroid to a specific radius.
                In this mode, only `to_radius` is required.
                No other parameters are allowed.
            4, "relative_to_node_index_to_radius":
                resize the motif relative to a node to a specific radius.
                In this mode, `relative_to_node_index` and `to_radius` are required.
                No other parameters are allowed.

        Args:
            operated_motif (BaseMotif): The motif to be resized.
            relative_to_node_index (Optional[int]): The index of the node to resize relative to.
                Notice: this is the index in the motif, not in the whole structure. The description
                will use the index in the whole structure for clarity, though.
            scale_by (Optional[float]): Scale factor to apply to the motif's radius.
            to_radius (Optional[float]): The new radius for the motif. Unit is Angstroms.
        """
        # Just for linting.
        self.scale_by=None
        self.relative_to_node_index=None
        self.to_radius=None
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=None, # Does not need a relative motif.
            relative_to_node_index=relative_to_node_index,
            scale_by=scale_by,
            to_radius=to_radius,
        )

    def __post_init__(self):
        """Post-initialization to validate parameters."""
        self._check_operated_motif_in_atoms()

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
            scale = self.to_radius / motif.radius
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
        # _check_operated_motif_in_atoms has been called in __post_init__,
        # so the indices are guaranteed to be valid.
        indices = self.operated_motif.indices

        other_indices = np.sort(np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int),
            indices, assume_unique=True
        )).tolist()
        motif_atoms = self.operated_motif.get_atoms().copy()
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
                is overwritten by this parameter.
            motif_desc_kwargs (Optional[dict]): Additional keyword arguments for the
                motif.describe method.
        Returns:
            str: A string description of the resize action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}

        # Update motif description kwargs. Prevent using addition mode.
        motif_desc_params = inspect.signature(self.operated_motif.describe).parameters
        if "precision" in motif_desc_params:
            motif_desc_kwargs["precision"] = precision

        is_pair = (
                isinstance(self.operated_motif, BaseSiteCollectionMotif) and
                len(self.operated_motif) == 2
        )
        size_word = "length" if is_pair else "radius"
        if "relative_to_centroid" in self.mode_flag:
            relative_word = "its centroid"
        elif "relative_to_node_index" in self.mode_flag:
            relative_word = (
                f"the atom at index {self.operated_motif.indices[self.relative_to_node_index]}"
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        is_region = isinstance(self.operated_motif, BaseRegionMotif)

        if "scale_by" in self.mode_flag:
            scale_word = f"by a scale factor of {self.scale_by:.{precision}f}"
            is_enlarge = (self.scale_by > 1)
        elif "to_radius" in self.mode_flag:
            if is_region:
                scale_word = f"to {self.to_radius:.{precision}f} angstroms"
            else:
                scale_word = f"to a {size_word} of {self.to_radius:.{precision}f} angstroms"
            is_enlarge = (self.to_radius > self.operated_motif.radius)
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

        if is_region:
            if is_enlarge:
                op_word = "away from"
            else:
                op_word = "towards"
        else:
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

        if is_region:
            return (
                f"move [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" {op_word} {relative_word}, such that their distances to {relative_word}"
                f" are changed {scale_word}."
                f" update atom coordinates only, do not change their order in structure."
            )
        else:
            return (
                f"{op_word} [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" {scale_word} by moving its atoms relative to {relative_word}."
                f" update atom coordinates only, do not change their order in structure."
            )

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
        ) -> 'ResizeMotifAction':
        """Get a random ResizeMotifAction.

        Args:
            operated_atoms (Atoms): The atoms to operate on.
            seed (Optional[int]): Random seed for reproducibility.
        Returns:
            ResizeMotifAction: A random ResizeMotifAction instance.
        """
        rng = np.random.default_rng(seed)

        # Pick a random motif for operation.
        class_alias = rng.choice(
            ["bond", "cluster", "sphere"]
        )
        operated_motif_kwargs = {
            "class_alias": class_alias,
            "atoms": operated_atoms,
            "seed": seed,
        }
        if class_alias == "cluster":
            operated_motif_kwargs["cluster_size"] = rng.integers(2, 5)
            operated_motif_kwargs["max_cluster_radius"] = 4.0
        elif class_alias == "bond":
            operated_motif_kwargs["max_cluster_radius"] = 4.0
        elif class_alias == "sphere":
            motif_style = rng.choice(
                ["center_around_atom_index", "center_around_coordinates"],
                p = [0.3, 0.7], # Prefer coordinates to avoid always picking existing atoms.
            )
            operated_motif_kwargs["style"] = motif_style

        operated_motif = get_random_motif(**operated_motif_kwargs)

        kwargs = {
            "operated_motif": operated_motif,
        }

        # Randomly choose relative mode.
        if class_alias != "sphere":
            relative_style = rng.choice(
                ["relative_to_centroid", "relative_to_node_index"],
                p = [0.6, 0.4], # Prefer centroid to avoid always picking existing atoms.
            )
            if relative_style == "relative_to_node_index":
                relative_to_node_index = int(rng.integers(0, len(operated_motif)))
                kwargs["relative_to_node_index"] = relative_to_node_index

        # Randomly choose size mode.
        size_style = rng.choice(
            ["scale_by", "to_radius"],
            p = [0.6, 0.4], # Emphasize scale_by to avoid too small or too large sizes.
        )
        if size_style == "scale_by":
            scale_by = float(rng.uniform(0.8, 1.2))
            kwargs["scale_by"] = scale_by
        elif size_style == "to_radius":
            to_radius = float(rng.uniform(
                operated_motif.radius * 0.8,
                operated_motif.radius * 1.2
            ))
            kwargs["to_radius"] = to_radius

        return cls(**kwargs)
