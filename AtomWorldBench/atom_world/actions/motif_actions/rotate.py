"""Rotate motif action."""
from typing import Optional
import inspect

from ase import Atoms
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .base import BaseMotifAction
from .utils import get_random_motif
from ...motifs.base import BaseMotif
from ...motifs.site_collections.base import BaseSiteCollectionMotif
from ...motifs.regions.base import BaseRegionMotif
from ...motifs.site_collections.bond import BondMotif

from ....utils.coord_utils import check_coordinates_shape
from ....utils.description_utils import describe_arraylike
from ....utils.atoms_utils import merge_atoms

from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....common.registry import register


def _check_rotation_axis_vector(r):
    """Normalize rotation axis vector."""
    r = check_coordinates_shape(
        r, "rotation_axis_vector", expected_1d=True, allow_none=True
    )
    if r is None:
        return None
    if np.linalg.norm(r) < 1e-8:
        raise ValueError("rotation_axis_vector cannot be a zero vector.")
    return r / np.linalg.norm(r)

def _check_operated_motif_compatibility(m, mode_flag):
    """Check if the operated motif is compatible for rotation."""
    if isinstance(m, BondMotif):
        raise ValueError("Bond motifs are not allowed as operated motifs for rotation.")
    if "self" in mode_flag:
        if not hasattr(m, "get_centroid"):
            raise ValueError(
                "Operated motif must support centroid calculation for self-relative rotation."
            )
        if isinstance(m, BaseSiteCollectionMotif) and len(m) < 2:
            raise ValueError(
                "Operated site collection motifs must have at least two sites"
                " for self-relative rotation to be meaningful."
            )
    if isinstance(m, BaseRegionMotif):
        if not "self" in mode_flag:
            raise ValueError(
                "Region motifs can only be used as operated motifs in self-relative rotation."
            )
    return m

def _check_relative_motif_compatibility(m):
    """Check if the relative motif is compatible for rotation."""
    if m is None:
        return m
    if isinstance(m, BaseRegionMotif):
        raise ValueError("Region motifs are not allowed as relative motifs for rotation.")
    return m


# Can only be called "rotate-motif" as "rotate" may conflict with RotateStructureAction.
@register(BaseMotifAction, ["rotate", "rotate-motif"])
class RotateMotifAction(BaseMotifAction):
    """Action to rotate a motif in the structure.

    Notice: this operation only allows relative style.
    """
    kwargs_formatting_functions = {
        "euler_angles": lambda x: check_coordinates_shape(
            x, "euler_angles", expected_1d=True, allow_none=True
        ),
        "rotation_axis_vector": _check_rotation_axis_vector,  # Rotation axis vector normalized.
        "relative_to_position": lambda x: check_coordinates_shape(
            x, "relative_to_position", expected_1d=True, allow_none=True
        ),
        "operated_motif": _check_operated_motif_compatibility,
        "relative_to_motif": _check_relative_motif_compatibility,
    }
    mode_definitions = {
        # operated_motif and operated_atoms are always required.
        # position_fractional does not need to be checked.
        "_excluded": ["operated_motif", "position_fractional"],
        "euler_relative_to_position": {
            "euler_angles": None,
            "relative_to_position": None,
        },
        "euler_relative_to_motif": {
            "euler_angles": None,
            "relative_to_motif": None,
            "relative_style": (
                lambda s: s == "centroid_distance",
                "Relative style must be centroid_distance for euler_relative_to_motif mode."
            ),
        },
        "euler_relative_to_self": {
            "euler_angles": None,
            "relative_style": (
                lambda s: s == "self",
                "Relative style must be self for euler_relative_to_self mode."
            ),
        },
        "axis_relative_to_self":{
            "rotation_axis_vector": None,
            "rotation_axis_angle": (
                lambda x: isinstance(x, (int, float)),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_style": (
                lambda s: s == "self",
                "Relative style must be self for axis_relative_to_self mode."
            ),
        },
        "axis_relative_to_position": {
            "rotation_axis_vector": None,
            "rotation_axis_angle": (
                lambda x: isinstance(x, (int, float)),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_to_position": None,
        },
        "axis_relative_to_regular_motif":{
            "rotation_axis_vector": None,  # Need to provide axis, motif only used as center.
            "rotation_axis_angle": (
                lambda x: isinstance(x, (int, float)),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_to_motif": None,
            "relative_style": (
                lambda s: s == "centroid_distance",
                "Relative style must be centroid_distance for axis_relative_to_regular_motif mode."
            ),
        },
        "axis_relative_to_pair_motif": {
            "rotation_axis_angle": (
                lambda x: isinstance(x, (int, float)),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_to_motif": (
                lambda m: isinstance(m, BaseSiteCollectionMotif) and len(m) == 2,
                "Only pair site-collection motifs are allowed for axis_relative_pair_motif mode."
            ),
            "relative_axis_origin_index": (
                lambda i: i in [0, 1],
                "Relative atom index must be provided as 0 or 1 for"
                " axis_relative_pair_motif mode."
            ),
            "relative_style": (
                lambda s: s == "rotation_axis",
                "Relative style must be rotation_axis for axis_relative_to_pair_motif mode."
            )
        }
    }
    mode_probabilities = {
        "euler_relative_to_position": 0.1,
        "euler_relative_to_motif": 0.1,
        "euler_relative_to_self": 0.1,
        "axis_relative_to_self": 0.2,
        "axis_relative_to_position": 0.2,
        "axis_relative_to_regular_motif": 0.1,
        "axis_relative_to_pair_motif": 0.2,
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            # The rest of parameters are optional, depending on the mode.
            relative_to_motif: Optional[BaseMotif] = None,
            euler_angles: Optional[ArrayLike] = None,  # In degrees. Always ZXZ, active convention.
            rotation_axis_vector: Optional[ArrayLike] = None,
            rotation_axis_angle: Optional[float] = None,  # In degrees.
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: Optional[bool] = False,
            relative_style: str = None,
            relative_axis_origin_index: Optional[int] = None,
    ):
        """Initialize the RotateMotifAction with a relative motif and style.

       `operated_motif` and `operated_atoms` are always required.
        For the test of parameters, currently, allows 7 modes of operation:
            1, `euler_relative_to_position`: Perform a rotation using Euler angles
                around a specified center position. In this mode, provide `euler_angles`,
                `relative_to_position`. No other parameters should be given except
                `position_fractional`.
            2,  `euler_relative_to_motif`: Perform a rotation using Euler angles
                around a specified motif's centroid. In this mode, provide `euler_angles`,
                `relative_to_motif`, and `relative_style`=="centroid_distance".
                No other parameters should be give except `position_fractional`.
            3, `euler_relative_to_self`: Perform a rotation using Euler angles
                around the centroid of the provided motif in the `execute` method itself.
                In this mode, provide `euler_angles`, and set `relative_style`=="self".
                No other parameters should be given except `position_fractional`.
            4, `axis_relative_to_self`: Perform a rotation around a specified rotation
                axis vector and the centroid of the provided motif in the `execute` method itself.
                In this mode, provide `rotation_axis_vector`, `rotation_axis_angle`,
                and set `relative_style`=="self".
            5, `axis_relative_to_position`: Perform a rotation around a specified rotation
                axis vector and a specified center position. In this mode, provide
                `rotation_axis_vector`, `rotation_axis_angle`, and `relative_to_position`.
                No other parameters should be given except `position_fractional`.
            6, `axis_relative_to_regular_motif`: Perform a rotation around a specified
                rotation axis vector and a specified motif's centroid. In this mode,
                provide `rotation_axis_vector`, `rotation_axis_angle`, `relative_to_motif`,
                and `relative_style`=="centroid_distance".
                No other parameters should be given except `position_fractional`.
            7, `axis_relative_to_pair_motif`: Perform a rotation around a specified
                pair motif's direction vector (pointing from the origin atom to the other atom)
                and the pair motif's centroid as the rotation center. In this mode,
                provide `rotation_axis_angle`, `relative_to_motif`, `relative_axis_origin_index`,
                and `relative_style`=="rotation_axis". `relative_to_motif` must be a motif with
                two sites.
                No other parameters should be given except  `position_fractional`.
        Note that all euler rotations are in active and intrinsic ZXZ convention.
        Euler angles and rotation angles are in degrees, using counter-clockwise
        direction.
        Args:
            operated_motif (BaseMotif):
                The motif that this action operates on. Required.
            euler_angles (Optional[ArrayLike]):
                Euler angles for rotation in degrees (ZXZ intrinsic convention, active rotation,
                counter-clockwise direction).
                Unit in degrees. Must be a 1D array-like of length 3.
            rotation_axis_vector (Optional[ArrayLike]):
                Vector defining the rotation axis. Rotation will be computed from
                Rodriguez' rotation formula. Must be a 1D array-like of length 3.
                This is not affected by `position_fractional`, always in cartesian metric.
            rotation_axis_angle (Optional[float]):
                Angle of counter-clockwise rotation around the rotation axis in degrees.
            relative_to_position (Optional[ArrayLike]):
                Rotation center position.
            position_fractional (Optional[bool]):
                Whether all positions provided in arguments are fractional. If False, will
                be cartesian. This will also affect the description style of the action.
                Default is False.
            relative_style (str):
                Style of the relative action.
            relative_to_motif (Optional[BaseMotif]):
                Motif to rotate relative to. If rotating in Euler angles, this motif's centroid
                will be used as the rotation center. If rotating around a vector, this motif must
                be a pair motif, with its rotation vector calculated from the line of the pair motif,
                pointing from the specified origin atom to the other atom, and its centroid as the
                rotation center.
            relative_axis_origin_index(Optional[int]):
                When the relative motif is a pair motif, this index specifies which atom
                will be used as the origin of the rotation vector. Must be provided if working
                 in `position_in_line` mode.
        """
        self.relative_to_position=None
        self.position_fractional=None
        self.rotation_axis_vector=None
        self.relative_axis_origin_index=None
        self.euler_angles=None
        self.rotation_axis_angle=None
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=relative_to_motif,
            euler_angles=euler_angles,
            rotation_axis_vector=rotation_axis_vector,
            rotation_axis_angle=rotation_axis_angle,
            relative_to_position=relative_to_position,
            position_fractional=position_fractional,
            relative_style=relative_style,
            relative_axis_origin_index=relative_axis_origin_index,
        )

    def __post_init__(self):
        """Post-initialization to validate parameters."""
        self._check_operated_motif_in_atoms()
        self._check_relative_motif_in_atoms()
        # Prevent single-site motifs from rotating around their own centroid.
        if "relative_to_position" in self.mode_flag:
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            ref_centroid = (
                self.relative_to_position @ self.operated_atoms.cell.complete()
                if self.position_fractional else self.relative_to_position
            )
            if (
                    np.allclose(operated_centroid, ref_centroid, atol=1e-8)
                    and len(self.operated_motif) == 1
            ):
                raise ValueError(
                    "Rotation center position cannot be the same"
                    " as the operated motif's centroid, when the"
                    " operated motif has only one atom."
                )
        if (
                ("relative_to_regular_motif" in self.mode_flag)
                or ("relative_to_pair_motif" in self.mode_flag)
        ):
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            ref_centroid = self.relative_to_motif.get_centroid(fractional=False)
            if (
                    np.allclose(operated_centroid, ref_centroid, atol=1e-8)
                    and len(self.operated_motif) == 1
            ):
                raise ValueError(
                    "Rotation center motif's centroid cannot be the same as"
                    " the operated motif's centroid, when the operated motif"
                    " has only one atom."
                )

    def _get_rotation_center(self):
        """Helper function to get the rotation center based on the mode.

        Return cartesian coordinates of the rotation center.
        """
        if self.mode_flag in ("euler_relative_to_position", "axis_relative_to_position"):
            position = self.relative_to_position
            if self.position_fractional:
                position = position @ self.operated_atoms.cell.complete()
        elif self.mode_flag in (
            "euler_relative_to_motif",
            "axis_relative_to_regular_motif",
            "axis_relative_to_pair_motif",
        ):
            position = self.relative_to_motif.get_centroid(fractional=False)
        elif self.mode_flag in ["euler_relative_to_self", "axis_relative_to_self"]:
            position = self.operated_motif.get_centroid(fractional=False)
        else:
            raise ValueError(
                f"Unknown mode {self.mode_flag} for rotation center."
            )
        return position

    def _get_rotation_axis(self):
        """Helper function to get the rotation axis based on the mode.

        Return cartesian coordinates of the rotation axis.
        """
        if self.mode_flag in (
            "axis_relative_to_position",
            "axis_relative_to_regular_motif",
            "axis_relative_to_self",
        ):
            return self.rotation_axis_vector
        elif self.mode_flag == "axis_relative_to_pair_motif":
            origin_position = self.relative_to_motif.cart_coords[self.relative_axis_origin_index]
            end_position = self.relative_to_motif.cart_coords[1 - self.relative_axis_origin_index]
            vec = end_position - origin_position
            vec /= np.linalg.norm(vec)
            return vec
        else:
            raise ValueError(
                f"Mode {self.mode_flag} not allowed for rotation axis."
            )

    def execute(self) -> Atoms:
        """Execute the rotation action on the atoms and motif.

        Returns:
            Atoms: The Atoms object with motif rotated.
        """
        # Call to __post_init__ to ensure all indices are set correctly.
        indices = self.operated_motif.indices
        other_indices = np.sort(np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int), indices, assume_unique=True
        )).tolist()

        center = self._get_rotation_center()
        motif_atoms = self.operated_motif.get_atoms().copy()
        if self.mode_flag in (
                "euler_relative_to_position",
                "euler_relative_to_motif",
                "euler_relative_to_self",
        ):
            alpha, beta, gamma = self.euler_angles
            rotation = Rotation.from_euler(
                seq="ZXZ", angles=[alpha, beta, gamma], degrees=True
            )
            new_motif_positions = rotation.apply(
                motif_atoms.get_positions(wrap=False).copy() - center
            ) + center
            motif_atoms.set_positions(new_motif_positions)
        elif self.mode_flag in (
                "axis_relative_to_self",
                "axis_relative_to_position",
                "axis_relative_to_regular_motif",
                "axis_relative_to_pair_motif",
        ):
            motif_atoms.rotate(
                a=self.rotation_axis_angle,
                v=self._get_rotation_axis(),
                center=center,
                rotate_cell=False,
            )  # Rotate atoms only, do not rotate cell.
        else:
            raise NotImplementedError("Invalid mode_flag: {self.mode_flag}.")

        return merge_atoms(
            [self.operated_atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_desc_kwargs: Optional[dict] = None,
            relative_motif_desc_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to translate a motif.

         Note that motif and relative motif description styles are not affected by the action's
        `position_fractional` attribute.
        Args:
            precision (int): The precision for formatting numerical values in the description in decimals.
                Default is set in `globals.py`, typically 4.
                Note that the precision in the description of the operated motif and the
                relative motif will be overwritten by this parameter.
            motif_desc_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_desc_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}
        relative_motif_desc_kwargs = relative_motif_desc_kwargs or {}

        # Update motif description kwargs.
        motif_desc_params = inspect.signature(self.operated_motif.describe).parameters
        relative_motif_desc_params = inspect.signature(
            self.relative_to_motif.describe
        ).parameters if self.relative_to_motif is not None else {}
        if "precision" in motif_desc_params:
            motif_desc_kwargs["precision"] = precision
        if "precision" in relative_motif_desc_params:
            relative_motif_desc_kwargs["precision"] = precision

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = (
            "update atom coordinates only, do not change their order in structure."
        )

        operated_motif_desc, operated_other_notes = self.operated_motif.describe(**motif_desc_kwargs)
        if relative_motif_desc_kwargs is not None and self.relative_to_motif is not None:
            relative_motif_desc, relative_other_notes = self.relative_to_motif.describe(
                **relative_motif_desc_kwargs
            )
            # check the other notes are same or not
            if relative_other_notes == operated_other_notes:
                relative_other_notes = ""

        if self.mode_flag == "euler_relative_to_position":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, right-hand counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees"
                f" around a center position in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                f" {operated_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_motif":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, right-hand counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" around the centroid of"
                f" {relative_motif_desc}"
                f" as the rotation center."
                f" {operated_other_notes}"
                f" {relative_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_self":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, right-hand counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" around its own centroid as the rotation center."
                f" {operated_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_self":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees right-hand counter-clockwise around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" around its own centroid as the rotation center."
                f" {operated_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_position":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees right-hand counter-clockwise around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" around a rotation center in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                f" {operated_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_regular_motif":
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees right-hand counter-clockwise around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" around the centroid of"
                f" {relative_motif_desc}"
                f" as the rotation center."
                f" {operated_other_notes}"
                f" {relative_other_notes}"
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_pair_motif":
            relative_motif_desc_kwargs.update({"style": "index"})
            return (
                f"rotate {operated_motif_desc}"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees right-hand counter-clockwise around a rotation axis defined by the line of"
                f" {relative_motif_desc},"
                f" pointing from the atom"
                f" with index {self.relative_to_motif.indices[self.relative_axis_origin_index]}"
                f" to the other atom, around the centroid of the axis pair"
                f" as the rotation center."
                f" {operated_other_notes}"
                f" {relative_other_notes}"
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")

    # python
    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Get a random instance of RotateMotifAction with compatibility checks.

        Ensures the chosen operated_motif is compatible with the randomly selected mode,
        e.g. modes containing "self" get a multi-site operated motif (cluster).
        """
        rng = np.random.default_rng(seed)

        # some hyperparameters
        max_cluster_radius = 4.0
        max_cluster_size = min(4, len(operated_atoms) - 1)

        # Pick mode first so we can produce compatible motifs/params.
        mode_flag = cls.get_random_mode(seed)

        # Choose operated_motif in a mode-aware way:
        if "self" in mode_flag:
            # For self-relative modes require motifs that support centroid and are multi-site.
            operated_class_alias = rng.choice(["cluster", "sphere"])
        else:
            # For other modes allow site/cluster/sphere as before (avoid bond as operated motif).
            operated_class_alias = rng.choice(["site", "cluster"])
        operated_motif_kwargs = {
            "class_alias": operated_class_alias,
            "atoms": operated_atoms,
            "seed": seed,
        }
        if operated_class_alias == "cluster":
            operated_motif_kwargs["cluster_size"] = rng.integers(2, max_cluster_size + 1)
            operated_motif_kwargs["max_cluster_radius"] = max_cluster_radius
        if operated_class_alias == "sphere":
            motif_style = rng.choice(
                ["center_around_atom_index", "center_around_coordinates"],
                p=[0.3, 0.7],
            )
            operated_motif_kwargs["style"] = motif_style

        operated_motif = get_random_motif(**operated_motif_kwargs)
        kwargs = {
            "operated_motif": operated_motif
        }

        # Rotation parameters per mode.
        if mode_flag in (
                "euler_relative_to_position",
                "euler_relative_to_motif",
                "euler_relative_to_self",
        ):
            kwargs["euler_angles"] = rng.uniform(-180, 180, size=3)
        elif mode_flag in (
                "axis_relative_to_self",
                "axis_relative_to_position",
                "axis_relative_to_regular_motif",
        ):
            rotation_axis_vector = rng.normal(size=3)
            rotation_axis_vector /= np.linalg.norm(rotation_axis_vector)
            rotation_axis_angle = float(rng.uniform(-180, 180))
            kwargs["rotation_axis_vector"] = rotation_axis_vector
            kwargs["rotation_axis_angle"] = rotation_axis_angle
        elif mode_flag == "axis_relative_to_pair_motif":
            rotation_axis_angle = float(rng.uniform(-180, 180))
            kwargs["rotation_axis_angle"] = rotation_axis_angle

        # Position-related params.
        if mode_flag in ("euler_relative_to_position", "axis_relative_to_position"):
            use_fractional = bool(rng.choice([True, False]))
            relative_to_fractional = rng.uniform(size=3)
            if len(operated_motif) == 1:
                relative_to_motif_centroid_fractional = operated_motif.get_centroid(
                    fractional=True
                )
                # Prevent overlap.
                if np.allclose(
                        relative_to_fractional,
                        relative_to_motif_centroid_fractional,
                        atol=1e-4,
                ):
                    relative_to_fractional += np.array([0.01, 0.01, 0.01])
            if use_fractional:
                kwargs["relative_to_position"] = relative_to_fractional
            else:
                cell = operated_atoms.cell.complete()
                relative_to_position = relative_to_fractional @ cell
                kwargs["relative_to_position"] = relative_to_position
            kwargs["position_fractional"] = use_fractional

        # Relative motif for centroid-based or pair-axis modes.
        if mode_flag in (
                "euler_relative_to_motif",
                "axis_relative_to_regular_motif",
        ):
            relative_class_alias = rng.choice(["site", "cluster", "bond"])
            relative_motif_kwargs = {
                "class_alias": relative_class_alias,
                "atoms": operated_atoms,
                "seed": seed + 1 if seed is not None else seed,  # Prevent overlap.
            }
            if relative_class_alias == "cluster":
                relative_motif_kwargs["cluster_size"] = rng.integers(2, 5)
                relative_motif_kwargs["max_cluster_radius"] = 4.0
            operated_motif_centroid = operated_motif.get_centroid(fractional=False)
            # Generate until not overlapping.
            relative_motif = get_random_motif(**relative_motif_kwargs)
            relative_to_motif_centroid = relative_motif.get_centroid(fractional=False)
            n_try = 0
            while np.allclose(
                    operated_motif_centroid,
                    relative_to_motif_centroid,
                    atol=1e-4,
            ) and len(operated_motif) == 1 and n_try < 20:
                relative_motif_kwargs["seed"] = (
                    relative_motif_kwargs["seed"] + 1
                    if relative_motif_kwargs["seed"] is not None
                    else None
                )
                n_try += 1
                relative_motif = get_random_motif(**relative_motif_kwargs)
                relative_to_motif_centroid = relative_motif.get_centroid(fractional=False)
            if n_try >= 20:
                raise ValueError(
                    "Failed to generate a non-overlapping relative motif"
                    " after 20 attempts. Please check the operated_atoms provided."
                )
            kwargs["relative_to_motif"] = relative_motif
        elif mode_flag == "axis_relative_to_pair_motif":
            # Need a pair motif (bond or cluster size 2)
            relative_class_alias = rng.choice(["bond", "cluster"])
            relative_motif_kwargs = {
                "class_alias": relative_class_alias,
                "atoms": operated_atoms,
                "seed": seed,
            }
            if relative_class_alias == "cluster":
                relative_motif_kwargs["cluster_size"] = 2
                relative_motif_kwargs["max_cluster_radius"] = 4.0
            else:
                relative_motif_kwargs["max_cluster_radius"] = 4.0
            operated_motif_centroid = operated_motif.get_centroid(fractional=False)
            # Generate until not overlapping.
            relative_motif = get_random_motif(**relative_motif_kwargs)
            relative_to_motif_centroid = relative_motif.get_centroid(fractional=False)
            n_try = 0
            while np.allclose(
                    operated_motif_centroid,
                    relative_to_motif_centroid,
                    atol=1e-8,
            ) and len(operated_motif) == 1 and n_try < 20:
                relative_motif_kwargs["seed"] = (
                    relative_motif_kwargs["seed"] + 1
                    if relative_motif_kwargs["seed"] is not None
                    else None
                )
                n_try += 1
                relative_motif = get_random_motif(**relative_motif_kwargs)
                relative_to_motif_centroid = relative_motif.get_centroid(fractional=False)
            if n_try >= 20:
                raise ValueError(
                    "Failed to generate a non-overlapping relative motif"
                    " after 20 attempts. Please check the operated_atoms provided."
                )
            kwargs["relative_to_motif"] = relative_motif
            kwargs["relative_axis_origin_index"] = int(rng.choice([0, 1]))

        # Determine relative_style when needed.
        if mode_flag in (
                "euler_relative_to_motif",
                "axis_relative_to_regular_motif",
        ):
            kwargs["relative_style"] = "centroid_distance"
        elif mode_flag in (
                "euler_relative_to_self",
                "axis_relative_to_self",
        ):
            kwargs["relative_style"] = "self"
        elif mode_flag == "axis_relative_to_pair_motif":
            kwargs["relative_style"] = "rotation_axis"

        return cls(**kwargs)
