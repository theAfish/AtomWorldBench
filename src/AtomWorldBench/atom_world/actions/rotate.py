"""Rotate motif action."""
from typing import Optional
from numbers import Number
import inspect

from ase import Atoms
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .base import BaseAction
from ..motifs.base import BaseMotif

from ...utils.coord_utils import check_coordinates_shape
from ...utils.description_utils import describe_arraylike
from ...utils.atoms_utils import merge_atoms

from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


def _check_rotation_axis_vector(r):
    """Normalize rotation axis vector."""
    r = check_coordinates_shape(
        r, "rotation_axis_vector", expected_1d=True, allow_none=True
    )
    if r is None:
        return None
    return r / np.linalg.norm(r)


class RotateAction(BaseAction):
    """Action to rotate a motif in the structure.

    Notice: this operation only allows relative style.
    """
    kwargs_formating_functions = {
        "euler_angles": lambda x: check_coordinates_shape(
            x, "euler_angles", expected_1d=True, allow_none=True
        ),
        "rotation_axis_vector": _check_rotation_axis_vector,  # Rotation axis vector normalized.
        "relative_to_position": lambda x: check_coordinates_shape(
            x, "relative_to_position", expected_1d=True, allow_none=True
        ),
    }
    mode_definitions = {
        # operated_motif and operated_atoms are always required.
        # position_fractional does not need to be checked.
        "_excluded": ["operated_motif", "operated_atoms", "position_fractional"],
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
        "axis_relative_to_position": {
            "rotation_axis_vector": None,
            "rotation_axis_angle": (
                lambda x: isinstance(x, Number),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_to_position": None,
        },
        "axis_relative_to_regular_motif":{
            "rotation_axis_vector": None,  # Need to provide axis, motif only used as center.
            "rotation_axis_angle": (
                lambda x: isinstance(x, Number),
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
                lambda x: isinstance(x, Number),
                "Rotation axis angle must be a number in degrees.",
            ),
            "relative_to_motif": (
                lambda m: len(m) == 2,
                "Only pair motifs are allowed for axis_relative_pair_motif mode."
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

    def __init__(
            self,
            operated_motif: BaseMotif,
            operated_atoms: Atoms,
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
        For the test of parameters, currently, allows 6 modes of operation:
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
            4, `axis_relative_to_position`: Perform a rotation around a specified rotation
                axis vector and a specified center position. In this mode, provide
                `rotation_axis_vector`, `rotation_axis_angle`, and `relative_to_position`.
                No other parameters should be given except `position_fractional`.
            5, `axis_relative_to_regular_motif`: Perform a rotation around a specified
                rotation axis vector and a specified motif's centroid. In this mode,
                provide `rotation_axis_vector`, `rotation_axis_angle`, `relative_to_motif`,
                and `relative_style`=="centroid_distance".
                No other parameters should be given except `position_fractional`.
            6, `axis_relative_to_pair_motif`: Perform a rotation around a specified
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
            operated_atoms (Atoms):
                The atoms that this action operates on. Required.
            euler_angles (Optional[ArrayLike]):
                Euler angles for rotation in degrees (ZXZ intrinsic convention, active rotation,
                counter-clockwise direction).
                Unit in degrees. Must be a 1D array-like of length 3.
            rotation_axis_vector (Optional[ArrayLike]):
                Vector defining the rotation axis. Rotation will be computed from
                Rodriguez' rotation formula. Must be a 1D array-like of length 3.
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
        super().__init__(
            operated_motif=operated_motif,
            operated_atoms=operated_atoms,
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
        self.__check_operated_motif_compatibility()
        self.__check_operated_motif_in_atoms()
        self.__check_relative_motif_in_atoms()
        if "self" in self.mode_flag:
            if len(self.operated_motif) == 1:
                raise ValueError(
                    "Cannot use self-relative rotation with a point motif."
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
        elif self.mode_flag == "euler_relative_to_self":
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
                f"Mode {self.mode_flag} not allowed in rotation center."
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
        motif_atoms = self.operated_motif.get_atoms()
        if self.mode_flag in (
                "euler_relative_to_position",
                "euler_relative_to_motif",
                "euler_relative_to_self",
        ):
            alpha, beta, gamma = self.euler_angles
            rotation = Rotation.from_euler(
                seq="ZXZ", angles=[alpha, beta, gamma], degrees=True
            )
            new_motif_positions = rotation.apply(motif_atoms.positions - center) + center
            motif_atoms.set_positions(new_motif_positions)
        elif self.mode_flag in (
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
            raise f"Invalid mode_flag: {self.mode_flag}."

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
                relative motif is controlled by the `motif_desc_kwargs` and
                `relative_motif_desc_kwargs` parameters, respectively, not by this parameter!
            motif_desc_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_desc_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}
        relative_motif_desc_kwargs = relative_motif_desc_kwargs or {}

        # Update motif description kwargs.
        motif_desc_params = inspect.signature(self.motif.describe).parameters
        relative_motif_desc_params = inspect.signature(
            self.relative_to_motif.describe
        ).parameters if self.relative_to_motif is not None else {}
        # Never use addition mode for rotation.
        if "is_addition" in motif_desc_params:
            motif_desc_kwargs["is_addition"] = False
        if "is_addition" in relative_motif_desc_params:
            relative_motif_desc_kwargs["is_addition"] = False

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = (
            "update atom coordinates only, do not change their order in structure."
        )

        if self.mode_flag == "euler_relative_to_position":
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees"
                f" around a center position in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_motif":
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" using the centroid of"
                f" [{self.relative_to_motif.describe(**relative_motif_desc_kwargs)}]"
                f" as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_self":
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by euler angles (Z-X-Z intrinsic convention,"
                f" active rotation, counter-clockwise direction) in"
                f" {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" using its own centroid as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_position":
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees counter-clockwise around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" and a rotation center in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_regular_motif":
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees counter-clockwise around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" and using the centroid of"
                f" [{self.relative_to_motif.describe(**relative_motif_desc_kwargs)}]"
                f" as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_pair_motif":
            relative_motif_desc_kwargs.update({"style": "index"})
            return (
                f"rotate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" in the structure by {self.rotation_axis_angle:.{precision}f}"
                f" degrees around a rotation axis defined by the line of"
                f" [{self.relative_to_motif.describe(**relative_motif_desc_kwargs)}],"
                f" pointing from the atom"
                f" with index {self.relative_to_motif.indices[self.relative_axis_origin_index]}"
                f" to the other atom, and using the centroid of the axis pair"
                f" as the rotation center."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
