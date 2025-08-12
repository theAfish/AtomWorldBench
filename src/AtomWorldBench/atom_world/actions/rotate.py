"""Rotate motif action."""
from typing import Optional, Tuple
from numbers import Number

from ase import Atoms
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .base import BaseAction
from ..motifs.site_collections.base import BaseSiteCollectionMotif

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


class RotateMotifAction(BaseAction):
    """Action to rotate a motif in the structure.

    Notice: this operation only allows relative style.
    """
    kwargs_and_formating_functions = {
        "euler_angles": lambda x: check_coordinates_shape(
            x, "euler_angles", expected_1d=True, allow_none=True
        ),
        "rotation_axis_vector": _check_rotation_axis_vector,  # Rotation axis vector normalized.
        "roration_axis_angle": lambda x: float(x) if x is not None else None,
        "relative_to_position": lambda x: check_coordinates_shape(
            x, "relative_to_position", expected_1d=True, allow_none=True
        ),
    }
    mode_definitions = {
        "_excluded": ["position_fractional"],
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
            "self_relative": lambda x: x is True,
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
            "relative_pair_origin_index": (
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
            euler_angles: Optional[ArrayLike] = None,  # In degrees. Always ZXZ, active convention.
            rotation_axis_vector: Optional[ArrayLike] = None,
            rotation_axis_angle: Optional[float] = None,  # In degrees.
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: Optional[bool] = True,
            relative_style: str = None,
            relative_to_motif: Optional[BaseSiteCollectionMotif] = None,
            relative_pair_origin_index: Optional[int] = None,
            self_relative: Optional[bool] = None,
    ):
        """Initialize the RotateMotifAction with a relative motif and style.

        Currently, allows 6 modes of operation:
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
                In this mode, provide `euler_angles`, and set `self_relative` to True.
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
                provide `rotation_axis_angle`, `relative_to_motif`, `relative_pair_origin_index`,
                and `relative_style`=="rotation_axis". `relative_to_motif` must be a motif with
                two sites.
                No other parameters should be given except  `position_fractional`.
        Note that all euler rotations are in active and intrinsic ZXZ convention.
        Euler angles and rotation angles are in degrees, using counter-clockwise
        direction.
        Args:
            euler_angles (Optional[ArrayLike]):
                Euler angles for rotation in degrees (ZXZ intrinsic convention, active rotation).
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
                be cartesian.
                This will also affect the description style of the action. Default is True
            relative_style (str):
                Style of the relative action.
            relative_to_motif (Optional[BaseSiteCollectionMotif]):
                Motif to rotate relative to. If rotating in Euler angles, this motif's centroid
                will be used as the rotation center. If rotating around a vector, this motif must
                be a pair motif, with its rotation vector calculated from the line of the pair motif,
                pointing from the specified origin atom to the other atom, and its centroid as the
                rotation center.
            relative_pair_origin_index(Optional[int]):
                When the relative motif is a pair motif, this index specifies which atom
                will be used as the origin of the rotation vector. Must be provided if working
                 in `position_in_line` mode.
            self_relative (bool):
                If True, Euler rotate the motif provided in `execute` method itself, using its
                own centroid as the rotation center.
        """
        self.euler_angles = None
        self.rotation_axis_vector = None
        self.rotation_axis_angle = None
        self.relative_to_position = None
        self.position_fractional = None
        self.relative_style = None
        self.relative_to_motif = None
        self.relative_pair_origin_index = None
        self.self_relative = None
        super().__init__(
            euler_angles=euler_angles,
            rotation_axis_vector=rotation_axis_vector,
            rotation_axis_angle=rotation_axis_angle,
            relative_to_position=relative_to_position,
            position_fractional=position_fractional,
            relative_style=relative_style,
            relative_to_motif=relative_to_motif,
            relative_pair_origin_index=relative_pair_origin_index,
            self_relative=self_relative,
        )

    def _check_compatibility(self, atoms: Atoms, motif: BaseSiteCollectionMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given atoms and motif.

        In this action, only checks whether motif is in atoms.
        Args:
            atoms (Atoms): The Atoms object to check compatibility with.
            motif (BaseSiteCollectionMotif): The motif to check compatibility with.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating compatibility
                and a message describing the compatibility status.
        """
        # Check if the relative motif is in the structure.
        indices, message = motif.find_indices_in_atoms(
            atoms,
            modify_indices_in_place=True
        )
        return indices is not None, f"operated not found in structure: {message}"

    def _get_rotation_center(self, motif: BaseSiteCollectionMotif):
        """Helper function to get the rotation center based on the mode.

        Return cartesian coordinates of the rotation center.
        """
        if self.mode_flag in ("euler_relative_to_position", "axis_relative_to_position"):
            position = self.relative_to_position
            if self.position_fractional:
                position = position @ motif.cell.complete()
        elif self.mode_flag in (
            "euler_relative_to_motif",
            "axis_relative_to_regular_motif",
            "axis_relative_to_pair_motif",
        ):
            position = self.relative_to_motif.get_centroid(fractional=False)
        elif self.mode_flag == "euler_relative_to_self":
            position = motif.get_centroid(fractional=False)
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
            origin_position = self.relative_to_motif.cart_coords[self.relative_pair_origin_index]
            end_position = self.relative_to_motif.cart_coords[1 - self.relative_pair_origin_index]
            vec = end_position - origin_position
            vec /= np.linalg.norm(vec)
            return vec
        else:
            raise ValueError(
                f"Mode {self.mode_flag} not allowed in rotation center."
            )

    def _execute(
            self,
            atoms: Atoms,
            motif: BaseSiteCollectionMotif
    ) -> Atoms:
        """Execute the rotation action on the atoms and motif.

        Args:
            atoms (Atoms): The Atoms object to rotate.
            motif (BaseSiteCollectionMotif): The motif to rotate.

        Returns:
            Atoms: The Atoms object with motif rotated.
        """
        indices, _ = motif.find_indices_in_atoms(
            atoms,
            modify_indices_in_place=True
        )
        other_indices = np.setdiff1d(
            np.arange(len(atoms), dtype=int), indices, assume_unique=True
        ).tolist()

        center = self._get_rotation_center(motif)
        motif_atoms = motif.get_atoms()
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
            [atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            motif: BaseSiteCollectionMotif,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_kwargs: Optional[dict] = None,
            relative_motif_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to translate a motif.

         Note that motif and relative motif description styles are not affected by the action's
        `position_fractional` attribute.
        Args:
            motif (BaseSiteCollectionMotif): The motif being translated.
            precision (int): The precision for formatting numerical values in the description in decimals.
                Default is set in `globals.py`, typically 4.
                Will overwrite motif and relative motif description precision settings.
            motif_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_kwargs = motif_kwargs or {}
        relative_motif_kwargs = relative_motif_kwargs or {}

        motif_kwargs.update({"precision": precision, "is_addition": False})
        relative_motif_kwargs.update({"precision": precision, "is_addition": False})

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = "modify coordinates only, do not change the order of atoms in structure."

        if self.mode_flag == "euler_relative_to_position":
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by euler angles (Z-X-Z intrinsic convention, active rotation)"
                f" in {describe_arraylike(self.euler_angles, precision=precision)} degrees"
                f" around a center position in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_motif":
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by euler angles (Z-X-Z intrinsic convention, active rotation)"
                f" in {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" using the centroid of [{self.relative_to_motif.describe(**relative_motif_kwargs)}]"
                f" as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "euler_relative_to_self":
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by euler angles (Z-X-Z intrinsic convention, active rotation)"
                f" in {describe_arraylike(self.euler_angles, precision=precision)} degrees,"
                f" using its own centroid as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_position":
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by {self.rotation_axis_angle:.{precision}f} degrees around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" and a rotation center position in {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_regular_motif":
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by {self.rotation_axis_angle:.{precision}f} degrees around a rotation axis"
                f" defined by the cartesian vector"
                f" {describe_arraylike(self.rotation_axis_vector, precision=precision)}"
                f" and using the centroid of"
                f" [{self.relative_to_motif.describe(**relative_motif_kwargs)}]"
                f" as the rotation center."
                + " " + common_instruction
            )
        elif self.mode_flag == "axis_relative_to_pair_motif":
            relative_motif_kwargs.update({"style": "index"})
            return (
                f"rotate [{motif.describe(**motif_kwargs)}] in the structure"
                f" by {self.rotation_axis_angle:.{precision}f} degrees around a rotation axis"
                f" defined by the direction of the pair formed by"
                f" [{self.relative_to_motif.describe(**relative_motif_kwargs)}],"
                f" pointing from the origin atom"
                f" (index {self.relative_to_motif.indices[self.relative_pair_origin_index]})"
                f" to the other atom,"
                f" and using the centroid of the aforementioned pair"
                f" as the rotation center."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
