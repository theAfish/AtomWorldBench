"""Rotate motif action."""
from typing import Optional

from numpy.typing import ArrayLike

from .base import BaseAction
from ..motifs.base import BaseMotif

from ...utils.coord_utils import check_coordinates_shape


class RotateMotifAction(BaseAction):
    """Action to rotate a motif in the structure.

    Notice: this operation only allows relative style.
    """
    allowed_relative_styles = [
        "centroid_distance",  # Rotate around the centroid of a relative motif.
        "rotation_axis",  # Rotate around a specific axis.
    ]

    def __init__(
            self,
            euler_angles: Optional[ArrayLike] = None,  # In degrees. Always ZXZ, active convention.
            rotation_axis_vector: Optional[ArrayLike] = None,
            rotation_axis_angle: Optional[float] = None,  # In degrees.
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: Optional[bool] = True,
            relative_style: str = None,
            relative_to_motif: Optional[BaseMotif] = None,
            relative_to_pair_motif_origin_index: int = 0,
            self_relative: bool = False,
    ):
        """Initialize the RotateMotifAction with a relative motif and style.

        Allows Euler angles rotation in ZXZ intrinsic convention (active rotation), with
        rotation center specified by either a position or the centroid of a relative motif.
        Also allows counter-clockwise rotation around a specified axis vector and angle, with
        rotation center specified by either a position or the centroid of a relative motif,
        and rotation vector specified by either a vector or the line of a pair motif.
        Args:
            euler_angles (Optional[ArrayLike]):
                Euler angles for rotation in degrees (ZXZ intrinsic convention, active rotation).
                Unit in degrees. Must be a 1D array-like of length 3.
            rotation_axis_vector (Optional[ArrayLike]):
                Vector defining the rotation axis. Will overwrite `euler_angles` and relative parameters
                when provided.
            rotation_axis_angle (Optional[float]):
                Angle of counter-clockwise rotation around the rotation axis in degrees.
                If rotating around a vector or a pair motif, this must be provided.
            relative_to_position (Optional[ArrayLike]):
                Center position to rotate around.
            position_fractional (Optional[bool]):
                Whether the position is given in fractional coordinates. Defaults to True.
            relative_style (str):
                Style of the relative action, must be one of `allowed_relative_styles` of this
                action and the relative motif.
            relative_to_motif (Optional[BaseMotif]):
                Motif to rotate relative to. If rotating in Euler angles, this motif's centroid
                will be used as the rotation center. If rotating around a vector, this motif must
                be a pair motif, with its origin index used as the rotation center, and the rotation
                vector calculated from the line of the pair motif, pointing from the origin atom
                to the other atom.
            relative_to_pair_motif_origin_index(int):
                When the relative motif is a pair motif, this index specifies which atom
                will be used as the rotation center and the origin of the rotation vector.
                Defaults to 0, which means the first atom in the pair motif object.
            self_relative (bool):
                If True, rotate the motif itself, using its own centroid as the rotation center.
                Can only be used with the Euler angle rotation.
        """
        super().__init__(relative_to_motif=relative_to_motif, relative_style=relative_style)
        if euler_angles is not None and rotation_axis_angle is not None:
            raise ValueError("Can only specify one of Euler angles and rotation axis angle.")
        if euler_angles is None and rotation_axis_vector is None:
            raise ValueError("Must specify either Euler angles or rotation axis vector.")
        if relative_to_position is not None and relative_to_motif is not None:
            raise ValueError(
                "Cannot specify both relative_to_position and relative_to_motif."
            )
        if relative_to_position is None and relative_to_motif is None and not self_relative:
            raise ValueError(
                "Must specify either relative_to_position or relative_to_motif, "
                "or set self_relative to True."
            )
        if euler_angles is not None:
            self.rotation_axis_vector = None
            self.rotation_axis_angle = None
            self.euler_angles = check_coordinates_shape(
                euler_angles, "euler_angles", expected_1d=True
            )






