"""Rotate the entire structure by given angles around the x, y, and z axes and center."""

from typing import Optional

from ase import Atoms
from ase.cell import Cell
from numpy.typing import ArrayLike
import numpy as np
from scipy.spatial.transform import Rotation

from .base import BaseStructureAction
from ..common.registry import register
from ..common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ..utils.coord_utils import check_coordinates_shape
from ..utils.description_utils import describe_arraylike


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


@register(BaseStructureAction, ["rotate-structure"])
class RotateStructureAction(BaseStructureAction):
    """Rotate the entire structure along with cell matrix."""
    kwargs_formatting_functions = {
        "euler_angles": lambda x: check_coordinates_shape(
            x, "euler_angles", expected_1d=True, allow_none=True
        ),
        "rotation_axis_vector": _check_rotation_axis_vector,
    }

    mode_definitions = {
        "_excluded": ["operated_atoms"],
        "euler": {"euler_angles": None,},
        "axis": {"rotation_axis_vector": None, "rotation_axis_angle": None,},
    }

    def __init__(
            self,
            operated_atoms: Atoms,
            euler_angles: Optional[ArrayLike] = None,
            rotation_axis_vector: Optional[ArrayLike] = None,
            rotation_axis_angle: Optional[float] = None,
    ):
        """Initialize the RotateStructureAction.

        operated_atoms is always required. relative_to_position is optional.
        For other parameters, support two modes:
            1. `euler`: Perform a rotation using Euler angles in active and intrinsic ZXZ convention.
                In this mode, provide `euler_angles` as a sequence of three angles (in degrees), no
                other parameters should be provided.
            2. `axis`: Perform a rotation around a specified axis by a specified angle
             counter-clockwise. In this mode, provide `rotation_axis_vector` as a sequence of three
             floats indicating the axis of rotation, and `rotation_axis_angle` as a float indicating
             the angle (in degrees) to rotate around that axis. No other parameters should be provided.
        Args:
            operated_atoms (Atoms): The atomic structure to be rotated.
            euler_angles (Optional[ArrayLike]): A sequence of three angles (in degrees) for rotation
                around the x, y, and z axes, respectively. Required if using `euler` mode.
                Use active intrinsic ZXZ convention.
            rotation_axis_vector (Optional[ArrayLike]): A sequence of three floats indicating the axis of
                rotation. Required if using `axis` mode. Will be normalized if not already.
            rotation_axis_angle (Optional[float]): The angle (in degrees) to rotate around the specified axis.
                Required if using `axis` mode. Rotation is performed counter-clockwise.
        """
        self.euler_angles = None
        self.rotation_axis_vector = None
        self.rotation_axis_angle = None
        self.relative_to_position = None
        super().__init__(
            operated_atoms=operated_atoms,
            euler_angles=euler_angles,
            rotation_axis_vector=rotation_axis_vector,
            rotation_axis_angle=rotation_axis_angle,
        )

    def __post_init__(self):
        pass

    def execute(self) -> Atoms:
        """Execute the rotation action on the structure.

        Returns:
            Atoms: The rotated atomic structure.
        """
        atoms = self.operated_atoms.copy()
        if self.mode_flag == "euler":
            alpha, beta, gamma = self.euler_angles
            rotation = Rotation.from_euler('ZXZ', [alpha, beta, gamma], degrees=True)
            # Rotate positions. Center at origin (0, 0, 0).
            new_positions = rotation.apply(atoms.get_positions(wrap=False))
            # Rotate cell.
            new_cell = Cell(
                rotation.apply(atoms.cell.complete().array.copy())  # Rotate cell vectors.
            )
            atoms.set_cell(new_cell, scale_atoms=False)
            atoms.set_positions(new_positions)
        elif self.mode_flag == "axis":
            atoms.rotate(
                a=self.rotation_axis_angle,
                v=self.rotation_axis_vector,
                center=(0.0, 0.0, 0.0),
                rotate_cell=True,
            )
        else:
            raise NotImplementedError(
                f"Invalid mode_flag '{self.mode_flag}'."
            )
        return atoms

    def describe(
            self,
            precision = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Generate a description of the rotation action performed.

        Args:
            precision (int): The number of decimal places to include in the description.
                Default is set in `globals.py`, typically 4.

        Returns:
            str: A textual description of the rotation action.
        """
        if self.mode_flag == "euler":
            desc = (
                f"rotate the entire structure (position and cell vectors) by Euler angles "
                f" (Z-X-Z intrinsic convention,"
                f" active rotation, right-hand counter-clockwise direction)"
                f" {describe_arraylike(self.euler_angles, precision=precision)} "
                f"degrees around the origin (0.0, 0.0, 0.0)."
            )
        elif self.mode_flag == "axis":
            desc = (
                f"rotate the entire structure (position and cell vectors)"
                f" right-hand counter-clockwise by "
                f"{self.rotation_axis_angle:.{precision}f} degrees "
                f"around the axis defined by the vector "
                f"{describe_arraylike(self.rotation_axis_vector, precision=precision)}, "
                f"centered at the origin (0.0, 0.0, 0.0)."
            )
        else:
            raise NotImplementedError(
                f"Invalid mode_flag '{self.mode_flag}'."
            )
        return desc

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Generate a random RotateStructureAction instance.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on.
            seed (int, optional):
                An optional random seed for reproducibility.

        Returns:
            RotateStructureAction:
                A randomly generated RotateStructureAction instance.
        """
        rng = np.random.default_rng(seed)
        mode = cls.get_random_mode(seed)
        if mode == "euler":
            # Euler mode.
            euler_angles = rng.uniform(-180, 180, size=3)
            return cls(
                operated_atoms=operated_atoms,
                euler_angles=euler_angles,
            )
        elif mode == "axis":
            # Axis mode.
            rotation_axis_vector = rng.normal(size=3)
            norm = np.linalg.norm(rotation_axis_vector)
            if norm < 1e-8:
                rotation_axis_vector = np.array([1.0, 0.0, 0.0])
            else:
                rotation_axis_vector /= norm
            rotation_axis_angle = rng.uniform(-180, 180)
            return cls(
                operated_atoms=operated_atoms,
                rotation_axis_vector=rotation_axis_vector,
                rotation_axis_angle=rotation_axis_angle,
            )
        else:
            raise NotImplementedError(
                f"Invalid randomly selected mode '{mode}'."
            )
