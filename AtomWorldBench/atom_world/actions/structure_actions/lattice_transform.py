from typing import Optional, Sequence

from ase import Atoms
from ase.cell import Cell
from numpy.typing import ArrayLike
import numpy as np

from .base import BaseStructureAction
from ....common.registry import register
from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....utils.coord_utils import check_lattice_matrix_shape
from ....utils.description_utils import describe_arraylike


def _check_lattice_parameters(parameters: Optional[Sequence] = None):
    if parameters is not None:
        if not (isinstance(parameters, (list, tuple)) and len(parameters) == 6):
            raise ValueError(
                "set_to_lattice_parameters must be a list or tuple of length 6."
            )
        parameters = np.asarray(parameters, dtype=float)
        if np.any(parameters[:3] <= 0):
            raise ValueError("Lattice lengths must be positive.")
        if np.any(parameters[3:] <= 0) or np.any(parameters[3:] >= 180):
            raise ValueError("Lattice angles must be in the range (0, 180).")
    return parameters


def _check_size_scale_factor(factor: Optional[float | Sequence] = None):
    if factor is not None:
        if isinstance(factor, (list, tuple)):
            if len(factor) != 3:
                raise ValueError(
                    "If size_scale_factor is a sequence, it must have length 3."
                )
            factor = np.asarray(factor, dtype=float)
            if np.any(factor <= 0):
                raise ValueError(
                    "All elements of size_scale_factor must be positive."
                )
        elif isinstance(factor, (int, float)):
            if factor <= 0:
                raise ValueError("size_scale_factor must be positive.")
        else:
            raise ValueError(
                "size_scale_factor must be a float or a sequence of three floats."
            )
    return factor


@register(BaseStructureAction, ["lattice-transform"])
class LatticeTransformAction(BaseStructureAction):
    """An action that applies a lattice transformation to a crystal structure.

    This action modifies the lattice parameters of the input structure according to
    a specified transformation matrix. The transformation can include scaling,
    rotation, or shear operations, any transformation that changes the lattice matrix.
    """
    kwargs_formating_functions = {
        "transformation_matrix": lambda x: check_lattice_matrix_shape(
            x, "transformation_matrix", allow_none=True
        ),
        "set_to_lattice_matrix": lambda x: check_lattice_matrix_shape(
            x, "set_to_lattice_matrix", allow_none=True
        ),
        "set_to_lattice_parameters": _check_lattice_parameters,
        "size_scale_factor": _check_size_scale_factor,
    }

    mode_definitions = {
        "_excluded": ["operated_atoms"],
        "by_matrix": {"transformation_matrix": None},
        "by_size_scale_factor": {"size_scale_factor": None},
        "to_lattice_matrix": {"set_to_lattice_matrix": None},
        "to_lattice_parameters": {"set_to_lattice_parameters": None},
    }
    def __init__(
            self,
            operated_atoms: Atoms,
            transformation_matrix: Optional[ArrayLike] = None,
            size_scale_factor: Optional[float | Sequence] = None,
            set_to_lattice_matrix: Optional[ArrayLike] = None,
            set_to_lattice_parameters: Optional[Sequence] = None,
    ):
        """Initialize the LatticeTransformAction.

        `operated_atoms` is always required, other arguments are mutually exclusive.
        Currently, supports four modes of lattice transformation:
            1, `by_matrix`: Apply a lattice transformation defined by a 3x3 matrix.
            2, `by_size_scale_factor`: Uniformly scale the lattice by a given factor.
            3, `to_lattice_matrix`: Set the lattice directly to a specified 3x3 matrix.
            4, `to_lattice_parameters`: Set the lattice directly to specified parameters
               (a, b, c, alpha, beta, gamma).

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on. Required.
            transformation_matrix (Optional[ArrayLike]):
                A 3x3 matrix defining the lattice transformation to apply.
                The new cell matrix will be obtained by transformation_matrix
                 @ old_cell.
                If provided, this matrix will be used to transform the lattice.
                Default is None.
            size_scale_factor (Optional[float | Sequence]):
                A uniform scaling factor to apply to the lattice vectors.
                If provided, the lattice will be scaled by this factor.
                If a sequence of three factors is provided, they will be applied
                to the a, b, and c lattice vectors respectively.
                Default is None.
            set_to_lattice_matrix (Optional[ArrayLike]):
                A 3x3 matrix defining the target lattice to set directly.
                If provided, this matrix will replace the current lattice.
                Default is None.
            set_to_lattice_parameters (Optional[Sequence]):
                A sequence of six values defining the target lattice parameters
                (a, b, c, alpha, beta, gamma) lengths and angles to set directly.
                If provided, these parameters will replace the current lattice parameters.
                Default is None.
        """
        self.operated_atoms = None
        self.transformation_matrix = None
        self.size_scale_factor = None
        self.set_to_lattice_matrix = None
        self.set_to_lattice_parameters = None
        super().__init__(
            operated_atoms=operated_atoms,
            transformation_matrix=transformation_matrix,
            size_scale_factor=size_scale_factor,
            set_to_lattice_matrix=set_to_lattice_matrix,
            set_to_lattice_parameters=set_to_lattice_parameters,
        )

    def __post_init__(self):
        # No specific compatibility checks needed for this action.
        pass

    @property
    def cell(self) -> Cell:
        """Return the cell of the new atoms."""
        if self.mode_flag == "by_matrix":
            return Cell(self.transformation_matrix @ self.operated_atoms.cell.array)
        elif self.mode_flag == "by_size_scale_factor":
            if isinstance(self.size_scale_factor, np.ndarray):
                return Cell(np.diag(self.size_scale_factor) @ self.operated_atoms.cell.array)
            else:
                return Cell(self.size_scale_factor * self.operated_atoms.cell.array)
        elif self.mode_flag == "to_lattice_matrix":
            return Cell(self.set_to_lattice_matrix)
        elif self.mode_flag == "to_lattice_parameters":
            return Cell.fromcellpar(self.set_to_lattice_parameters)
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")

    def execute(self) -> Atoms:
        """Execute the lattice transformation on the structure.

        Returns:
            Atoms: A new Atoms object with the transformed lattice.
        """
        new_atoms = self.operated_atoms.copy()
        new_atoms.set_cell(self.cell, scale_atoms=True)
        return new_atoms

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Describe the action for LLM prompting.

        Args:
            precision (int): The precision for formatting numerical values in the description
                in decimals. Default is set in `globals.py`, typically 4.
        Returns:
            str: A description of the action.
        """
        if self.mode_flag == "by_matrix":
            matrix_str = describe_arraylike(
                self.transformation_matrix, precision=precision
            )
            desc = (
                f"transform the current by the matrix {matrix_str} such that the new"
                f" lattice is obtained by multiplying this matrix with the current lattice."
            )
        elif self.mode_flag == "by_size_scale_factor":
            if isinstance(self.size_scale_factor, np.ndarray):
                factor_str = describe_arraylike(
                    self.size_scale_factor, precision=precision
                )
                desc = (
                    f"scale the current lattice by factors of {factor_str} along the"
                    f" a, b, and c lattice vectors respectively."
                )
            else:
                desc = (
                    f"uniformly scale the current lattice by a factor of"
                    f" {self.size_scale_factor:.{precision}f}."
                )
        elif self.mode_flag == "to_lattice_matrix":
            matrix_str = describe_arraylike(
                self.set_to_lattice_matrix, precision=precision
            )
            desc = f"set the lattice directly to the matrix {matrix_str}."
        elif self.mode_flag == "to_lattice_parameters":
            params_str = describe_arraylike(
                self.set_to_lattice_parameters, precision=precision
            )
            desc = (
                f"reset the lattice matrix such that its lattice vector lengths and angles"
                f" (a, b, c, alpha, beta, gamma) ="
                f" {params_str}."
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")

        desc += (
            " note that the atomic positions should be scaled accordingly so that"
            " their fractional coordinates are unchanged."
        )
        if "matrix" in self.mode_flag:
            desc += (
                " the rows of lattice matrix represent the a, b, c"
                " lattice vectors, respectively."
            )

        return desc
