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

LATTICE_VOLUME_TOL = 1e-6


def _check_lattice_parameters(parameters: Optional[Sequence] = None):
    if parameters is not None:
        if not (isinstance(parameters, (list, tuple, np.ndarray)) and len(parameters) == 6):
            raise ValueError(
                "set_to_lattice_parameters must be a list or tuple of length 6."
            )
        parameters = np.asarray(parameters, dtype=float)
        if np.any(parameters[:3] <= 0):
            raise ValueError("Lattice lengths must be positive.")
        if np.any(parameters[3:] <= 0) or np.any(parameters[3:] >= 180):
            raise ValueError("Lattice angles must be in the range (0, 180).")
        cos_alpha, cos_beta, cos_gamma = np.cos(np.deg2rad(parameters[3:]))
        volume_term = 1 + 2 * cos_alpha * cos_beta * cos_gamma - (
            cos_alpha ** 2 + cos_beta ** 2 + cos_gamma ** 2
        )
        if volume_term <= LATTICE_VOLUME_TOL:
            raise ValueError(
                "Lattice lengths/angles yield a non-physical cell (zero or imaginary volume)."
            )
    return parameters


def _check_size_scale_factor(factor: Optional[float | Sequence] = None):
    if factor is not None:
        if isinstance(factor, (list, tuple, np.ndarray)):
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
    kwargs_formatting_functions = {
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

    mode_probabilities = {
        "by_matrix": 0.2,
        "by_size_scale_factor": 0.4,  # Prefer scaling transformations more often.
        "to_lattice_matrix": 0.2,
        "to_lattice_parameters": 0.2,
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
            return Cell(self.transformation_matrix @ self.operated_atoms.cell.complete())
        elif self.mode_flag == "by_size_scale_factor":
            if isinstance(self.size_scale_factor, np.ndarray):
                return Cell(np.diag(self.size_scale_factor) @ self.operated_atoms.cell.array)
            else:
                return Cell(self.size_scale_factor * self.operated_atoms.cell.array)
        elif self.mode_flag == "to_lattice_matrix":
            return Cell(self.set_to_lattice_matrix)
        elif self.mode_flag == "to_lattice_parameters":
            orig_vec_a = self.operated_atoms.cell.complete()[0]
            orig_vec_b = self.operated_atoms.cell.complete()[1]
            ab_norm_vec = np.cross(orig_vec_a, orig_vec_b)
            ab_norm_vec = ab_norm_vec / np.linalg.norm(ab_norm_vec)
            orig_unit_a = orig_vec_a / np.linalg.norm(orig_vec_a)
            # Fix a and ab plane normal direction to avoid arbitrary rotation.
            try:
                return Cell.fromcellpar(
                    self.set_to_lattice_parameters,
                    ab_normal=ab_norm_vec,
                    a_direction=orig_unit_a
                )
            except Exception as e:
                print("Error in setting lattice from parameters:", e)
                print("set_to_lattice_parameters: ", self.set_to_lattice_parameters)
                print("ab_normal: ", ab_norm_vec)
                print("a_direction: ", orig_unit_a)
                raise e
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
                f"deform the current lattice by the matrix {matrix_str}"
                " suppose the original lattice matrix is L_old, then the new lattice"
                " matrix L_new = transformation_matrix @ L_old."
            )
        elif self.mode_flag == "by_size_scale_factor":
            if isinstance(self.size_scale_factor, np.ndarray):
                factor_str = describe_arraylike(
                    self.size_scale_factor, precision=precision
                )
                desc = (
                    f"scale the current lattice by factors of {factor_str} along the"
                    " a, b, and c lattice vectors."
                )
            else:
                desc = (
                    f"uniformly scale the current lattice vectors by a factor of"
                    f" {self.size_scale_factor:.{precision}f}."
                )
            desc += " do not rotate or shear the lattice."
        elif self.mode_flag == "to_lattice_matrix":
            matrix_str = describe_arraylike(
                self.set_to_lattice_matrix, precision=precision
            )
            desc = (
                f"set the lattice directly to the matrix {matrix_str}."
                " the matrix rows correspond to the new lattice vectors a, b, and c,"
                " respectively."
            )
        elif self.mode_flag == "to_lattice_parameters":
            params_str = describe_arraylike(
                self.set_to_lattice_parameters, precision=precision
            )
            desc = (
                f"reset the lattice matrix such that its lattice vector lengths and angles"
                f" (a, b, c, alpha, beta, gamma) ="
                f" {params_str}."
                " the new lattice vector a should be aligned with the original"
                " lattice vector a, and the normal vector of the ab plane should be"
                " aligned with that of the original ab plane."
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")

        desc += (
            " note that the atomic positions should be scaled accordingly so that"
            " their fractional coordinates are unchanged."
        )

        return desc

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
            n_attempts: int = 10,
    ) -> "LatticeTransformAction":
        """Generate a random LatticeTransformAction instance.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on.
            seed (int, optional):
                Seed for random number generator for reproducibility. Optional.
                Will also influence the choice of mode.

        Returns:
            LatticeTransformAction:
                A randomly generated LatticeTransformAction instance.
        """
        rng = np.random.default_rng(seed)

        chosen_mode = cls.get_random_mode(seed)
        if chosen_mode == "by_matrix":
            # Apply a small random deformation to the identity matrix.
            random_matrix = np.eye(3) + rng.normal(size=(3, 3)) * 0.1
            return cls(
                operated_atoms=operated_atoms,
                transformation_matrix=random_matrix,
            )
        elif chosen_mode == "by_size_scale_factor":
            if rng.random() < 0.5:
                # Uniform scaling
                scale_factor = float(rng.uniform(0.5, 1.5))
            else:                # Anisotropic scaling
                scale_factor = rng.uniform(0.5, 1.5, size=3).tolist()
            return cls(
                operated_atoms=operated_atoms,
                size_scale_factor=scale_factor,
            )
        elif chosen_mode == "to_lattice_matrix":
            deformation_matrix = np.eye(3) + rng.normal(size=(3, 3)) * 0.1
            random_matrix = deformation_matrix @ operated_atoms.cell.complete()
            return cls(
                operated_atoms=operated_atoms,
                set_to_lattice_matrix=random_matrix,
            )
        elif chosen_mode == "to_lattice_parameters":
            try:
                for _ in range(n_attempts):
                    a, b, c = rng.uniform(2.0, 10.0, size=3).tolist()
                    alpha, beta, gamma = rng.uniform(30.0, 120.0, size=3).tolist()
                    params = [a, b, c, alpha, beta, gamma]
                    try:
                        _check_lattice_parameters(params)
                    except ValueError:
                        continue
                    return cls(
                        operated_atoms=operated_atoms,
                        set_to_lattice_parameters=params,
                    )
            except Exception as e:
                print("Failed to generate valid lattice parameters:", e)
                raise e
        else:
            raise ValueError(f"Invalid mode: {chosen_mode}")
