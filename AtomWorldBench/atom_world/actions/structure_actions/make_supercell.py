from typing import Optional

from ase import Atoms
from ase.build import make_supercell
from numpy.typing import ArrayLike
import numpy as np

from .base import BaseStructureAction
from ....common.registry import register
from ....utils.description_utils import describe_arraylike


def _check_supercell_matrix(
        supercell_matrix: ArrayLike
    ) -> Optional[np.ndarray]:
    if supercell_matrix is not None:
        supercell_matrix = np.array(supercell_matrix)
        if supercell_matrix.shape != (3, 3) and supercell_matrix.shape != (3, ):
            raise ValueError("supercell_matrix must be a 3x3 matrix or a length-3 vector.")
        if not np.all(np.equal(np.mod(supercell_matrix, 1), 0)):
            raise ValueError("All elements of supercell_matrix must be integers.")
        # Supercell matrix determinant must not be zero.
        if supercell_matrix.shape == (3, 3):
            if np.linalg.det(supercell_matrix) == 0:
                raise ValueError("supercell_matrix must have a non-zero determinant.")
        else:
            if np.any(supercell_matrix == 0):
                raise ValueError("supercell_matrix must have a non-zero determinant.")
        supercell_matrix = supercell_matrix.astype(int)
    return supercell_matrix


def _sample_integer_diagonal(
    max_det=8,
    allow_identity=True
):
    """
    Generate (a, d, f) such that:
    - a, d, f > 0
    - a*d*f <= max_det
    - if allow_identity=False, (1,1,1) is excluded
    """
    triples = []

    for a in range(1, max_det + 1):
        for d in range(1, max_det + 1):
            for f in range(1, max_det + 1):
                if a * d * f > max_det:
                    continue
                if not allow_identity and (a, d, f) == (1, 1, 1):
                    continue
                triples.append((a, d, f))

    if not triples:
        raise RuntimeError("No valid diagonal triples found.")

    return triples

_diag_lists = _sample_integer_diagonal(max_det=8, allow_identity=True)


@register(BaseStructureAction, ["make-supercell"])
class MakeSupercellAction(BaseStructureAction):
    """An action that creates a supercell from a crystal structure.

    This action expands the input structure by replicating it along its lattice
    vectors according to the specified size scale factors.
    """
    kwargs_formatting_functions = {
        "supercell_matrix": _check_supercell_matrix,
    }

    mode_definitions = {
        "_excluded": ["operated_atoms"],
        "default": {"supercell_matrix": None},
    }

    def __init__(
            self,
            operated_atoms: Atoms,
            supercell_matrix: ArrayLike,
    ):
        """Initialize the MakeSupercellAction.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on. Required.
            supercell_matrix (ArrayLike, optional):
                A 3x3 matrix or length-3 vector defining the supercell expansion
                along each lattice vector. If a length-3 vector is provided,
                it is interpreted as a diagonal matrix. Required.
        """
        self.operated_atoms = None
        self.supercell_matrix = None
        super().__init__(
            operated_atoms=operated_atoms,
            supercell_matrix=supercell_matrix,
        )

    def __post_init__(self):
        """Check compatibility of the action with operated motif and atoms."""
        pass

    def execute(self) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        supercell_matrix = self.supercell_matrix
        if supercell_matrix.shape == (3, ):
            diag_matrix = np.zeros((3, 3), dtype=int)
            np.fill_diagonal(diag_matrix, supercell_matrix)
            supercell_matrix = diag_matrix
        return make_supercell(self.operated_atoms, supercell_matrix, wrap=False)

    def describe(
            self) -> str:
        """Generate a description of the action performed.

        Returns:
            str:
                A textual description of the supercell creation action.
        """
        diagonal = "diagonal" if self.supercell_matrix.shape == (3, ) else ""
        description = (
            f"create a supercell by expanding the original structure using the {diagonal}"
            f" supercell matrix {describe_arraylike(self.supercell_matrix, precision=0)}."
            f" use cell-major convention for ordering the generated atoms in supercell"
            f" (i.e., first over all the atoms in cell1 and then move to cell2, etc.)."
        )  # Describe as integer.
        return description

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ):
        """Generate a random MakeSupercellAction instance.

        Args:
            operated_atoms (Atoms):
                The Atoms object that this action operates on.
            seed (int, optional):
                An optional random seed for reproducibility.

        Returns:
            MakeSupercellAction:
                A randomly generated MakeSupercellAction instance.
        """
        rng = np.random.default_rng(seed)
        # Randomly choose scale factors between 2 and 4 for each lattice vector.
        if rng.random() < 0.5:
            # Diagonal supercell matrix.
            # without 111
            scale_factors = _diag_lists[
                rng.integers(1, len(_diag_lists))
            ]
            return cls(
                operated_atoms=operated_atoms,
                supercell_matrix=scale_factors,
            )
        else:
            # Upper triangular supercell matrix. This guarantees a non-zero determinant.
            scale_factors = rng.integers(-2, 3, size=(3, 3))
            # Set the diagonal elements to be the choosen scale factors
            diag_factors = _diag_lists[
                rng.integers(0, len(_diag_lists))
            ]
            scale_factors[0, 0] = diag_factors[0]
            scale_factors[1, 1] = diag_factors[1]
            scale_factors[2, 2] = diag_factors[2]
            scale_factors[1, 0] = 0
            scale_factors[2, 0] = 0
            scale_factors[2, 1] = 0
            return cls(
                operated_atoms=operated_atoms,
                supercell_matrix=scale_factors,
            )
