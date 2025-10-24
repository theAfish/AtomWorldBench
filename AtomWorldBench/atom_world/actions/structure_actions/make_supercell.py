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
        if supercell_matrix.shape != (3, 3) or supercell_matrix.shape != (3, ):
            raise ValueError("supercell_matrix must be a 3x3 matrix or a length-3 vector.")
        if not np.all(np.equal(np.mod(supercell_matrix, 1), 0)):
            raise ValueError("All elements of supercell_matrix must be integers.")
        supercell_matrix = supercell_matrix.astype(int)
    return supercell_matrix


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
        description = (
            f"create a supercell by expanding the original structure using the"
            f" supercell matrix {describe_arraylike(self.supercell_matrix, precision=0)}."
            f" use cell-major convention for ordering the generated atoms in supercell"
            f" (i.e., first over all the atoms in cell1 and then move to cell2, etc.)."
        )  # Describe as integer.
        return description
