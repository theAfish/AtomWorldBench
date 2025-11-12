import pytest
from ase import Atoms
import numpy as np
import numpy.testing as npt

from AtomWorldBench.utils.atoms_utils import merge_atoms

def test_merge_atoms():
    """Test the merge_atoms function."""
    # Create two simple Atoms objects
    atoms1 = Atoms(
        symbols = ["H", "C"],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        cell=[3.0, 3.0, 3.0], pbc=True
    )
    atoms2 = Atoms(
        symbols = ["O"],
        positions=[(0.5, 0.0, 0.5)],
        cell=[3.0, 3.0, 3.0], pbc=True
    )

    # Non-indexed merge (simple concatenation)
    merged1 = merge_atoms([atoms1, atoms2])
    assert len(merged1) == 3
    assert merged1 == (atoms1 + atoms2)
    assert merged1.get_chemical_symbols() == ["H", "C", "O"]

    # Indexed merge (with reordering).
    merged2 = merge_atoms(
        [atoms1, atoms2],
        all_atoms_indices=[[2, 1], [0]]  # Reorder to have O at the end
    )
    assert len(merged2) == 3
    assert merged2.get_chemical_symbols() == ["O", "C", "H"]
    npt.assert_almost_equal(
        merged2.get_positions(wrap=False),
        np.array(
            [(0.5, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]
        )
    )

    # Error case 1: Different cell or pbc
    atoms3 = Atoms(
        symbols=["S"],
        positions=[(1.0, 1.0, 1.0)],
        cell=[4.0, 4.0, 4.0], pbc=True
    )
    with pytest.raises(ValueError):
        _ = merge_atoms([atoms1, atoms3])

    # Error case  2: indices length does not match atoms length.
    with pytest.raises(ValueError):
        _ = merge_atoms(
            [atoms1, atoms2],
            all_atoms_indices=[[0], [0]]  # Incorrect length for atoms1
        )

    # Error case 3: indices do not form a complete permutation.
    with pytest.raises(ValueError):
        _ = merge_atoms(
            [atoms1, atoms2],
            all_atoms_indices=[[0, 1], [1]]  # Duplicate index 1.
        )

    with pytest.raises(ValueError):
        _ = merge_atoms(
            [atoms1, atoms2],
            all_atoms_indices=[[3, 1], [0]]  # Discontinuous index 3.
        )
