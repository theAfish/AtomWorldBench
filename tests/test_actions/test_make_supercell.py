"""Comprehensive test suite for MakeSupercellAction."""

import pytest

from AtomWorldBench.atom_world.actions.structure_actions.base import BaseStructureAction
from AtomWorldBench.atom_world.actions.structure_actions.make_supercell import MakeSupercellAction
from AtomWorldBench.common.registry import get_registered


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseStructureAction)["make-supercell"]
    assert action_class is MakeSupercellAction


def test_make_supercell(orig_atoms):
    """Test making a supercell."""
    supercell_matrix = [2, 2, 2]
    action = MakeSupercellAction(
        operated_atoms=orig_atoms,
        supercell_matrix=supercell_matrix,
    )
    new_atoms = action.execute()
    expected_num_atoms = len(orig_atoms) * 8  # 2x2x2 = 8
    assert len(new_atoms) == expected_num_atoms
    orig_symbols = orig_atoms.get_chemical_symbols()
    expected_new_symbols = orig_symbols * 8
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == expected_new_symbols

    desc = action.describe()
    assert f"create a supercell by expanding the original structure using the diagonal" in desc
    assert "cell-major convention" in desc

    # Test with full 3x3 matrix
    supercell_matrix = [[2, 1, -1], [0, 3, 2], [0, 0, 1]]  # Determinant = 2*3*1 = 6
    action = MakeSupercellAction(
        operated_atoms=orig_atoms,
        supercell_matrix=supercell_matrix,
    )
    new_atoms = action.execute()
    expected_num_atoms = len(orig_atoms) * 6  # Determinant = 6
    assert len(new_atoms) == expected_num_atoms
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == orig_symbols * 6
    orig_symbols = orig_atoms.get_chemical_symbols()
    expected_new_symbols = orig_symbols * 6
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == expected_new_symbols


def test_invalid_supercell_matrix(orig_atoms):
    """Test invalid supercell matrix raises error."""
    # Single integer is invalid.
    supercell_matrix = 2
    with pytest.raises(ValueError, match="supercell_matrix must be a 3x3 matrix or a length-3 vector."):
        MakeSupercellAction(
            operated_atoms=orig_atoms,
            supercell_matrix=supercell_matrix,
        )
    # 2*2 matrix is invalid
    supercell_matrix = [[2, 0], [0, 2]]
    with pytest.raises(ValueError, match="supercell_matrix must be a 3x3 matrix or a length-3 vector."):
        MakeSupercellAction(
            operated_atoms=orig_atoms,
            supercell_matrix=supercell_matrix,
        )
    # Non-integer values are invalid
    supercell_matrix = [2.5, 2, 2]
    with pytest.raises(ValueError, match="All elements of supercell_matrix must be integers."):
        MakeSupercellAction(
            operated_atoms=orig_atoms,
            supercell_matrix=supercell_matrix,
        )
    # zero determinant is invalid
    supercell_matrix = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="supercell_matrix must have a non-zero determinant."):
        MakeSupercellAction(
            operated_atoms=orig_atoms,
            supercell_matrix=supercell_matrix,
        )

    supercell_matrix = [2, 0, 2]
    with pytest.raises(ValueError, match="supercell_matrix must have a non-zero determinant."):
        MakeSupercellAction(
            operated_atoms=orig_atoms,
            supercell_matrix=supercell_matrix,
        )


def test_get_random_one(orig_atoms):
    """Test getting a random MakeSupercellAction."""
    all_appeared_scale_modes = set()
    for _ in range(100):
        action = MakeSupercellAction.get_random_one(operated_atoms=orig_atoms)
        assert action.mode_flag == "default"
        assert action.supercell_matrix is not None
        if action.supercell_matrix.shape == (3,):
            scale_mode = "diagonal"
            assert action.supercell_matrix.dtype == int
        else:
            scale_mode = "full"
            # Assert upper-triangular.
            assert action.supercell_matrix[1, 0] == 0
            assert action.supercell_matrix[2, 0] == 0
            assert action.supercell_matrix[2, 1] == 0
            # Assert dtype is int.
            assert action.supercell_matrix.dtype == int
        all_appeared_scale_modes.add(scale_mode)
    assert "diagonal" in all_appeared_scale_modes
    assert "full" in all_appeared_scale_modes
