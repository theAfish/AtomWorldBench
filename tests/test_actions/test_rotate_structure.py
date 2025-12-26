"""Comprehensive test suite for RotateStructureAction."""

import pytest

import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation

from AtomWorldBench.atom_world.actions.structure_actions.base import BaseStructureAction
from AtomWorldBench.atom_world.actions.structure_actions.rotate import RotateStructureAction
from AtomWorldBench.common.registry import get_registered


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseStructureAction)["rotate-structure"]
    assert action_class is RotateStructureAction


def test_rotate_structure_euler(orig_atoms):
    """Test rotating structure using Euler angles."""
    euler_angles = [45, -45, 20]  # Rotate 90 degrees around z-axis
    action = RotateStructureAction(
        operated_atoms=orig_atoms,
        euler_angles=euler_angles,
    )
    assert action.mode_flag == "euler"
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms)
    rotation = Rotation.from_euler('ZXZ', euler_angles, degrees=True)
    orig_positions = orig_atoms.get_positions(scale=False)
    expected_positions = rotation.apply(orig_positions)
    new_positions = new_atoms.get_positions(scale=False)
    npt.assert_allclose(new_positions, expected_positions, atol=1e-6)
    orig_cell_array = orig_atoms.cell.complete().array
    expected_cell = rotation.apply(orig_cell_array)
    new_cell_array = new_atoms.cell.complete().array
    npt.assert_allclose(new_cell_array, expected_cell, atol=1e-6)
    # Order of atoms not changed.
    orig_symbols = orig_atoms.get_chemical_symbols()
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == orig_symbols

    desc = action.describe(precision=2)
    assert "rotate the entire structure (position and cell vectors) by Euler angles" in desc
    assert "(45.00, -45.00, 20.00) degrees around the origin" in desc

def test_rotate_structure_axis(orig_atoms):
    """Test rotating structure using axis-angle representation."""
    rotation_axis_vector = [0, 1, 1]  # z-axis
    rotation_axis_angle = 90  # degrees
    action = RotateStructureAction(
        operated_atoms=orig_atoms,
        rotation_axis_vector=rotation_axis_vector,
        rotation_axis_angle=rotation_axis_angle,
    )
    assert action.mode_flag == "axis"
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms)
    orig_positions = orig_atoms.get_positions(scale=False)
    norm = np.linalg.norm(rotation_axis_vector)
    rotation = Rotation.from_rotvec(
        np.radians(rotation_axis_angle) * np.array(rotation_axis_vector) / norm
    )
    expected_positions = rotation.apply(orig_positions)
    new_positions = new_atoms.get_positions(scale=False)
    npt.assert_allclose(new_positions, expected_positions, atol=1e-6)
    orig_cell_array = orig_atoms.cell.complete().array
    expected_cell = rotation.apply(orig_cell_array)
    new_cell_array = new_atoms.cell.complete().array
    npt.assert_allclose(new_cell_array, expected_cell, atol=1e-6)
    # Order of atoms not changed.
    orig_symbols = orig_atoms.get_chemical_symbols()
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == orig_symbols

    desc = action.describe(precision=1)
    assert (
            "rotate the entire structure (position and cell vectors)"
            " right-hand counter-clockwise by 90.0 degrees" in desc
    )
    assert "around the axis defined by the vector (0.0, 0.7, 0.7), centered at the origin" in desc


def test_invalid_rotation_parameters(orig_atoms):
    """Test that invalid rotation parameters raise errors."""
    # Invalid euler_angles length.
    with pytest.raises(ValueError, match="euler_angles expected 1D array of shape"):
        RotateStructureAction(
            operated_atoms=orig_atoms,
            euler_angles=[0, 0],
        )
    # Invalid rotation_axis_vector length.
    with pytest.raises(ValueError, match="rotation_axis_vector expected 1D array of shape"):
        RotateStructureAction(
            operated_atoms=orig_atoms,
            rotation_axis_vector=[0, 1],
            rotation_axis_angle=90,
        )
    # Zero rotation_axis_vector.
    with pytest.raises(ValueError):
        RotateStructureAction(
            operated_atoms=orig_atoms,
            rotation_axis_vector=[0, 0, 0],
            rotation_axis_angle=90,
        )


def test_get_random_one(orig_atoms):
    """Test generating random RotateStructureAction instances."""
    all_appeared_modes = set()
    for _ in range(100):
        action = RotateStructureAction.get_random_one(operated_atoms=orig_atoms)
        if action.mode_flag == "euler":
            assert action.euler_angles is not None
            assert np.all(action.euler_angles >= -180 - 1e-6) and np.all(action.euler_angles <= 180 + 1e-6)
            assert action.rotation_axis_vector is None
            assert action.rotation_axis_angle is None
        elif action.mode_flag == "axis":
            assert action.euler_angles is None
            assert action.rotation_axis_vector is not None
            assert np.isclose(np.linalg.norm(action.rotation_axis_vector), 1.0)
            assert action.rotation_axis_angle is not None
            assert -180 - 1e-6 <= action.rotation_axis_angle <= 180 + 1e-6
        else:
            raise AssertionError(f"Unexpected mode_flag: {action.mode_flag}")
        all_appeared_modes.add(action.mode_flag)
    assert all_appeared_modes == {"euler", "axis"}