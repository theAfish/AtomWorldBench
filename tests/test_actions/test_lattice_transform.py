"""Comprehensive test suite for LatticeTransformAction."""

import pytest
import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation as R

from AtomWorldBench.atom_world.actions.structure_actions.base import BaseStructureAction
from AtomWorldBench.atom_world.actions.structure_actions.lattice_transform import LatticeTransformAction
from AtomWorldBench.common.registry import get_registered


def test_registry():
    """Test that LatticeTransformAction is registered correctly."""
    action_class = get_registered(BaseStructureAction)["lattice-transform"]
    assert action_class is LatticeTransformAction


@pytest.fixture
def orig_atoms_cell_rotated(orig_atoms):
    """Fixture that returns orig_atoms with a rotated cell."""
    rotation_matrix = R.from_euler('zxz', (45, 30, -15), degrees=True).as_matrix()
    new_cell = rotation_matrix @ orig_atoms.cell.complete()
    new_atoms = orig_atoms.copy()
    new_atoms.set_cell(new_cell.T, scale_atoms=True)
    return new_atoms


def test_lattice_transform_by_matrix(orig_atoms_cell_rotated):
    """Test applying a lattice transformation using a transformation matrix."""
    transformation_matrix = np.array([[1.2, -0.1, 0.2],
                                       [0.0, 0.9, 0.0],
                                       [0.05, 0.1, 0.8]])
    action = LatticeTransformAction(
        operated_atoms=orig_atoms_cell_rotated,
        transformation_matrix=transformation_matrix,
    )
    assert action.mode_flag == "by_matrix"
    new_atoms = action.execute()
    expected_cell = transformation_matrix @ orig_atoms_cell_rotated.cell.complete()
    npt.assert_allclose(new_atoms.cell.complete(), expected_cell, atol=1e-8)
    npt.assert_allclose(
        new_atoms.get_scaled_positions(wrap=False),
        orig_atoms_cell_rotated.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
    assert new_atoms.get_chemical_symbols() == orig_atoms_cell_rotated.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "deform the current lattice by the matrix" in desc
    assert "suppose the original lattice matrix is L_old" in desc
    assert "(0.050, 0.100, 0.800)" in desc  # Check for formatted matrix in description


def test_lattice_transform_to_target(orig_atoms_cell_rotated):
    """Test applying a lattice transformation to match a target lattice."""
    target_cell = np.array([[5.0, 1.0, 0.0],
                            [0.0, 4.5, 1.0],
                            [1.0, 0.0, 6.0]])
    action = LatticeTransformAction(
        operated_atoms=orig_atoms_cell_rotated,
        set_to_lattice_matrix=target_cell,
    )
    assert action.mode_flag == "to_lattice_matrix"
    new_atoms = action.execute()
    npt.assert_allclose(new_atoms.cell.complete(), target_cell, atol=1e-8)
    npt.assert_allclose(
        new_atoms.get_scaled_positions(wrap=False),
        orig_atoms_cell_rotated.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
    assert new_atoms.get_chemical_symbols() == orig_atoms_cell_rotated.get_chemical_symbols()

    desc = action.describe(precision=2)
    assert "set the lattice directly to the matrix" in desc
    assert "the matrix rows correspond to the new lattice vectors" in desc
    assert "(1.00, 0.00, 6.00)" in desc  # Check for formatted matrix in description


def test_lattice_transform_by_scale_factor(orig_atoms_cell_rotated):
    """Test applying a lattice transformation using a scale factor."""
    # Number scale factor.
    scale_factor = 1.1
    action = LatticeTransformAction(
        operated_atoms=orig_atoms_cell_rotated,
        size_scale_factor=scale_factor,
    )
    assert action.mode_flag == "by_size_scale_factor"
    new_atoms = action.execute()
    expected_cell = orig_atoms_cell_rotated.cell.complete() * scale_factor
    npt.assert_allclose(new_atoms.cell.complete(), expected_cell, atol=1e-8)
    npt.assert_allclose(
        new_atoms.get_scaled_positions(wrap=False),
        orig_atoms_cell_rotated.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
    assert new_atoms.get_chemical_symbols() == orig_atoms_cell_rotated.get_chemical_symbols()

    desc = action.describe(precision=2)
    assert f"uniformly scale the current lattice vectors by a factor of 1.10" in desc
    assert "do not rotate or shear the lattice" in desc

    # Vector scale factor.
    scale_factors = [1.2, 0.9, 1.1]
    action = LatticeTransformAction(
        operated_atoms=orig_atoms_cell_rotated,
        size_scale_factor=scale_factors,
    )
    new_atoms = action.execute()
    expected_cell = np.diag(scale_factors) @ orig_atoms_cell_rotated.cell.complete()
    npt.assert_allclose(new_atoms.cell.complete(), expected_cell, atol=1e-8)
    npt.assert_allclose(
        new_atoms.get_scaled_positions(wrap=False),
        orig_atoms_cell_rotated.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
    assert new_atoms.get_chemical_symbols() == orig_atoms_cell_rotated.get_chemical_symbols()
    desc = action.describe(precision=2)
    assert "scale the current lattice by factors of (1.20, 0.90, 1.10)" in desc
    assert "along the a, b, and c" in desc
    assert "do not rotate or shear the lattice" in desc


def test_lattice_transform_to_lattice_parameters(orig_atoms_cell_rotated):
    """Test applying a lattice transformation to match target lattice parameters."""
    target_lengths = [5.0, 4.5, 6.0]
    target_angles = [90.0, 100.0, 110.0]
    action = LatticeTransformAction(
        operated_atoms=orig_atoms_cell_rotated,
        set_to_lattice_parameters=target_lengths + target_angles,
    )
    assert action.mode_flag == "to_lattice_parameters"
    new_atoms = action.execute()
    new_cell = new_atoms.cell.complete()
    lengths = np.linalg.norm(new_cell, axis=1)
    a, b, c = new_cell
    alpha = np.degrees(np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))))
    beta = np.degrees(np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))))
    gamma = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    angles = [alpha, beta, gamma]
    npt.assert_allclose(lengths, target_lengths, atol=1e-8)
    npt.assert_allclose(angles, target_angles, atol=1e-8)
    # Check whether a direction aligns with the original a direction.
    original_a_dir = orig_atoms_cell_rotated.cell[0] / np.linalg.norm(orig_atoms_cell_rotated.cell[0])
    new_a_dir = new_atoms.cell[0] / np.linalg.norm(new_atoms.cell[0])
    cos_angle = np.dot(original_a_dir, new_a_dir)
    assert np.isclose(cos_angle, 1.0, atol=1e-6)
    # Check whether original ab plane normal vector is still normal to the new ab plane.
    original_ab_normal = np.cross(orig_atoms_cell_rotated.cell[0], orig_atoms_cell_rotated.cell[1])
    new_ab_normal = np.cross(new_atoms.cell[0], new_atoms.cell[1])
    original_ab_normal /= np.linalg.norm(original_ab_normal)
    new_ab_normal /= np.linalg.norm(new_ab_normal)
    cos_normal_angle = np.dot(original_ab_normal, new_ab_normal)
    assert np.isclose(cos_normal_angle, 1.0, atol=1e-6)

    npt.assert_allclose(
        new_atoms.get_scaled_positions(wrap=False),
        orig_atoms_cell_rotated.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
    assert new_atoms.get_chemical_symbols() == orig_atoms_cell_rotated.get_chemical_symbols()

    desc = action.describe(precision=2)
    assert "reset the lattice matrix such that its lattice vector lengths and angles" in desc
    assert "should be aligned with" in desc
    assert "the normal vector of the ab plane" in desc


def test_lattice_transform_invalid_lattice_parameters(orig_atoms_cell_rotated):
    """Test that invalid lattice parameters raise an error."""
    invalid_params = [5.0, 4.5, 6.0, 90.0, 100.0]  # Only 5 parameters instead of 6
    with pytest.raises(ValueError, match="must be a list or tuple of length 6."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            set_to_lattice_parameters=invalid_params,
        )
    # Negative value provided.
    invalid_params = [5.0, -4.5, 6.0, 90.0, 100.0, 110.0]
    with pytest.raises(ValueError, match="Lattice lengths must be positive."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            set_to_lattice_parameters=invalid_params,
        )
    # Angle out of range.
    invalid_params = [5.0, 4.5, 6.0, 90.0, 100.0, 200.0]
    with pytest.raises(ValueError, match="Lattice angles must be in the range \(0, 180\)."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            set_to_lattice_parameters=invalid_params,
        )

def test_lattice_transform_invalid_scale_factor(orig_atoms_cell_rotated):
    """Test that invalid scale factors raise an error."""
    # Negative scale factor.
    with pytest.raises(ValueError, match="size_scale_factor must be positive."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            size_scale_factor=-1.0,
        )
    # Vector scale factor with incorrect length.
    with pytest.raises(ValueError, match="If size_scale_factor is a sequence, it must have length 3."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            size_scale_factor=[1.0, 0.9],
        )
    # One of the vector scale factors is non-positive.
    with pytest.raises(ValueError, match="All elements of size_scale_factor must be positive."):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            size_scale_factor=[1.0, 0.0, 1.1],
        )
    # Non-numeric scale factor.
    with pytest.raises(
            ValueError,
            match="size_scale_factor must be a float or a sequence of three floats."
    ):
        LatticeTransformAction(
            operated_atoms=orig_atoms_cell_rotated,
            size_scale_factor="large",
        )
