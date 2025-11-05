"""Comprehensive test suite for RotateMotifAction."""

import pytest
import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.rotate import RotateMotifAction
from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif
from AtomWorldBench.atom_world.motifs.regions.box import BoxRegionMotif
from AtomWorldBench.atom_world.motifs.regions.sphere import SphereRegionMotif
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["rotate-motif"]
    assert action_class is RotateMotifAction
    action_class = get_registered(BaseMotifAction)["rotate"]
    assert action_class is RotateMotifAction

### --- Self relative modes ---


@pytest.fixture(params=["cluster", "sphere"])
def allowed_relative_self_motif(request, orig_atoms):
    # Allowed operated motifs in every mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=42,
    )


def test_rotate_motif_euler_relative_self(allowed_relative_self_motif):
    """Test RotateMotifAction with Euler angles in relative self mode."""
    motif = allowed_relative_self_motif
    action = RotateMotifAction(
        operated_motif=motif,
        euler_angles=[20, 30, 40],
        relative_style="self",
    )
    assert action.mode_flag == "euler_relative_to_self"
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    rot = Rotation.from_euler("ZXZ", [20, 30, 40], degrees=True)
    expected_motif_positions = (
        rot.apply(motif.cart_coords - motif_center)
        + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by euler angles" in desc
    assert "counter-clockwise" in desc
    assert "(20.000, 30.000, 40.000) degrees" in desc
    assert "around its own centroid as the rotation center" in desc


def test_rotate_motif_axis_relative_to_self(allowed_relative_self_motif):
    """Test RotateMotifAction with axis-angle in relative self mode."""
    motif = allowed_relative_self_motif
    action = RotateMotifAction(
        operated_motif=motif,
        rotation_axis_vector=[1, 1, 0],
        rotation_axis_angle=45,
        relative_style="self",
    )
    assert action.mode_flag == "axis_relative_to_self"
    npt.assert_allclose(
        action.rotation_axis_vector,
        np.array([1, 1, 0]) / np.sqrt(2)
    )
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    rot = Rotation.from_rotvec(
        np.radians(45) * np.array([1, 1, 0]) / np.linalg.norm([1, 1, 0])
    )
    expected_motif_positions = (
        rot.apply(motif.cart_coords - motif_center)
        + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by 45.000 degrees counter-clockwise" in desc
    assert ("around a rotation axis defined by the cartesian"
            " vector (0.707, 0.707, 0.000)") in desc  # Already normalized.
    assert "around its own centroid as the rotation center" in desc


@pytest.fixture(params=["bond", "box", "site"])
def forbidden_relative_self_motif(request, orig_atoms):
    # Forbidden operated motifs in relative self mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=42,
    )


def test_forbidden_relative_self_motif(
    forbidden_relative_self_motif,
):
    """Test that forbidden motifs in relative self mode raise errors."""
    motif = forbidden_relative_self_motif
    if isinstance(motif, BoxRegionMotif):
        with pytest.raises(ValueError, match="Operated motif must support centroid calculation"):
            RotateMotifAction(
                operated_motif=motif,
                euler_angles=[10, 20, 30],
                relative_style="self",
            )
    elif isinstance(motif, BondMotif):
        with pytest.raises(ValueError, match="Bond motifs are not allowed"):
            RotateMotifAction(
                operated_motif=motif,
                rotation_axis_vector=[0, 0, 1],
                rotation_axis_angle=90,
                relative_style="self",
            )
    elif isinstance(motif, SiteMotif):
        with pytest.raises(ValueError, match="must have at least two sites"):
            RotateMotifAction(
                operated_motif=motif,
                euler_angles=[5, 15, 25],
                relative_style="self",
            )


### --- Relative to position modes ---
@pytest.fixture(params=["cluster", "site"])
def allowed_relative_position_motif(request, orig_atoms):
    # Allowed operated motifs in every mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=123,
    )


def test_rotate_motif_euler_relative_to_position_cartesian(
    allowed_relative_position_motif,
):
    """Test RotateMotifAction with Euler angles in relative to position mode."""
    motif = allowed_relative_position_motif
    rotation_center = np.array([-1.0, 1.0, 2.5])
    action = RotateMotifAction(
        operated_motif=motif,
        euler_angles=[40, 52, -60],
        relative_to_position=rotation_center,
    )
    assert action.mode_flag == "euler_relative_to_position"
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    rot = Rotation.from_euler("ZXZ", [40, 52, -60], degrees=True)
    expected_motif_positions = (
        rot.apply(motif.cart_coords - rotation_center)
        + rotation_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by euler angles" in desc
    assert "active rotation, counter-clockwise" in desc
    assert "(40.000, 52.000, -60.000) degrees" in desc
    assert ("around a center position in cartesian coordinates"
            " (-1.000, 1.000, 2.500)") in desc


def test_rotate_motif_euler_relative_to_position_fractional(
    allowed_relative_position_motif,
    orig_atoms,
):
    """Test RotateMotifAction with Euler angles in relative to position mode."""
    motif = allowed_relative_position_motif
    frac_center = np.array([0.2, 0.5, 0.75])
    cart_center = frac_center @ orig_atoms.cell.complete()
    action = RotateMotifAction(
        operated_motif=motif,
        euler_angles=[-30, 60, 90],
        relative_to_position=frac_center,
        position_fractional=True,
    )
    assert action.mode_flag == "euler_relative_to_position"
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    rot = Rotation.from_euler("ZXZ", [-30, 60, 90], degrees=True)
    expected_motif_positions = (
        rot.apply(motif.cart_coords - cart_center)
        + cart_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by euler angles" in desc
    assert "active rotation, counter-clockwise" in desc
    assert "(-30.000, 60.000, 90.000) degrees" in desc
    assert ("around a center position in fractional coordinates"
            " (0.200, 0.500, 0.750)") in desc


def test_rotate_motif_axis_relative_to_position_cartesian(
    allowed_relative_position_motif,
):
    """Test RotateMotifAction with axis-angle in relative to position mode."""
    motif = allowed_relative_position_motif
    rotation_center = np.array([0.2, -2.0, 1.0])
    action = RotateMotifAction(
        operated_motif=motif,
        rotation_axis_vector=[0.2, -0.5, 1],
        rotation_axis_angle=-55,
        relative_to_position=rotation_center,
    )
    assert action.mode_flag == "axis_relative_to_position"
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    rot = Rotation.from_rotvec(
        np.radians(-55) * np.array([0.2, -0.5, 1]) / np.linalg.norm([0.2, -0.5, 1])
    )
    expected_motif_positions = (
        rot.apply(motif.cart_coords - rotation_center)
        + rotation_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by -55.000 degrees counter-clockwise" in desc
    assert ("around a rotation axis defined by the cartesian"
            " vector (0.176, -0.440, 0.880)") in desc  # Already normalized.
    assert ("around a rotation center in cartesian coordinates"
            " (0.200, -2.000, 1.000)") in desc


def test_rotate_motif_axis_relative_to_position_fractional(
    allowed_relative_position_motif,
    orig_atoms,
):
    """Test RotateMotifAction with axis-angle in relative to position mode."""
    motif = allowed_relative_position_motif
    frac_center = np.array([0.75, 0.3, 0.1])
    cart_center = frac_center @ orig_atoms.cell.complete()
    action = RotateMotifAction(
        operated_motif=motif,
        rotation_axis_vector=[-1, 0, 1],
        rotation_axis_angle=120,
        relative_to_position=frac_center,
        position_fractional=True,
    )
    assert action.mode_flag == "axis_relative_to_position"
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    rot = Rotation.from_rotvec(
        np.radians(120) * np.array([-1, 0, 1]) / np.linalg.norm([-1, 0, 1])
    )
    expected_motif_positions = (
        rot.apply(motif.cart_coords - cart_center)
        + cart_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by 120.000 degrees counter-clockwise" in desc
    assert ("around a rotation axis defined by the cartesian"
            " vector (-0.707, 0.000, 0.707)") in desc  # Already normalized.
    assert ("around a rotation center in fractional coordinates"
            " (0.750, 0.300, 0.100)") in desc


@pytest.fixture(params=["bond", "box"])
def forbidden_relative_position_operated_motif(request, orig_atoms):
    # Forbidden operated motifs in relative to position mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=123,
    )


def test_forbidden_relative_to_position_motif(
    forbidden_relative_position_operated_motif,
):
    """Test that forbidden motifs in relative to position mode raise errors."""
    motif = forbidden_relative_position_operated_motif
    if isinstance(motif, BoxRegionMotif):
        with pytest.raises(
                ValueError,
                match="Region motifs can only be used as operated motifs in self-relative"
        ):
            RotateMotifAction(
                operated_motif=motif,
                euler_angles=[15, 25, 35],
                relative_to_position=[0.0, 0.0, 0.0],
            )
    elif isinstance(motif, BondMotif):
        with pytest.raises(ValueError, match="Bond motifs are not allowed"):
            RotateMotifAction(
                operated_motif=motif,
                rotation_axis_vector=[1, 0, 0],
                rotation_axis_angle=90,
                relative_to_position=[1.0, 1.0, 1.0],
            )


def test_relative_to_position_overlap(
        allowed_relative_position_motif
):
    """Test that overlap between motif and relative_to_position raises error."""
    motif = allowed_relative_position_motif
    # Cartesian case.
    rotation_center = motif.get_centroid(fractional=False)
    if len(motif) == 1:
        with pytest.raises(
                ValueError,
                match="Rotation center position cannot be the same as"
        ):
            RotateMotifAction(
                operated_motif=motif,
                euler_angles=[10, 20, 30],
                relative_to_position=rotation_center,
            )
    else:
        action = RotateMotifAction(
            operated_motif=motif,
            euler_angles=[10, 20, 30],
            relative_to_position=rotation_center,
        )
        assert action.mode_flag == "euler_relative_to_position"
    # Fractional case.
    rotation_center_frac = motif.get_centroid(fractional=True)
    if len(motif) == 1:
        with pytest.raises(
                ValueError,
                match="Rotation center position cannot be the same as"
        ):
            RotateMotifAction(
                operated_motif=motif,
                rotation_axis_vector=[0, 1, 0],
                rotation_axis_angle=45,
                relative_to_position=rotation_center_frac,
                position_fractional=True,
            )
    else:
        action = RotateMotifAction(
            operated_motif=motif,
            rotation_axis_vector=[0, 1, 0],
            rotation_axis_angle=45,
            relative_to_position=rotation_center_frac,
            position_fractional=True,
        )
        assert action.mode_flag == "axis_relative_to_position"


### --- Relative to regular motif modes ---
@pytest.fixture(params=["cluster", "site"])
def allowed_relative_to_motif_operated_motif(request, orig_atoms):
    # Allowed operated motifs in relative to motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=456,
    )


@pytest.fixture(params=["cluster", "site", "bond"])
def allowed_relative_to_motif_relative_motif(request, orig_atoms):
    # Allowed relative motifs in relative to motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=789,
    )


def test_rotate_motif_euler_relative_to_motif(
    allowed_relative_to_motif_operated_motif,
    allowed_relative_to_motif_relative_motif,
):
    """Test RotateMotifAction with Euler angles in relative to motif mode."""
    operated_motif = allowed_relative_to_motif_operated_motif
    relative_motif = allowed_relative_to_motif_relative_motif
    action = RotateMotifAction(
        operated_motif=operated_motif,
        euler_angles=[25, -35, 45],
        relative_to_motif=relative_motif,
        relative_style="centroid_distance",
    )
    assert action.mode_flag == "euler_relative_to_motif"
    new_atoms = action.execute()
    operated_positions = new_atoms.get_positions(scale=False)[operated_motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        operated_motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    relative_center = relative_motif.get_centroid(fractional=False)
    rot = Rotation.from_euler("ZXZ", [25, -35, 45], degrees=True)
    expected_operated_positions = (
        rot.apply(operated_motif.cart_coords - relative_center)
        + relative_center
    )
    npt.assert_allclose(
        operated_positions,
        expected_operated_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        operated_motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert operated_motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by euler angles" in desc
    assert "counter-clockwise" in desc
    assert "(25.000, -35.000, 45.000) degrees" in desc
    assert ("around the centroid of"
            f" [{relative_motif.describe()}] as the rotation center") in desc


def test_rotate_motif_axis_relative_to_motif(
    allowed_relative_to_motif_operated_motif,
    allowed_relative_to_motif_relative_motif,
):
    """Test RotateMotifAction with axis-angle in relative to motif mode."""
    operated_motif = allowed_relative_to_motif_operated_motif
    relative_motif = allowed_relative_to_motif_relative_motif
    action = RotateMotifAction(
        operated_motif=operated_motif,
        rotation_axis_vector=[-1, 2, 1],
        rotation_axis_angle=75,
        relative_to_motif=relative_motif,
        relative_style="centroid_distance",
    )
    assert action.mode_flag == "axis_relative_to_regular_motif"
    npt.assert_allclose(
        action.rotation_axis_vector,
        np.array([-1, 2, 1]) / np.sqrt(6)
    )
    new_atoms = action.execute()
    operated_positions = new_atoms.get_positions(scale=False)[operated_motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        operated_motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    relative_center = relative_motif.get_centroid(fractional=False)
    rot = Rotation.from_rotvec(
        np.radians(75) * np.array([-1, 2, 1]) / np.linalg.norm([-1, 2, 1])
    )
    expected_operated_positions = (
        rot.apply(operated_motif.cart_coords - relative_center)
        + relative_center
    )
    npt.assert_allclose(
        operated_positions,
        expected_operated_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        operated_motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert operated_motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by 75.000 degrees counter-clockwise" in desc
    assert ("around a rotation axis defined by the cartesian"
            " vector (-0.408, 0.816, 0.408)") in desc  # Already normalized.
    assert ("around the centroid of"
            f" [{relative_motif.describe()}] as the rotation center") in desc


@pytest.fixture(params=["box", "bond", "sphere"])
def forbidden_relative_to_motif_operated_motif(request, orig_atoms):
    # Forbidden operated motif in relative to motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=456,
    )


def test_relative_to_motif_forbidden_operated_motif(
    forbidden_relative_to_motif_operated_motif,
    allowed_relative_to_motif_relative_motif,
):
    """Test that forbidden operated motifs in relative to motif mode raise errors."""
    operated_motif = forbidden_relative_to_motif_operated_motif
    relative_motif = allowed_relative_to_motif_relative_motif
    if isinstance(operated_motif, BoxRegionMotif):
        with pytest.raises(
                ValueError,
                match="Region motifs can only be used as operated motifs in self-relative"
        ):
            RotateMotifAction(
                operated_motif=operated_motif,
                euler_angles=[5, 15, 25],
                relative_to_motif=relative_motif,
                relative_style="centroid_distance",
            )
    elif isinstance(operated_motif, SphereRegionMotif):
        with pytest.raises(
                ValueError,
                match="Region motifs can only be used as operated motifs in self-relative"
        ):
            RotateMotifAction(
                operated_motif=operated_motif,
                euler_angles=[-10, 20, -30],
                relative_to_motif=relative_motif,
                relative_style="centroid_distance",
            )
    elif isinstance(operated_motif, BondMotif):
        with pytest.raises(ValueError, match="Bond motifs are not allowed"):
            RotateMotifAction(
                operated_motif=operated_motif,
                rotation_axis_vector=[0, 1, 0],
                rotation_axis_angle=60,
                relative_to_motif=relative_motif,
                relative_style="centroid_distance",
            )


@pytest.fixture(params=["box", "sphere"])
def forbidden_relative_to_motif_relative_motif(request, orig_atoms):
    # Forbidden relative motif in relative to motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=789,
    )


def test_relative_to_motif_forbidden_relative_motif(
    allowed_relative_to_motif_operated_motif,
    forbidden_relative_to_motif_relative_motif,
):
    """Test that forbidden relative motifs in relative to motif mode raise errors."""
    operated_motif = allowed_relative_to_motif_operated_motif
    relative_motif = forbidden_relative_to_motif_relative_motif
    with pytest.raises(
            ValueError,
            match="Region motifs are not allowed as relative motifs for rotation."
    ):
        RotateMotifAction(
            operated_motif=operated_motif,
            euler_angles=[-15, 25, -35],
            relative_to_motif=relative_motif,
            relative_style="centroid_distance",
        )


@pytest.fixture(params=["site"])
def site_motif(request, orig_atoms):
    # Site motif for overlap test.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=101112,
    )


def test_relative_to_site_motif_overlap(
        site_motif
):
    """Test that overlap between operated motif and relative_to_motif raises error."""
    operated_motif = site_motif
    relative_motif = site_motif
    # Operated motif and relative motif are the same.
    with pytest.raises(
            ValueError,
            match="Rotation center motif's centroid cannot"
                  " be the same as the operated motif's centroid"
    ):
        RotateMotifAction(
            operated_motif=operated_motif,
            rotation_axis_vector=[1, 0, 0],
            rotation_axis_angle=30,
            relative_to_motif=relative_motif,
            relative_style="centroid_distance",
        )


### --- Relative to pair motif modes ---
@pytest.fixture(params=["cluster", "site"])
def allowed_relative_to_pair_motif_operated_motif(request, orig_atoms):
    # Allowed operated motifs in relative to pair motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=131415,
    )

@pytest.fixture(params=["bond", "cluster"])
def allowed_relative_to_pair_motif_relative_motif(request, orig_atoms):
    # Allowed relative pair motifs in relative to pair motif mode.
    if request.param == "bond":
        return get_random_motif(
            class_alias=request.param,
            atoms=orig_atoms,
            seed=161718,
        )
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        cluster_size=2,
        seed=161718,
    )


# Pair motif relative only allows axis-angle rotation for now.
def test_rotate_motif_axis_relative_to_pair_motif(
    allowed_relative_to_pair_motif_operated_motif,
    allowed_relative_to_pair_motif_relative_motif,
):
    """Test RotateMotifAction with axis-angle in relative to pair motif mode."""
    operated_motif = allowed_relative_to_pair_motif_operated_motif
    relative_motif = allowed_relative_to_pair_motif_relative_motif
    action = RotateMotifAction(
        operated_motif=operated_motif,
        rotation_axis_angle=90,
        relative_to_motif=relative_motif,
        relative_style="rotation_axis",
        relative_axis_origin_index=1,
    )
    assert action.mode_flag == "axis_relative_to_pair_motif"
    assert action.rotation_axis_vector is None
    new_atoms = action.execute()
    operated_positions = new_atoms.get_positions(scale=False)[operated_motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        operated_motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    pair_center = relative_motif.get_centroid(fractional=False)
    # Origin 1, point to 0.
    rotation_axis_vector = relative_motif.cart_coords[0] - relative_motif.cart_coords[1]
    rotation_axis_vector /= np.linalg.norm(rotation_axis_vector)
    rot = Rotation.from_rotvec(
        np.radians(90) * rotation_axis_vector
    )
    expected_operated_positions = (
        rot.apply(operated_motif.cart_coords - pair_center)
        + pair_center
    )
    npt.assert_allclose(
        operated_positions,
        expected_operated_positions,
        atol=1e-6,
    )
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        operated_motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check that chemical symbols are unchanged.
    assert operated_motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()

    desc = action.describe(precision=3)
    assert "in the structure by 90.000 degrees counter-clockwise" in desc
    assert "around a rotation axis defined by the line of" in desc
    idx1 = relative_motif.indices[1]
    assert f"pointing from the atom with index {idx1} to the other atom" in desc
    assert "around the centroid of the axis pair" in desc


@pytest.fixture(params=["bond", "box", "sphere"])
def forbidden_relative_to_pair_motif_operated_motif(request, orig_atoms):
    # Forbidden operated motif in relative to pair motif mode.
    return get_random_motif(
        class_alias=request.param,
        atoms=orig_atoms,
        seed=131415,
    )


def test_relative_to_pair_motif_forbidden_operated_motif(
    forbidden_relative_to_pair_motif_operated_motif,
    allowed_relative_to_pair_motif_relative_motif,
):
    """Test that forbidden operated motifs in relative to pair motif mode raise errors."""
    operated_motif = forbidden_relative_to_pair_motif_operated_motif
    relative_motif = allowed_relative_to_pair_motif_relative_motif
    if isinstance(operated_motif, BoxRegionMotif):
        with pytest.raises(
                ValueError,
                match="Region motifs can only be used as operated motifs in self-relative"
        ):
            RotateMotifAction(
                operated_motif=operated_motif,
                rotation_axis_angle=45,
                relative_to_motif=relative_motif,
                relative_style="rotation_axis",
                relative_axis_origin_index=0,
            )
    elif isinstance(operated_motif, SphereRegionMotif):
        with pytest.raises(
                ValueError,
                match="Region motifs can only be used as operated motifs in self-relative"
        ):
            RotateMotifAction(
                operated_motif=operated_motif,
                rotation_axis_angle=60,
                relative_to_motif=relative_motif,
                relative_style="rotation_axis",
                relative_axis_origin_index=1,
            )
    elif isinstance(operated_motif, BondMotif):
        with pytest.raises(ValueError, match="Bond motifs are not allowed"):
            RotateMotifAction(
                operated_motif=operated_motif,
                rotation_axis_angle=60,
                relative_to_motif=relative_motif,
                relative_style="rotation_axis",
                relative_axis_origin_index=0,
            )


@pytest.fixture(params=["box", "sphere", "site", "cluster"])
def forbidden_relative_to_pair_motif_relative_motif(request, orig_atoms):
    # Forbidden relative pair motif in relative to pair motif mode.
    if request.param != "cluster":
        return get_random_motif(
            class_alias=request.param,
            atoms=orig_atoms,
            seed=161718,
        )
    else:
        return get_random_motif(
            class_alias=request.param,
            atoms=orig_atoms,
            cluster_size=3,  # Not a pair.
            seed=161718,
        )


def test_relative_to_pair_motif_forbidden_relative_motif(
    allowed_relative_to_pair_motif_operated_motif,
    forbidden_relative_to_pair_motif_relative_motif,
):
    """Test that forbidden relative motifs in relative to pair motif mode raise errors."""
    operated_motif = allowed_relative_to_pair_motif_operated_motif
    relative_motif = forbidden_relative_to_pair_motif_relative_motif
    with pytest.raises(
            ValueError,
            match="Only pair site-collection motifs are allowed for axis_relative_pair_motif mode."
    ):
        RotateMotifAction(
            operated_motif=operated_motif,
            rotation_axis_angle=45,
            relative_to_motif=relative_motif,
            relative_style="rotation_axis",
            relative_axis_origin_index=0,
        )


def test_relative_to_pair_motif_invalid_origin_index(
    allowed_relative_to_pair_motif_operated_motif,
    allowed_relative_to_pair_motif_relative_motif,
):
    """Test that invalid origin index in relative to pair motif mode raises error."""
    operated_motif = allowed_relative_to_pair_motif_operated_motif
    relative_motif = allowed_relative_to_pair_motif_relative_motif
    with pytest.raises(
            ValueError,
            match="Relative atom index must be provided as 0 or 1"
    ):
        RotateMotifAction(
            operated_motif=operated_motif,
            rotation_axis_angle=30,
            relative_to_motif=relative_motif,
            relative_style="rotation_axis",
            relative_axis_origin_index=2,
        )


def test_relative_to_pair_motif_invalid_rotation_angle(
    allowed_relative_to_pair_motif_operated_motif,
    allowed_relative_to_pair_motif_relative_motif,
):
    """Test that missing rotation angle in relative to pair motif mode raises error."""
    operated_motif = allowed_relative_to_pair_motif_operated_motif
    relative_motif = allowed_relative_to_pair_motif_relative_motif
    with pytest.raises(
            ValueError,
            match="Rotation axis angle must be a number"
    ):
        RotateMotifAction(
            operated_motif=operated_motif,
            rotation_axis_angle="not a number",
            relative_to_motif=relative_motif,
            relative_style="rotation_axis",
            relative_axis_origin_index=0,
)