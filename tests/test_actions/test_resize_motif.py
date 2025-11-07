"""Comprehensive test suite for ResizeMotifAction."""

import pytest
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.resize import ResizeMotifAction
from AtomWorldBench.atom_world.motifs.regions.sphere import SphereRegionMotif
from AtomWorldBench.atom_world.motifs.regions.box import BoxRegionMotif
from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["resize-motif"]
    assert action_class is ResizeMotifAction
    action_class = get_registered(BaseMotifAction)["resize"]
    assert action_class is ResizeMotifAction


@pytest.fixture(params=["cluster", "sphere", "bond"])
def allowed_motif(request, orig_atoms):
    """Fixture for allowed motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms)
    return motif


@pytest.fixture(params=["site", "box"])
def forbidden_motif(request, orig_atoms):
    """Fixture for forbidden motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms)
    return motif


def test_resize_motif_action_relative_to_centroid_scale_by_enlarge(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    motif = allowed_motif

    # Enlarge.
    scale_factor = 1.5

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        scale_by=scale_factor,
    )
    assert action.mode_flag == "relative_to_centroid_scale_by"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
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
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, SphereRegionMotif):
        op_word = "away from"
    elif isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "enlarge"
    elif isinstance(motif, BondMotif):
        op_word = "elongate"
    else:
        raise ValueError("Unexpected motif type.")
    assert "its centroid" in desc
    assert "by a scale factor of 1.500" in desc
    assert op_word in desc
    if isinstance(motif, SphereRegionMotif):
        assert "such that their distances to" in desc
    else:
        assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_centroid_scale_by_shrink(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    motif = allowed_motif

    # Enlarge.
    scale_factor = 0.8

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        scale_by=scale_factor,
    )
    assert action.mode_flag == "relative_to_centroid_scale_by"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
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
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, SphereRegionMotif):
        op_word = "towards"
    elif isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "shrink"
    elif isinstance(motif, BondMotif):
        op_word = "shorten"
    else:
        raise ValueError("Unexpected motif type.")
    assert "its centroid" in desc
    assert "by a scale factor of 0.800" in desc
    assert op_word in desc
    if isinstance(motif, SphereRegionMotif):
        assert "such that their distances to" in desc
    else:
        assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_centroid_to_radius_enlarge(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    motif = allowed_motif

    # Enlarge.
    scale_factor = 1.5
    init_radius = motif.radius
    to_radius = init_radius * scale_factor

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        to_radius=to_radius,
    )
    assert action.mode_flag == "relative_to_centroid_to_radius"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, SphereRegionMotif):
        op_word = "away from"
        size_word = "radius"
    elif isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "enlarge"
        size_word = "radius"
    elif isinstance(motif, BondMotif):
        op_word = "elongate"
        size_word = "length"
    else:
        raise ValueError("Unexpected motif type.")
    assert "its centroid" in desc
    assert op_word in desc
    if isinstance(motif, SphereRegionMotif):
        assert f"to {to_radius:.3f} angstroms" in desc
        assert "such that their distances to" in desc
    else:
        assert f"to a {size_word} of {to_radius:.3f} angstroms" in desc
        assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_centroid_to_radius_shrink(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    motif = allowed_motif

    # Enlarge.
    scale_factor = 0.8
    init_radius = motif.radius
    to_radius = init_radius * scale_factor

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        to_radius=to_radius,
    )
    assert action.mode_flag == "relative_to_centroid_to_radius"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, SphereRegionMotif):
        op_word = "towards"
        size_word = "radius"
    elif isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "shrink"
        size_word = "radius"
    elif isinstance(motif, BondMotif):
        op_word = "shorten"
        size_word = "length"
    else:
        raise ValueError("Unexpected motif type.")
    assert "its centroid" in desc
    assert op_word in desc
    if isinstance(motif, SphereRegionMotif):
        assert f"to {to_radius:.3f} angstroms" in desc
        assert "such that their distances to" in desc
    else:
        assert f"to a {size_word} of {to_radius:.3f} angstroms" in desc
        assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_index_scale_by_enlarge(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    if isinstance(allowed_motif, SphereRegionMotif):
        with pytest.raises(ValueError, match="operated_motif cannot be a region motif for"):
            _ = ResizeMotifAction(
                operated_motif=allowed_motif,
                relative_to_node_index=0,
                scale_by=1.2,
            )
        return

    motif = allowed_motif

    # Enlarge.
    scale_factor = 1.2

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        relative_to_node_index=0,
        scale_by=scale_factor,
    )
    assert action.mode_flag == "relative_to_node_index_scale_by"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.cart_coords[action.relative_to_node_index]
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "enlarge"
    elif isinstance(motif, BondMotif):
        op_word = "elongate"
    else:
        raise ValueError("Unexpected motif type.")
    assert f"the atom at index {motif.indices[0]}" in desc
    assert "by a scale factor of 1.200" in desc
    assert op_word in desc
    assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_index_scale_by_shrink(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    if isinstance(allowed_motif, SphereRegionMotif):
        with pytest.raises(ValueError, match="operated_motif cannot be a region motif for"):
            _ = ResizeMotifAction(
                operated_motif=allowed_motif,
                relative_to_node_index=0,
                scale_by=1.2,
            )
        return

    motif = allowed_motif

    # Enlarge.
    scale_factor = 0.9

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        relative_to_node_index=0,
        scale_by=scale_factor,
    )
    assert action.mode_flag == "relative_to_node_index_scale_by"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.cart_coords[action.relative_to_node_index]
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "shrink"
    elif isinstance(motif, BondMotif):
        op_word = "shorten"
    else:
        raise ValueError("Unexpected motif type.")
    assert f"the atom at index {motif.indices[0]}" in desc
    assert "by a scale factor of 0.900" in desc
    assert op_word in desc
    assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_index_to_radius_enlarge(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    if isinstance(allowed_motif, SphereRegionMotif):
        with pytest.raises(ValueError, match="operated_motif cannot be a region motif for"):
            _ = ResizeMotifAction(
                operated_motif=allowed_motif,
                relative_to_node_index=0,
                to_radius=1.2,
            )
        return

    motif = allowed_motif

    # Enlarge.
    scale_factor = 1.2
    init_radius = motif.radius
    to_radius = init_radius * scale_factor

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        relative_to_node_index=0,
        to_radius=to_radius,
    )
    assert action.mode_flag == "relative_to_node_index_to_radius"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.cart_coords[action.relative_to_node_index]
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "enlarge"
        size_word = "radius"
    elif isinstance(motif, BondMotif):
        op_word = "elongate"
        size_word = "length"
    else:
        raise ValueError("Unexpected motif type.")
    assert f"the atom at index {motif.indices[0]}" in desc
    assert f"to a {size_word} of {to_radius:.3f} angstroms" in desc
    assert op_word in desc
    assert "by moving its atoms relative to " in desc


def test_resize_motif_action_relative_to_index_to_radius_shrink(allowed_motif):
    """Test ResizeMotifAction with allowed motifs."""
    if isinstance(allowed_motif, SphereRegionMotif):
        with pytest.raises(ValueError, match="operated_motif cannot be a region motif for"):
            _ = ResizeMotifAction(
                operated_motif=allowed_motif,
                relative_to_node_index=0,
                to_radius=0.2,
            )
        return

    motif = allowed_motif

    # Enlarge.
    scale_factor = 0.9
    init_radius = motif.radius
    to_radius = init_radius * scale_factor

    action = ResizeMotifAction(
        operated_motif=allowed_motif,
        relative_to_node_index=0,
        to_radius=to_radius,
    )
    assert action.mode_flag == "relative_to_node_index_to_radius"
    new_atoms = action.execute()
    # Check that the motif size has increased.
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.cart_coords[action.relative_to_node_index]
    expected_motif_positions = (
        (motif.cart_coords - motif_center) * scale_factor + motif_center
    )
    npt.assert_allclose(
        motif_positions,
        expected_motif_positions,
        atol=1e-6,
    )
    # Check that chemical symbols are unchanged.
    assert motif.in_atoms.get_chemical_symbols() == new_atoms.get_chemical_symbols()
    # Check that other atoms have not moved.
    npt.assert_allclose(
        other_positions,
        motif.in_atoms.get_positions(scale=False)[other_indices],
    )
    # Check description is correct.
    desc = action.describe(precision=3)
    if isinstance(motif, ClusterMotif) and (not isinstance(motif, BondMotif)):
        # Default size > 2.
        op_word = "shrink"
        size_word = "radius"
    elif isinstance(motif, BondMotif):
        op_word = "shorten"
        size_word = "length"
    else:
        raise ValueError("Unexpected motif type.")
    assert f"the atom at index {motif.indices[0]}" in desc
    assert f"to a {size_word} of {to_radius:.3f} angstroms" in desc
    assert op_word in desc
    assert "by moving its atoms relative to " in desc


# --- Test failed cases ---
def test_resize_motif_action_forbidden_motifs(forbidden_motif):
    """Test ResizeMotifAction raises errors for forbidden motifs."""
    motif = forbidden_motif

    if isinstance(forbidden_motif, SiteMotif):
        with pytest.raises(ValueError, match="operated_motif must have at least 2 atoms"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                scale_by=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif must have at least 2 atoms"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                to_radius=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif must have at least 2 atoms"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                relative_to_node_index=0,
                scale_by=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif must have at least 2 atoms"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                relative_to_node_index=0,
                to_radius=1.2,
            )

    elif isinstance(forbidden_motif, BoxRegionMotif):
        with pytest.raises(ValueError, match="operated_motif must have get_centroid method"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                scale_by=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif must have get_centroid method"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                to_radius=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif cannot be a region motif for"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                relative_to_node_index=0,
                scale_by=1.2,
            )
        with pytest.raises(ValueError, match="operated_motif must have radius attribute"):
            _ = ResizeMotifAction(
                operated_motif=motif,
                relative_to_node_index=0,
                to_radius=1.2,
            )
    else:
        raise ValueError("Unexpected forbidden motif type.")


def test_operated_motif_not_in_atoms(orig_atoms):
    """Test ResizeMotifAction raises error if operated_motif not in atoms."""
    # Create a motif in additive_mode, i.e, not having in_atoms.
    motif_additive = get_random_motif(
        "cluster", orig_atoms, seed=42, additive_mode=True
    )

    with pytest.raises(ValueError, match="must be attached to an Atoms object"):
        _ = ResizeMotifAction(
            operated_motif=motif_additive,
            scale_by=1.2,
        )


def test_get_random_one(orig_atoms):
    all_appeared_modes = set()
    for _ in range(200):
        action = ResizeMotifAction.get_random_one(
            operated_atoms=orig_atoms,
            seed=None,
        )
        assert isinstance(action, ResizeMotifAction)
        assert isinstance(action.operated_motif, (ClusterMotif, SphereRegionMotif, BondMotif))
        all_appeared_modes.add(action.mode_flag)
        # Sphere does not support relative_to_node_index.
        if isinstance(action.operated_motif, SphereRegionMotif):
            assert action.relative_to_node_index is None
            assert "relative_to_node_index" not in action.mode_flag
        if "scale_by" in action.mode_flag:
            assert action.scale_by is not None
            assert action.to_radius is None
        if "to_radius" in action.mode_flag:
            assert action.to_radius is not None
            assert action.scale_by is None
    expected_modes = {
        "relative_to_centroid_scale_by",
        "relative_to_centroid_to_radius",
        "relative_to_node_index_scale_by",
        "relative_to_node_index_to_radius",
    }
    assert all_appeared_modes == expected_modes
