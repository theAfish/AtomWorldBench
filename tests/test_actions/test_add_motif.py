"""Comprehensive test suite for AddMotifAction."""

import pytest
import numpy as np
import numpy.testing as npt
from ase import Atoms

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.add import AddMotifAction
from AtomWorldBench.atom_world.motifs.regions.base import BaseRegionMotif
from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


@pytest.fixture(
    params=["box", "sphere", "bond"] # Region and Bond motif cannot be added to structure.
)
def forbidden_operated_motif(request, orig_atoms):
    """Fixture to provide forbidden operated motifs."""
    return get_random_motif(request.param, orig_atoms)


@pytest.fixture(
    params=["cluster", "site"]
)
def allowed_operated_motif(request, orig_atoms):
    """Fixture to provide allowed operated motifs."""
    return get_random_motif(request.param, orig_atoms, additive_mode=True)


@pytest.fixture(
    params=["cluster", "site", "bond"]
)
def allowed_relative_to_motif(request, orig_atoms):
    """Fixture to provide allowed relative_to motifs."""
    return get_random_motif(request.param, orig_atoms)


@pytest.fixture(
    params=["bond", "cluster"]
)
def bond_motif(request, orig_atoms):
    """Fixture to provide a bond motif."""
    if request.param == "cluster":
        return get_random_motif(request.param, orig_atoms, cluster_size=2)
    return get_random_motif(request.param, orig_atoms)


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["add-motif"]
    assert action_class is AddMotifAction
    action_class = get_registered(BaseMotifAction)["add"]
    assert action_class is AddMotifAction


def test_init_invalid_operated_motif(forbidden_operated_motif, orig_atoms):
    """Test initialization with invalid operated motif."""
    with pytest.raises(ValueError, match="must be a non-bond site collection motif"):
        _ = AddMotifAction(
            operated_motif=forbidden_operated_motif,
            operated_atoms=orig_atoms,
            at_position=[0.5, 0.5, 0.5],
            position_fractional=True
        )

# ---- Test operations under modes ----

def test_at_position_mode(allowed_operated_motif, orig_atoms):
    """Test initialization in at_position mode."""
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        at_position=[0.5, 0.5, 0.5],
        position_fractional=True
    )
    assert action.mode_flag == "absolute"
    assert np.allclose(action.at_position, [0.5, 0.5, 0.5])
    assert action.position_fractional is True
    new_atoms = action.execute()
    assert hasattr(allowed_operated_motif, "__len__")
    n_added = len(allowed_operated_motif)
    assert len(new_atoms) == len(orig_atoms) + n_added
    # Check that motif has been correctly appended to the desired place.
    added_atoms = new_atoms[-n_added:]
    assert added_atoms.get_chemical_symbols() == allowed_operated_motif.get_atoms().get_chemical_symbols()
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        np.array([0.5, 0.5, 0.5]) @ orig_atoms.cell.complete()
    )
    # Translated, rather than rotated.
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )
    desc = action.describe(precision=3)
    assert "with its centroid located at fractional coordinates (0.500, 0.500, 0.500)" in desc
    assert "newly added motif should be appended to the end" in desc


def test_relative_to_motif_mode(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test initialization in relative_to_motif_centroid mode."""
    random_shift = np.random.RandomState(42).randn(3)
    shift_string = f"({random_shift[0]:.3f}, {random_shift[1]:.3f}, {random_shift[2]:.3f})"
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_motif=allowed_relative_to_motif,
        relative_shift=random_shift,  # Not fractional.
        relative_style="centroid_distance"
    )
    assert action.mode_flag == "relative_to_motif_centroid"
    n_added = len(allowed_operated_motif)
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms) + n_added
    # Check that motif has been correctly appended to the desired place.
    added_atoms = new_atoms[-n_added:]
    assert added_atoms.get_chemical_symbols() == allowed_operated_motif.get_atoms().get_chemical_symbols()
    relative_to_centroid = allowed_relative_to_motif.get_centroid(fractional=False)
    expected_centroid = relative_to_centroid + random_shift
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        expected_centroid
    )
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )
    desc = action.describe(precision=3)
    assert f"with its centroid shifted in cartesian coordinates by {shift_string}" in desc
    assert "relative to the centroid of" in desc


def test_relative_to_position(allowed_operated_motif, orig_atoms):
    """Test initialization in relative_to_position mode."""
    random_shift = np.random.RandomState(42).randn(3)
    shift_string = f"({random_shift[0]:.3f}, {random_shift[1]:.3f}, {random_shift[2]:.3f})"
    # Test fractional here.
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_position=[0.2, 0.3, 0.4],  # Fractional
        position_fractional=True,
        relative_shift=random_shift,  # Fractional.
    )
    assert action.mode_flag == "relative_to_position"
    n_added = len(allowed_operated_motif)
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms) + n_added
    # Check that motif has been correctly appended to the desired place.
    added_atoms = new_atoms[-n_added:]
    assert added_atoms.get_chemical_symbols() == allowed_operated_motif.get_atoms().get_chemical_symbols()
    relative_to_position_cart = np.array([0.2, 0.3, 0.4]) @ orig_atoms.cell.complete()
    expected_centroid = relative_to_position_cart + random_shift @ orig_atoms.cell.complete()
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        expected_centroid
    )
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )
    desc = action.describe(precision=3)
    assert f"with its centroid shifted in fractional coordinates by {shift_string}" in desc
    assert "relative to a reference point at fractional coordinates (0.200, 0.300, 0.400)" in desc


def test_relative_to_pair_motif(allowed_operated_motif, bond_motif, orig_atoms):
    """Test initialization in at_bond_center mode."""
    random_shift = np.random.RandomState(42).randn()
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_motif=bond_motif,
        relative_shift=random_shift,  # Not fractional.
        relative_style="position_in_line",
        relative_atom_index=1,
        position_fractional=True,  # Fractional should not affect distance calculation.
    )
    assert action.mode_flag == "relative_to_pair_motif"
    n_added = len(allowed_operated_motif)
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms) + n_added
    # Check that motif has been correctly appended to the desired place.
    added_atoms = new_atoms[-n_added:]
    assert added_atoms.get_chemical_symbols() == allowed_operated_motif.get_atoms().get_chemical_symbols()
    bond_normal_vector = bond_motif.cart_coords[0] - bond_motif.cart_coords[1]
    bond_normal_vector /= np.linalg.norm(bond_normal_vector)
    expected_centroid = bond_motif.cart_coords[1] + random_shift * bond_normal_vector
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        expected_centroid
    )
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )
    desc = action.describe(precision=3)
    assert f"with its centroid located on the line between" in desc
    assert (
               f"at {random_shift:.3f} angstroms away from the atom indexed"
               f" {bond_motif.indices[1]} (in the original structure)."
    ) in desc


def test_init_missing_required_params(allowed_operated_motif, orig_atoms):
    """Test that missing required parameters raises error."""
    with pytest.raises(ValueError, match="No mode detected"):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms
        )


def test_init_ambiguous_mode(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test that providing parameters for multiple modes raises error."""
    with pytest.raises(ValueError, match="No mode detected"):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms,
            at_position=[0.5, 0.5, 0.5],
            relative_to_motif=allowed_relative_to_motif,
            relative_to_position=[1.0, 0.0, 0.0]
        )


def test_init_invalid_at_position_shape(allowed_operated_motif, orig_atoms):
    """Test that invalid at_position shape raises error."""
    with pytest.raises(ValueError, match="expected 1D array of shape \(3,\)"):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms,
            at_position=[[0.5, 0.5], [0.5, 0.5]],
            position_fractional=True
        )


def test_init_invalid_relative_position_shape(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test that invalid relative_position shape raises error."""
    with pytest.raises(ValueError, match="expected 1D array of shape \(3,\)"):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms,
            relative_to_motif=allowed_relative_to_motif,
            relative_shift=[1.0, 0.0],  # Wrong length
            relative_style="centroid_distance"
        )


def test_init_non_bond_relative_to_bond(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test that non-bond motif for at_bond_center raises error."""
    if isinstance(allowed_relative_to_motif, BaseRegionMotif) or len(allowed_relative_to_motif) != 2:
        with pytest.raises(ValueError, match="Only pair motifs are allowed"):
            _ = AddMotifAction(
                operated_motif=allowed_operated_motif,
                operated_atoms=orig_atoms,
                relative_to_motif=allowed_relative_to_motif,  # Default size is 3 or 1, cannot be a bond.
                relative_style="position_in_line",
                relative_atom_index=0,
                position_fractional=False,
                relative_shift=0.1,
            )

# ---- Test operated_motif validation ----

def test_operated_motif_not_additive(orig_atoms):
    """Test that non-additive operated motif raises error."""
    non_additive_motif = SiteMotif(
        in_atoms=orig_atoms,
        indices=[0]
    )
    with pytest.raises(ValueError, match="Inserted motif must be in additive mode."):
        _ = AddMotifAction(
            operated_motif=non_additive_motif,
            operated_atoms=orig_atoms,
            at_position=[0.5, 0.5, 0.5],
            position_fractional=True
        )

# ---- Test relative_to_motif validation ----

def test_relative_motif_in_wrong_atoms(orig_atoms):
    """Test that relative motif belonging to different atoms raises error."""
    other_atoms = Atoms('H2', positions=[(0, 0, 0), (1, 1, 1)], cell=[10, 10, 10], pbc=True)
    wrong_motif = SiteMotif(
        in_atoms=other_atoms,
        indices=[0]
    )
    operated_motif = SiteMotif(
        atoms=Atoms("H", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True),
    )
    with pytest.raises(ValueError, match="must be attached to the same Atoms"):
        _ = AddMotifAction(
            operated_motif=operated_motif,
            operated_atoms=orig_atoms,
            relative_to_motif=wrong_motif,
            relative_shift=[0.5, 0.5, 0.5],
            relative_style="centroid_distance",
            position_fractional=True
        )


# ---- Test at_bond_center validation ----
def test_relative_to_pair_wrong_size(allowed_operated_motif, orig_atoms):
    """Test that bond with wrong size raises error."""
    triplet = ClusterMotif.detect_random_one(
        orig_atoms,
        cluster_size=3,
        seed=42
    )
    with pytest.raises(ValueError, match="Only pair motifs are allowed for relative_to_pair_motif mode."):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms,
            relative_to_motif=triplet,
            relative_style="position_in_line",
            relative_shift=0.1,
            position_fractional=True,
        )

# ---- Test mode_flag immutability ----

def test_mode_flag_immutable(allowed_operated_motif, orig_atoms):
    """Test that mode_flag cannot be changed after initialization."""
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        at_position=[0.5, 0.5, 0.5],
        position_fractional=True
    )
    with pytest.raises(AttributeError, match="immutable"):
        action._mode_flag = "different_mode"

# ---- Test edge cases ----

def test_large_relative_shift(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test with large relative shift values. Should not have wrapped."""
    large_pos = [100.0, 100.0, 100.0]
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_motif=allowed_relative_to_motif,
        relative_shift=large_pos,
        relative_style="centroid_distance"
    )
    assert np.allclose(action.relative_shift, large_pos)
    n_insert = len(allowed_operated_motif)
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms) + n_insert
    relative_centroid = allowed_relative_to_motif.get_centroid(fractional=False)
    added_atoms = new_atoms[-n_insert:]
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        relative_centroid + np.array(large_pos)
    )
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )


def test_at_position_outside_cell(allowed_operated_motif, orig_atoms):
    """Test adding motif at position outside the unit cell. Should still work."""
    outside_pos = [2.0, 2.0, 2.0]  # Fractional > 1.0
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        at_position=outside_pos,
        position_fractional=True
    )
    assert np.allclose(action.at_position, outside_pos)
    n_insert = len(allowed_operated_motif)
    new_atoms = action.execute()
    assert len(new_atoms) == len(orig_atoms) + n_insert
    added_atoms = new_atoms[-n_insert:]
    added_centroid = np.mean(added_atoms.get_positions(wrap=False), axis=0)
    npt.assert_allclose(
        added_centroid,
        np.array(outside_pos) @ orig_atoms.cell.complete()
    )
    npt.assert_allclose(
        added_atoms.get_positions() - added_centroid,
        allowed_operated_motif.cart_coords - allowed_operated_motif.get_centroid(fractional=False)
    )

# ---- Test attribute access ----

def test_attributes_set_correctly_at_position(allowed_operated_motif, orig_atoms):
    """Test that all attributes are set correctly in at_position mode."""
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        at_position=[0.3, 0.4, 0.5],
        position_fractional=True
    )
    assert hasattr(action, "operated_motif")
    assert hasattr(action, "operated_atoms")
    assert hasattr(action, "at_position")
    assert hasattr(action, "position_fractional")
    assert hasattr(action, "mode_flag")


def test_attributes_set_correctly_relative_to_motif(allowed_operated_motif, allowed_relative_to_motif, orig_atoms):
    """Test that all attributes are set correctly in relative_to_motif mode."""
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_motif=allowed_relative_to_motif,
        relative_shift=[1.5, 0.0, 0.0],
        relative_style="centroid_distance"
    )
    assert hasattr(action, "operated_motif")
    assert hasattr(action, "operated_atoms")
    assert hasattr(action, "relative_to_motif")
    assert hasattr(action, "relative_shift")
    assert hasattr(action, "position_fractional")
    assert hasattr(action, "relative_style")
    assert hasattr(action, "mode_flag")


def test_attributes_set_correctly_relative_to_bond(allowed_operated_motif, bond_motif, orig_atoms):
    """Test that all attributes are set correctly in at_bond_center mode."""
    action = AddMotifAction(
        operated_motif=allowed_operated_motif,
        operated_atoms=orig_atoms,
        relative_to_motif=bond_motif,
        relative_shift=0.2,
        relative_style="position_in_line",
        relative_atom_index=0,
        position_fractional=False
    )
    assert hasattr(action, "operated_motif")
    assert hasattr(action, "operated_atoms")
    assert hasattr(action, "relative_to_motif")
    assert hasattr(action, "relative_shift")
    assert hasattr(action, "relative_atom_index")
    assert hasattr(action, "position_fractional")
    assert hasattr(action, "relative_style")
    assert hasattr(action, "mode_flag")

# ---- Test None values ----

def test_none_for_optional_param_raises_error(allowed_operated_motif, orig_atoms):
    """Test that None for required parameter raises error."""
    with pytest.raises(ValueError, match="No mode detected"):
        _ = AddMotifAction(
            operated_motif=allowed_operated_motif,
            operated_atoms=orig_atoms,
            at_position=None,
            position_fractional=True
        )

# ---- Test get random one ----
def test_get_random_one(orig_atoms):
    """Test the detect_random_one method."""
    all_appeared_modes = []
    for _ in range(200):
        action = AddMotifAction.get_random_one(
            operated_atoms=orig_atoms,
            seed=None,
        )
        assert isinstance(action, AddMotifAction)
        all_appeared_modes.append(action.mode_flag)

        # Check operated motif.
        assert not isinstance(action.operated_motif, BaseRegionMotif)
        assert not isinstance(action.operated_motif, BondMotif)
        assert action.operated_motif.is_additive is True

        # Mode specific checks.
        if action.mode_flag == "relative_to_motif_centroid":
            assert getattr(action, "relative_to_motif", None) is not None
            assert isinstance(action.relative_to_motif, BaseSiteCollectionMotif)
            assert not action.relative_to_motif.is_additive
            assert getattr(action, "relative_shift", None) is not None
            assert action.relative_style == "centroid_distance"

        if action.mode_flag == "relative_to_pair_motif":
            assert getattr(action, "relative_to_motif", None) is not None
            assert isinstance(action.relative_to_motif, (BondMotif, ClusterMotif))
            assert len(action.relative_to_motif) == 2
            assert not action.relative_to_motif.is_additive
            assert getattr(action, "relative_shift", None) is not None
            assert isinstance(action.relative_shift, (int, float))
            assert action.relative_style == "position_in_line"
            assert getattr(action, "relative_atom_index", None) in [0, 1]

        if action.mode_flag == "relative_to_position":
            assert getattr(action, "relative_to_position", None) is not None
            assert len(action.relative_to_position) == 3
            assert getattr(action, "relative_shift", None) is not None
            assert len(action.relative_to_position) == 3

        if action.mode_flag == "absolute":
            assert getattr(action, "at_position", None) is not None
            assert len(action.at_position) == 3

    # Check that all modes have appeared.
    expected_modes = set(
        k for k in AddMotifAction._flattened_mode_definitions.keys()
        if not k.startswith("_")
    )
    assert set(all_appeared_modes) == expected_modes
