"""Comprehensive test suite for TranslateMotifAction."""

import pytest
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.translate import TranslateMotifAction
from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["translate-motif"]
    assert action_class is TranslateMotifAction
    action_class = get_registered(BaseMotifAction)["translate"]
    assert action_class is TranslateMotifAction


@pytest.fixture(params=["cluster", "site"])
def allowed_operated_motif(request, orig_atoms):
    """Fixture for allowed motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms, seed=42)
    assert isinstance(motif, BaseSiteCollectionMotif) and (not isinstance(motif, BondMotif))
    return motif


@pytest.fixture(params=["box", "sphere", "bond"])
def forbidden_operated_motif(request, orig_atoms):
    """Fixture for forbidden motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms, seed=42)
    return motif


@pytest.fixture(params=["cluster", "site"])
def allowed_relative_motif(request, orig_atoms):
    """Fixture for allowed motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms, seed=123)
    return motif


@pytest.fixture(params=["box", "sphere", "bond"])
def forbidden_relative_motif(request, orig_atoms):
    """Fixture for forbidden motifs."""
    motif_type = request.param
    motif = get_random_motif(motif_type, orig_atoms, seed=123)
    return motif


def test_translate_motif_absolute_cartesian(allowed_operated_motif):
    """Test absolute translation of motifs."""
    motif = allowed_operated_motif
    action = TranslateMotifAction(
        operated_motif=motif,
        to_position=np.array([1.0, 2.0, 3.0]),
        position_fractional=False,
    )
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center = motif.get_centroid(fractional=False)
    expected_motif_positions = (
        motif.cart_coords - motif_center + np.array([1.0, 2.0, 3.0])
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
    assert "so as to relocate its centroid at cartesian coordinates (1.000, 2.000, 3.000)" in desc
    assert "do not change their order in structure" in desc


def test_translate_motif_absolute_fractional(allowed_operated_motif):
    """Test absolute translation of motifs in fractional coordinates."""
    motif = allowed_operated_motif
    action = TranslateMotifAction(
        operated_motif=motif,
        to_position=np.array([0.25, 0.5, 0.75]),
        position_fractional=True,
    )
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    motif_center_frac = motif.get_centroid(fractional=True)
    lattice = motif.in_atoms.cell.complete()
    expected_motif_positions = (
        motif.frac_coords - motif_center_frac + np.array([0.25, 0.5, 0.75])
    ) @ lattice
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
    assert "so as to relocate its centroid at fractional coordinates (0.250, 0.500, 0.750)" in desc


def test_translate_motif_relative_to_motif(allowed_operated_motif, allowed_relative_motif):
    """Test absolute translation of motifs."""
    if np.allclose(
        allowed_operated_motif.get_centroid(fractional=False),
        allowed_relative_motif.get_centroid(fractional=False),
    ):
        pytest.skip("Operated and relative motifs have the same centroid; skipping test.")
    for d in [-0.1, 0.1]:
        motif = allowed_operated_motif
        action = TranslateMotifAction(
            operated_motif=motif,
            relative_to_motif=allowed_relative_motif,
            translation_vector=d
        )
        new_atoms = action.execute()
        motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
        other_indices = np.setdiff1d(
            np.arange(len(new_atoms)),
            motif.indices,
        )
        other_positions = new_atoms.get_positions(scale=False)[other_indices]
        motif_center = motif.get_centroid(fractional=False)
        relative_center = allowed_relative_motif.get_centroid(fractional=False)
        dv = relative_center - motif_center
        nv = dv / np.linalg.norm(dv)
        expected_motif_positions = (
            motif.cart_coords - nv * d
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
        op_word = "away from" if d > 0 else "towards"
        assert f"so as to move its centroid {d:.3f} angstroms {op_word} the centroid of" in desc
        assert "do not change their order in structure" in desc


def test_translate_motif_relative_to_position_cartesian(allowed_operated_motif):
    for d in [-0.1, 0.1]:
        motif = allowed_operated_motif
        action = TranslateMotifAction(
            operated_motif=motif,
            relative_to_position=np.array([1.0, 2.0, 3.0]),
            position_fractional=False,
            translation_vector=d
        )
        new_atoms = action.execute()
        motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
        other_indices = np.setdiff1d(
            np.arange(len(new_atoms)),
            motif.indices,
        )
        other_positions = new_atoms.get_positions(scale=False)[other_indices]
        motif_center = motif.get_centroid(fractional=False)
        relative_center = np.array([1.0, 2.0, 3.0])
        dv = relative_center - motif_center
        nv = dv / np.linalg.norm(dv)
        expected_motif_positions = (
                motif.cart_coords - nv * d
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
        op_word = "away from" if d > 0 else "towards"
        assert (
                f"so as to move its centroid {d:.3f} angstroms {op_word}"
                f" a reference point at cartesian coordinates"
                in desc
        )
        assert "do not change their order in structure" in desc


def test_translate_motif_relative_to_position_fractional(allowed_operated_motif):
    for d in [-0.1, 0.1]:
        motif = allowed_operated_motif
        action = TranslateMotifAction(
            operated_motif=motif,
            relative_to_position=np.array([0.25, 0.5, 0.75]),
            position_fractional=True,
            translation_vector=d
        )
        new_atoms = action.execute()
        motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
        other_indices = np.setdiff1d(
            np.arange(len(new_atoms)),
            motif.indices,
        )
        other_positions = new_atoms.get_positions(scale=False)[other_indices]
        motif_center = motif.get_centroid(fractional=False)
        relative_center = np.array([0.25, 0.5, 0.75]) @ motif.in_atoms.cell.complete()
        dv = relative_center - motif_center
        nv = dv / np.linalg.norm(dv)
        expected_motif_positions = (
                motif.cart_coords - nv * d
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
        op_word = "away from" if d > 0 else "towards"
        assert (
                f"so as to move its centroid {d:.3f} angstroms {op_word}"
                f" a reference point at fractional coordinates"
                in desc
        )
        assert "do not change their order in structure" in desc


def test_translate_motif_relative_to_self_cartesian(allowed_operated_motif):
    """Test error when relative_to_motif is the same as operated_motif."""
    motif = allowed_operated_motif
    action = TranslateMotifAction(
        operated_motif=motif,
        translation_vector=np.array([1.0, 2.0, 3.0]),
        position_fractional=False,
    )
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    expected_motif_positions = (
        motif.cart_coords + np.array([1.0, 2.0, 3.0])
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
    assert "by (1.000, 2.000, 3.000) in cartesian coordinates" in desc
    assert "do not change their order in structure" in desc


def test_translate_motif_relative_to_self_fractional(allowed_operated_motif):
    """Test error when relative_to_motif is the same as operated_motif."""
    motif = allowed_operated_motif
    action = TranslateMotifAction(
        operated_motif=motif,
        translation_vector=np.array([0.1, 0.2, 0.3]),
        position_fractional=True,
    )
    new_atoms = action.execute()
    motif_positions = new_atoms.get_positions(scale=False)[motif.indices]
    other_indices = np.setdiff1d(
        np.arange(len(new_atoms)),
        motif.indices,
    )
    other_positions = new_atoms.get_positions(scale=False)[other_indices]
    expected_motif_positions = (
        motif.cart_coords + np.array([0.1, 0.2, 0.3]) @ motif.in_atoms.cell.complete()
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
    assert "by (0.100, 0.200, 0.300) in fractional coordinates" in desc
    assert "do not change their order in structure" in desc


def test_forbidden_operated_motifs(forbidden_operated_motif):
    """Test that forbidden operated motifs raise errors."""
    motif = forbidden_operated_motif
    with pytest.raises(ValueError, match="must be a non-bond site collection motif"):
        TranslateMotifAction(
            operated_motif=motif,
            to_position=np.array([1.0, 2.0, 3.0]),
            position_fractional=False,
        )


def test_forbidden_relative_motifs(allowed_operated_motif, forbidden_relative_motif):
    """Test that forbidden relative motifs raise errors."""
    motif = allowed_operated_motif
    relative_motif = forbidden_relative_motif
    with pytest.raises(ValueError, match="must be a non-bond site collection motif"):
        TranslateMotifAction(
            operated_motif=motif,
            relative_to_motif=relative_motif,
            translation_vector=0.1
        )


def test_relative_to_motif_overlap(allowed_operated_motif):
    """Test that overlapping operated and relative motifs raise errors."""
    motif = allowed_operated_motif
    with pytest.raises(ValueError, match="The centroids of the operated motif and the relative motif are"):
        TranslateMotifAction(
            operated_motif=motif,
            relative_to_motif=motif,
            translation_vector=0.1
        )

def test_relative_to_position_overlap(allowed_operated_motif):
    """Test that overlapping operated motif and relative position raise errors."""
    motif = allowed_operated_motif
    motif_center = motif.get_centroid(fractional=False)
    with pytest.raises(ValueError, match="The centroid of the operated motif and the reference point are"):
        TranslateMotifAction(
            operated_motif=motif,
            relative_to_position=motif_center,
            position_fractional=False,
            translation_vector=0.1
        )


def test_relative_to_motif_too_long_inward(allowed_operated_motif, allowed_relative_motif):
    """Test that too long inward translation raises errors."""
    motif = allowed_operated_motif
    if np.allclose(
        motif.get_centroid(fractional=False),
        allowed_relative_motif.get_centroid(fractional=False),
    ):
        pytest.skip("Operated and relative motifs have the same centroid; skipping test.")
    with pytest.raises(ValueError, match="The translation distance to move inward is larger"):
        TranslateMotifAction(
            operated_motif=motif,
            relative_to_motif=allowed_relative_motif,
            translation_vector=-10000.0
        )


def test_relative_to_position_too_long_inward(allowed_operated_motif):
    """Test that too long inward translation raises errors."""
    motif = allowed_operated_motif
    motif_center = motif.get_centroid(fractional=False)
    with pytest.raises(ValueError, match="The translation distance to move inward is larger"):
        TranslateMotifAction(
            operated_motif=motif,
            relative_to_position=motif_center + 2.0,
            position_fractional=False,
            translation_vector=-10000.0
        )
