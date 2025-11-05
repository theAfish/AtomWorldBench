"""Comprehensive test suite for ReplaceMotif action."""

import pytest
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.replace import ReplaceMotifAction
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


@pytest.fixture(params=["cluster", "site"])
def allowed_remove_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif_kwargs = {}
    if motif_class_alias == "sphere":
        motif_kwargs = {"radius": 4.0}
    # Not in additive mode.
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123, **motif_kwargs)
    assert not hasattr(motif, "is_additive") or not motif.is_additive
    assert motif.in_atoms == orig_atoms
    return motif

@pytest.fixture(params=["cluster", "site"])
def additive_remove_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123, additive_mode=True)
    return motif

@pytest.fixture(params=["bond", "sphere", "box"])
def forbidden_remove_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123)
    return motif


@pytest.fixture(
    params=["cluster", "site"]
)
def allowed_added_motif(request, orig_atoms):
    """Fixture to provide allowed operated motifs."""
    return get_random_motif(request.param, orig_atoms, additive_mode=True)

@pytest.fixture(
    params=["cluster", "site"]
)
def non_additive_motif(request, orig_atoms):
    """Fixture to provide allowed operated motifs."""
    return get_random_motif(request.param, orig_atoms, additive_mode=False)


@pytest.fixture(
    params=["box", "sphere", "bond"] # Region and Bond motif cannot be added to structure.
)
def forbidden_added_motif(request, orig_atoms):
    """Fixture to provide forbidden operated motifs."""
    return get_random_motif(request.param, orig_atoms)


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["replace-motif"]
    assert action_class is ReplaceMotifAction
    action_class = get_registered(BaseMotifAction)["replace"]
    assert action_class is ReplaceMotifAction


def test_replace_motif_action(allowed_added_motif, allowed_remove_motif):
    """Test initialization of ReplaceMotifAction."""
    action = ReplaceMotifAction(
        operated_motif=allowed_added_motif,
        relative_to_motif=allowed_remove_motif,
    )
    assert isinstance(action, ReplaceMotifAction)
    assert isinstance(action, BaseMotifAction)
    assert action.operated_motif is allowed_added_motif
    assert action.replaced_motif is allowed_remove_motif
    assert action.mode_flag == "default"

    action._check_relative_motif_in_atoms()
    assert allowed_added_motif.is_additive
    assert not allowed_remove_motif.is_additive
    new_atoms = action.execute()
    remove_indices = np.unique(allowed_remove_motif.indices)
    assert len(new_atoms) == len(action.operated_atoms) - len(remove_indices) + len(allowed_added_motif)
    remaining_indices = np.sort(np.setdiff1d(
        np.arange(len(action.operated_atoms)),
        allowed_remove_motif.indices
    ))
    # Added motif positions should be translated to the centroid of the removed motif.
    # New atomic order is remaining atoms followed by added motif atoms.
    removed_centroid = allowed_remove_motif.get_centroid(fractional=False)
    added_centroid = allowed_added_motif.get_centroid(fractional=False)
    translation_vector = removed_centroid - added_centroid
    expected_positions = np.vstack((
        action.operated_atoms.get_positions(wrap=False)[remaining_indices],
        allowed_added_motif.cart_coords + translation_vector
    ))
    npt.assert_allclose(
        new_atoms.get_positions(wrap=False),
        expected_positions
    )
    original_chemical_symbols = action.operated_atoms.get_chemical_symbols()
    expected_chemical_symbols = (
        [original_chemical_symbols[ii] for ii in remaining_indices] +
        allowed_added_motif.get_atoms().get_chemical_symbols()
    )
    assert (
        new_atoms.get_chemical_symbols() == expected_chemical_symbols
    )


def test_replace_motif_action_add_forbidden(forbidden_added_motif, allowed_remove_motif):
    """Test that forbidden motifs raise errors in ReplaceMotifAction."""
    with pytest.raises(ValueError, match="must be a non-bond site collection motif."):
        ReplaceMotifAction(
            operated_motif=forbidden_added_motif,
            relative_to_motif=allowed_remove_motif,
        )


def test_replace_motif_action_add_non_additive(non_additive_motif, allowed_remove_motif):
    """Test that non-additive motifs raise errors in ReplaceMotifAction."""
    with pytest.raises(ValueError, match="Inserted motif must be in is_additive mode."):
        ReplaceMotifAction(
            operated_motif=non_additive_motif,
            relative_to_motif=allowed_remove_motif,
        )


def test_replace_motif_action_remove_forbidden(allowed_added_motif, forbidden_remove_motif):
    """Test that forbidden motifs raise errors in ReplaceMotifAction."""
    with pytest.raises(ValueError, match="must be a non-bond site collection motif."):
        ReplaceMotifAction(
            operated_motif=allowed_added_motif,
            relative_to_motif=forbidden_remove_motif,
        )

def test_replace_motif_action_remove_additive(allowed_added_motif, additive_remove_motif):
    """Test that additive motifs raise errors in ReplaceMotifAction."""
    with pytest.raises(ValueError, match="The motif to be replaced must NOT be in is_additive mode."):
        ReplaceMotifAction(
            operated_motif=allowed_added_motif,
            relative_to_motif=additive_remove_motif,
        )