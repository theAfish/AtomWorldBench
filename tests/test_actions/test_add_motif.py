"""Comprehensive test suite for AddMotifAction."""

import pytest

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.add import AddMotifAction
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.common.registry import get_registered


def get_random_motif(class_alias, atoms, seed=42, **kwargs):
    """Helper function to get a random motif of a given class alias."""
    motif_class = get_registered(BaseMotif)[class_alias]
    assert issubclass(motif_class, BaseMotif)
    return motif_class.detect_random_one(atoms, seed=seed, **kwargs)


@pytest.fixture(
    params=["sphere", "bond"] # Region and Bond motif cannot be added to structure.
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
    params=["cluster", "site", "sphere", "bond"]
)
def allowed_relative_to_motif(request, orig_atoms):
    """Fixture to provide allowed relative_to motifs."""
    return get_random_motif(request.param, orig_atoms)


@pytest.fixture(
    params=["bond", "cluster"]
)
def bond_motif(request, orig_atoms):
    """Fixture to provide a bond motif."""
    return get_random_motif(request.param, orig_atoms, cluster_size=2)


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
