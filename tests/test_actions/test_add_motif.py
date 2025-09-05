"""Comprehensive test suite for AddMotifAction."""

import pytest
from numpy.ma.core import indices

from AtomWorldBench.atom_world.actions.motif_actions.add import AddMotifAction
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.common.registry import get_registered


def get_random_motif(class_alias, atoms, seed=42):
    """Helper function to get a random motif of a given class alias."""
    motif_class = get_registered(BaseMotif)[class_alias]
    assert isinstance(motif_class, BaseMotif)
    return motif_class.detect_random_one(atoms, seed=seed)


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
    return get_random_motif(request.param, orig_atoms)


@pytest