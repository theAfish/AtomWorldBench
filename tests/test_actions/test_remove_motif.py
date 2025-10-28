"""Comprehensive test suite for RemoveMotif action."""

import pytest
import numpy as np
import numpy.testing as npt
from ase import Atoms

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.remove import RemoveMotifAction
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.atom_world.motifs.regions.base import BaseRegionMotif
from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif

from ..utils import get_random_motif


# All types of motifs can be removed.
@pytest.fixture(params=["cluster", "site", "box", "sphere"])
def allowed_remove_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif_kwargs = {}
    if motif_class_alias == "sphere":
        motif_kwargs = {"radius": 4.0}
    # Not in additive mode.
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123, **motif_kwargs)
    assert not hasattr(motif, "is_additive") or not motif.is_additive
    return motif


@pytest.fixture(params=["bond"])
def forbidden_remove_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123)
    return motif


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["remove-motif"]
    assert action_class is RemoveMotifAction
    action_class = get_registered(BaseMotifAction)["remove"]
    assert action_class is RemoveMotifAction


def test_remove_motif_action(allowed_remove_motif):
    """Test initialization of RemoveMotifAction."""
    action = RemoveMotifAction(operated_motif=allowed_remove_motif)
    assert isinstance(action, RemoveMotifAction)
    assert isinstance(action, BaseMotifAction)
    assert action.operated_motif is allowed_remove_motif
    assert action.mode_flag == "default"

    action._check_operated_motif_in_atoms()
    new_atoms = action.execute()
    remove_indices = np.unique(allowed_remove_motif.indices)
    assert len(new_atoms) == len(action.operated_atoms) - len(remove_indices)
    remaining_indices = np.sort(np.setdiff1d(
        np.arange(len(action.operated_atoms)),
        allowed_remove_motif.indices
    ))
    npt.assert_array_equal(
        new_atoms.get_positions(wrap=False),
        action.operated_atoms.get_positions(wrap=False)[remaining_indices]
    )
    original_chemical_symbols = action.operated_atoms.get_chemical_symbols()
    assert (
        new_atoms.get_chemical_symbols() ==
        [original_chemical_symbols[ii] for ii in remaining_indices]
    )

    desc = action.describe()
    assert "remove" in desc
    assert "from the structure. do not change the order of remaining atoms" in desc


def test_remove_motif_action_forbidden(forbidden_remove_motif):
    """Test that forbidden motifs raise errors in RemoveMotifAction."""
    with pytest.raises(ValueError, match="cannot be removed directly"):
        RemoveMotifAction(operated_motif=forbidden_remove_motif)



