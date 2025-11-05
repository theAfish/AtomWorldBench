"""Comprehensive test suite for SwapMotif action."""

from ase import Atoms
import pytest
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.actions.motif_actions.base import BaseMotifAction
from AtomWorldBench.atom_world.actions.motif_actions.swap import SwapMotifAction
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif
from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif

from AtomWorldBench.common.registry import get_registered

from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseMotifAction)["swap-motif"]
    assert action_class is SwapMotifAction
    action_class = get_registered(BaseMotifAction)["swap"]
    assert action_class is SwapMotifAction


@pytest.fixture(params=["cluster", "site"])
def allowed_motifs_pair(request, orig_atoms):
    motif_class_alias = request.param
    if motif_class_alias == "cluster":
        n_choice1 = max(1, len(orig_atoms) // 4)
        indices1 = np.random.choice(len(orig_atoms), size=n_choice1, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(orig_atoms)), indices1)
        n_choice2 = max(1, len(remaining_indices) // 4)
        indices2 = np.random.choice(remaining_indices, size=n_choice2, replace=False)
        return (
            ClusterMotif(
                in_atoms=orig_atoms,
                indices=indices1,
            ),
            ClusterMotif(
                in_atoms=orig_atoms,
                indices=indices2,
            )
        )
    elif motif_class_alias == "site":
        n_choice1 = 1
        indices1 = np.random.choice(len(orig_atoms), size=n_choice1, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(orig_atoms)), indices1)
        n_choice2 = 1
        indices2 = np.random.choice(remaining_indices, size=n_choice2, replace=False)
        return (
            SiteMotif(
                in_atoms=orig_atoms,
                indices=indices1,
            ),
            SiteMotif(
                in_atoms=orig_atoms,
                indices=indices2,
            )
        )
    else:
        raise ValueError(f"Unknown motif class alias: {motif_class_alias}")

@pytest.fixture
def allowed_motif1(allowed_motifs_pair):
    return allowed_motifs_pair[0]

@pytest.fixture
def allowed_motif2(allowed_motifs_pair):
    return allowed_motifs_pair[1]


# This will fail at initialization.
@pytest.fixture(params=["cluster", "site"])
def overlapping_motif_pair(request, orig_atoms):
    motif_class_alias = request.param
    if motif_class_alias == "cluster":
        n_choice1 = max(1, len(orig_atoms) // 4)
        indices1 = np.random.choice(len(orig_atoms), size=n_choice1, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(orig_atoms)), indices1)
        n_choice2 = max(1, n_choice1 // 2)
        indices2_rest = np.random.choice(remaining_indices, size=n_choice2, replace=False)
        overlap_indices = np.random.choice(indices1, size=1, replace=False)
        indices2 = np.concatenate((overlap_indices, indices2_rest)).tolist()
        return (
            ClusterMotif(
                in_atoms=orig_atoms,
                indices=indices1,
            ),
            ClusterMotif(
                in_atoms=orig_atoms,
                indices=indices2,
            )
        )
    elif motif_class_alias == "site":
        indices1 = np.random.choice(len(orig_atoms), size=1, replace=False)
        indices2 = indices1
        return (
            SiteMotif(
                in_atoms=orig_atoms,
                indices=indices1,
            ),
            SiteMotif(
                in_atoms=orig_atoms,
                indices=indices2,
            )
        )
    else:
        raise ValueError(f"Unknown motif class alias: {motif_class_alias}")


# This will fail at initialization.
@pytest.fixture(params=["bond", "sphere", "box"])
def forbidden_motif(request, orig_atoms):
    motif_class_alias = request.param
    motif = get_random_motif(motif_class_alias, orig_atoms, seed=123)
    return motif

# This will fail at initialization.
@pytest.fixture(params=["cluster", "site"])
def motif_from_another_atoms(request):
    motif_class_alias = request.param
    other_atoms = Atoms(
        symbols="Nb3",
        positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)],
        cell=[5, 5, 5]
    )
    motif = get_random_motif(motif_class_alias, other_atoms, seed=123)
    return motif


def test_swap_motif_action(allowed_motif1, allowed_motif2):
    """Test the SwapMotifAction with valid motifs."""
    action = SwapMotifAction(
        operated_motif=allowed_motif1,
        relative_to_motif=allowed_motif2,
    )
    assert isinstance(action, SwapMotifAction)
    assert isinstance(action, BaseMotifAction)
    assert action.operated_motif is allowed_motif1
    assert action.relative_to_motif is allowed_motif2
    assert action.mode_flag == "default"
    new_atoms = action.execute()

    # Check that the positions of the motifs have been swapped.
    centroid_1 = allowed_motif1.get_centroid(fractional=False)
    centroid_2 = allowed_motif2.get_centroid(fractional=False)
    print("centroid 1:", centroid_1)
    print("centroid 2:", centroid_2)
    print("coords1:", allowed_motif1.cart_coords)
    print("coords2:", allowed_motif2.cart_coords)

    expected_positions_1 = allowed_motif1.cart_coords - centroid_1 + centroid_2
    expected_positions_2 = allowed_motif2.cart_coords - centroid_2 + centroid_1

    # Require that the order of atoms does not change after swap operation.
    actual_positions_1 = new_atoms[allowed_motif1.indices].get_positions(wrap=False)
    actual_positions_2 = new_atoms[allowed_motif2.indices].get_positions(wrap=False)
    print("expected positions 1:", expected_positions_1)
    print("actual positions 1:", actual_positions_1)
    print("expected positions 2:", expected_positions_2)
    print("actual positions 2:", actual_positions_2)

    npt.assert_allclose(expected_positions_1, actual_positions_1)
    npt.assert_allclose(expected_positions_2, actual_positions_2)

    other_indices = np.setdiff1d(
        np.arange(len(action.operated_atoms)),
        np.concatenate((allowed_motif1.indices, allowed_motif2.indices))
    )
    npt.assert_allclose(
        action.operated_atoms[other_indices].get_positions(wrap=False),
        new_atoms[other_indices].get_positions(wrap=False)
    )

    # Check that chemical symbols remain unchanged.
    npt.assert_array_equal(
        action.operated_atoms.get_chemical_symbols(),
        new_atoms.get_chemical_symbols()
    )


def test_init_forbidden_motif(allowed_motif1, forbidden_motif):
    """Test that initializing SwapMotifAction with forbidden motifs raises an error."""
    with pytest.raises(ValueError, match="must be a non-bond site collection motif."):
        SwapMotifAction(
            operated_motif=allowed_motif1,
            relative_to_motif=forbidden_motif,
        )
    with pytest.raises(ValueError, match="must be a non-bond site collection motif."):
        SwapMotifAction(
            operated_motif=forbidden_motif,
            relative_to_motif=allowed_motif1
        )

def test_init_motif_from_another_atoms(allowed_motif1, motif_from_another_atoms):
    """Test that initializing SwapMotifAction with motifs from different atoms raises an error."""
    with pytest.raises(ValueError, match="must be attached to the same Atoms object."):
        SwapMotifAction(
            operated_motif=allowed_motif1,
            relative_to_motif=motif_from_another_atoms,
        )
    with pytest.raises(ValueError, match="must be attached to the same Atoms object."):
        SwapMotifAction(
            operated_motif=motif_from_another_atoms,
            relative_to_motif=allowed_motif1
        )

def test_init_same_motif(allowed_motif1):
    """Test that initializing SwapMotifAction with the same motif raises an error."""
    with pytest.raises(ValueError, match="The two motifs to swap must not share any atoms."):
        SwapMotifAction(
            operated_motif=allowed_motif1,
            relative_to_motif=allowed_motif1,
        )

def test_init_overlapping_motifs(overlapping_motif_pair):
    """Test that initializing SwapMotifAction with overlapping motifs raises an error."""
    motif1, motif2 = overlapping_motif_pair
    with pytest.raises(ValueError, match="The two motifs to swap must not share any atoms."):
        SwapMotifAction(
            operated_motif=motif1,
            relative_to_motif=motif2,
        )
