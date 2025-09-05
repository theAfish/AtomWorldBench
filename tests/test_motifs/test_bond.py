"""Comprehensive tests for bond motif"""
from ase import Atoms
import numpy as np
import numpy.testing as npt
import pytest

from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif

def test_resigstry():
    """Test that BondMotif is registered in the motif registry."""
    from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif
    from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
    from AtomWorldBench.common.registry import _REGISTRY
    motif_registry = _REGISTRY[BaseSiteCollectionMotif]
    assert 'bond' in motif_registry
    assert motif_registry['bond'] is BondMotif
    assert "bond-motif" in motif_registry
    assert motif_registry["bond-motif"] is BondMotif
    assert "bond_motif" in motif_registry
    assert motif_registry["bond_motif"] is BondMotif
    assert "bondmotif" in motif_registry
    assert motif_registry["bondmotif"] is BondMotif
    assert "BondMotif" in motif_registry
    assert motif_registry["BondMotif"] is BondMotif

@pytest.fixture
def simple_atoms():
    """2-atom cubic cell for basic tests."""
    cell = np.eye(3) * 3.0
    atoms = Atoms(
        "NaCl",
        positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        cell=cell,
        pbc=True
    )
    atoms.set_initial_charges([+1, -1])
    return atoms


def test_forbidden_actions(simple_atoms):
    """Test that forbidden actions are correctly set."""
    motif = BondMotif(simple_atoms, indices=[0, 1])
    expected_forbidden = [
        "add-motif", "remove-motif", "replace-motif", "rotate-motif", "translate-motif"
    ]
    assert set(motif.forbidden_actions) == set(expected_forbidden)


def test_post_init(simple_atoms):
    """Test that BondMotif raises ValueError if not exactly two sites."""

    # Valid case
    motif = BondMotif(simple_atoms, indices=[0, 1])
    assert len(motif) == 2

    # Invalid case: more than two sites
    with pytest.raises(ValueError, match="BondMotif must contain exactly two sites"):
        BondMotif(simple_atoms, indices=[0, 1, 0])

    # Invalid case: less than two sites
    with pytest.raises(ValueError, match="BondMotif must contain exactly two sites"):
        BondMotif(simple_atoms, indices=[0])


def test_default_name(simple_atoms):
    """Test that the default name is generated correctly."""
    motif = BondMotif(simple_atoms, indices=[0, 1])
    expected_name = "a bond between Na+ and Cl-"
    assert motif.name == expected_name


def test_from_cluster_motif(simple_atoms):
    """Test creating a BondMotif from a ClusterMotif."""
    from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif

    # Valid case
    cluster = ClusterMotif(simple_atoms, indices=[0, 1])
    bond = BondMotif.from_cluster_motif(cluster)
    assert isinstance(bond, BondMotif)
    assert bond.indices == [0, 1]
    npt.assert_allclose(bond.cart_coords, cluster.cart_coords)

    # Invalid case: ClusterMotif with more than two sites
    cluster_invalid = ClusterMotif(simple_atoms, indices=[0, 1, 0])
    with pytest.raises(ValueError, match="ClusterMotif must contain exactly two sites"):
        BondMotif.from_cluster_motif(cluster_invalid)

    # Invalid case: ClusterMotif with less than two sites
    cluster_invalid = ClusterMotif(simple_atoms, indices=[0])
    with pytest.raises(ValueError, match="ClusterMotif must contain exactly two sites"):
        BondMotif.from_cluster_motif(cluster_invalid)


def test_detect_random_one(simple_atoms):
    """Test random bond motif detection."""
    bond = BondMotif.detect_random_one(simple_atoms, seed=42)
    cluster = BondMotif.detect_random_one(simple_atoms, seed=42)
    assert isinstance(bond, BondMotif)
    assert len(bond) == 2
    assert set(bond.indices).issubset({0, 1})

    bond_from_cluster = BondMotif.from_cluster_motif(cluster)
    # Equivalence because bond generation just calls cluster generation internally.
    assert bond_from_cluster == bond
