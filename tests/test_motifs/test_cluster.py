"""Comprehensive pytest suite for and ClusterMotif."""
import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch, MagicMock

from ase import Atoms

from AtomWorldBench.atom_world.motifs.site_collections.cluster import (
    ClusterMotif,
    detect_neighbor_sites_around_site_index
)
from AtomWorldBench.atom_world.motifs.site_collections.site import (
    SiteMotif
)


# -----------------------------
# Fixtures
# -----------------------------
def test_resigstry():
    """Test that ClusterMotif is registered in the motif registry."""
    from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif
    from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
    from AtomWorldBench.common.registry import _REGISTRY
    motif_registry = _REGISTRY[BaseSiteCollectionMotif]
    assert 'cluster' in motif_registry
    assert motif_registry['cluster'] is ClusterMotif
    assert "atom-cluster" in motif_registry
    assert motif_registry["atom-cluster"] is ClusterMotif
    assert "cluster-motif" in motif_registry
    assert motif_registry["cluster-motif"] is ClusterMotif
    assert "cluster_motif" in motif_registry
    assert motif_registry["cluster_motif"] is ClusterMotif
    assert "clustermotif" in motif_registry
    assert motif_registry["clustermotif"] is ClusterMotif
    assert "ClusterMotif" in motif_registry
    assert motif_registry["ClusterMotif"] is ClusterMotif
    assert "clus" not in motif_registry
    assert "Clustermotif" not in motif_registry


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


@pytest.fixture
def complex_atoms():
    """More complex structure for advanced tests."""
    cell = np.eye(3) * 5.0
    atoms = Atoms(
        "NaClNaCl",
        positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 2.0, 0.0), (1.5, 2.0, 0.0)],
        cell=cell,
        pbc=True
    )
    atoms.set_initial_charges([+1, -1, +1, -1])
    return atoms


@pytest.fixture
def atoms_with_offsets():
    """Atoms structure that will test cell offsets functionality."""
    cell = np.eye(3) * 2.0
    atoms = Atoms(
        "HH",
        positions=[(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)],  # Second atom outside unit cell
        cell=cell,
        pbc=True
    )
    atoms.set_initial_charges([0, 0])
    return atoms


def test_post_init_empty_cluster(simple_atoms):
    """Test that empty clusters are not allowed."""
    with pytest.raises(ValueError, match="ClusterMotif must contain at least one atom"):
        ClusterMotif(
            in_atoms=simple_atoms,
            indices=[]
        )

def test_default_names():
    """Test default name generation for different cluster sizes."""
    cell = np.eye(3) * 3.0

    # Point (1 atom)
    atoms1 = Atoms("H", positions=[(0, 0, 0)], cell=cell, pbc=True)
    cluster1 = ClusterMotif(in_atoms=atoms1, indices=[0])
    assert cluster1.name == "a point of atoms/species H"

    # Pair (2 atoms)
    atoms2 = Atoms("HH", positions=[(0, 0, 0), (1, 0, 0)], cell=cell, pbc=True)
    cluster2 = ClusterMotif(in_atoms=atoms2, indices=[0, 1])
    assert cluster2.name == "a pair of atoms/species H, H"

    # Triplet (3 atoms)
    atoms3 = Atoms("HCH", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], cell=cell, pbc=True)
    cluster3 = ClusterMotif(in_atoms=atoms3, indices=[0, 1, 2])
    assert cluster3.name == "a triplet of atoms/species H, C, H"

    # Quadruplet (4 atoms)
    atoms4 = Atoms(
        "HHHH",
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        cell=cell,
        pbc=True
    )
    cluster4 = ClusterMotif(in_atoms=atoms4, indices=[0, 1, 2, 3])
    assert "quadruplet" in cluster4._get_default_name()

    # Large cluster
    atoms_large = Atoms("H" * 10, positions=np.random.rand(10, 3), cell=cell, pbc=True)
    cluster_large = ClusterMotif(in_atoms=atoms_large, indices=list(range(10)))
    assert "10-sites cluster" in cluster_large._get_default_name()


def test_site_motifs_property(simple_atoms):
    """Test site_motifs property."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    site_motifs = cluster.site_motifs
    assert len(site_motifs) == 2

    # Check that each site motif has correct properties
    for i, site_motif in enumerate(site_motifs):
        assert isinstance(site_motif, SiteMotif)
        assert site_motif.indices == [cluster.indices[i]]
        npt.assert_array_equal(site_motif.cell_offsets, [cluster.cell_offsets[i]])
        npt.assert_allclose(site_motif.cart_coords, cluster.cart_coords[[i]])


def test_detect_random_one_success(monkeypatch, simple_atoms):
    # mock neighbor site.
    mock_site = MagicMock(spec=SiteMotif)
    mock_site.indices = [1]
    mock_site.in_atoms = simple_atoms
    mock_site.cell_offsets = np.array([[0, 0, 0]])

    with (
        patch('AtomWorldBench.atom_world.motifs.site_collections.cluster.detect_neighbor_sites_around_site_index')
        as mock_detect
    ):
        mock_detect.return_value = [mock_site]

        # fix radius to 2.0 (effective in this test only)
        monkeypatch.setattr(ClusterMotif, 'radius', property(lambda self: 2.0))

        cluster = ClusterMotif.detect_random_one(
            simple_atoms, cluster_size=2, max_cluster_radius=3.0, seed=42
        )
        assert cluster.radius == 2.0
        mock_detect.assert_called()


@patch('AtomWorldBench.atom_world.motifs.site_collections.cluster.detect_neighbor_sites_around_site_index')
def test_detect_random_one_failure(mock_detect, simple_atoms):
    """Test failure case for random cluster detection."""
    # Mock to return no neighbors
    mock_detect.return_value = []

    with pytest.raises(RuntimeError, match="Failed to detect a valid"):
        ClusterMotif.detect_random_one(
            simple_atoms,
            cluster_size=3,
            n_attempts=2,
            max_cluster_radius=1.0
        )


@patch('AtomWorldBench.atom_world.motifs.site_collections.cluster.detect_neighbor_sites_around_site_index')
def test_detect_random_one_with_symbols(mock_detect, simple_atoms):
    """Test random cluster detection with symbol filtering."""
    mock_detect.return_value = []

    try:
        ClusterMotif.detect_random_one(
            simple_atoms,
            cluster_size=2,
            randomize_symbols=True,
            seed=42,
            n_attempts=1
        )
    except RuntimeError:
        pass  # Expected to fail, but we want to check the call

    # Verify that symbols were passed to the detection function
    assert mock_detect.call_count > 0


def test_detect_random_one_radius_filter(monkeypatch, simple_atoms):
    """Test that clusters exceeding max radius are filtered out."""
    mock_site = MagicMock(spec=SiteMotif)
    mock_site.indices = [1]
    mock_site.in_atoms = simple_atoms
    mock_site.cell_offsets = np.array([[0, 0, 0]])

    with (
        patch('AtomWorldBench.atom_world.motifs.site_collections.cluster.detect_neighbor_sites_around_site_index')
        as mock_detect
    ):
        mock_detect.return_value = [mock_site]

        # fix radius to 10.0, exceed detection range (effective in this test only)
        monkeypatch.setattr(ClusterMotif, 'radius', property(lambda self: 10.0))

        with pytest.raises(RuntimeError, match="Failed to detect a valid"):
            _ = ClusterMotif.detect_random_one(
                simple_atoms, cluster_size=2, max_cluster_radius=3.0, seed=42, n_attempts=10
            )

# -----------------------------
# Tests for detect_neighbor_sites_around_site_index
# -----------------------------
def test_detect_with_dummy_atom_error():
    """Test error when structure contains dummy atom 'X'."""
    atoms = Atoms("HX", positions=[(0, 0, 0), (0.5, 0, 0)], cell=np.eye(3), pbc=True)

    with pytest.raises(ValueError, match="already contains a dummy atom with symbol 'X'"):
        detect_neighbor_sites_around_site_index(atoms, site_index=0, cutoff=1.0)


@patch('AtomWorldBench.atom_world.motifs.site_collections.cluster.detect_indices_offsets_around_frac_coords')
def test_detect_basic_functionality(mock_detect_indices):
    """Test basic neighbor detection functionality."""
    atoms = Atoms("HHH", positions=[(0, 0, 0), (0.5, 0, 0), (2.0, 0, 0)], cell=np.eye(3), pbc=True)
    atoms.set_initial_charges([0, 0, 0])

    # Mock the low-level detection function
    mock_detect_indices.return_value = (
        np.array([0, 1, 2]),  # indices
        np.zeros((3, 3))  # offsets
    )

    result = detect_neighbor_sites_around_site_index(atoms, site_index=1, cutoff=1.0)

    # Should exclude the site itself (index 1)
    assert len(result) == 2

    # Check that indices are properly converted to integers
    for site_motif in result:
        assert isinstance(site_motif.indices[0], (int, np.integer))
        assert site_motif.indices[0] != 1  # Should not include the passed in site


def test_detect_with_symbol_filtering():
    """Test neighbor detection with symbol filtering."""
    atoms = Atoms(
        "HHCH",
        positions=[(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0)],
        cell=np.eye(3) * 2,
        pbc=True
    )
    atoms.set_initial_charges([0, 0, 0, 0])

    result = detect_neighbor_sites_around_site_index(
        atoms,
        site_index=1,
        cutoff=0.6,
        symbols=["H"]
    )
    assert len(result) == 1
    all_result_indices = [site.indices[0] for site in result]
    # Should only include site 0 as it is the only H within cutoff.
    assert set(all_result_indices) == {0}


# -----------------------------
# Integration Tests
# -----------------------------
def test_cluster_workflow(complex_atoms):
    """Test a complete workflow with ClusterMotif."""
    # Create a cluster
    cluster = ClusterMotif(
        in_atoms=complex_atoms,
        indices=[0, 1, 2],
        name="test_cluster"
    )

    # Test basic properties
    assert len(cluster) == 3
    assert cluster.name == "test_cluster"

    # Test site motifs
    site_motifs = cluster.site_motifs
    assert len(site_motifs) == 3

    # Test copy and equality
    cluster_copy = cluster.copy()
    assert cluster_copy == cluster

    # Test describe
    desc = cluster.describe(style="coord")
    assert "test_cluster" in desc

    # Test conversion to atoms
    atoms = cluster.get_atoms()
    assert len(atoms) == 3


def test_motif_with_offsets_workflow(atoms_with_offsets):
    """Test workflow with cell offsets."""
    offsets = np.array([[0, 0, 0], [1, 0, 0]])  # Second atom is in next unit cell

    cluster = ClusterMotif(
        in_atoms=atoms_with_offsets,
        indices=[0, 1],
        offsets=offsets
    )

    # Check that offsets and pre_init wrap are properly applied
    expected_offsets = np.array([[0, 0, 0], [1, 0, 0]])
    npt.assert_array_equal(cluster.cell_offsets, expected_offsets)

    # Test describe with offsets
    desc = cluster.describe(style="index")
    assert "offsets" in desc


def test_edge_cases(simple_atoms):
    """Test various edge cases."""
    # Single atom cluster
    single_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0]
    )
    assert len(single_cluster) == 1
    assert single_cluster.radius == 0.0


def test_describe_addition_mode(simple_atoms):
    """Test describe method in addition mode."""
    single_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0]
    )

    # Single site in addition mode should return just the name
    desc = single_cluster.describe(style="coord", is_addition=True)
    assert desc == single_cluster.name

    # Multi-site should still use full description
    multi_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    desc_multi = multi_cluster.describe(style="coord", is_addition=True)
    assert "coordinates" in desc_multi
