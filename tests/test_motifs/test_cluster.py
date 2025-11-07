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
def test_registry():
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
    atoms_large = Atoms(
        "H" * 10, positions=np.random.rand(10, 3).tolist(), cell=cell, pbc=True
    )
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


def test_random_one_from_orig_atoms(orig_atoms):
    all_detected_indices = set()
    for _ in range(200):
        rng = np.random.default_rng(None)
        size = int(rng.integers(2, 5))
        cluster = ClusterMotif.detect_random_one(
            orig_atoms,
            cluster_size=size,
            max_cluster_radius=4.0,
            seed=None
        )
        assert len(cluster) == size
        assert cluster.radius <= 4.0
        all_detected_indices.add(tuple(sorted(cluster.indices)))
    assert len(all_detected_indices) > 1  # Ensure multiple unique clusters detected
    all_detected_lengths = set(len(indices) for indices in all_detected_indices)
    assert all_detected_lengths == {2, 3, 4}
    assert len(all_detected_indices) > 3  # Ensure multiple unique clusters detected
    # print(len(all_detected_indices))
    # print(all_detected_indices)
    # assert False # Temporary fail to inspect output


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


def test_detect_random_one_with_excluded_indices():
    """Test random cluster detection with excluded indices."""
    # Not using monkeypatch, test in a real atoms.
    atom = Atoms(
        "HHHH",
        positions=[(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0)],
        cell=np.eye(3) * 2,
        pbc=True
    )
    atom.set_initial_charges([0, 0, 0, 0])

    # Exclude index 1
    detected_possible_indices = []
    seed = 42
    for _ in range(10):
        cluster = ClusterMotif.detect_random_one(
            atom,
            cluster_size=2,
            max_cluster_radius=4.0,
            excluded_site_indices=[1],
            seed=seed,
            n_attempts=10
        )
        seed += 1
        # Check 1 not in cluster indices.
        assert 1 not in cluster.indices
        detected_possible_indices.append(sorted(cluster.indices))
    # Ensure we detect more than one unique combination.
    assert len(set(tuple(x) for x in detected_possible_indices)) > 1

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
    additive_cluster = ClusterMotif(
        atoms=simple_atoms[[0]],
    )

    # Single site in addition mode should return just the name
    desc = additive_cluster.describe(style="coord")
    assert desc == additive_cluster.name

    # Multi-site should still use full description
    single_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0]
    )
    desc_single = single_cluster.describe(style="coord")
    assert "coordinates" in desc_single


# Add these new tests to test_cluster.py

def test_detect_random_one_additive_mode(simple_atoms):
    """Test random cluster detection in additive mode."""
    cluster = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=3,
        max_cluster_radius=2.5,
        seed=42
    )

    # Verify additive mode properties
    assert cluster.is_additive is True
    assert cluster.indices is None
    assert cluster.cell_offsets is None
    assert cluster._atoms is not None
    assert len(cluster) == 3

    # Verify radius constraint
    assert cluster.radius <= 2.5

    # Verify atoms object properties
    assert len(cluster.get_atoms()) == 3
    assert cluster.get_atoms().cell is not None
    npt.assert_allclose(cluster.get_atoms().cell.complete(), simple_atoms.cell.complete())


def test_detect_random_one_additive_with_allowed_symbols(simple_atoms):
    """Test additive mode with specific allowed symbols."""
    allowed_symbols = ['Na', 'Cl']
    cluster = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        additive_mode_allowed_symbols=allowed_symbols,
        cluster_size=4,
        max_cluster_radius=3.0,
        seed=42
    )

    # Verify all symbols are from allowed list
    symbols = cluster.get_atoms().get_chemical_symbols()
    assert all(s in allowed_symbols for s in symbols)
    assert len(symbols) == 4


def test_detect_random_one_additive_randomize_symbols(simple_atoms):
    """Test additive mode with randomized symbols."""
    for _ in range(100):
        cluster = ClusterMotif.detect_random_one(
            simple_atoms,
            additive_mode=True,
            cluster_size=3,
            randomize_symbols=True,
        )

        # Verify cluster properties
        assert cluster.is_additive is True
        assert len(cluster) == 3

        # Verify symbols come from original atoms
        from ase.data import chemical_symbols
        original_symbols = set(chemical_symbols)
        cluster_symbols = set(cluster.get_atoms().get_chemical_symbols())
        assert cluster_symbols.issubset(original_symbols)
        assert "X" not in cluster_symbols  # No dummy atoms


def test_detect_random_one_additive_positions_within_sphere(simple_atoms):
    """Test that additive mode generates positions within the specified radius."""
    max_radius = 4.0
    cluster = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=3,
        max_cluster_radius=max_radius,
        seed=42
    )

    # Calculate all pairwise distances
    positions = cluster.get_atoms().get_positions()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            assert distance <= max_radius, f"Distance {distance} exceeds max_radius {max_radius}"


def test_detect_random_one_non_additive_vs_additive(simple_atoms):
    """Test that non-additive and additive modes produce different results."""
    # Non-additive mode
    non_additive = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=False,
        cluster_size=2,
        max_cluster_radius=3.0,
        seed=42
    )

    # Additive mode
    additive = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=2,
        seed=42
    )

    # Verify different properties
    assert non_additive.is_additive is False
    assert additive.is_additive is True
    assert non_additive.indices is not None
    assert additive.indices is None
    assert non_additive.in_atoms is not None
    assert additive.in_atoms is None


def test_additive_cluster_describe(simple_atoms):
    """Test describe method for additive clusters."""
    cluster = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=2,
        seed=42
    )

    # Additive clusters should force coord style
    desc = cluster.describe(style="index")  # Should be overridden to coord
    assert "coordinates" in desc.lower() or cluster.name in desc

    # Single atom additive should return name only
    single_cluster = cluster[0]
    desc_single = single_cluster.describe(style="coord")
    assert desc_single == single_cluster.name


def test_additive_cluster_operations(simple_atoms):
    """Test that additive clusters can be copied and extended."""
    cluster1 = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=2,
        seed=42
    )

    cluster2 = ClusterMotif.detect_random_one(
        simple_atoms,
        additive_mode=True,
        cluster_size=2,
        seed=43
    )

    # Test copy
    cluster_copy = cluster1.copy()
    assert cluster_copy.is_additive is True
    assert len(cluster_copy) == len(cluster1)

    # Test extend with another additive cluster
    original_len = len(cluster1)
    cluster1.extend(cluster2)
    assert len(cluster1) == original_len + len(cluster2)
    assert cluster1.is_additive is True
