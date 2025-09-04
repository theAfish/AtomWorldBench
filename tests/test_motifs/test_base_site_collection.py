"""Comprehensive pytest suite for BaseSiteCollectionMotif."""
import numpy as np
import numpy.testing as npt
import pytest
from collections import Counter

from ase import Atoms

from AtomWorldBench.atom_world.motifs.site_collections.cluster import (
    ClusterMotif,
)
from AtomWorldBench.atom_world.motifs.site_collections.site import (
    SiteMotif
)


# -----------------------------
# Fixtures
# -----------------------------

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



def test_init_basic(simple_atoms):
    """Test basic initialization through ClusterMotif."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1],
        name="test_motif"
    )
    assert len(cluster) == 2
    assert cluster.indices == [0, 1]
    assert cluster.name == "test_motif"
    wrap_atoms = simple_atoms.copy()
    wrap_atoms.wrap()
    assert cluster.in_atoms == wrap_atoms

def test_init_with_offsets(simple_atoms):
    """Test initialization with cell offsets."""
    offsets = np.array([[1, 0, 0], [0, 1, 0]])
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1],
        offsets=offsets
    )
    # Check that positions were updated with offsets
    expected_pos = simple_atoms.get_positions(wrap=False) + offsets @ simple_atoms.cell.complete()
    npt.assert_allclose(cluster.cart_coords, expected_pos)
    assert "pair" in cluster.name # Default name should reflect size

def test_init_point(simple_atoms):
    """Test initialization of a single-atom cluster."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[1]
    )
    assert len(cluster) == 1
    assert cluster.indices == [1]
    assert "point" in cluster.name

def test_init_without_offsets(simple_atoms):
    """Test initialization without offsets (should default to zeros)."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    npt.assert_allclose(cluster.cart_coords, simple_atoms.get_positions()[[0, 1]])

def test_species_strings(simple_atoms):
    """Test species_strings property."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    assert cluster.species_strings == ["Na+", "Cl-"]

def test_composition(simple_atoms):
    """Test composition property."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    expected = Counter({"Na+": 1, "Cl-": 1})
    assert cluster.composition == expected

def test_coordinates_properties(simple_atoms):
    """Test frac_coords, cart_coords, and cell_offsets properties."""
    # Apply a non-zero offset to simple atoms for testing.
    orig_positions = simple_atoms.get_positions(wrap=False).copy()
    simple_atoms_cp = simple_atoms.copy()
    simple_atoms_cp.set_positions(
        orig_positions +
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) @ simple_atoms.cell.complete()
    )  # Will be wrapped back into cell later.
    cluster = ClusterMotif(
        in_atoms=simple_atoms_cp,
        indices=[1, 0],
        offsets=[[0, 1, 0], [1, 0, 1]]
    )

    # Test fractional coordinates (unwrapped)
    expected_frac = simple_atoms_cp.get_scaled_positions(wrap=False)[[1, 0]]
    npt.assert_allclose(cluster.frac_coords, expected_frac)

    # Test Cartesian coordinates (unwrapped)
    expected_cart = simple_atoms_cp.get_positions(wrap=False)[[1, 0]]
    npt.assert_allclose(cluster.cart_coords, expected_cart)

    # Test cell offsets. Original atoms already wrapped, so offsets should match input.
    expected_offsets = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
    npt.assert_array_equal(cluster.cell_offsets, expected_offsets)


def test_centroid(simple_atoms):
    """Test centroid calculation in both coordinate systems."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    # Cartesian centroid
    cart_centroid = cluster.get_centroid(fractional=False)
    expected_cart = np.array([0.75, 0.0, 0.0])
    npt.assert_allclose(cart_centroid, expected_cart)

    # Fractional centroid
    frac_centroid = cluster.get_centroid(fractional=True)
    expected_frac = np.array([0.25, 0.0, 0.0])
    npt.assert_allclose(frac_centroid, expected_frac)


def test_radius(simple_atoms):
    """Test radius calculation."""
    # Single atom cluster
    single_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0]
    )
    assert single_cluster.radius == 0.0

    # Two atom cluster
    double_cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    expected_radius = 0.75  # Half the distance between the two atoms
    assert double_cluster.radius == pytest.approx(expected_radius)


def test_radius_complex(complex_atoms):
    """Test radius calculation in a more complex cluster."""
    cluster = ClusterMotif(
        in_atoms=complex_atoms,
        indices=[0, 1, 2]
    )
    assert "triplet" in cluster.name
    # Manually compute expected radius
    positions = complex_atoms.get_positions()[[0, 1, 2]]
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    expected_radius = np.max(distances)
    assert cluster.radius == pytest.approx(expected_radius)


def test_indices_property(simple_atoms):
    """Test indices getter and setter."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[1, 0]
    )
    assert cluster.indices == [1, 0]

def test_update_indices_offsets(simple_atoms):
    """Test updating indices and offsets."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    wrap_atoms = simple_atoms.copy()
    wrap_atoms.wrap()
    assert cluster.indices == [0, 1]
    npt.assert_allclose(cluster.cell_offsets, 0)
    assert cluster.in_atoms == wrap_atoms
    assert cluster.get_atoms() == wrap_atoms[[0, 1]]

    # Update with new indices and offsets
    new_offsets = np.array([[1, 0, 0], [0, 0, 1]])
    cluster.indices = [1, 0]
    assert cluster.indices == [1, 0]
    npt.assert_allclose(cluster.cell_offsets, 0)
    assert cluster.in_atoms == wrap_atoms
    assert cluster.get_atoms() == wrap_atoms[[1, 0]]
    cluster.cell_offsets = new_offsets
    assert cluster.indices == [1, 0]
    npt.assert_array_equal(cluster.cell_offsets, new_offsets)
    assert cluster.in_atoms == wrap_atoms
    wrap_atoms_offset = wrap_atoms.copy()[[1, 0]]
    wrap_atoms_offset.set_scaled_positions(
        wrap_atoms_offset.get_scaled_positions(wrap=False) + new_offsets
    )
    assert cluster.get_atoms() == wrap_atoms_offset

    # Try update cell offsets only.
    newer_offsets = np.array([[0, 1, 0], [0, 0, -1]])
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )
    cluster.cell_offsets = newer_offsets
    assert cluster.indices == [0, 1]
    npt.assert_array_equal(cluster.cell_offsets, newer_offsets)
    assert cluster.in_atoms == wrap_atoms
    wrap_atoms_offset = wrap_atoms.copy()[[0, 1]]
    wrap_atoms_offset.set_scaled_positions(
        wrap_atoms_offset.get_scaled_positions(wrap=False) + newer_offsets
    )
    assert cluster.get_atoms() == wrap_atoms_offset

    # Try update with wrong indices format.
    with pytest.raises(ValueError, match="Indices must be a list of integers."):
        cluster.indices = "not a list"
    with pytest.raises(ValueError, match="Indices must be a list of integers."):
        cluster.indices = [0, "a"]

    # Try update with wrong offsets format.
    with pytest.raises(ValueError, match="Cell offsets must be a 2D array with shape"):
        cluster.cell_offsets = [[0, 0], [1, 0]]
    with pytest.raises(ValueError, match="Cell offsets must have the same length as indices."):
        cluster.cell_offsets = [[0, 0, 0]]


def test_describe_coord_style(simple_atoms):
    """Test describe method with coordinate style."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    # Cartesian coordinates
    desc_cart = cluster.describe(style="coord", coord_fractional=False)
    assert "cartesian" in desc_cart.lower()

    # Fractional coordinates
    desc_frac = cluster.describe(style="coord", coord_fractional=True)
    assert "fractional" in desc_frac.lower()


def test_describe_index_style(simple_atoms):
    """Test describe method with index style."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    # With zero offsets
    desc = cluster.describe(style="index")
    assert "and offsets" not in desc.lower()

    # With non-zero offsets
    offsets = np.array([[1, 0, 0], [0, -1, 0]])
    cluster.cell_offsets = offsets
    desc_offsets = cluster.describe(style="index")
    assert "and offsets" in desc_offsets.lower()
    assert "((1, 0, 0), (0, -1, 0))" in desc_offsets


def test_describe_invalid_style(simple_atoms):
    """Test describe method with invalid style."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    with pytest.raises(NotImplementedError, match="Description style.*not implemented"):
        cluster.describe(style="invalid")


def test_get_atoms(simple_atoms):
    """Test conversion back to ASE Atoms object."""
    cluster = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1]
    )

    atoms = cluster.get_atoms()
    assert isinstance(atoms, Atoms)

    # Check that reserved arrays are not copied
    assert "site_indices" not in atoms.arrays

    # Check that other properties are preserved
    npt.assert_allclose(atoms.cell.array, cluster.in_atoms.cell.array)
    assert np.array_equal(atoms.pbc, cluster.in_atoms.pbc)
    npt.assert_allclose(atoms.get_positions(wrap=False), cluster.cart_coords)
    assert atoms.get_chemical_symbols() == cluster.in_atoms[cluster.indices].get_chemical_symbols()
    npt.assert_array_equal(
        atoms.get_initial_charges(),
        cluster.in_atoms[cluster.indices].get_initial_charges()
    )
    assert atoms == simple_atoms[cluster.indices]


def test_extend_and_add(simple_atoms):
    """Test extending and adding clusters."""
    cluster1 = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0],
        name="cluster1"
    )
    cluster2 = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[1],
        name="cluster2"
    )
    site2 = SiteMotif(
        in_atoms=simple_atoms,
        indices=[1]
    )

    # Test addition
    combined = cluster1 + cluster2
    print("combined indices:", combined.indices)
    assert len(combined) == 2
    assert "pair" in combined.name  # Name should be reset to default.

    # Test in-place addition
    cluster3 = cluster1.copy()
    cluster3 += cluster2
    assert len(cluster3) == 2
    assert "pair" in combined.name  # Name should be reset to default.

    combined2 = cluster1 + site2
    assert len(combined2) == 2
    assert isinstance(combined2, ClusterMotif)

    with pytest.raises(
            ValueError,
            match="SiteMotif must contain exactly one site, but got 2 sites."
    ):
        _ = site2 + cluster1


def test_extend_atoms_mismatch(simple_atoms, complex_atoms):
    """Test extend with mismatched indices."""
    cluster1 = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0]
    )
    cluster2 = ClusterMotif(
        in_atoms=complex_atoms,
        indices=[1]
    )

    with pytest.raises(ValueError, match="Can only extend motifs from the same original structure"):
        cluster1.extend(cluster2)


def test_copy(simple_atoms):
    """Test copying clusters."""
    original = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1],
        name="original"
    )

    copy_cluster = original.copy()
    assert copy_cluster == original
    assert copy_cluster is not original
    assert copy_cluster.indices == original.indices
    npt.assert_array_equal(copy_cluster.cell_offsets, original.cell_offsets)
    assert copy_cluster.in_atoms == original.in_atoms
    assert copy_cluster.name == original.name
    npt.assert_allclose(copy_cluster.cart_coords, original.cart_coords)


def test_getitem(complex_atoms):
    """Test slicing and indexing."""
    cluster = ClusterMotif(
        in_atoms=complex_atoms,
        indices=[0, 1],
        name="original"
    )

    # Single index
    sub_cluster = cluster[0]
    assert isinstance(sub_cluster, ClusterMotif)
    assert len(sub_cluster) == 1
    assert sub_cluster.indices == [0]
    assert "point" in sub_cluster.name  # Name should be reset to default.

    # Slice
    sub_cluster = cluster[1:]
    assert len(sub_cluster) == 1
    assert sub_cluster.indices == [1]

    assert cluster[[0, 1]] == cluster # Fancy indexing returns identical cluster


def test_equality(simple_atoms, complex_atoms):
    """Test equality comparison."""
    cluster1 = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1],
        name="cluster1",
    )
    cluster2 = ClusterMotif(
        in_atoms=simple_atoms,
        indices=[0, 1],
        name="cluster2",
    )
    cluster3 = ClusterMotif(
        in_atoms=complex_atoms,
        indices=[0, 1]
    )

    # Same clusters should be equal, even with different names.
    assert cluster1 == cluster2

    # Different source atoms should not be equal
    assert cluster1 != cluster3

    # Different class should not be equal
    assert cluster1 != "not a cluster"
