import numpy as np
import numpy.testing as npt
import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motifs.regions.sphere import SphereRegionMotif


def test_registry():
    """Test that SphereRegionMotif is registered correctly."""
    from AtomWorldBench.atom_world.motifs.regions.base import BaseRegionMotif
    from AtomWorldBench.atom_world.motifs.base import BaseMotif
    from AtomWorldBench.common.registry import get_registered

    region_classes = get_registered(BaseRegionMotif)
    motif_classes = get_registered(BaseMotif)

    assert "sphere" in region_classes
    assert "sphere" in motif_classes
    assert region_classes["sphere"] is SphereRegionMotif
    assert motif_classes["sphere"] is SphereRegionMotif

    assert "sphere-region" in region_classes
    assert "sphere-region" in motif_classes
    assert region_classes["sphere-region"] is SphereRegionMotif
    assert motif_classes["sphere-region"] is SphereRegionMotif

# Test input validations and error handling.
def test_invalid_radius_negative(orig_atoms):
    """Test that negative radius raises ValueError."""
    with pytest.raises(ValueError, match="radius must be a positive number"):
        SphereRegionMotif(orig_atoms, radius=-1.0, center=[0, 0, 0])

def test_invalid_radius_zero(orig_atoms):
    """Test that zero radius raises ValueError."""
    with pytest.raises(ValueError, match="radius must be a positive number"):
        SphereRegionMotif(orig_atoms, radius=0.0, center=[0, 0, 0])

def test_invalid_center_id_negative(orig_atoms):
    """Test that negative center_id raises ValueError."""
    with pytest.raises(ValueError, match="center_id .* must be a non-negative integer"):
        SphereRegionMotif(orig_atoms, radius=5.0, center_id=-1)

def test_invalid_center_id_out_of_bounds(orig_atoms):
    """Test that center_id beyond atoms length raises ValueError."""
    with pytest.raises(ValueError, match="center_id .* is out of bounds"):
        SphereRegionMotif(orig_atoms, radius=5.0, center_id=len(orig_atoms))

def test_invalid_center_coordinates_wrong_shape(orig_atoms):
    """Test that wrong shape center coordinates raise ValueError."""
    with pytest.raises(ValueError):
        SphereRegionMotif(orig_atoms, radius=5.0, center=[0, 0])  # Only 2 coords

    with pytest.raises(ValueError):
        SphereRegionMotif(orig_atoms, radius=5.0, center=[0, 0, 0, 0])  # 4 coords


@pytest.fixture
def sphere_motif_fractional(orig_atoms):
    """Fixture for a SphereRegionMotif instance."""
    return SphereRegionMotif(
        orig_atoms,
        radius=4.0,
        center=[0.5, 0.5, 0.5],
        center_is_fractional=True
    )


@pytest.fixture
def sphere_motif_cartesian(orig_atoms):
    """Fixture for a SphereRegionMotif instance."""
    cart_center = np.array([0.5, 0.5, 0.5]) @ orig_atoms.cell.complete()
    return SphereRegionMotif(
        orig_atoms,
        radius=4.0,
        center=cart_center,
        center_is_fractional=False
    )


@pytest.fixture
def sphere_motif_index(orig_atoms):
    """Fixture for a SphereRegionMotif instance."""
    return SphereRegionMotif(
        orig_atoms,
        radius=4.0,
        center_id=len(orig_atoms) // 2  # Center around and atom.
    )


@pytest.fixture
def sphere_motif(sphere_motif_fractional):
    """Fixture for a SphereRegionMotif instance.

    Using fractional coordinates for consistency in tests.
    """
    return sphere_motif_fractional


@pytest.fixture
def sphere_motif_partial(orig_atoms):
    """Fixture for a SphereRegionMotif instance with partial symbols."""
    symbols = list(set(orig_atoms.get_chemical_symbols()))
    return SphereRegionMotif(
        orig_atoms,
        radius=4.0,
        center=[0.5, 0.5, 0.5],
        center_is_fractional=True,
        symbols=symbols[:len(symbols) // 2]  # Only use half of the symbols.
    )


def test_mode_flag(
        sphere_motif_fractional,
        sphere_motif_cartesian,
        sphere_motif_index
):
    """Test mode flag assignment."""
    assert sphere_motif_fractional.mode_flag == "center_around_coordinates"
    assert sphere_motif_cartesian.mode_flag == "center_around_coordinates"
    assert sphere_motif_index.mode_flag == "center_around_atom_index"


def test_get_centroid(sphere_motif):
    """Test centroid calculation in Cartesian coordinates."""
    centroid_frac = sphere_motif.get_centroid(fractional=True)
    assert isinstance(centroid_frac, np.ndarray)
    assert centroid_frac.shape == (3,)
    centroid_cart = sphere_motif.get_centroid(fractional=False)
    assert isinstance(centroid_cart, np.ndarray)
    assert centroid_cart.shape == (3,)
    npt.assert_almost_equal(
        centroid_frac @ sphere_motif.in_atoms.cell.complete(), centroid_cart
    )

def test_get_site_indices_in_atoms(sphere_motif):
    """Test site indices detection."""
    orig_atoms = sphere_motif.in_atoms
    assert isinstance(orig_atoms, Atoms)
    indices, offsets = sphere_motif._get_site_indices_offsets_in_atoms()
    assert isinstance(indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in indices)
    assert all(0 <= idx < len(orig_atoms) for idx in indices)
    subset = orig_atoms[indices]
    orig_positions = subset.get_positions(wrap=False)
    new_positions = orig_positions + offsets @ orig_atoms.cell.complete()
    subset.set_positions(new_positions)
    distances = np.linalg.norm(
        subset.get_positions(wrap=False) -
        sphere_motif.get_centroid(fractional=False),
        axis=1
    )
    assert np.all(distances <= sphere_motif.radius + 1e-6)  # Allow small numerical tolerance


def test_get_site_indices_in_atoms_partial(sphere_motif_partial):
    """Test site indices detection."""
    orig_atoms = sphere_motif_partial.in_atoms
    assert isinstance(orig_atoms, Atoms)
    indices, offsets = sphere_motif_partial._get_site_indices_offsets_in_atoms()
    assert isinstance(indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in indices)
    assert all(0 <= idx < len(orig_atoms) for idx in indices)
    subset = orig_atoms[indices]
    orig_positions = subset.get_positions(wrap=False)
    new_positions = orig_positions + offsets @ orig_atoms.cell.complete()
    subset.set_positions(new_positions)
    distances = np.linalg.norm(
        subset.get_scaled_positions(wrap=False) -
        sphere_motif_partial.get_centroid(fractional=True),
        axis=1
    )
    assert np.all(distances <= sphere_motif_partial.radius + 1e-6)
    assert all(
        elem in sphere_motif_partial.symbols for elem in subset.get_chemical_symbols()
    )


def test_describe_fractional(sphere_motif_fractional):
    """Test string description generation."""
    # Test fractional coordinates description.
    desc_frac = sphere_motif_fractional.describe(precision=3)
    assert isinstance(desc_frac, str)
    assert "fractional coordinates" in desc_frac
    assert "radius 4.000 angstroms" in desc_frac
    assert "all atoms" in desc_frac
    assert "with element symbols" not in desc_frac


def test_describe_cartesian(sphere_motif_cartesian):
    """Test string description generation."""
    # Test Cartesian coordinates description.
    desc_cart = sphere_motif_cartesian.describe(precision=2)
    assert isinstance(desc_cart, str)
    assert "cartesian coordinates" in desc_cart
    assert "radius 4.00 angstroms" in desc_cart
    assert "all atoms" in desc_cart
    assert "with element symbols" not in desc_cart


def test_describe_index(sphere_motif_index):
    """Test string description generation."""
    # Test index-based description.
    desc_index = sphere_motif_index.describe(precision=1)
    assert isinstance(desc_index, str)
    assert "atom with index" in desc_index
    assert f"radius 4.0 angstroms" in desc_index
    assert "all atoms" in desc_index
    assert "with element symbols" not in desc_index


def test_describe_partial_symbols(sphere_motif_partial):
    """Test string description generation with partial symbols."""
    desc_partial = sphere_motif_partial.describe(precision=2)
    assert isinstance(desc_partial, str)
    assert "fractional coordinates" in desc_partial
    assert "radius 4.00 angstroms" in desc_partial
    assert "with element symbols" in desc_partial
    assert "all atoms" in desc_partial


def test_detect_random_default(orig_atoms):
    #  Test default
    motif = SphereRegionMotif.detect_random_one(orig_atoms, seed=42)
    assert isinstance(motif, SphereRegionMotif)
    assert motif.radius > 0
    assert motif.mode_flag == "center_around_atom_index"
    assert motif.symbols is None  # Default is no symbols filtering

    # Test with radius.
    radius = 3.5
    motif = SphereRegionMotif.detect_random_one(orig_atoms, radius=radius, seed=42)
    assert motif.radius == radius
    assert motif.mode_flag == "center_around_atom_index"

    # Test around coord.
    motif = SphereRegionMotif.detect_random_one(
        orig_atoms, style="center_around_coordinates", seed=42
    )
    assert motif.mode_flag == "center_around_coordinates"
    assert not motif.center_is_fractional  # Default is Cartesian

    # Test with randomized symbols.
    motif = SphereRegionMotif.detect_random_one(
        orig_atoms, randomize_symbols=True, seed=42
    )
    assert motif.symbols is not None
    assert isinstance(motif.symbols, list)
    assert len(motif.symbols) >= 1


def test_detect_random_empty_atoms():
    """Test random detection with empty atoms raises error."""
    empty_atoms = Atoms()
    with pytest.raises(ValueError, match="Atoms object is empty"):
        SphereRegionMotif.detect_random_one(empty_atoms)


def test_detect_random_invalid_style(orig_atoms):
    """Test random detection with invalid style raises error."""
    with pytest.raises(NotImplementedError, match="Invalid style"):
        SphereRegionMotif.detect_random_one(orig_atoms, style="invalid_style")


def test_detect_random_reproducibility(orig_atoms):
    """Test that same seed produces same results."""
    motif1 = SphereRegionMotif.detect_random_one(orig_atoms, seed=123)
    motif2 = SphereRegionMotif.detect_random_one(orig_atoms, seed=123)

    assert motif1.radius == motif2.radius
    if motif1.mode_flag == "center_around_atom_index":
        assert motif1.center_id == motif2.center_id
    else:
        np.testing.assert_array_almost_equal(motif1.get_centroid(), motif1.get_centroid())


def test_coordinate_vs_index_consistency(orig_atoms):
    """Test that coordinate and index modes give same center when equivalent."""
    atom_idx = 0
    atom_pos = orig_atoms.get_positions()[atom_idx]

    motif_index = SphereRegionMotif(orig_atoms, radius=5.0, center_id=atom_idx)
    motif_coord = SphereRegionMotif(
        orig_atoms, radius=5.0, center=atom_pos, center_is_fractional=False
    )

    centroid_index = motif_index.get_centroid()
    centroid_coord = motif_coord.get_centroid()

    np.testing.assert_array_almost_equal(centroid_index, centroid_coord)

def test_fractional_vs_cartesian_consistency(orig_atoms):
    """Test fractional vs Cartesian coordinate consistency."""
    frac_coords = [0.5, 0.5, 0.5]
    cart_coords = np.array(frac_coords) @ orig_atoms.cell.complete()

    motif_frac = SphereRegionMotif(
        orig_atoms, radius=5.0, center=frac_coords, center_is_fractional=True
    )
    motif_cart = SphereRegionMotif(
        orig_atoms, radius=5.0, center=cart_coords, center_is_fractional=False
    )

    centroid_frac = motif_frac.get_centroid()
    centroid_cart = motif_cart.get_centroid()

    np.testing.assert_array_almost_equal(centroid_frac, centroid_cart)


def test_get_atoms(sphere_motif):
    """Test that get_atoms returns correct subset."""
    atoms_subset = sphere_motif.get_atoms()
    assert isinstance(atoms_subset, Atoms)

    # Check that all atoms in the subset are within the radius
    distances = np.linalg.norm(
        atoms_subset.get_positions(wrap=False) -
        sphere_motif.get_centroid(fractional=False),
        axis=1
    )
    assert np.all(distances <= sphere_motif.radius + 1e-6)  # Allow small numerical tolerance

    # Check that the returned atoms are indeed part of the original atoms
    # Don't use the "in expression" to avoid issues with ASE Atoms object,
    # ase.Atom did not implement __eq__ method.
    # assert all(atom in sphere_motif.in_atoms for atom in atoms_subset)

    npt.assert_array_equal(sphere_motif.frac_coords,
                           atoms_subset.get_scaled_positions(wrap=False))
    npt.assert_array_equal(sphere_motif.cart_coords,
                           atoms_subset.get_positions(wrap=False))


def test_default_name(sphere_motif):
    """Test that default name is generated correctly."""
    default_name = sphere_motif._get_default_name()
    assert isinstance(default_name, str)
    assert default_name == "a sphere region"


def test_forbidden_actions(sphere_motif):
    """Test that forbidden actions include expected ones."""
    forbidden = sphere_motif.forbidden_actions
    assert isinstance(forbidden, list)
    assert "add-motif" in forbidden
    assert "remove-motif" not in forbidden
    assert "translate-motif" in forbidden
    assert "resize-motif" not in forbidden
    assert "replace-motif" in forbidden
