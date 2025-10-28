import numpy as np
import numpy.testing as npt
import pytest
from ase import Atoms
from ase.build import make_supercell

from AtomWorldBench.atom_world.motifs.regions.box import BoxRegionMotif


@pytest.fixture
def test_atoms(orig_atoms):
    return make_supercell(orig_atoms, np.eye(3) * 6)


def test_registry():
    """Test that BoxRegionMotif is registered correctly."""
    from AtomWorldBench.atom_world.motifs.regions.base import BaseRegionMotif
    from AtomWorldBench.atom_world.motifs.base import BaseMotif
    from AtomWorldBench.common.registry import get_registered

    region_classes = get_registered(BaseRegionMotif)
    motif_classes = get_registered(BaseMotif)

    assert "box" in region_classes
    assert "box" in motif_classes
    assert region_classes["box"] is BoxRegionMotif
    assert motif_classes["box"] is BoxRegionMotif

    assert "box-region" in region_classes
    assert "box-region" in motif_classes
    assert region_classes["box-region"] is BoxRegionMotif
    assert motif_classes["box-region"] is BoxRegionMotif


@pytest.fixture
def box_motif_fractional(test_atoms):
    """Fixture for a BoxRegionMotif instance with fractional coordinates."""
    return BoxRegionMotif(
        test_atoms,
        xmin=0.2,
        xmax=0.8,
        ymin=0.2,
        ymax=0.8,
        zmin=0.2,
        zmax=0.8,
    )


@pytest.fixture
def box_motif_partial_boundaries(test_atoms):
    """Fixture for a BoxRegionMotif with only some boundaries defined."""
    return BoxRegionMotif(
        test_atoms,
        xmin=0.3,
        ymax=0.7,
        zmin=0.1,
    )


@pytest.fixture
def box_motif_partial_symbols(test_atoms):
    """Fixture for a BoxRegionMotif with partial symbols."""
    symbols = list(set(test_atoms.get_chemical_symbols()))
    return BoxRegionMotif(
        test_atoms,
        xmin=0.2,
        xmax=0.8,
        ymin=0.2,
        ymax=0.8,
        zmin=0.2,
        zmax=0.8,
        symbols=symbols[:max(1, len(symbols) // 2)]
    )


def test_get_site_indices_test_atoms_fractional(box_motif_fractional):
    """Test site indices detection with fractional boundaries."""
    oatoms = box_motif_fractional.in_atoms
    assert isinstance(oatoms, Atoms)

    indices, offsets = box_motif_fractional._get_site_indices_offsets_in_atoms()
    assert isinstance(indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in indices)
    assert all(0 <= idx < len(oatoms) for idx in indices)

    # Verify atoms are within boundaries
    frac_coords = oatoms.get_scaled_positions(wrap=True)[indices]
    assert np.all(frac_coords[:, 0] >= box_motif_fractional.xmin - box_motif_fractional.tol)
    assert np.all(frac_coords[:, 0] <= box_motif_fractional.xmax + box_motif_fractional.tol)
    assert np.all(frac_coords[:, 1] >= box_motif_fractional.ymin - box_motif_fractional.tol)
    assert np.all(frac_coords[:, 1] <= box_motif_fractional.ymax + box_motif_fractional.tol)
    assert np.all(frac_coords[:, 2] >= box_motif_fractional.zmin - box_motif_fractional.tol)
    assert np.all(frac_coords[:, 2] <= box_motif_fractional.zmax + box_motif_fractional.tol)

    npt.assert_array_equal(offsets, 0)


def test_get_site_indices_partial_boundaries(box_motif_partial_boundaries):
    """Test site indices with only partial boundaries defined."""
    indices, offsets = box_motif_partial_boundaries._get_site_indices_offsets_in_atoms()
    assert isinstance(indices, list)

    # Verify only specified boundaries are enforced
    frac_coords = box_motif_partial_boundaries.in_atoms.get_scaled_positions(wrap=True)[indices]
    assert np.all(frac_coords[:, 0] >= box_motif_partial_boundaries.xmin - box_motif_partial_boundaries.tol)
    assert np.all(frac_coords[:, 1] <= box_motif_partial_boundaries.ymax + box_motif_partial_boundaries.tol)
    assert np.all(frac_coords[:, 2] >= box_motif_partial_boundaries.zmin - box_motif_partial_boundaries.tol)
    assert box_motif_partial_boundaries.xmax is None
    assert box_motif_partial_boundaries.ymin is None
    assert box_motif_partial_boundaries.zmax is None


def test_get_site_indices_partial_symbols(box_motif_partial_symbols):
    """Test site indices with symbol filtering."""
    indices, offsets = box_motif_partial_symbols._get_site_indices_offsets_in_atoms()
    subset = box_motif_partial_symbols.in_atoms[indices]

    # Verify only specified symbols are included
    assert all(
        elem in box_motif_partial_symbols.symbols
        for elem in subset.get_chemical_symbols()
    )


def test_describe_fractional(box_motif_fractional):
    """Test string description with fractional boundaries."""
    desc = box_motif_fractional.describe(precision=2)
    assert isinstance(desc, str)
    assert "fractional" in desc
    assert "0.20 ≤ x ≤ 0.80" in desc
    assert "0.20 ≤ y ≤ 0.80" in desc
    assert "0.20 ≤ z ≤ 0.80" in desc
    assert "all atoms" in desc


def test_describe_partial_boundaries(box_motif_partial_boundaries):
    """Test description with only some boundaries defined."""
    desc = box_motif_partial_boundaries.describe(precision=2)
    assert isinstance(desc, str)
    assert "x ≥" in desc
    assert "y ≤" in desc
    assert "z ≥" in desc
    # xmax, ymin, zmax should not be in description
    assert ("x ≤" not in desc) and ("0.30" in desc)  # Only xmin constraint


def test_describe_partial_symbols(box_motif_partial_symbols):
    """Test description with symbol filtering."""
    desc = box_motif_partial_symbols.describe(precision=2)
    assert isinstance(desc, str)
    assert "fractional" in desc
    assert "with element symbols" in desc
    assert "all atoms" in desc


def test_detect_random_default(test_atoms):
    """Test default random detection."""
    for _ in range(20):
        motif = BoxRegionMotif.detect_random_one(test_atoms)
        assert isinstance(motif, BoxRegionMotif)
        assert motif.symbols is None
        assert 0.0 - 1e-6 <= motif.xmin <= 0.4 + 1e-6
        assert 0.0 - 1e-6 <= motif.ymin <= 0.4 + 1e-6
        assert 0.0 - 1e-6 <= motif.zmin <= 0.4 + 1e-6

        assert motif.xmax - motif.xmin >= 0.2 - 1e-6
        assert motif.ymax - motif.ymin >= 0.2 - 1e-6
        assert motif.zmax - motif.zmin >= 0.2 - 1e-6

def test_detect_random_with_boundaries(test_atoms):
    """Test random detection with specified boundary ranges."""
    none_appeared = False
    for _ in range(20):
        motif = BoxRegionMotif.detect_random_one(
            test_atoms,
            randomize_boundaries=True
        )
        none_appeared = (
                motif.xmin is None or motif.xmax is None or
                motif.ymin is None or motif.ymax is None or
                motif.zmin is None or motif.zmax is None
        )
        if none_appeared:
            break
    assert none_appeared, "Expected at least one boundary to be None over multiple trials."


def test_detect_random_with_symbols(test_atoms):
    """Test random detection with symbol randomization."""
    motif = BoxRegionMotif.detect_random_one(
        test_atoms, randomize_symbols=True, seed=42
    )
    assert motif.symbols is not None
    assert isinstance(motif.symbols, list)
    assert len(motif.symbols) >= 1


def test_detect_random_reproducibility(test_atoms):
    """Test that same seed produces same results."""
    motif1 = BoxRegionMotif.detect_random_one(
        test_atoms,
        randomize_boundaries=True,
        randomize_symbols=True,
        seed=123
    )
    motif2 = BoxRegionMotif.detect_random_one(
        test_atoms,
        randomize_boundaries=True,
        randomize_symbols=True,
        seed=123
    )

    assert motif1.xmin == motif2.xmin
    assert motif1.xmax == motif2.xmax
    assert motif1.ymin == motif2.ymin
    assert motif1.ymax == motif2.ymax
    assert motif1.zmin == motif2.zmin
    assert motif1.zmax == motif2.zmax
    assert motif1.symbols == motif2.symbols


def test_get_atoms(box_motif_fractional):
    """Test that get_atoms returns correct subset."""
    atoms_subset = box_motif_fractional.get_atoms()
    assert isinstance(atoms_subset, Atoms)

    npt.assert_allclose(
        box_motif_fractional.frac_coords,
        atoms_subset.get_scaled_positions(wrap=False)
    )
    npt.assert_allclose(
        box_motif_fractional.cart_coords,
        atoms_subset.get_positions(wrap=False)
    )


def test_default_name(box_motif_fractional):
    """Test that default name is generated correctly."""
    default_name = box_motif_fractional._get_default_name()
    assert isinstance(default_name, str)
    assert default_name == "a box region"



def test_no_boundaries(test_atoms):
    """Test BoxRegionMotif with no boundaries (selects all atoms)."""
    motif = BoxRegionMotif(test_atoms)
    assert len(motif.indices) == len(test_atoms)


def test_single_boundary(test_atoms):
    """Test with only a single boundary constraint."""
    motif = BoxRegionMotif(test_atoms, xmin=0.5)
    indices = motif.indices
    frac_coords = test_atoms.get_scaled_positions(wrap=True)[indices]
    assert np.all(frac_coords[:, 0] >= 0.5 - motif.tol)
