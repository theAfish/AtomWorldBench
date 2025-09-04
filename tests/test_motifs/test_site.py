"""Comprehensive pytest suite for and SiteMotif."""
import pytest
from ase import Atoms
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif


def test_resigstry():
    """Test that SiteMotif is registered in the motif registry."""
    from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif
    from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
    from AtomWorldBench.common.registry import _REGISTRY
    motif_registry = _REGISTRY[BaseSiteCollectionMotif]
    assert 'site' in motif_registry
    assert motif_registry['site'] is SiteMotif
    assert "single-site" in motif_registry
    assert motif_registry["single-site"] is SiteMotif
    assert "site-motif" in motif_registry
    assert motif_registry["site-motif"] is SiteMotif
    assert "site_motif" in motif_registry
    assert motif_registry["site_motif"] is SiteMotif
    assert "sitemotif" in motif_registry
    assert motif_registry["sitemotif"] is SiteMotif
    assert "SiteMotif" in motif_registry
    assert motif_registry["SiteMotif"] is SiteMotif


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


def test_post_init(simple_atoms):
    """Test that SiteMotif raises ValueError if not exactly one site."""

    # Valid case
    motif = SiteMotif(simple_atoms, indices=[0])
    assert len(motif) == 1

    # Invalid case: more than one site
    with pytest.raises(ValueError, match="SiteMotif must contain exactly one site"):
        SiteMotif(simple_atoms, indices=[0, 1])

    # Invalid case: zero sites
    with pytest.raises(ValueError, match="SiteMotif must contain exactly one site"):
        SiteMotif(simple_atoms, indices=[])


def test_default_name(simple_atoms):
    """Test default name generation based on species and coordinates."""
    motif = SiteMotif(simple_atoms, indices=[0])
    expected_name = "a species Na+"
    assert motif.name == expected_name

    motif2 = SiteMotif(simple_atoms, indices=[1])
    expected_name2 = "a species Cl-"
    assert motif2.name == expected_name2

    simple_atoms_no_charge = simple_atoms.copy()
    simple_atoms_no_charge.set_initial_charges([0, 0])
    motif3 = SiteMotif(simple_atoms_no_charge, indices=[0])
    expected_name3 = "an atom Na"
    assert motif3.name == expected_name3


def test_forbidden_actions(simple_atoms):
    """Test that forbidden actions are correctly set."""
    motif = SiteMotif(simple_atoms, indices=[0])
    assert "resize" in motif.forbidden_actions


def test_detect_random_one(simple_atoms):
    """Test random site motif detection."""
    motif = SiteMotif.detect_random_one(simple_atoms, seed=42)
    assert isinstance(motif, SiteMotif)
    assert len(motif) == 1
    assert motif.indices[0] in [0, 1]  # Should be one of the two atoms
    npt.assert_array_equal(motif.cell_offsets, np.array([[0, 0, 0]]))
    # No offsets in site autodetect case.
