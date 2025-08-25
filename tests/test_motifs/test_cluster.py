# test_motifs.py
# Pytest suite covering base.py (BaseSiteCollectionMotif) and cluster.py (ClusterMotif)
# The tests are designed to work even if you only have these two files by
# dynamically loading them under a synthetic package and stubbing their external
# dependencies.

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif


# -----------------------------
# Tests for BaseSiteCollectionMotif
# -----------------------------

from ase import Atoms

@pytest.fixture
def simple_atoms():
    # 2-atom cubic cell for many tests
    cell = np.eye(3) * 3.0
    a = Atoms(
        "NaCl",
        positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        cell=cell,
        pbc=True
    )
    a.set_initial_charges([+1, -1])
    return a


@pytest.fixture
def simple_cluster(simple_atoms):
    a = simple_atoms
    m = ClusterMotif(
        symbols=a.get_chemical_symbols(),
        positions=a.get_positions(),
        cell=a.cell,
        pbc=a.pbc,
        charges=a.get_initial_charges()
    )
    return m


def test_init_requires_positions():
    with pytest.raises(ValueError, match="Motif must be initialized with positions"):
        _ = ClusterMotif(symbols=["H"])  # no positions


def test_basic_properties_species_groups_composition(simple_atoms):
    a = simple_atoms
    m = ClusterMotif(
        symbols=a.get_chemical_symbols(),
        positions=a.get_positions(),
        cell=a.cell,
        pbc=a.pbc,
        charges=a.get_initial_charges()
    )
    # species strings pick up charges from stub get_species_string
    assert m.species_strings == ["Na+", "Cl-"]
    # groups
    g = m.species_groups
    assert set(g.keys()) == {"Na+", "Cl-"}
    assert g["Na+"].tolist() == [0]
    assert g["Cl-"].tolist() == [1]
    # composition
    from collections import Counter
    assert m.composition == Counter({"Na+": 1, "Cl-": 1})


def test_coords_and_offsets(orig_atoms):
    a = orig_atoms
    m = ClusterMotif(
        symbols=a.get_chemical_symbols(),
        positions=a.get_positions(),
        cell=a.cell,
        pbc=a.pbc,
        charges=a.get_initial_charges()
    )
    # frac coords unwrapped
    npt.assert_allclose(m.frac_coords, a.get_scaled_positions(wrap=False))
    # offsets floor(frac)
    npt.assert_array_equal(m.cell_offsets, np.floor(m.frac_coords).astype(int))
    # cart coords
    npt.assert_allclose(m.cart_coords, a.get_positions(wrap=False))


def test_centroid_and_radius(simple_atoms):
    a = simple_atoms
    # single-site radius zero
    m1 = BaseSiteCollectionMotif(symbols=["Na"], positions=[a.positions[0]], cell=a.cell, pbc=a.pbc, charges=[+1])
    assert m1.radius == 0.0
    # two sites
    m2 = BaseSiteCollectionMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges())
    c_cart = m2.get_centroid()
    c_frac = m2.get_centroid(fractional=True)
    # centroid is midpoint between the two
    np.testing.assert_allclose(c_cart, np.array([0.75, 0.0, 0.0]))
    np.testing.assert_allclose(c_frac @ m2.cell.complete(), c_cart)
    assert m2.radius > 0


def test_edge_lengths(BaseSiteCollectionMotif):
    a = simple_atoms()
    m = BaseSiteCollectionMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges())
    edges = m.edge_lengths
    assert edges[(0, 1)] == pytest.approx(np.linalg.norm(a.positions[1] - a.positions[0]))


def test_indices_set_get_and_describe(BaseSiteCollectionMotif):
    a = simple_atoms()
    m = BaseSiteCollectionMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges())
    # describe index without indices -> error
    with pytest.raises(ValueError):
        m.describe(style="index")
    m.indices = [0, 1]
    # coord description
    s = m.describe(style="coord", coord_fractional=False, precision=2)
    assert "cartesian" in s and "Na" in s and "Cl" in s
    # index description, zero offsets -> "." ending
    s2 = m.describe(style="index")
    assert s2.endswith(".")
    # fractional coord description
    s3 = m.describe(style="coord", coord_fractional=True, precision=3)
    assert "fractional" in s3
    # invalid style -> NotImplementedError
    with pytest.raises(NotImplementedError):
        m.describe(style="unknown")


def test_from_atoms_and_get_atoms_roundtrip(BaseSiteCollectionMotif):
    a = simple_atoms()
    m = BaseSiteCollectionMotif.from_atoms(a, indices=[0, 1])
    assert isinstance(m, BaseSiteCollectionMotif)
    assert m.indices == [0, 1]
    a2 = m.get_atoms()
    # arrays except reserved copied
    assert set(a2.arrays.keys()) == set(a.arrays.keys()) - {"site_indices"}


def test_find_indices_and_get_site_indices(BaseSiteCollectionMotif):
    a = simple_atoms()
    m = BaseSiteCollectionMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges())
    idx, msg = m.find_indices_in_atoms(a, modify_indices_in_place=True)
    assert msg == ""
    assert idx == [0, 1]
    assert m.indices == [0, 1]
    # cell mismatch -> not found
    b = a.copy()
    b.set_cell(np.eye(3) * 4.0)
    idx2, msg2 = m.find_indices_in_atoms(b)
    assert idx2 is None and "cell" in msg2
    with pytest.raises(ValueError):
        m.get_site_indices_in_atoms(b)


def test_add_iadd_extend_copy_and_eq(BaseSiteCollectionMotif):
    a = simple_atoms()
    m1 = BaseSiteCollectionMotif(symbols=["Na"], positions=[a.positions[0]], cell=a.cell, pbc=a.pbc, charges=[+1])
    m2 = BaseSiteCollectionMotif(symbols=["Cl"], positions=[a.positions[1]], cell=a.cell, pbc=a.pbc, charges=[-1])
    # indices mismatch on extend
    m1.indices = [0]
    m2.indices = None
    with pytest.raises(ValueError):
        m1.extend(m2)
    # Proper add with both indices None
    m1.indices = None
    m2.indices = None
    m3 = m1 + m2
    assert len(m3) == 2
    # inplace add
    m1 += m2
    assert len(m1) == 2
    # copy equals original (by our __eq__ logic)
    c = m1.copy()
    assert c == m1
    # modify composition -> not equal
    m_diff = BaseSiteCollectionMotif(symbols=["Na", "Na"], positions=m1.cart_coords, cell=m1.cell, pbc=m1.pbc, charges=[+1, +1])
    assert (m_diff == m1) is False


def test_slice_and_repeat_reset_name(BaseSiteCollectionMotif):
    a = simple_atoms()
    m = BaseSiteCollectionMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges(), name="foo", indices=[0, 1])
    sub = m[0]
    assert sub.name != "foo"  # reset to default
    assert sub.indices == [0]
    m *= 2
    assert m.name is None  # name reset
    assert len(m) == 4


# -----------------------------
# Tests for ClusterMotif
# -----------------------------


def test_cluster_post_init_and_default_names(ClusterMotif):
    a = simple_atoms()
    # zero atoms should fail via ASE before motif, but we simulate directly
    with pytest.raises(ValueError):
        ClusterMotif(symbols=[], positions=[], cell=a.cell, pbc=a.pbc)

    # name prefixes by size
    c1 = ClusterMotif(symbols=["H"], positions=[[0, 0, 0]], cell=np.eye(3), pbc=[0, 0, 0])
    assert c1._get_default_name().startswith("a point")
    c2 = ClusterMotif(symbols=["H", "He"], positions=[[0, 0, 0], [0, 0, 1]], cell=np.eye(3), pbc=[0, 0, 0])
    assert c2._get_default_name().startswith("a pair")
    c3 = ClusterMotif(symbols=["H", "He", "Li"], positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=np.eye(3), pbc=[0, 0, 0])
    assert c3._get_default_name().startswith("a triplet")


def test_cluster_site_motifs(ClusterMotif):
    a = simple_atoms()
    c = ClusterMotif(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.cell, pbc=a.pbc, charges=a.get_initial_charges(), indices=[0, 1])
    sites = c.site_motifs
    assert len(sites) == 2
    assert [int(s.indices[0]) for s in sites] == [0, 1]


def test_detect_random_one_success_and_failure(ClusterMotif, imported_modules, monkeypatch):
    a = simple_atoms()

    # Monkeypatch neighbor detection to a controlled behavior
    cluster_mod = imported_modules[1]

    class DummySite:
        def __init__(self, idx, pos, cell, pbc):
            self._idx = [idx]
            self._pos = [pos]
            self._cell = cell
            self._pbc = pbc
            self.indices = [idx]

        @classmethod
        def from_atoms(cls, a, indices):
            return cls(indices[0], a.get_positions()[0], a.cell, a.pbc)

        def __eq__(self, other):
            return isinstance(other, DummySite) and self._idx == other._idx

        # Allow addition with ClusterMotif in filtering path
        def __radd__(self, other):
            return other.__class__.from_atoms(
                other.get_atoms() + a[[self._idx[0]]], indices=(other.indices or []) + [self._idx[0]]
            )

    # Replace SiteMotif used inside cluster with DummySite
    monkeypatch.setattr(cluster_mod, "SiteMotif", DummySite, raising=False)

    def fake_detect_neighbors(struct, site_index, cutoff, symbols=None):
        # Always return the immediate other index as neighbor so it can grow
        others = [i for i in range(len(struct)) if i != site_index]
        return [DummySite(i, struct.positions[i], struct.cell, struct.pbc) for i in others]

    monkeypatch.setattr(cluster_mod, "detect_neighbor_sites_around_site_index", fake_detect_neighbors, raising=False)

    cm = ClusterMotif.detect_random_one(a, cluster_size=2, max_cluster_radius=10.0, seed=42)
    assert isinstance(cm, ClusterMotif)
    assert len(cm) == 2

    # Failure path: neighbor detector returns none so it cannot grow
    def no_neighbors(struct, site_index, cutoff, symbols=None):
        return []

    monkeypatch.setattr(cluster_mod, "detect_neighbor_sites_around_site_index", no_neighbors, raising=False)
    with pytest.raises(RuntimeError):
        ClusterMotif.detect_random_one(a, cluster_size=3, n_attempts=2, max_cluster_radius=1.0, seed=0)


def test_detect_neighbor_sites_validation(imported_modules, monkeypatch):
    cluster_mod = imported_modules[1]
    a = Atoms("HX", positions=[[0,0,0],[0.5,0,0]], cell=np.eye(3), pbc=True)
    # Presence of 'X' should trigger error
    with pytest.raises(ValueError):
        cluster_mod.detect_neighbor_sites_around_site_index(a, site_index=0, cutoff=1.0)

    # For a clean structure, stub the low-level neighbor function to control output
    b = Atoms("HHH", positions=[[0,0,0],[0.5,0,0],[2.0,0,0]], cell=np.eye(3), pbc=True)

    def stub_detect(atoms, frac_coords, cutoff, symbols=None):
        # Pretend all but center are within cutoff; return indices and zero offsets
        idx = np.array([0, 1, 2], dtype=int)
        off = np.zeros((3,3), dtype=int)
        return idx, off

    monkeypatch.setattr(cluster_mod, "detect_indices_offests_around_frac_coords", stub_detect, raising=False)
    res = cluster_mod.detect_neighbor_sites_around_site_index(b, site_index=1, cutoff=1.0)
    # exclude self -> 2 neighbors as SiteMotif/DummySite
    assert len(res) == 2
    # verify indices assigned as ints
    assert all(isinstance(m.indices[0], (int, np.integer)) for m in res)
