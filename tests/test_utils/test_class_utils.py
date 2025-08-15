"""Tests for class utilities in AtomWorldBench.

Adapted from Luis Barroso-Luque's smol project.
"""


from abc import ABC, abstractmethod

import pytest
from ase import Atoms

from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.atom_world.motifs.regions.base import BaseRegionMotif
from AtomWorldBench.atom_world.motifs.regions.sphere import SphereRegionMotif
from AtomWorldBench.atom_world.motifs.site_collections.base import BaseSiteCollectionMotif
from AtomWorldBench.atom_world.motifs.site_collections.cluster import ClusterMotif
from AtomWorldBench.atom_world.motifs.site_collections.site import SiteMotif
from AtomWorldBench.atom_world.motifs.site_collections.bond import BondMotif

from AtomWorldBench.atom_world.actions.base import BaseAction
from AtomWorldBench.atom_world.actions.add import AddAction
from AtomWorldBench.atom_world.actions.remove import RemoveAction
from AtomWorldBench.atom_world.actions.replace import ReplaceAction
from AtomWorldBench.atom_world.actions.rotate import RotateAction
from AtomWorldBench.atom_world.actions.translate import TranslateAction
from AtomWorldBench.atom_world.actions.resize import ResizeAction

from AtomWorldBench.mixin_classes import MultiModeInitMixin

from AtomWorldBench.utils.class_utils import (
    class_name_from_str,
    derived_class_factory,
    get_subclasses,
)


@pytest.mark.parametrize(
    "class_str",
    [
        "TheBestClass",
        "The-Best-Class",
        "The-Best-class",
        "The-best-Class",
        "the-Best-Class",
        "The-best-class",
        "the-Best-class",
        "the-best-Class",
        "the-best-class",
    ],
)
def test_class_name_from_str(class_str):
    assert class_name_from_str(class_str) == "TheBestClass"


def test_get_subclasses():
    # test a few dummy classes
    class DummyABC(ABC):
        @abstractmethod
        def do_stuff(self):
            pass

    class DummyDummyABC(DummyABC):
        pass

    class DummyParent(DummyABC):
        def do_stuff(self):
            print("I'm doing stuff!")

    class DummyChild(DummyParent):
        def do_stuff(self):
            print("I'm doing more stuff!")

    assert all(
        c in get_subclasses(DummyABC).values() for c in [DummyParent, DummyChild]
    )
    assert DummyABC not in get_subclasses(DummyABC).values()
    assert DummyDummyABC not in get_subclasses(DummyABC).values()

    # now test classes in AtomWorldBench.
    assert all(
        c in get_subclasses(BaseSiteCollectionMotif).values()
        for c in [SiteMotif, ClusterMotif, BondMotif]
    )
    assert SphereRegionMotif in get_subclasses(BaseRegionMotif).values()
    assert all(
        c in get_subclasses(BaseMotif).values()
        for c in [SiteMotif, ClusterMotif, BondMotif, SphereRegionMotif]
    )
    # Abstract classes should never appear in subclasses search.
    assert not any (
        c in get_subclasses(BaseMotif).values()
        for c in [BaseRegionMotif, BaseSiteCollectionMotif]
    )

    assert all(
        c in get_subclasses(BaseAction).values()
        for c in [
            AddAction, RemoveAction, ReplaceAction,
            RotateAction, TranslateAction, ResizeAction
        ]
    )
    # All these actions should be MultiModeInitMixin.
    assert all(
        issubclass(c, MultiModeInitMixin)
        for c in [
            AddAction, RemoveAction, ReplaceAction,
            RotateAction, TranslateAction, ResizeAction
        ]
    )


def test_derived_class_factory(orig_atoms: Atoms):
    sphere = derived_class_factory(
        'sphere-region-motif',
        BaseRegionMotif,
        orig_atoms,
        center=[0.0, 0.0, 0.0],
        radius=2.0,
    )
    assert isinstance(sphere, SphereRegionMotif)
    assert sphere.mode_flag == "center_around_coordinates"

    site_from_atom = derived_class_factory(
        'site-motif',
        BaseMotif,
        orig_atoms[0],
        indices=[0],
    )
    site_from_atoms = derived_class_factory(
        'site-motif',
        BaseMotif,
        orig_atoms[[0]],
        indices=[0],
    )
    site_from_scratch = derived_class_factory(
        'site-motif',
        BaseMotif,
        symbols=[orig_atoms.get_chemical_symbols()[0]],
        positions=[orig_atoms.get_positions(wrap=False)[0]],
        cell=orig_atoms.cell,
        pbc=orig_atoms.pbc,
        charges=[orig_atoms.get_initial_charges()[0]],
        indices=[0],
    )
    assert isinstance(site_from_atom, SiteMotif)
    assert isinstance(site_from_atoms, SiteMotif)
    assert isinstance(site_from_scratch, SiteMotif)
    assert site_from_atom == site_from_atoms == site_from_scratch

    bond_from_atoms = derived_class_factory(
        'bond-motif',
        BaseMotif,
        orig_atoms[[0, 1]],
        indices=[0, 1],
    )
    bond_from_scratch = derived_class_factory(
        'bond-motif',
        BaseMotif,
        symbols=orig_atoms.get_chemical_symbols()[:2],
        positions=orig_atoms.get_positions(wrap=False)[:2],
        cell=orig_atoms.cell,
        pbc=orig_atoms.pbc,
        charges=orig_atoms.get_initial_charges()[:2],
        indices=[0, 1],
    )
    assert isinstance(bond_from_atoms, BondMotif)
    assert isinstance(bond_from_scratch, BondMotif)
    assert bond_from_atoms == bond_from_scratch

    cluster_from_atoms = derived_class_factory(
        'cluster-motif',
        BaseMotif,
        orig_atoms[[0, 1, 2]],
        indices=[0, 1, 2],
    )
    cluster_from_scratch = derived_class_factory(
        'cluster-motif',
        BaseMotif,
        symbols=orig_atoms.get_chemical_symbols()[:3],
        positions=orig_atoms.get_positions(wrap=False)[:3],
        cell=orig_atoms.cell,
        pbc=orig_atoms.pbc,
        charges=orig_atoms.get_initial_charges()[:3],
        indices=[0, 1, 2],
    )
    assert isinstance(cluster_from_atoms, ClusterMotif)
    assert isinstance(cluster_from_scratch, ClusterMotif)
    assert cluster_from_atoms == cluster_from_scratch
