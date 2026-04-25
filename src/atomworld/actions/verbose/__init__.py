"""Verbose (motif-based) actions for AtomWorld.

This package provides a rich motif-centric API for describing and executing
atomic structure manipulations with expressive natural-language descriptions.

Key sub-packages
----------------
motifs
    Geometric elements (site, cluster, bond, sphere region, box region) that
    act as operands for the verbose actions.
motif_actions
    Actions that operate on a motif within a structure
    (add, remove, replace, translate, rotate, swap, resize).
structure_actions
    Whole-structure transformations independent of individual motifs
    (change element, lattice transform, make supercell, rotate structure).
common
    Shared infrastructure: registry, multi-mode init mixin, globals.
utils
    Coordinate, description, atom and neighbour utilities.
"""
# Motif primitives
from .motifs.base import BaseMotif
from .motifs.site_collections.base import BaseSiteCollectionMotif
from .motifs.site_collections.site import SiteMotif
from .motifs.site_collections.cluster import ClusterMotif
from .motifs.site_collections.bond import BondMotif
from .motifs.regions.base import BaseRegionMotif
from .motifs.regions.sphere import SphereRegionMotif
from .motifs.regions.box import BoxRegionMotif

# Motif actions
from .motif_actions import (
    BaseMotifAction,
    AddMotifAction,
    RemoveMotifAction,
    ReplaceMotifAction,
    TranslateMotifAction,
    RotateMotifAction,
    SwapMotifAction,
    ResizeMotifAction,
    motif_action_factory,
)

# Structure actions
from .structure_actions import (
    BaseStructureAction,
    ChangeElementAction,
    LatticeTransformAction,
    MakeSupercellAction,
    RotateStructureAction,
    structure_action_factory,
)

__all__ = [
    # motifs
    "BaseMotif",
    "BaseSiteCollectionMotif",
    "SiteMotif",
    "ClusterMotif",
    "BondMotif",
    "BaseRegionMotif",
    "SphereRegionMotif",
    "BoxRegionMotif",
    # motif actions
    "BaseMotifAction",
    "AddMotifAction",
    "RemoveMotifAction",
    "ReplaceMotifAction",
    "TranslateMotifAction",
    "RotateMotifAction",
    "SwapMotifAction",
    "ResizeMotifAction",
    "motif_action_factory",
    # structure actions
    "BaseStructureAction",
    "ChangeElementAction",
    "LatticeTransformAction",
    "MakeSupercellAction",
    "RotateStructureAction",
    "structure_action_factory",
]
