"""AtomWorld actions — unified public API.

Two action families are available:

``simple``
    Index-based actions that directly address atoms by their position in the
    ASE Atoms list.  Lightweight, no extra dependencies beyond ASE.

``verbose``
    Motif-centric actions inspired by the AtomWorldBench-2 design.  Each action
    operates on a named geometric element (site, cluster, bond, region …) and
    produces an expressive natural-language description suitable for LLM prompting.
    Requires ``scipy`` for some rotation actions.

Quick-start examples::

    # -- simple --
    from atomworld.actions.simple import AddAtomAction, RemoveAtomAction

    # -- verbose --
    from atomworld.actions.verbose import (
        SiteMotif, AddMotifAction, ChangeElementAction
    )
"""
from .simple import (
    BaseAction,
    DEFAULT_CONFIG,
    AddAtomAction,
    RemoveAtomAction,
    DeleteBelowAtomAction,
    DeleteAroundAtomAction,
    MoveAtomAction,
    MoveTowardsAtomAction,
    MoveSelectedAtomsAction,
    MoveAroundAtomAction,
    MoveAllAction,
    ChangeAtomAction,
    SwapAtomsAction,
    InsertBetweenAtomsAction,
    RotateAroundAtomAction,
    RotateWholeAction,
    SuperCellAction,
)

from .verbose import (
    # motifs
    BaseMotif,
    BaseSiteCollectionMotif,
    SiteMotif,
    ClusterMotif,
    BondMotif,
    BaseRegionMotif,
    SphereRegionMotif,
    BoxRegionMotif,
    # motif actions
    BaseMotifAction,
    AddMotifAction,
    RemoveMotifAction,
    ReplaceMotifAction,
    TranslateMotifAction,
    RotateMotifAction,
    SwapMotifAction,
    ResizeMotifAction,
    motif_action_factory,
    # structure actions
    BaseStructureAction,
    ChangeElementAction,
    LatticeTransformAction,
    MakeSupercellAction,
    RotateStructureAction,
    structure_action_factory,
)

__all__ = [
    # ---- simple ----
    "BaseAction",
    "DEFAULT_CONFIG",
    "AddAtomAction",
    "RemoveAtomAction",
    "DeleteBelowAtomAction",
    "DeleteAroundAtomAction",
    "MoveAtomAction",
    "MoveTowardsAtomAction",
    "MoveSelectedAtomsAction",
    "MoveAroundAtomAction",
    "MoveAllAction",
    "ChangeAtomAction",
    "SwapAtomsAction",
    "InsertBetweenAtomsAction",
    "RotateAroundAtomAction",
    "RotateWholeAction",
    "SuperCellAction",
    # ---- verbose motifs ----
    "BaseMotif",
    "BaseSiteCollectionMotif",
    "SiteMotif",
    "ClusterMotif",
    "BondMotif",
    "BaseRegionMotif",
    "SphereRegionMotif",
    "BoxRegionMotif",
    # ---- verbose motif actions ----
    "BaseMotifAction",
    "AddMotifAction",
    "RemoveMotifAction",
    "ReplaceMotifAction",
    "TranslateMotifAction",
    "RotateMotifAction",
    "SwapMotifAction",
    "ResizeMotifAction",
    "motif_action_factory",
    # ---- verbose structure actions ----
    "BaseStructureAction",
    "ChangeElementAction",
    "LatticeTransformAction",
    "MakeSupercellAction",
    "RotateStructureAction",
    "structure_action_factory",
]
