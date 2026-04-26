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

# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------
# Maps every known AtomWorld action name (as used in data folders / CLI) to
# the subfolder of results/AtomWorld/ it should be saved under.
# When a new action category is introduced, add a block here and the rest
# of the framework will pick it up automatically.
ACTION_CATEGORIES: dict[str, str] = {
    # ---- simple (index-based) actions ----
    "add_atom_action": "simple",
    "change_atom_action": "simple",
    "delete_around_atom_action": "simple",
    "delete_below_atom_action": "simple",
    "insert_between_atoms_action": "simple",
    "insert_between_atoms_action_natoms": "simple",  # data-folder variant
    "move_all_action": "simple",
    "move_around_atom_action": "simple",
    "move_atom_action": "simple",
    "move_selected_atoms_action": "simple",
    "move_towards_atom_action": "simple",
    "remove_atom_action": "simple",
    "rotate_around_atom_action": "simple",
    "rotate_whole_action": "simple",
    "swap_atoms_action": "simple",
    "super_cell_action": "simple",
    # ---- verbose (motif-based) actions ----
    "add_motif_action": "verbose",
    "remove_motif_action": "verbose",
    "replace_motif_action": "verbose",
    "translate_motif_action": "verbose",
    "rotate_motif_action": "verbose",
    "swap_motif_action": "verbose",
    "resize_motif_action": "verbose",
    "change_element_action": "verbose",
    "lattice_transform_action": "verbose",
    "make_supercell_action": "verbose",
    "rotate_structure_action": "verbose",
}


def get_action_category(action_name: str, default: str = "simple") -> str:
    """Return the results subfolder category for the given AtomWorld action name.

    If the action name is not recognised, *default* is returned so that new or
    custom actions still produce a valid (if generic) output path.
    """
    return ACTION_CATEGORIES.get(action_name, default)


__all__ = [
    "get_action_category",
    "ACTION_CATEGORIES",
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
