import pytest

from data_generation.cif_action_generator import (
    ready_actions,
    resolve_action_classes,
)


def test_resolve_action_classes_snake_case():
    resolved = resolve_action_classes(["add_atom_action"], ready_actions)
    assert [cls.__name__ for cls in resolved] == ["AddAtomAction"]


def test_resolve_action_classes_alias_forms():
    resolved = resolve_action_classes(
        ["AddAtomAction", "move_atom", "super_cell_action"],
        ready_actions,
    )
    assert [cls.__name__ for cls in resolved] == [
        "AddAtomAction",
        "MoveAtomAction",
        "SuperCellAction",
    ]


def test_resolve_action_classes_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown action name"):
        resolve_action_classes(["not_a_real_action"], ready_actions)
