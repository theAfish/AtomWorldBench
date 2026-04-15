from utils.dataloader import _resolve_action_file


def test_resolve_json_snake_to_camel_action_name():
    files = ["AddAtomAction.json", "MoveAtomAction.json"]
    assert _resolve_action_file("add_atom_action", files) == "AddAtomAction.json"


def test_resolve_json_short_name_without_action_suffix():
    files = ["AddAtomAction.json", "MoveAtomAction.json"]
    assert _resolve_action_file("add_atom", files) == "AddAtomAction.json"


def test_resolve_json_camel_short_name():
    files = ["AddAtomAction.json", "MoveAtomAction.json"]
    assert _resolve_action_file("AddAtom", files) == "AddAtomAction.json"


def test_resolve_csv_with_action_suffix():
    files = ["AddAtomAction.csv", "MoveAtomAction.csv"]
    assert _resolve_action_file("add_atom_action", files) == "AddAtomAction.csv"


def test_resolve_returns_none_when_not_found():
    files = ["AddAtomAction.json"]
    assert _resolve_action_file("delete_below_atom", files) is None
