import os
import numpy as np
import pytest
from ase import Atoms
import ase.io

from atom_world.actions import *


def _find_cif(name='Li6PS5Cl.cif'):
    """Return the CIF path located in a `cifs/` folder next to this test file.

    This keeps the test deterministic: the test data is expected to live next to
    the tests, so running pytest from anywhere still finds it.
    """
    test_dir = os.path.dirname(__file__)
    candidate = os.path.join(test_dir, 'cifs', name)
    candidate = os.path.abspath(candidate)
    return candidate if os.path.exists(candidate) else None


def _ensure_outputs_dir():
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def test_add_atom_action():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = AddAtomAction(atoms, 'O', np.array([1.0, 2.0, 3.0]))
    result = action.execute()
    assert len(result) == 2
    assert result[-1].symbol == 'O'
    np.testing.assert_array_equal(result[-1].position, [1.0, 2.0, 3.0])


def test_remove_atom_action():
    atoms = Atoms('HO', positions=[[0, 0, 0], [1, 0, 0]])
    action = RemoveAtomAction(atoms, 0)
    result = action.execute()
    assert len(result) == 1
    assert result[0].symbol == 'O'


def test_remove_atom_action_out_of_bounds():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = RemoveAtomAction(atoms, 5)
    with pytest.raises(IndexError):
        action.execute()


def test_move_atom_action():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = MoveAtomAction(atoms, 0, np.array([1.0, 1.0, 1.0]))
    result = action.execute()
    np.testing.assert_array_equal(result[0].position, [1.0, 1.0, 1.0])


def test_move_atom_action_out_of_bounds():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = MoveAtomAction(atoms, 2, np.array([1.0, 1.0, 1.0]))
    with pytest.raises(IndexError):
        action.execute()


def test_modify_atom_action():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = ChangeAtomAction(atoms, 0, 'C')
    result = action.execute()
    assert result[0].symbol == 'C'


def test_modify_atom_action_out_of_bounds():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    action = ChangeAtomAction(atoms, 2, 'C')
    with pytest.raises(IndexError):
        action.execute()


def test_str_methods():
    atoms = Atoms('H', positions=[[0, 0, 0]])
    add = AddAtomAction(atoms, 'O', np.array([1, 2, 3]))
    remove = RemoveAtomAction(atoms, 0)
    move = MoveAtomAction(atoms, 0, np.array([1, 1, 1]))
    modify = ChangeAtomAction(atoms, 0, 'C')
    # be permissive: check for key words rather than exact phrases
    assert 'add' in str(add).lower()
    assert 'remove' in str(remove).lower()
    assert 'move' in str(move).lower()
    assert 'change' in str(modify).lower()


def test_insert_atom_actions():
    cif_path = _find_cif('Li6PS5Cl.cif')
    if cif_path is None:
        pytest.skip('Li6PS5Cl.cif not found in known locations; skipping CIF-dependent test')

    atoms = ase.io.read(cif_path)

    insert_between_action = InsertBetweenAtomsAction(atoms, 0, 1, 'O', 0.4)
    result = insert_between_action.execute()

    out_dir = _ensure_outputs_dir()
    ase.io.write(os.path.join(out_dir, 'Li6PS5Cl_insert_between.cif'), result)

    assert len(result) == len(atoms) + 1


def test_delete_below():
    cif_path = _find_cif('Li6PS5Cl.cif')
    if cif_path is None:
        pytest.skip('Li6PS5Cl.cif not found in known locations; skipping CIF-dependent test')

    atoms = ase.io.read(cif_path)

    delete_below_action = DeleteBelowAtomAction(atoms, 10)
    result = delete_below_action.execute()

    out_dir = _ensure_outputs_dir()
    ase.io.write(os.path.join(out_dir, 'Li6PS5Cl_delete_below.cif'), result)

    assert len(result) < len(atoms)  # Atoms should be removed


def test_delete_around():
    cif_path = _find_cif('Li6PS5Cl.cif')
    if cif_path is None:
        pytest.skip('Li6PS5Cl.cif not found in known locations; skipping CIF-dependent test')

    atoms = ase.io.read(cif_path)

    delete_around_action = DeleteAroundAtomAction(atoms, 6, 3.0)
    result = delete_around_action.execute()

    out_dir = _ensure_outputs_dir()
    ase.io.write(os.path.join(out_dir, 'Li6PS5Cl_delete_around.cif'), result)

    assert len(result) < len(atoms)  # Atoms should be removed around the specified atom


def test_move_around():
    cif_path = _find_cif('Li6PS5Cl.cif')
    if cif_path is None:
        pytest.skip('Li6PS5Cl.cif not found in known locations; skipping CIF-dependent test')

    atoms = ase.io.read(cif_path)

    move_around_action = MoveAroundAtomAction(atoms, 6, 8.0, np.array([12.0, 0.0, 0.0]))
    result = move_around_action.execute()

    out_dir = _ensure_outputs_dir()
    ase.io.write(os.path.join(out_dir, 'Li6PS5Cl_move_around.cif'), result)

    assert len(result) == len(atoms)  # Number of atoms should remain the same