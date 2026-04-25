from atomworld.actions import (
    AddAtomAction, RemoveAtomAction, MoveAtomAction, ChangeAtomAction,
    SwapAtomsAction, DeleteBelowAtomAction,
)
from atomworld.actions import SuperCellAction
import os
import numpy as np
import pytest
from ase.io import read, write
from ase import Atoms


def test_randomize_add_atom():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                  cell=[[5, 1, 0], [0, 4, 0], [0.5, 0, 3]], pbc=True)
    rng = np.random.default_rng(42)
    params = AddAtomAction.randomize(atoms, rng=rng)
    assert "atoms" in params
    assert "symbol" in params
    assert "position" in params
    assert len(params["position"]) == 3


def test_randomize_remove_atom():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                  cell=[[5, 1, 0], [0, 4, 0], [0.5, 0, 3]], pbc=True)
    rng = np.random.default_rng(42)
    params = RemoveAtomAction.randomize(atoms, rng=rng)
    assert 0 <= params["index"] < len(atoms)


def test_randomize_change_atom():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                  cell=[[5, 1, 0], [0, 4, 0], [0.5, 0, 3]], pbc=True)
    rng = np.random.default_rng(42)
    params = ChangeAtomAction.randomize(atoms, rng=rng)
    assert params["symbol"] != atoms[params["index"]].symbol


def test_apply_random_add_atom():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                  cell=[[5, 1, 0], [0, 4, 0], [0.5, 0, 3]], pbc=True)
    rng = np.random.default_rng(42)
    action, result = AddAtomAction.apply_random(atoms, rng=rng)
    assert action is not None
    assert len(result) == len(atoms) + 1
    # Original should be unchanged (copy=True)
    assert len(atoms) == 3


def test_apply_random_supercell():
    atoms = Atoms('NaCl', positions=[(0, 0, 0), (1.5, 1.5, 1.5)],
                  cell=[3, 3, 3], pbc=True)
    rng = np.random.default_rng(42)
    action, result = SuperCellAction.apply_random(atoms, rng=rng)
    assert action is not None
    assert len(result) > len(atoms)


def test_apply_random_delete_below_returns_none():
    # Single atom should return None
    atoms = Atoms('H', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    rng = np.random.default_rng(42)
    action, result = DeleteBelowAtomAction.apply_random(atoms, rng=rng)
    assert action is None
    assert result is None