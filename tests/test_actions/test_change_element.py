"""Comprehensive test suite for ChangeElementAction."""
from ase import Atoms
from ase.data import chemical_symbols

import pytest
import numpy as np
import numpy.testing as npt

from AtomWorldBench.atom_world.actions.structure_actions.base import BaseStructureAction
from AtomWorldBench.atom_world.actions.structure_actions.change_element import ChangeElementAction
from AtomWorldBench.common.registry import get_registered


def test_registry():
    """Test that AddMotifAction is registered correctly."""
    action_class = get_registered(BaseStructureAction)["change-element"]
    assert action_class is ChangeElementAction


def test_change_element_replace(orig_atoms):
    """Test changing one element to another."""
    all_elements = set(orig_atoms.get_chemical_symbols())
    from_element = all_elements.pop()
    to_element = "Og"  # Oganesson, a noble gas not in orig_atoms
    action = ChangeElementAction(
        operated_atoms=orig_atoms,
        from_element=from_element,
        to_element=to_element,
    )
    assert action.mode_flag == "replace_element"
    orig_symbols = orig_atoms.get_chemical_symbols()
    expected_new_symbols = [
        to_element if sym == from_element else sym for sym in orig_symbols
    ]
    new_atoms = action.execute()
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == expected_new_symbols
    npt.assert_allclose(
        new_atoms.get_positions(wrap=False),
        orig_atoms.get_positions(wrap=False),
        atol=1e-8,
    )

    desc = action.describe()
    assert f"replace all atoms of element {from_element} with {to_element}." in desc


def test_change_element_remove(orig_atoms):
    """Test removing an element."""
    all_elements = set(orig_atoms.get_chemical_symbols())
    from_element = all_elements.pop()
    action = ChangeElementAction(
        operated_atoms=orig_atoms,
        from_element=from_element,
    )
    assert action.mode_flag == "remove_element"
    orig_symbols = orig_atoms.get_chemical_symbols()
    expected_new_symbols = [sym for sym in orig_symbols if sym != from_element]
    expected_new_positions = orig_atoms.get_positions(wrap=False)[
        np.array(orig_symbols) != from_element
    ]
    new_atoms = action.execute()
    new_symbols = new_atoms.get_chemical_symbols()
    assert new_symbols == expected_new_symbols
    npt.assert_allclose(
        new_atoms.get_positions(wrap=False),
        expected_new_positions,
        atol=1e-8,
    )

    desc = action.describe()
    assert (
               f"remove all atoms of element {from_element}"
               f" without affecting the order of other atoms"
    ) in desc


def test_change_element_remove_all_atoms_rejected():
    """Ensure we do not allow removal that empties the structure."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="Removing all atoms would leave an empty structure"):
        ChangeElementAction(operated_atoms=atoms, from_element='H')


def test_change_element_from_not_in_atoms(orig_atoms):
    """Test error when from_element is not in operated_atoms."""
    with pytest.raises(ValueError, match="from_element 'whatever' not found in operated_atoms."):
        ChangeElementAction(
            operated_atoms=orig_atoms,
            from_element="whatever",  # Non-existent element
            to_element="Og",
        )


def test_change_element_to_same_as_from(orig_atoms):
    """Test error when to_element is the same as from_element."""
    all_elements = set(orig_atoms.get_chemical_symbols())
    from_element = all_elements.pop()
    with pytest.raises(ValueError, match="to_element must be different from from_element."):
        ChangeElementAction(
            operated_atoms=orig_atoms,
            from_element=from_element,
            to_element=from_element,
        )


def test_only_to_element_provided(orig_atoms):
    """Test error when only to_element is provided."""
    to_element = "Og"
    with pytest.raises(TypeError):
        ChangeElementAction(
            operated_atoms=orig_atoms,
            to_element=to_element,
        )


def test_get_random_one(orig_atoms):
    """Test the get_random_one class method."""
    all_appeared_modes = set()
    all_appeared_to_elements = set()
    for _ in range(100):
        action = ChangeElementAction.get_random_one(operated_atoms=orig_atoms)
        assert isinstance(action, ChangeElementAction)
        assert action.operated_atoms == orig_atoms
        all_elements = set(orig_atoms.get_chemical_symbols())
        assert action.from_element in all_elements
        if action.mode_flag == "replace_element":
            assert action.to_element is not None
            assert action.to_element != action.from_element
            assert action.to_element in chemical_symbols
            all_appeared_to_elements.add(action.to_element)
        else:
            assert action.to_element is None
        all_appeared_modes.add(action.mode_flag)

    expected_modes = {"replace_element", "remove_element"}
    assert all_appeared_modes == expected_modes
    assert len(all_appeared_to_elements) > 1  # Ensure variety in to_elements
