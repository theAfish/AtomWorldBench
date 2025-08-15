from atom_world.data_generator import ActionInputGenerator, DataGenerator
import os
import pytest
from ase.io import read, write
from ase import Atoms


def test_action_input_generator():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], cell=[[5, 1, 0], [0, 4, 0], [0.5, 0, 3]])
    generator = ActionInputGenerator(atoms)

    # Test random index
    index = generator.random_index()
    assert 0 <= index < len(atoms)

    # Test random indices
    indices = generator.random_indices(2)
    assert len(indices) == 2
    for idx in indices:
        assert 0 <= idx < len(atoms)

    # Test random radius
    radius = generator.random_radius()
    assert radius >= 0.5 and radius <= 5.0

    # Test random distance
    distance = generator.random_distance()
    assert distance >= 0.0 and distance <= 1.0

    # Test random symbol
    symbol = generator.random_symbol()

    # Test random position
    position = generator.random_position()
    assert len(position) == 3