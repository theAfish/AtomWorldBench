from AtomWorldBench.utils.description_utils import (
    get_species_string,
    describe_arraylike,
)

import numpy as np

def test_get_species_string():
    assert get_species_string("H") == "H"
    assert get_species_string("H", 1) == "H+"
    assert get_species_string("O", 2) == "O2+"
    assert get_species_string("Na", -1) == "Na-"
    assert get_species_string("Cl", 0) == "Cl"
    assert get_species_string("Fe", 3) == "Fe3+"
    assert get_species_string("Cu", -2) == "Cu2-"

def test_describe_arraylike():
    assert describe_arraylike([1, 2, 3], precision=0) == "(1, 2, 3)"
    assert describe_arraylike(np.array([1, 2, 3]), precision=0) == "(1, 2, 3)"
    assert describe_arraylike([1, 2, 3], precision=2) == "(1.00, 2.00, 3.00)"
    assert describe_arraylike((1.23456, 2.34567), precision=3) == "(1.235, 2.346)"
    assert describe_arraylike(np.array((1.23456, 2.34567)), precision=3) == "(1.235, 2.346)"
    assert describe_arraylike([[1, 2], [3, 4]], precision=0) == "((1, 2), (3, 4))"
    assert describe_arraylike("Hello") == "'Hello'"
    assert describe_arraylike(42, precision=0) == "42"
    assert describe_arraylike(3.14159, precision=2) == "3.14"
    assert describe_arraylike([1.0, [2, 3]], precision=2) == "(1.00, (2.00, 3.00))"
