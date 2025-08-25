import numpy as np
import numpy.testing as npt
from ase import Atoms

from AtomWorldBench.utils.neighbor_utils import detect_indices_offsets_around_frac_coords

from tests.utils import assert_array_permuated_equal

# Test 2 standard cases with simple structures
def test_detect_neighbors():
    simple_cubic_cu = Atoms(
        "Cu",
        positions=[
            (0, 0, 0),
        ],
        cell=[2, 2, 2],
        pbc=True
    )

    indices, offsets = detect_indices_offsets_around_frac_coords(simple_cubic_cu, frac_coords=np.array([0.5, 0.5, 0.5]),
                                                                 cutoff=1.5)  # Nearest neighbor is at 1.732 away, so no neighbors should be detected.
    assert len(indices) == 0
    assert len(offsets) == 0

    indices, offsets = detect_indices_offsets_around_frac_coords(simple_cubic_cu, frac_coords=np.array([0.5, 0.5, 0.5]),
                                                                 cutoff=2.0)
    # Nearest neighbor is at 1.732 away, 2-nd nearest at 3.316,
    # so only the first one should be detected. Total of 8 neighbors.
    assert len(indices) == 8
    assert len(offsets) == 8
    # All detected zero.
    npt.assert_array_equal(
        indices, 0
    )
    assert_array_permuated_equal(
        offsets,
        np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
    )

    indices, offsets = detect_indices_offsets_around_frac_coords(simple_cubic_cu, frac_coords=np.array([0.5, 0.5, 0.5]),
                                                                 cutoff=2.0, symbols=["Ag"])
    # Ag does not exist in the structure, so no neighbors should be detected.
    assert len(indices) == 0
    assert len(offsets) == 0

    cux = Atoms(
        ["Cu", "X"],
        positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
        ],
        cell=[2, 2, 2],
        pbc=True
    )
    indices, offsets = detect_indices_offsets_around_frac_coords(cux, frac_coords=np.array([0.5, 0.5, 0.5]), cutoff=1.5)
    # Even though X is at the center, it should not be detected.
    assert len(indices) == 0
    assert len(offsets) == 0
