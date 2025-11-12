from AtomWorldBench.utils.coord_utils import (
    check_coordinates_shape,
    check_lattice_matrix_shape,
    check_integer_translation,
    find_coordinate_subset_indices,
)

import numpy as np
import numpy.testing as npt


def test_check_lattice_matrix():
    # Test valid 3x3 matrix
    arr = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    result = check_lattice_matrix_shape(arr, "lattice")
    npt.assert_array_equal(result, arr)

    # Test invalid shape
    arr_invalid = np.array([[1.0, 0.0],
                            [0.0, 1.0]])
    try:
        check_lattice_matrix_shape(arr_invalid, "lattice")
    except ValueError as e:
        assert str(e) == (
            "lattice expected 2D array with shape (3, 3), got shape (2, 2)"
        )

    # Test None.
    result = check_lattice_matrix_shape(
        None, "lattice", allow_none=True
    )
    assert result is None
    try:
        check_lattice_matrix_shape(
            None, "lattice", allow_none=False
        )
    except ValueError as e:
        assert str(e) == "lattice cannot be None."


def test_check_coordinates_shape():
    # Test valid 1D array
    arr = np.array([1.0, 2.0, 3.0])
    result = check_coordinates_shape([1.0, 2.0, 3.0], "test", expected_1d=True)
    npt.assert_array_equal(result, arr)

    # Test valid 2D array with shape (1, 3)
    arr_2d = np.array([[1.0, 2.0, 3.0]])
    result = check_coordinates_shape(arr_2d, "test", expected_1d=False)
    npt.assert_array_equal(result, arr_2d)

    # Test invalid shape
    arr_invalid = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        check_coordinates_shape(arr_invalid, "test", expected_1d=True)
    except ValueError as e:
        assert str(e) == "test expected 1D array of shape (3,), got shape (2, 2)"
    try:
        check_coordinates_shape(arr_invalid, "test", expected_1d=False)
    except ValueError as e:
        assert str(e) == (
            "test expected 2D array with second dimension of size 3,"
            " got shape (2, 2)"
        )

    # Test None.
    result = check_coordinates_shape(None, "test", allow_none=True)
    assert result is None
    try:
        check_coordinates_shape(None, "test", allow_none=False)
    except ValueError as e:
        assert str(e) == "test cannot be None."


def test_check_integer_translation():
    # Test random valid integer translation.
    for _ in range(100):
        n = np.random.randint(1, 6)  # Random size for the arrays
        frac1 = np.random.rand(n, 3)
        shuffle_ids = np.random.permutation(n)
        rand_shift = np.random.randint(-5, 6, size=(3,))
        frac2 = frac1[shuffle_ids, :] + rand_shift
        sort_ids1, sort_ids2, shift = check_integer_translation(frac1, frac2)
        npt.assert_array_equal(
            sort_ids1,
            np.lexsort((frac1[:, 2], frac1[:, 1], frac1[:, 0]))
        )
        npt.assert_array_equal(
            sort_ids2,
            np.lexsort((frac2[:, 2], frac2[:, 1], frac2[:, 0]))
        )
        npt.assert_array_almost_equal(shift, rand_shift)
        npt.assert_array_almost_equal(frac2[sort_ids2], frac1[sort_ids1] + shift)

    # Test random invalid non-integer translation.
    for _ in range(30):
        n = np.random.randint(1, 6)  # Random size for the arrays
        frac1 = np.random.rand(n, 3)
        rand_shift = (np.random.rand(3) + np.pi / 100) * np.pi  # Guaranteed to be non-integer.
        frac2 = frac1 + rand_shift
        result = check_integer_translation(frac1, frac2)
        assert result is None

    # Test random invalid integer translation. (each row has a different shift)
    for _ in range(50):
        n = np.random.randint(1, 6)  # Random size for the arrays
        frac1 = np.random.rand(n, 3)
        rand_shift = np.random.randint(-5, 6, size=(n, 3))
        if np.allclose(rand_shift, rand_shift[0]):
            continue  # Ensure not all shifts are the same, though probability is low.
        frac2 = frac1 + rand_shift
        result = check_integer_translation(frac1, frac2)
        assert result is None

    # Test fully random invalid case.
    for _ in range(50):
        n = np.random.randint(1, 6)  # Random size for the arrays
        frac1 = np.random.rand(n, 3)
        frac2 = np.random.rand(n, 3)  # Completely random, unlikely to be related.
        if np.allclose(frac1, frac2):
            continue # Ensure not the same, though probability is low.
        result = check_integer_translation(frac1, frac2)
        assert result is None

def test_find_coordinate_subset_indices():
    # Test unwrapped case.
    for _ in range(20):
        all_coords = np.random.rand(20, 3)
        subset = np.random.randint(20, size=5)
        subset_coords = all_coords[subset]
        found_indices = find_coordinate_subset_indices(subset_coords, all_coords, wrap=False)
        npt.assert_array_equal(subset, found_indices)

    # Test wrapped case.
    for _ in range(20):
        all_coords = np.random.rand(20, 3)
        subset = np.random.randint(20, size=5)
        subset_shifts = np.random.randint(-5, 5, size=(5, 3))
        subset_coords = all_coords[subset] + subset_shifts
        found_indices = find_coordinate_subset_indices(subset_coords, all_coords, wrap=True)
        npt.assert_array_equal(subset, found_indices)

    # Test not found case.
    for _ in range(20):
        all_coords = np.random.rand(20, 3)
        subset = np.random.randint(20, size=5)
        subset_coords = all_coords[subset] + np.random.rand(5, 3) * 0.1 + 10.0
        # Shifted far away, will not match
        found_indices = find_coordinate_subset_indices(subset_coords, all_coords, wrap=True)
        assert found_indices is None

        # Shift only one coordinate, will not match
        subset_coords = all_coords[subset]
        subset_coords[np.random.randint(5), np.random.randint(3)] += 0.1
        found_indices = find_coordinate_subset_indices(subset_coords, all_coords, wrap=True)
        assert found_indices is None
