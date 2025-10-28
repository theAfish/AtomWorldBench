import numpy as np
from AtomWorldBench.common.registry import get_registered
from AtomWorldBench.atom_world.motifs.base import BaseMotif


def assert_array_permuated_equal(arr1: np.ndarray, arr2: np.ndarray):
    """Assert that two arrays are equal up to a permutation of their rows.

    Args:
        arr1 (np.ndarray): The first array to compare.
        arr2 (np.ndarray): The second array to compare.

    Raises:
        AssertionError: If the arrays are not equal up to a permutation of their rows.
    """
    if arr1.shape != arr2.shape:
        raise AssertionError(f"Arrays have different shapes: {arr1.shape} vs {arr2.shape}")

    # Sort the rows of both arrays
    sorted_arr1 = np.array(sorted(map(tuple, arr1)))
    sorted_arr2 = np.array(sorted(map(tuple, arr2)))

    # Compare the sorted arrays
    if not np.array_equal(sorted_arr1, sorted_arr2):
        raise AssertionError("Arrays are not equal up to a permutation of their rows.")


def get_random_motif(class_alias, atoms, seed=42, **kwargs):
    """Helper function to get a random motif of a given class alias."""
    motif_class = get_registered(BaseMotif)[class_alias]
    assert issubclass(motif_class, BaseMotif)
    return motif_class.detect_random_one(atoms, seed=seed, **kwargs)
