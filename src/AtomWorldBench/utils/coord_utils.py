from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import ArrayLike


def check_coordinates_shape(
        coord: ArrayLike,
        name: Optional[str] = None,
        expected_1d: bool = True,
) -> np.ndarray:
    """Check if the shape of the fractional coordinates matches the expected shape.

    Returns np.ndarray if the shape matches, otherwise throws an error.
    Args:
        coord (ArrayLike): The fractional coordinates to check.
        name (Optional[str]): Optional name for the coordinates, used in error messages.
        expected_1d (bool): If True, expects a 1D array. If False, expects a 2D array.
            Default is True.
    Returns:
        bool: True if the shape matches, False otherwise.
    """
    coord = np.asarray(coord, dtype=float)
    name = name or "coordinates"
    if expected_1d:
        if coord.ndim != 1 or coord.shape[0] != 3:
            raise ValueError(
                f"{name} expected 1D array of shape (3,),"
                f" got shape {coord.shape}"
            )
    else:
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(
                f"{name} expected 2D array with second dimension of size 3,"
                f" got shape {coord.shape}"
            )
    return coord

def check_integer_translation(
        frac1: ArrayLike,
        frac2: ArrayLike,
        atol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Check if two sets of fractional coordinates are related by an integer translation.

    Permutations of the coordinates are allowed, but the translation vector must be unique.
    Args:
        frac1 (ArrayLike): First set of fractional coordinates.
        frac2 (ArrayLike): Second set of fractional coordinates.
        atol (float): Tolerance for floating point comparison. Default is 1e-6.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] | None: A tuple containing:
            - Lexicographically sorted indices of frac1 according to ascending x, y, z.
            - Lexicographically sorted indices of frac2 according to ascending x, y, z.
            - Translation vector if the two sets are related by an integer translation,
             otherwise None.
            Or None if the input arrays are not related.

    """
    a1 = np.asarray(frac1, dtype=float)
    a2 = np.asarray(frac2, dtype=float)
    if a1.shape != a2.shape:
        return None

    # Sort the arrays to handle permutations.
    idx1 = np.lexsort((a1[:, 2], a1[:, 1], a1[:, 0]))
    idx2 = np.lexsort((a2[:, 2], a2[:, 1], a2[:, 0]))

    diffs = a2[idx2] - a1[idx1]

    # Check if the differences are close to integers, and are unique.
    taus = np.round(diffs).astype(int)
    if (
            np.allclose(diffs, taus, atol=atol)
            and np.unique(taus, axis=0).shape[0] == 1
    ):
        return idx1, idx2, taus[0]
    return None


def find_coordinate_subset_indices(subset, fullset, wrap=True, atol=1e-8) -> List[int] | None:
    """Find indices of a subset of fractional coordinates in a full set.

    Args:
        subset (ArrayLike): The subset of fractional coordinates to find.
        fullset (ArrayLike): The full set of fractional coordinates.
        wrap (bool): Whether to wrap the coordinates before comparison.
            Default is True.
        atol (float): Absolute tolerance for floating point comparison. Default is 1e-8.
    Returns:
        List[int] | None: Indices of the subset in the full set if found, otherwise None.
    """
    indices = []
    subset = np.array(subset, dtype=float)
    fullset = np.array(fullset, dtype=float)
    if wrap:
        subset = np.mod(subset, 1.0)
        fullset = np.mod(fullset, 1.0)

    for a in subset:
        matches = np.where(np.all(np.isclose(fullset, a, atol=atol), axis=1))[0]
        if len(matches) == 0:
            return None
        indices.append(matches[0])  # 取第一个匹配
    return indices
