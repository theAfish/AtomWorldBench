"""Tool functions for describing objects in AtomWorldBench."""
from collections.abc import Iterable
from numpy.typing import ArrayLike

def format_arraylike(obj: ArrayLike, precision: int = 4):
    """
    Recursively format any array-like object into a string with adjustable float precision.

    Args:
        obj(ArrayLike): array-like structure (list, tuple, np.ndarray, nested or flat)
        precision(int): number of decimal places to format floats. Default is 4.

    Returns:
        str: formatted string of the array-like object.
    """
    if isinstance(obj, (int, float)):
        return f"{obj:.{precision}f}"
    elif isinstance(obj, str):
        return repr(obj)
    elif isinstance(obj, Iterable):
        inner = ", ".join(format_arraylike(item, precision) for item in obj)
        return f"({inner})"
    else:
        raise TypeError(f"Unsupported element type: {type(obj)}")