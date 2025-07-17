"""Tool functions for describing objects in AtomWorldBench."""
from collections.abc import Iterable
from typing import Optional

from numpy.typing import ArrayLike


def get_species_string(
        element_symbol: str, charge: Optional[int] = None,
):
    """Generate a string representation of species with optional charge.

    Args:
        element_symbol (str): Element symbol, e.g., "H".
        charge (Optional[int]): Charge of the species, e.g., -2. Default is None.

    Returns:
        str: Formatted species string, e.g., "H+"
    """
    if not isinstance(charge, (int, type(None))):
        raise TypeError("Charge must be an integer or None.")
    if charge is not None and charge != 0:
        if charge == 1:
            return f"{element_symbol} +"
        elif charge == -1:
            return f"{element_symbol} -"
        else:
            if charge > 0:
                return f"{element_symbol} {charge}+"
            else:
                return f"{element_symbol} {abs(charge)}-"
    return element_symbol


def format_arraylike(obj: ArrayLike, precision: int = 4):
    """Recursively format any array-like object into a string with adjustable float precision.

    Args:
        obj(ArrayLike): array-like structure (list, tuple, np.ndarray, nested or flat)
        precision(int): number of decimal places to format floats. Default is 4.
         If precision is 0, integers will be returned as integers.

    Returns:
        str: formatted string of the array-like object.
    """
    if isinstance(obj, (int, float)):
        if precision > 0:
            return f"{obj:.{precision}f}"
        else:
            return str(int(obj))
    elif isinstance(obj, str):
        return repr(obj)
    elif isinstance(obj, Iterable):
        inner = ", ".join(format_arraylike(item, precision) for item in obj)
        return f"({inner})"
    else:
        raise TypeError(f"Unsupported element type: {type(obj)}")