from typing import Optional

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