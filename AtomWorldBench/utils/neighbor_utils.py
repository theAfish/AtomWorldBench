"""Detect neighbors, etc."""
from ase import Atoms
from ase.neighborlist import neighbor_list

import numpy as np
from numpy.typing import ArrayLike


def detect_indices_offests_around_frac_coords(
        atoms: Atoms,
        frac_coords: ArrayLike,
        cutoff: float,
        symbols: list[str] = None,
):
    """Detect indices and offsets of atoms around the given fractional coordinates.

    Args:
        atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
        frac_coords (ArrayLike): Fractional coordinates for the atom detection center.
            Must be a one-dimensional array of shape (3,).
        cutoff (float): The radius within which to detect atoms.
        symbols (list[str]): If provided, only atoms with these symbols will be detected.
            Default is None, which means no filtering.
    Returns:
        tuple: A tuple containing:
            - indices_j_valid (ndarray[int]): Indices of atoms that are within the cutoff
                distance from the given fractional coordinates.
            - offsets_valid (ndarray[float]): Offsets of the detected atoms relative to the
                given fractional coordinates.
    """
    # Add a dummy atom to the structure at the given fractional coordinates to mark.
    atoms_modified = atoms.copy()
    cart_coords = frac_coords @ atoms_modified.cell.complete()
    atoms_modified += Atoms("X", positions=[cart_coords], cell=atoms.cell, pbc=atoms.pbc)
    # Dummy atom is added to the end of the atoms list.
    dummy_index = len(atoms_modified) - 1

    # Get the indices of atoms within the cutoff distance from the given fractional coordinates
    # No need to use NeighborList class, as it also checks for the whole structure.
    indices_i, indices_j, offsets = neighbor_list(
        "ijS",
        atoms_modified,
        cutoff=cutoff,
        self_interaction=False,
    )

    symbols_atoms = np.array(atoms_modified.get_chemical_symbols())
    # Filter indices to only include those that are within the cutoff distance from the dummy atom,
    # and that match the specified symbols if provided.
    if symbols is None:
        symbols = [sym for sym in np.unique(symbols_atoms).tolist() if sym != "X"]
    indices_valid = (
            (indices_i == dummy_index) &
            np.vectorize(lambda x: x in symbols, otypes=[bool])(
                symbols_atoms[indices_j]
            )
    )
    indices_j_valid = indices_j[indices_valid]
    offsets_valid = offsets[indices_valid, :]

    return indices_j_valid, offsets_valid