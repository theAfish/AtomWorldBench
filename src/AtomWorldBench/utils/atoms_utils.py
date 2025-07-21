from typing import List, Optional

from ase import Atoms
import numpy as np


def merge_atoms(
        all_atoms: List[Atoms],
        all_atoms_indices: Optional[List[List[int]]] = None,
) -> Atoms:
    """Merge two ASE Atoms objects based on specified indices.

    Args:
        all_atoms (List[Atoms]): A list of ASE Atoms objects to merge.
        all_atoms_indices (Optional[List[List[int]]]): A list of lists of indices
            for the sites of each Atoms object to be placed in the resulting merged
            Atoms object. If None, will simply concatenate the Atoms objects.

    Returns:
        Atoms: A new ASE Atoms object containing the merged atoms.
    """
    if all_atoms_indices is None:
        final_atoms = all_atoms[0].copy()
        for atoms in all_atoms[1:]:
            final_atoms += atoms
        return final_atoms

    for i, (atoms, indices) in enumerate(zip(all_atoms, all_atoms_indices)):
        if len(indices) != len(atoms):
            raise ValueError(
                f"Length of atoms at index {i} does not match length of its indices."
            )

    n_total = sum(len(a) for a in all_atoms)
    if not np.array_equal(
        np.arange(n_total, dtype=int),
        np.sort(np.concatenate(all_atoms_indices, axis=0))
    ):
        raise ValueError(
            "Indices in all_atoms_indices do not match"
            f" the expected range 0~{n_total}."
        )

    # Construct a list containing structure index and site index in structure for
    # each atom in the merged structure.
    merged_indices = []
    for i in range(n_total):
        for struct_id, indices in enumerate(all_atoms_indices):
            if i in indices:
                merged_indices.append((struct_id, indices.index(i)))
                break
        raise ValueError(
            f"Index {i} not found in any of the provided indices."
        )

    final_atoms = all_atoms[merged_indices[0][0]][[merged_indices[0][0]]].copy()
    for struct_id, site_id in merged_indices[1:]:
        final_atoms += all_atoms[struct_id][[site_id]]
    return final_atoms
