from typing import Sequence, Optional

from ase import Atoms
import numpy as np


def merge_atoms(
        all_atoms: Sequence[Atoms],
        all_atoms_indices: Optional[Sequence[Sequence[int]]] = None,
) -> Atoms:
    """Merge two ASE Atoms objects based on specified indices.

    Args:
        all_atoms (List[Atoms]): A list of ASE Atoms objects to merge.
        all_atoms_indices (Optional[List[List[int]]]): A list of lists of locations
            for the sites of each Atoms object to be placed in the resulting merged
            Atoms object. The indices must form a complete permutation of range(n_total),
            describing a global reordering of all atoms.
            If None, will simply concatenate the Atoms objects.

    Returns:
        Atoms: A new ASE Atoms object containing the merged atoms.
    """
    # Check if all atoms have the same cell and pbc.
    if not all(
            np.allclose(a.cell.complete(), all_atoms[0].cell.complete())
            and np.all(a.pbc == all_atoms[0].pbc) for a in all_atoms
    ):
        raise ValueError("All Atoms objects must have the same cell and pbc.")

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
    # Build reverse lookup: global_index → (structure_id, local_index)
    index_map = {}
    for struct_id, indices in enumerate(all_atoms_indices):
        for local_index, global_index in enumerate(indices):
            if global_index in index_map:
                raise ValueError(f"Duplicate global index {global_index}")
            index_map[global_index] = (struct_id, local_index)

    # Verify completeness
    if set(index_map.keys()) != set(range(n_total)):
        missing = set(range(n_total)) - set(index_map.keys())
        raise ValueError(f"Missing indices in all_atoms_indices: {missing}")

    # Now assemble merged list in correct order
    merged_indices = [index_map[i] for i in range(n_total)]

    final_atoms = all_atoms[merged_indices[0][0]][[merged_indices[0][1]]].copy()
    for struct_id, site_id in merged_indices[1:]:
        final_atoms += all_atoms[struct_id][[site_id]]
    return final_atoms
