from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import logging


def check_atom_counts(struct1, struct2):
    """
    Check if two structures have the same number of atoms for each atom type.
    Returns True if they match, False otherwise.
    """
    if len(struct1) != len(struct2):
        return False

    counts1 = struct1.composition.as_dict()
    counts2 = struct2.composition.as_dict()

    return counts1 == counts2


def match_structures(struct1, struct2, primitive_cell=False, stol=0.5, **kwargs):
    """
    Match two structures using StructureMatcher.
    """
    matcher = StructureMatcher(primitive_cell=primitive_cell, stol=stol, **kwargs)
    if matcher.fit(struct1, struct2):
        rmsd, max_dist = matcher.get_rms_dist(struct1, struct2)
        return rmsd, max_dist
    else:
        logging.info("Structures do NOT match within tolerances.")
        return None, None


def compute_exact_match_positional_metrics(struct1, struct2, stol=0.5):
    """Compute RMSD/max distance with minimum-image PBC, assuming identical lattices."""
    if len(struct1) != len(struct2):
        logging.info("Structures have different number of atoms; cannot compute RMSD.")
        return None, None

    for site1, site2 in zip(struct1.sites, struct2.sites):
        if site1.species_string != site2.species_string:
            logging.info("Atom types do not match in order; cannot compute RMSD.")
            return -1, -1

    lattice1 = struct1.lattice
    lattice2 = struct2.lattice
    if not np.allclose(lattice1.matrix, lattice2.matrix):
        logging.info(
            "Structures have different lattices; cannot compute RMSD under PBC assumption."
        )
        return -1, -1

    frac1 = np.array([site.frac_coords for site in struct1.sites])
    frac2 = np.array([site.frac_coords for site in struct2.sites])

    frac_diffs = frac1 - frac2
    frac_diffs -= np.round(frac_diffs)

    diffs = frac_diffs @ lattice1.matrix
    sq_norms = np.sum(diffs**2, axis=1)
    normalization = (len(struct1) / lattice1.volume) ** (1 / 3)
    rmsd = np.sqrt(np.mean(sq_norms)) * normalization
    max_dist = np.sqrt(np.max(sq_norms)) * normalization

    if max_dist > stol:
        logging.info(f"Max distance {max_dist} exceeds tolerance {stol}.")
        return -1, -1

    return rmsd, max_dist


def check_partially_occupied_sites(struct):
    """
    Check if a structure has partially occupied sites.
    Returns True if any site is partially occupied, False otherwise.
    """
    for site in struct.sites:
        for occu in site.species.values():
            if occu < 1.0:
                return True
    return False


def check_atoms_too_close(struct, threshold=0.5):
    """
    Check if any two atoms in the structure are closer than the given threshold (in Angstroms).
    Returns True if any pair is too close, False otherwise.
    """
    dists = struct.distance_matrix
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < threshold):
        return True
    return False
