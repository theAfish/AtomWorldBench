from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser
import numpy as np
import logging

def load_cif_file(cif_file):
    """
    Load a CIF file and return the first structure.
    If the file is not valid, return None.
    """
    try:
        parser = CifParser(cif_file)
        structures = parser.parse_structures(primitive=True)
        if structures:
            return structures[0]
        else:
            logging.info(f"No structures found in {cif_file}.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF file {cif_file}: {e}")
        return None

def load_cif_file_from_string(cif_string):
    """
    Load a CIF file from a string and return the first structure.
    If the string is not valid, return None.
    """
    try:
        parser = CifParser.from_str(cif_string)
        structures = parser.parse_structures(primitive=True)
        if structures:
            return structures[0]
        else:
            logging.info("No structures found in the CIF string.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF from string: {e}")
        return None

def check_atom_counts(struct1, struct2):
    """
    Check if two structures have the same number of atoms for each atom type.
    Returns True if they match, False otherwise.
    """
    if len(struct1) != len(struct2):
        return False

    # Count atoms in both structures
    counts1 = struct1.composition.as_dict()
    counts2 = struct2.composition.as_dict()

    # Compare counts
    return counts1 == counts2

# def match_structures(struct1, struct2):
#     """
#     Match two structures using StructureMatcher.
#     Returns True if they match, False otherwise.
#     """
#     matcher = StructureMatcher(primitive_cell=False, stol=0.5)
#     if matcher.fit(struct1, struct2):
#         rmsd = matcher.get_rms_dist(struct1, struct2)[0]
#         mapping = matcher.get_mapping(struct1, struct2)
#     else:
#         logging.info("Structures do NOT match within tolerances.")
#         return -1, -1
    
#     coords1 = struct1.cart_coords
#     coords2 = struct2.cart_coords

#     # Build ordered arrays based on mapping
#     coords2_matched = np.array([
#         coords2[mapping[i]] for i in range(len(coords1))
#     ])
#     # Compute RMSD without alignment
#     diff = coords1 - coords2_matched
#     rmsd = np.sqrt((diff**2).sum(axis=1).mean())
#     max_diff = np.max(np.linalg.norm(diff, axis=1))
#     return rmsd, max_diff

def match_structures(struct1, struct2):
    """
    Match two structures using StructureMatcher.
    """
    matcher = StructureMatcher(primitive_cell=False, stol=0.5)
    if matcher.fit(struct1, struct2):
        rmsd, max_dist = matcher.get_rms_dist(struct1, struct2)
        return rmsd, max_dist
    else:
        logging.info("Structures do NOT match within tolerances.")
        return -1, -1

# def get_rmsd_and_maxdiff(struct1, struct2):
#     """
#     Compute RMSD and maximum difference in positions between two structures.
#     """
#     coords1 = struct1.cart_coords
#     coords2 = struct2.cart_coords

#     # Ensure both structures have the same number of atoms
#     if len(coords1) != len(coords2):
#         raise ValueError("Structures must have the same number of atoms for RMSD calculation.")

#     # Compute the RMSD
#     diff = coords1 - coords2
#     rmsd = np.sqrt((diff**2).sum(axis=1).mean())
#     max_diff = np.max(np.linalg.norm(diff, axis=1))

#     return rmsd, max_diff