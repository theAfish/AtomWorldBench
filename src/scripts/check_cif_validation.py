# -*- coding: utf-8 -*-
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import os
import logging
from pathlib import Path
import glob

def check_partial_occupancy(structure):
    """
    Check if the structure has any sites with partial occupancy.
    Returns a list of sites with partial occupancy and their occupancy values.
    """
    partial_occ_sites = []
    for site in structure:
        species_and_occu = site.species.as_dict()
        if len(species_and_occu) > 1 or any(occ != 1.0 for occ in species_and_occu.values()):
            partial_occ_sites.append((site, species_and_occu))
    return partial_occ_sites

def check_close_atoms(structure, min_distance=0.5):
    """
    Check if there are any atoms that are too close to each other.
    min_distance is in Angstroms.
    Returns a list of pairs of sites that are too close and their distances.
    """
    too_close_pairs = []
    all_distances = structure.distance_matrix
    num_sites = len(structure)
    
    # Look at upper triangle of distance matrix to avoid duplicates
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            if 0 < all_distances[i,j] < min_distance:
                too_close_pairs.append((
                    structure[i], 
                    structure[j], 
                    all_distances[i,j]
                ))
    return too_close_pairs

def validate_cif_file(cif_path, min_distance=0.5):
    """
    Validate a CIF file for partially occupied sites and too-close atoms.
    Returns a tuple of (partial_occ_sites, too_close_pairs) or (None, None) if file can't be parsed.
    """
    try:
        parser = CifParser(cif_path)
        structure = parser.parse_structures(primitive=True)[0]
        
        partial_occ = check_partial_occupancy(structure)
        too_close = check_close_atoms(structure, min_distance)
        
        return partial_occ, too_close
    except Exception as e:
        logging.error("Error processing {}: {}".format(cif_path, str(e)))
        return None, None

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # Path to results directory
    results_dir = Path("results/StructPropBench")
    
    # Find all .cif files recursively in the results directory
    cif_files = glob.glob(str(results_dir / "**/*.cif"), recursive=True)

    # print out all the folders found
    logging.info(f"Found {len(cif_files)} CIF files in {results_dir}")
    
    # Process each CIF file
    for cif_path in cif_files:
        partial_occ, too_close = validate_cif_file(cif_path)
        
        if partial_occ is None:  # File couldn't be processed
            continue
            
        # Report findings for this file
        rel_path = Path(cif_path).relative_to(results_dir)
        
        if partial_occ:
            logging.warning("\nPartially occupied sites found in {}:".format(rel_path))
            for site, occupancies in partial_occ:
                logging.warning("  Site at {}: {}".format(site.coords, occupancies))
                
        if too_close:
            logging.warning("\nToo-close atoms found in {}:".format(rel_path))
            for site1, site2, distance in too_close:
                logging.warning(
                    "  Distance {:.3f} A between:"
                    "\n    {} at {}"
                    "\n    {} at {}".format(
                        distance,
                        site1.species_string, site1.coords,
                        site2.species_string, site2.coords
                    )
                )

if __name__ == "__main__":
    main()