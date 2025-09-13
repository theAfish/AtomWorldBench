# this code is used for doing bad things in cif files.
import os
import io
import pandas as pd

# missing means dead
essential_tags_for_missing = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c", 
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma", 
    "_atom_site_type_symbol", "_atom_site_label","_atom_site_symmetry_multiplicity",
    "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
    "_atom_site_occupancy"
]

# wrong spelling can cause error while loading
essential_tags_for_spelling = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c", 
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma", 
    "_atom_site_label",
    "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"
]

# # these tags have standard names, but do not influence the structure reading.
# wrong_spelling_is_ok_for_reading = [
#     "_atom_site_type_symbol", "_atom_site_symmetry_multiplicity", "_atom_site_occupancy"
# ]

misleading_changes = [
    {"_length_a":"_length_x", "_length_b":"_length_y", "_length_c":"_length_z"},
    {"_length_a":"_length_u", "_length_b":"_length_v", "_length_c":"_length_w"},
    {"_length_a":"_length_i", "_length_b":"_length_j", "_length_c":"_length_k"},
    {"_atom_site":"_atom"},
    {"_cell_length": "_cell", "_cell_angle":"_cell"},
    {"_cell": "_lattice"},
    {"_fract_x": "_frac_a","_fract_y": "_frac_b","_fract_z": "_frac_c"},
    {"_fract_x": "_frac_u","_fract_y": "_frac_v","_fract_z": "_frac_w"},
    {"_fract_x": "_frac_i","_fract_y": "_frac_j","_fract_z": "_frac_k"},
]


def read_raw_cif(cif_file):
    # read cif file as text
    with open(cif_file, 'r') as f:
        raw_cif = f.read()
    return raw_cif

def line_omission(cif_file, tag):
    """
    Omit a line containing the specified tag from the CIF file.
    Return the modified CIF content as a string. 
    If there's a numerical value in the deleted line, also return the value in string format as it is.
    Example:
        modified_cif = line_omission('example.cif', '_cell_length_a')
    """
    raw_cif = read_raw_cif(cif_file)
    modified_lines = []
    num_values = None
    for line in raw_cif.splitlines():
        if tag in line:
            # check if there's a numerical value in the line
            parts = line.split()
            for part in parts:
                try:
                    float(part)
                    num_values = part  # store the numerical value as string
                    break
                except ValueError:
                    continue
            continue
        modified_lines.append(line)
    modified_cif = "\n".join(modified_lines)
    return modified_cif, num_values

def tag_misspelling(cif, mapping):
    """
    Give some misleading misspellings to the cif tags.
    Return the modified CIF content as a string.
    """
    raw_cif = read_raw_cif(cif)
    modified_cif = raw_cif
    for old_tag, new_tag in mapping.items():
        modified_cif = modified_cif.replace(old_tag, new_tag)
    return modified_cif




if __name__ == "__main__":

    records = []
    cif_file = 'H2O.cif'
    for tag in essential_tags_for_missing:
        modified_cif, val = line_omission(cif_file, tag)
        records.append({
            'original_cif': read_raw_cif(cif_file),
            'modified_cif': modified_cif,
            'modification_type': 'line_omission',
            'modify_tag(s)': tag,
            'removed_value': val
        })

    for mapping in misleading_changes:
        modified_cif = tag_misspelling(cif_file, mapping)
        records.append({
            'original_cif': read_raw_cif(cif_file),
            'modified_cif': modified_cif,
            'modification_type': 'tag_misspelling',
            'modify_tag(s)': str(mapping),
            'removed_value': None
        })

    df = pd.DataFrame(records)
    df.to_csv('cif_modifications.csv', index=False)
