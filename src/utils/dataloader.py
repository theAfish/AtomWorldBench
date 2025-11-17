import os
import pandas as pd
from scripts.load_data_from_h5 import load_cifs_from_hdf5
from pymatgen.io.cif import CifParser
import logging


def load_cif_file(cif_file, primitive=True):
    """
    Load a CIF file and return the first structure.
    If the file is not valid, return None.
    """
    try:
        parser = CifParser(cif_file)
        structures = parser.parse_structures(primitive=primitive, check_occu=False)
        if structures:
            return structures[0]
        else:
            logging.info(f"No structures found in {cif_file}.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF file {cif_file}: {e}")
        return None

def load_cif_file_from_string(cif_string, primitive=True):
    """
    Load a CIF file from a string and return the first structure.
    If the string is not valid, return None.
    """
    try:
        parser = CifParser.from_str(cif_string)
        structures = parser.parse_structures(primitive=primitive, check_occu=False)
        if structures:
            return structures[0]
        else:
            logging.info("No structures found in the CIF string.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF from string: {e}")
        return None
    


def load_data(data_folder, action_name=None, input_cifs="input_cifs.hdf5"):
    """
    Loads data for training/analysis.
    Args:
        data_folder: Path to the folder containing input_cifs.hdf5, action CSVs, and HDF5s.
        action_name: If specified, only load data for this action (e.g., 'add_atom_action').
    Returns:
        pd.DataFrame with columns: input_cif, action_prompt, output_cif
    """
    # Load all input CIFs
    input_cifs_path = os.path.join(data_folder, input_cifs)
    input_cifs = load_cifs_from_hdf5(input_cifs_path)

    # Find all action CSVs
    files = os.listdir(data_folder)
    action_csvs = [f for f in files if f.endswith(".csv")]
    if action_name:
        action_csvs = [f"{action_name}.csv"] if f"{action_name}.csv" in action_csvs else []

    rows = []
    for csv_file in action_csvs:
        action_base = csv_file[:-4]
        h5_file = f"{action_base}.hdf5"
        h5_path = os.path.join(data_folder, h5_file)
        # Load output CIFs for this action
        output_cifs = load_cifs_from_hdf5(h5_path) if os.path.exists(h5_path) else {}
        # Read CSV
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # input_cif: get from input_cifs dict
            input_cif_path = os.path.basename(row["input_cif"])
            input_cif_str = input_cifs.get(input_cif_path, "")
            # action_prompt: from CSV
            action_prompt = row["action_prompt"]
            # output_cif: get from output_cifs dict
            output_cif_path = os.path.basename(row["output_cif"])
            output_cif_str = output_cifs.get(output_cif_path, "")
            rows.append({
                "input_cif": input_cif_str,
                "action_prompt": action_prompt,
                "output_cif": output_cif_str
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data_folder = "." 
    action_name = 'add_atom_action' 
    df = load_data(data_folder, action_name)
    print(df.head()['action_prompt'])