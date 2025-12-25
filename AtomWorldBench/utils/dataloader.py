import os
import json
import pandas as pd
import logging
from pymatgen.io.cif import CifParser


def load_cif_file(cif_file, primitive=True):
    """
    Load a CIF file and return the first structure.
    If the file is not valid, return None.
    """
    try:
        parser = CifParser(cif_file)
        structures = parser.parse_structures(primitive=primitive)
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
        structures = parser.parse_structures(primitive=primitive)
        if structures:
            return structures[0]
        else:
            logging.info("No structures found in the CIF string.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF from string: {e}")
        return None


def load_data(data_folder, action_name=None):
    """
    Loads data for training/analysis from JSON files.
    Args:
        data_folder: Path to the folder containing action JSON files.
        action_name: If specified, only load data for this action (e.g., 'AddMotifAction').
    Returns:
        pd.DataFrame with columns: input_cif, action_prompt, output_cif
    """
    # Find all action JSONs
    if not os.path.exists(data_folder):
        logging.error(f"Data folder {data_folder} does not exist.")
        return pd.DataFrame()

    files = os.listdir(data_folder)
    action_jsons = [f for f in files if f.endswith(".json")]
    
    if action_name:
        # Check if action_name has .json extension or not
        target_file = f"{action_name}.json" if not action_name.endswith(".json") else action_name
        if target_file in action_jsons:
            action_jsons = [target_file]
        else:
            logging.warning(f"Action {action_name} not found in {data_folder}.")
            return pd.DataFrame()

    rows = []
    for json_file in action_jsons:
        json_path = os.path.join(data_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # data is expected to be a list of dictionaries
            for item in data:
                rows.append({
                    "input_cif": item.get("input", ""),
                    "action_prompt": item.get("action_prompt", ""),
                    "output_cif": item.get("output", "")
                })
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")

    return pd.DataFrame(rows)

