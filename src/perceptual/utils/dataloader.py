import pandas as pd
from pymatgen.core.structure import Structure
import os
import json
from prompts.cif_gen_prompt import cif_gen_prompt


def load_data(data_file: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    The CSV is expected to have columns: 'original_cif', 'modified_cif'
    """
    df = pd.read_csv(data_file, sep=',', dtype={'removed_value': str})
    df.fillna(value={'removed_value': 'None'}, inplace=True)
    return df

def load_cif_gen_data(data_folder: str) -> pd.DataFrame:
    """
    Load the dataset for CIF generation from a folder containing cif files.
    The dataframe will have 'structure', 'prompt' columns.
    """
    cif_files = [f for f in os.listdir(data_folder) if f.endswith('.cif')]
    # load the description_card.json
    json_path = os.path.join(data_folder, 'description_cards.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"description_cards.json not found in {data_folder}")
    with open(json_path, 'r') as f:
        description_cards = json.load(f)
    
    data = []
    for cif_file in cif_files:
        cif_path = os.path.join(data_folder, cif_file)
        structure = Structure.from_file(cif_path)

        # the cif file name is in general like "MaterialName_StructureType.cif", 
        # if additional information provided, the name is like "MaterialName_StructureType_additionalinfo1_additionalinfo2_etc.cif"
        name_parts = cif_file[:-4].split('_')
        formula = name_parts[0]
        structure_type = name_parts[1]
        
        description_card = description_cards.get(structure_type, None)
        if description_card is None:
            raise ValueError(f"No description card found for structure type: {structure_type}")
        
        if "must_fields" in description_card:
            must_fields = description_card["must_fields"]  
        else:
            must_fields = []
        if "additional_must" in description_card and len(name_parts) > 2:
            must_fields += description_card["additional_must"]

        raw_additional_info = name_parts[2:] if len(name_parts) > 2 else None
        
        additional_info = {}

        additional_info['formula'] = formula
        additional_info['structure_type'] = structure_type

        if 'a' in must_fields:
            additional_info['a'] = structure.lattice.a
        if 'b' in must_fields:
            additional_info['b'] = structure.lattice.b
        if 'c' in must_fields:
            additional_info['c'] = structure.lattice.c
        
        # other info should be in the order of must_fields
        if raw_additional_info:
            idx = 0
            for field in must_fields:
                if field in ['a', 'b', 'c']:
                    continue
                if idx >= len(raw_additional_info):
                    raise ValueError(f"Not enough additional info provided in filename {cif_file}")
                additional_info[field] = raw_additional_info[idx]
                idx += 1

        prompt = cif_gen_prompt(additional_info)
        
        data.append({
            'structure': structure,
            'prompt': prompt
        })
    
    df = pd.DataFrame(data, columns=['structure', 'prompt'])
    return df
