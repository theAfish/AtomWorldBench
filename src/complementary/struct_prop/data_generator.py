import os
import re
from complementary.prompts.property_action_prompt import property_action_prompt
import pandas as pd
import random


properties = ["band gap", "bulk modulus"]
property_trends = ["increase", "decrease"]

property_head_map = {
    "band gap": "bandgap",
    "bulk modulus": "bulkmodulus_vrh"
}

structure_path = "./raw_data/"

def get_mp_number(filename):
    match = re.search(r'mp-(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def generate_input_data(prop, output_file):

    cif_files = [f for f in os.listdir(structure_path+'structures/non_metal/') if f.endswith('.cif')]
    cif_files = sorted(cif_files, key=get_mp_number)

    data_files = pd.read_csv(structure_path + 'non_metal.csv')

    rows = []
    for cif_file in cif_files:
        input_path = os.path.join(structure_path, 'structures/non_metal/', cif_file)
        with open(input_path, 'r') as f:
            cif_string = f.read()

        # randomly choose a trend
        trend = random.choice(property_trends)

        prompt = property_action_prompt(
            input_cif=cif_string,
            target_property=prop,
            target_trend=trend
        )

        # get the mp-id from the cif file name
        mp_id = get_mp_number(cif_file)
        # find the corresponding property value from the data files
        prop_value = data_files.loc[data_files['mpid'] == f'mp-{mp_id}', property_head_map[prop]].values[0]
        
        # pd dataframe column: input_cif_name, property, trend, prompt, origin_prop_value
        rows.append([cif_file, prop, trend, prompt, prop_value])

    df = pd.DataFrame(rows, columns=["input_cif_name", "property", "trend", "prompt", "origin_prop_value"])
    df.to_csv(output_file, index=False)




if __name__ == "__main__":
    for prop in properties:
        generate_input_data(prop=prop, output_file=f"{property_head_map[prop]}_nonmetal.csv")