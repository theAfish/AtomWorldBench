import json
from utils.dataloader import load_data
import pandas as pd
from prompts.cif_action_prompt_for_json import cif_action_prompt

action_names = [
    "add_atom_action",
    "remove_atom_action",
    "change_atom_action",
    "delete_below_atom_action",
    "swap_atoms_action",
    "insert_between_atoms_action",
    "move_atom_action",
    "swap_atoms_action",
    "super_cell_action",
    "rotate_around_atom_action",
]

for action_name in action_names:
    dataframs = load_data(data_folder="src/data", action_name=action_name)
    cif_csv_path = f"src/data/{action_name}.csv"
    cif_csv = pd.read_csv(cif_csv_path)
    # get the mp-ids from the csv (the input_cif is mp-id.cif)
    mp_ids = [mp_id.split(".")[0] for mp_id in cif_csv['input_cif'].tolist()]


    qa_data = []


    for index, row in dataframs.iterrows():
        prompt = cif_action_prompt(
            action_prompt=row['action_prompt'],
            output_format="cif"
        )
        input_cif=row['input_cif'],
        
        qa_pair = {
            "instruction": prompt,
            "input": row['input_cif'],
            "output": row['output_cif']
        }
        qa_data.append(qa_pair)

    with open(f"src/data/{action_name}_qa_format.json", "w") as f:
        json.dump(qa_data, f, indent=4)
        print(f"Saved {len(qa_data)} QA pairs to {action_name}_qa_format.json")

    