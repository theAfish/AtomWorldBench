from utils.dataloader import load_data
from evaluation.metrics import load_cif_file_from_string, load_cif_file


data_folder = "../data"  # Adjust this to your actual data folder
action_name = 'insert_between_atoms_action'  # Specify an action name if needed, e.g., 'add_atom_action'
df = load_data(data_folder, action_name)
# print(df['output_cif'][2])

cif = load_cif_file_from_string(df['output_cif'][2])
print(cif)

cif = load_cif_file("D:\Codes\AtomWorld\src\data/raw_data\output_cifs\insert_between_atoms_action\mp-1016569_processed.cif")
print(cif)