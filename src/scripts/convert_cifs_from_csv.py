import pandas as pd
from utils.dataloader import load_cif_file_from_string
import os


def convert_cifs_from_csv(csv_file, cif_col, cif_folder='cifs', name_col=None):
    df = pd.read_csv(csv_file)
    os.makedirs(cif_folder, exist_ok=True)
    for i, row in df.iterrows():
        if name_col is None:
            cif = load_cif_file_from_string(row[cif_col])
            name = cif.formula.replace(' ', '_') + '-0.cif'
        
        cif_path = os.path.join(cif_folder, name)
        with open(cif_path, 'w') as f:
            f.write(row[cif_col])





if __name__ == "__main__":
    file = "D:/Codes/AtomWorld/results/AtomWorld/deepseek_chat/super_cell_action/20250919_110406/evaluation_results.csv"
    convert_cifs_from_csv(file, "target_cif")