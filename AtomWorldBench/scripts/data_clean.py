import pandas as pd
import os
import glob

folder = "./src/data/"  # Change this to your folder path

for csv_file in glob.glob(os.path.join(folder, "*.csv")):
    df = pd.read_csv(csv_file)
    if "input_cif" in df.columns:
        df["input_cif"] = df["input_cif"].str.replace(r"^input_cifs\\", "", regex=True)
        df.to_csv(csv_file, index=False)