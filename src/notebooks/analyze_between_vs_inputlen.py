# load the csv data and check success rate
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from utils.dataloader import load_cif_file_from_string
from transformers import AutoTokenizer

model_name = "qwen3_4B"
action_name = "insert_between_atoms_action_natoms"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# get the datetime folder name
current_file = Path(__file__)
folder = current_file.parent / f"../../results/AtomWorld/{model_name}/{action_name}/"
datetime_folders = os.listdir(folder)
# get the latest folder
latest_folder = sorted(datetime_folders)[-1]
csv_path = os.path.join(folder, latest_folder, f"evaluation_results.csv")
csv_wrongs_path = os.path.join(folder, latest_folder, f"evaluation_wrongs.csv")
df = pd.read_csv(csv_path)
df_wrongs = pd.read_csv(csv_wrongs_path)

def count_tokens(s):
    if not isinstance(s, str):
        return 0
    return len(tokenizer.encode(s))

df["input_cif_token_len"] = df["input_cif"].apply(count_tokens)
df_wrongs["input_cif_token_len"] = df_wrongs["input_cif"].apply(count_tokens)

# Count items per bin
success_counts = df["input_cif_token_len"].value_counts().sort_index()
wrong_counts = df_wrongs["input_cif_token_len"].value_counts().sort_index()

# Combine into one DataFrame
bin_stats = pd.DataFrame({
    "success": success_counts,
    "wrong": wrong_counts
}).fillna(0)

# Compute success ratio
bin_stats["success_ratio"] = bin_stats["success"] / (bin_stats["success"] + bin_stats["wrong"])

print(bin_stats)

"""

model_name = "qwen3_32B"
action_name = "insert_between_atoms_action_natoms"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

                     success  wrong  success_ratio
input_cif_token_len                               
837                      9.0    1.0            0.9
1737                    10.0    0.0            1.0
3356                     6.0    4.0            0.6
6562                     4.0    6.0            0.4
10172                    0.0   10.0            0.0
15728                    0.0   10.0            0.0

model_name = "qwen3_4B"
action_name = "insert_between_atoms_action_natoms"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

                     success  wrong  success_ratio
input_cif_token_len                               
837                      7.0      3            0.7
1737                     2.0      8            0.2
3356                     0.0     10            0.0
6562                     0.0     10            0.0
10172                    0.0     10            0.0
15728                    0.0     10            0.0

"""