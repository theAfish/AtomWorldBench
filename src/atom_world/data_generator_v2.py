"""Simplified data generator using the new actions_v2 API.

Usage:
 - Create action instances (from actions_v2) and call generate_data(action_classes)

The generator will call action.apply_random(atoms, rng) and write outputs.
"""
import os
import csv
import traceback
import numpy as np
from ase.io import read, write
from ase import Atoms
from typing import List
from atom_world.actions_v2 import BaseActionV2


class DataGeneratorV2:
    def __init__(self, input_dir: str, output_dir: str, rng: np.random.Generator | None = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_data(self, action_classes: List[type[BaseActionV2]]):
        cif_files = [f for f in os.listdir(self.input_dir) if f.endswith('.cif')]
        for action_cls in action_classes:
            action_name = action_cls.__name__
            action_folder = os.path.join(self.output_dir, action_name)
            os.makedirs(action_folder, exist_ok=True)
            csv_path = os.path.join(self.output_dir, f"{action_name}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["input_cif", "action_prompt", "output_cif"])
                for cif_file in cif_files:
                    input_path = os.path.join(self.input_dir, cif_file)
                    try:
                        atoms = read(input_path)
                    except Exception as e:
                        print(f"Failed to read {input_path}: {e}")
                        continue
                    action = action_cls()
                    try:
                        out_atoms = action.apply_random(atoms, rng=self.rng, copy=True)
                    except Exception as e:
                        print(f"Action {action_name} failed on {cif_file}: {e}")
                        traceback.print_exc()
                        continue
                    name = os.path.splitext(cif_file)[0]
                    output_cif = os.path.join(action_folder, f"{name}_processed.cif")
                    try:
                        write(output_cif, out_atoms)
                    except Exception as e:
                        print(f"Failed to write output CIF for {cif_file}: {e}")
                        continue
                    prompt = str(action)
                    writer.writerow([f"{name}.cif", prompt, f"{name}_processed.cif"])
