# raw cif --> processed cif + action prompt
# the info are stored in a csv file, with columns:
# 'cif_file', 'action_prompt', 'processed_cif_file'
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from atom_world.actions import *
from atom_world.cell_actions import SuperCellAction
import csv
import traceback

from scripts.convert_cifs_to_h5 import convert_cifs_to_h5

# classification of actions
single_atom_actions = [
    AddAtomAction, RemoveAtomAction, MoveAtomAction, ChangeAtomAction,
]

double_atom_actions = [
    SwapAtomsAction, InsertBetweenAtomsAction, MoveTowardsAtomAction,
]

multiple_atom_actions = [
    DeleteBelowAtomAction, DeleteAroundAtomAction, MoveSelectedAtomsAction, 
    MoveAroundAtomAction, RotateAroundAtomAction,
]

cell_actions = [
    SuperCellAction,
]


class DataGenerator:
    def __init__(self, input_dir: str, output_dir: str, rng=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_data(self, action_classes):
        cif_files = [f for f in os.listdir(self.input_dir) if f.endswith('.cif')]
        for action_cls in action_classes:
            action_name = action_cls.__name__
            # convert class name to lowercase and add underscore between words
            action_name = ''.join(['_' + c.lower() if c.isupper() else c for c in action_name]).lstrip('_')
            action_folder = os.path.join(self.output_dir, action_name)
            os.makedirs(action_folder, exist_ok=True)
            csv_path = os.path.join(self.output_dir, f"{action_name}.csv")
            with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["input_cif", "action_prompt", "output_cif"])
                for cif_file in cif_files:
                    input_path = os.path.join(self.input_dir, cif_file)
                    atoms = read(input_path)
                    try:
                        action, processed_atoms = action_cls.apply_random(
                            atoms, rng=self.rng, copy=True
                        )
                        if action is None:
                            continue
                    except Exception as e:
                        print(f"Action {action_name} failed on {cif_file}: {e}")
                        traceback.print_exc()
                        continue
                    name = os.path.splitext(cif_file)[0]
                    output_cif = os.path.join(action_folder, f"{name}_processed.cif")
                    try:
                        write(output_cif, processed_atoms)
                    except Exception as e:
                        print(f"Failed to write output CIF for {cif_file}: {e}")
                        continue
                    prompt = str(action)
                    writer.writerow([f"{name}.cif", prompt, f"{name}_processed.cif"])
    
    def generate_analysis_data(self, action_cls, action_kwargs, analysis_name="analysis"):
        cif_files = [f for f in os.listdir(self.input_dir) if f.endswith('.cif')]
        action_name = action_cls.__name__
        
        action_name = ''.join(['_' + c.lower() if c.isupper() else c for c in action_name]).lstrip('_') + f"_{analysis_name}"
        action_folder = os.path.join(self.output_dir, action_name)
        os.makedirs(action_folder, exist_ok=True)
        csv_path = os.path.join(self.output_dir, f"{action_name}.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["input_cif", "action_prompt", "output_cif"])
            for cif_file in cif_files:
                input_path = os.path.join(self.input_dir, cif_file)
                atoms = read(input_path)
                kwargs = action_kwargs.copy()
                kwargs['atoms'] = atoms
                action = action_cls(**kwargs)
                processed_atoms = action.execute()
                name = os.path.splitext(cif_file)[0]
                output_cif = os.path.join(action_folder, f"{name}_processed.cif")
                try:
                    write(output_cif, processed_atoms)
                except Exception as e:
                    print(f"Failed to write output CIF for {cif_file}: {e}")
                    continue
                prompt = str(action)
                writer.writerow([f"{name}.cif", prompt, f"{name}_processed.cif"])


if __name__ == "__main__":
    # # for data generation
    # all_actions = [DeleteBelowAtomAction] #[SuperCellAction]
    # data_gen = DataGenerator("input_cifs", "output_cifs")
    # data_gen.generate_data(all_actions)

    # convert_cifs_to_h5(
    #     folder_path="output_cifs/delete_below_atom_action_bad/",
    #     hdf5_output_path="delete_below_atom_action_bad.hdf5"
    # )
    

    # for analysis data generation

    data_gen = DataGenerator("D:/Codes/AtomWorld/src/data/_raw_data/input_cifs", 
                             "D:/Codes/AtomWorld/src/data/_raw_data/output4analysis")
    data_gen.generate_analysis_data(
        MoveAtomAction,
        # action_kwargs={'index1': -2, 'index2': -1, 'distance_ratio': 0.45, 'symbol': 'H'},
        action_kwargs={'index': 0, 'd_pos': np.array([0.1, 0.1, 0.1])},
        analysis_name="late_pos"
    )

    # Convert CIFs to HDF5
    convert_cifs_to_h5(
        folder_path="D:/AI/PythonProjects/AtomWorldBench/src/data/_raw_data/input4analysis/pos",
        hdf5_output_path="analysis_input_pos.hdf5"
    )

    convert_cifs_to_h5(
        folder_path="D:/Codes/AtomWorld/src/data/_raw_data/output4analysis/insert_between_atoms_action_late_pos/",
        hdf5_output_path="insert_between_atoms_action_late_pos.hdf5"
    )