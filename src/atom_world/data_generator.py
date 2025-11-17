# raw cif --> processed cif + action prompt
# the info are stored in a csv file, with columns:
# 'cif_file', 'action_prompt', 'processed_cif_file'
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from atom_world.actions import *
from atom_world.cell_actions import SuperCellAction
from ase.data import chemical_symbols
import random
import inspect
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

class ActionInputGenerator:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.dpos_scale = 2.0  # scale for random displacements

    def random_indices(self, count=1):
        if len(self.atoms) < count:
            raise ValueError("Not enough atoms to select the required number of indices.")
        return random.sample(range(len(self.atoms)), count)

    def random_radius(self, min_r=1.0, max_r=4.0, index=None):
        # If index is provided, ensure radius does not delete all atoms
        if index is not None and len(self.atoms) > 1:
            # Compute all distances from index to others (with PBC)
            distances = self.atoms.get_distances(index, range(len(self.atoms)), mic=True)
            max_dist = np.max(distances[distances > 0])  # exclude self
            min_dist = np.min(distances[distances > 0])
            max_r = min(max_r, max_dist * 0.99)
            min_r = max(min_r, min_dist * 1.01)
            if max_r < min_r:
                max_r = max_dist * 0.5  # fallback
            if max_r < 0.1:
                max_r = 0.1
        return random.uniform(min_r, max_r)

    def random_distance(self, min_d=0.1, max_d=3.0):
        return random.uniform(min_d, max_d)

    def random_symbol(self):
        return random.choice(chemical_symbols[1:])  # skip index 0 (None)

    def random_position(self):
        # Generate a random position anywhere inside the cell
        frac = np.random.rand(3)  # random fractional coordinates in [0, 1)
        return np.dot(frac, self.atoms.get_cell())
    
    def random_dpos(self):
        # Generate a random displacement vector
        return np.random.randn(3) * self.dpos_scale
    
    def random_axis(self):
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        return axes[random.randint(0, len(axes) - 1)]
    
    def random_angle(self):
        # Generate a random angle in radians
        return random.uniform(45, 315)
    
    def generate_inputs_for_action(self, action_cls):
        sig = inspect.signature(action_cls.__init__)
        kwargs = {}
        used_indices = None
        for name, param in list(sig.parameters.items())[1:]:  # skip 'self'
            if name == 'atoms':
                kwargs[name] = self.atoms
            elif name in ('index1', 'index2'):
                # Ensure index1 and index2 are different and (for SwapAtomsAction) different types
                if used_indices is None:
                    if action_cls.__name__ == "SwapAtomsAction":
                        unique_types = set(atom.symbol for atom in self.atoms)
                        if len(unique_types) < 2:
                            raise ValueError("Not enough atom types to swap.")
                        max_tries = 20
                        for _ in range(max_tries):
                            idx1, idx2 = self.random_indices(2)
                            if self.atoms[idx1].symbol != self.atoms[idx2].symbol:
                                used_indices = [idx1, idx2]
                                break
                        else:
                            raise ValueError("Could not find two atoms of different types to swap.")
                    else:
                        used_indices = self.random_indices(2)
                kwargs[name] = used_indices[0] if name == 'index1' else used_indices[1]
            elif name == 'index':
                if action_cls.__name__ == "DeleteBelowAtomAction":
                    max_tries = 20
                    for _ in range(max_tries):
                        idx = self.random_indices(1)[0]
                        z_threshold = self.atoms[idx].position[2]
                        indices_below = [i for i, atom in enumerate(self.atoms) if atom.position[2] < z_threshold and i != idx]
                        if indices_below:
                            kwargs[name] = idx
                            break
                    else:
                        print(f"Warning: Could not find an atom with atoms below it to delete, skipping...")
                        return None 
                        # raise ValueError("Could not find an atom with atoms below it to delete.")
                else:
                    kwargs[name] = self.random_indices(1)[0]
            elif name == 'indices':
                # You can decide the count or randomize it
                count = random.randint(1, len(self.atoms))
                kwargs[name] = self.random_indices(count)
            elif name == 'radius':
                if 'index' in kwargs:
                    kwargs[name] = self.random_radius(index=kwargs['index'])
                else:
                    kwargs[name] = self.random_radius()
            elif name == 'distance':
                kwargs[name] = self.random_distance()
            elif name == 'symbol':
                kwargs[name] = self.random_symbol()
            elif name == 'position':
                kwargs[name] = self.random_position()
            elif name == 'd_pos':
                kwargs[name] = self.random_dpos()
            elif name == 'distance_ratio':
                kwargs[name] = self.random_distance(max_d=0.9)
            elif name == 'include_self':
                kwargs[name] = random.choice([True, False])
            elif name == 'axis':
                kwargs[name] = self.random_axis()
            elif name == 'angle':
                kwargs[name] = self.random_angle()
            else:
                # fallback: try to use default or skip
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"Don't know how to generate input for {name}")
        return kwargs



class DataGenerator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
                    input_gen = ActionInputGenerator(atoms.copy())
                    try:
                        kwargs = input_gen.generate_inputs_for_action(action_cls)
                        if kwargs is None:  # Skip if no valid inputs could be generated
                            continue
                        action = action_cls(**kwargs)
                        processed_atoms = action.execute()
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



class _DataGenerator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
                    input_gen = ActionInputGenerator(atoms.copy())
                    try:
                        kwargs = input_gen.generate_inputs_for_action(action_cls)
                        action = action_cls(**kwargs)
                        processed_atoms = action.execute()
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


if __name__ == "__main__":
    # # for data generation
    # all_actions = [MoveAllAction] #[SuperCellAction]
    # data_gen = DataGenerator("input_cifs", "output_cifs")
    # data_gen.generate_data(all_actions)

    # convert_cifs_to_h5(
    #     folder_path="output_cifs/move_all_action",
    #     hdf5_output_path="move_all_action.hdf5"
    # )
    

    # for analysis data generation

    data_gen = DataGenerator("D:/Codes/AtomWorld/src/data/_raw_data/input_cifs", 
                             "D:/Codes/AtomWorld/src/data/_raw_data/output4analysis")
    data_gen.generate_analysis_data(
        InsertBetweenAtomsAction,
        action_kwargs={'index1': -2, 'index2': -1, 'distance_ratio': 0.45, 'symbol': 'H'},
        analysis_name="late_pos"
    )

    # Convert CIFs to HDF5
    # convert_cifs_to_h5(
    #     folder_path="D:/AI/PythonProjects/AtomWorldBench/src/data/_raw_data/input4analysis/pos",
    #     hdf5_output_path="analysis_input_pos.hdf5"
    # )

    convert_cifs_to_h5(
        folder_path="D:/Codes/AtomWorld/src/data/_raw_data/output4analysis/insert_between_atoms_action_late_pos/",
        hdf5_output_path="insert_between_atoms_action_late_pos.hdf5"
    )