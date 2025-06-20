# raw cif --> processed cif + action prompt
# the info are stored in a csv file, with columns:
# 'cif_file', 'action_prompt', 'processed_cif_file'
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from atom_world.actions import *
from ase.data import chemical_symbols
import random
import inspect
import csv

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

class ActionInputGenerator:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.dpos_scale = 2.0  # scale for random displacements

    def random_index(self):
        return random.randint(0, len(self.atoms) - 1)

    def random_indices(self):
        count = random.randint(1, len(self.atoms))
        return random.sample(range(len(self.atoms)), count)

    def random_radius(self, min_r=1.5, max_r=5.0):
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
        # choose from [1,0,0], [0,1,0], [0,0,1] only
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return axes[random.randint(0, len(axes) - 1)]
    
    def random_angle(self):
        # Generate a random angle in radians
        return random.uniform(0, 2 * np.pi)
    
    def generate_inputs_for_action(self, action_cls):
        sig = inspect.signature(action_cls.__init__)
        kwargs = {}
        for name, param in list(sig.parameters.items())[1:]:  # skip 'self'
            if name == 'atoms':
                kwargs[name] = self.atoms
            elif name == 'index' or name.startswith('index'):
                kwargs[name] = self.random_index()
            elif name == 'indices':
                kwargs[name] = self.random_indices()
            elif name == 'radius':
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
                kwargs[name] = self.random_distance()
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
            csv_path = os.path.join(action_folder, f"{action_name}.csv")
            with open(csv_path, "w", newline='') as csvfile:
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
                        continue
                    output_cif = os.path.join(action_folder, f"{os.path.splitext(cif_file)[0]}_processed.cif")
                    write(output_cif, processed_atoms)
                    prompt = str(action)
                    writer.writerow([input_path, prompt, output_cif])





if __name__ == "__main__":
    all_actions = single_atom_actions
    data_gen = DataGenerator("input_cifs", "output_cifs")
    data_gen.generate_data(all_actions)
