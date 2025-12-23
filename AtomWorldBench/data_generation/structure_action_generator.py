import os
import io
import json
from typing import List, Dict, Any, Iterator, Optional
from ase.io import read, write
from ase import Atoms
from AtomWorldBench.data_generation.base_data_generator import BaseDataGenerator
from AtomWorldBench.atom_world.actions.structure_actions import BaseStructureAction
from AtomWorldBench.atom_world.actions.motif_actions import BaseMotifAction
from AtomWorldBench.common.registry import get_registered

class StructureActionGenerator(BaseDataGenerator):
    """
    Generator for structure-action pairs using CIF files and registered actions.
    """
    
    def __init__(
        self,
        cif_folder: str,
        action_names: List[str],
        seed: Optional[int] = 75,
        max_attempts: int = 10,
        is_random: bool = True,
        allow_repeat_structures: bool = True
    ):
        """
        Initialize the generator.

        Args:
            cif_folder (str): Path to the folder containing CIF files.
            action_names (List[str]): List of action names to sample from.
            seed (int, optional): Random seed.
            max_attempts (int): Max attempts to generate a valid sample per iteration.
            allow_repeat_structures (bool): If False, each structure (CIF file) will be used at most once
                                            across all generation calls until reset.
        """
        super().__init__(seed)
        self.cif_folder = cif_folder
        self.is_random = is_random
        self.allow_repeat_structures = allow_repeat_structures
        
        if not os.path.exists(cif_folder):
             raise ValueError(f"CIF folder not found: {cif_folder}")

        self.cif_files = [
            os.path.join(cif_folder, f) 
            for f in os.listdir(cif_folder) 
            if f.endswith('.cif')
        ]
        
        if not self.cif_files:
            raise ValueError(f"No CIF files found in {cif_folder}")
        
        # Generate indices for CIF files
        self.cif_indices = self._init_indices()
            
        self.action_names = action_names
        self.max_attempts = max_attempts
        
        # Pre-fetch registered actions
        self.structure_registry = get_registered(BaseStructureAction)
        self.motif_registry = get_registered(BaseMotifAction)
        
        self.available_actions = self._resolve_actions(action_names)

    def _resolve_actions(self, action_names: List[str]) -> Dict[str, Any]:
        resolved = {}
        for name in action_names:
            # Registry keys are typically lower-case kebab-case or similar.
            # We try exact match, lower case, or just check if it's in values.
            
            found = False
            # Try direct lookup in structure registry
            if name in self.structure_registry:
                resolved[name] = self.structure_registry[name]
                found = True
            # Try motif registry
            elif name in self.motif_registry:
                resolved[name] = self.motif_registry[name]
                found = True
            
            if not found:
                print(f"Warning: Action '{name}' not found in registries.")
        
        if not resolved:
            raise ValueError("No valid actions found from the provided list.")
            
        return resolved
    
    def _init_indices(self):
        data_size = len(self.cif_files)
        indices = list(range(data_size))
        if self.is_random:
            if not self.allow_repeat_structures:
                indices = self.rng.integers(0, data_size, size=data_size).tolist()
            else:
                self.rng.shuffle(indices)
        return indices

    def _atoms_to_cif_string(self, atoms: Atoms) -> str:
        with io.BytesIO() as buffer:
            write(buffer, atoms, format='cif')
            return buffer.getvalue().decode('utf-8')

    def generate(self, num_samples: int = -1, action_name: Optional[str] = None, **kwargs) -> Iterator[Dict[str, Any]]:
        count = 0
        idx = 0
        if num_samples < 0:
            num_samples = len(self.cif_indices)
        while count < num_samples:
            # Try to generate a valid sample
            for _ in range(self.max_attempts):
                # Check if we need to reset indices or stop
                if idx >= len(self.cif_indices):
                    if not self.allow_repeat_structures:
                        print("No more unique structures available. Stopping generation.")
                        return # Stop generation as we ran out of structures
                    else:
                        self.cif_indices = self._init_indices()
                        idx = 0

                try:
                    cif_idx = self.cif_indices[idx]
                    atoms = read(self.cif_files[cif_idx])
                    # Ensure atoms are wrapped so motifs see the same coordinates. This is a must!
                    atoms.wrap()
                    
                    
                    if action_name:
                        if action_name not in self.available_actions:
                            raise ValueError(f"Action {action_name} not available.")
                        action_key = action_name
                    else:
                        action_key = self.rng.choice(list(self.available_actions.keys()))
                        
                    action_cls = self.available_actions[action_key]
                    
                    # Generate random action instance
                    action = action_cls.get_random_one(operated_atoms=atoms)
                    
                    if action is None:
                        # Move to next structure if action generation failed
                        idx += 1
                        continue
                        
                    action_prompt = action.describe()
                    input_cif = self._atoms_to_cif_string(atoms)
                    
                    result_atoms = action.execute()
                    output_cif = self._atoms_to_cif_string(result_atoms)
                    
                    yield {
                        "action_prompt": action_prompt,
                        "input": input_cif,
                        "output": output_cif
                    }

                    # Move to next structure after success
                    idx += 1
                    count += 1
                    break
                except Exception as e:
                    # assert False
                    print(f"Error generating sample: {e}")
                    print("File path: ", self.cif_files[self.cif_indices[idx]])
                    # Move to next structure on error
                    idx += 1
                    continue

    def generate_and_save_per_action(self, output_dir: str, num_samples_per_action: int):
        """
        Generates samples for each action and saves them to separate JSON files.
        
        Args:
            output_dir (str): Directory to save the JSON files.
            num_samples_per_action (int): Number of samples to generate per action.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for action_name in self.available_actions:
            samples = []
            # Generate samples for this specific action
            gen = self.generate(num_samples_per_action, action_name=action_name)
            for sample in gen:
                samples.append(sample)
            
            if samples:
                out_file = os.path.join(output_dir, f"{action_name}.json")
                with open(out_file, 'w') as f:
                    json.dump(samples, f, indent=2)
                print(f"Saved {len(samples)} samples for {action_name} to {out_file}")
            else:
                print(f"No samples generated for {action_name}")


# debug
if __name__ == "__main__":
    cif_folder = "D:\\Codes\\AtomWorld\\src\\data\\_raw_data\\input_cifs"
    output_dir = "D:\\Codes\\AtomWorld\\debug\\output_cifs"
    action_names = [
        "ChangeElementAction",
        "LatticeTransformAction",
        "MakeSupercellAction",
        "RotateStructureAction",
        "AddMotifAction",
        "RemoveMotifAction",
        "ReplaceMotifAction",
        "ResizeMotifAction",
        "RotateMotifAction",
        "SwapMotifAction",
        "TranslateMotifAction"
    ]
    
    generator = StructureActionGenerator(
        cif_folder=cif_folder,
        action_names=action_names,
        max_attempts=5,
        is_random=True,
        allow_repeat_structures=False
    )
    
    generator.generate_and_save_per_action(
        output_dir=output_dir,
        num_samples_per_action=1000
    )

    # # load and print a sample
    # sample_file = os.path.join(output_dir, "AddMotifAction.json")
    # with open(sample_file, 'r') as f:
    #     samples = json.load(f)
    #     print(f"Loaded {len(samples)} samples from {sample_file}")
    #     if samples:
    #         print("Sample 0:")
    #         print("Action Prompt:", samples[0]['action_prompt'])
    #         print("Input CIF:", samples[0]['input'])
    #         print("Output CIF:", samples[0]['output'])
