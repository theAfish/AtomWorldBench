import os
import io
import json
from typing import List, Dict, Any, Iterator, Optional, Type
import uuid
from ase.io import read, write
from ase import Atoms
from data_generation.base_data_generator import BaseDataGenerator
from atom_world.actions import (
    BaseAction,
    AddAtomAction, RemoveAtomAction, MoveAtomAction, ChangeAtomAction,
    SwapAtomsAction, InsertBetweenAtomsAction, MoveTowardsAtomAction,
    DeleteBelowAtomAction, DeleteAroundAtomAction, MoveSelectedAtomsAction,
    MoveAroundAtomAction, RotateAroundAtomAction,
)
from atom_world.cell_actions import SuperCellAction

ready_actions = [
    AddAtomAction,
    RemoveAtomAction,
    MoveAtomAction,
    ChangeAtomAction,
    SwapAtomsAction,
    InsertBetweenAtomsAction,
    MoveTowardsAtomAction,
    DeleteBelowAtomAction,
    DeleteAroundAtomAction,
    MoveSelectedAtomsAction,
    MoveAroundAtomAction,
    RotateAroundAtomAction,
    SuperCellAction,
]


class CIFActionGenerator(BaseDataGenerator):
    """
    Generator for cif-action pairs using CIF files and registered actions.
    Produces per-action JSON files with inline CIF strings.
    """

    def __init__(
        self,
        cif_folder: str,
        action_classes: Optional[List[Type]] = None,
        seed: Optional[int] = 75,
        max_attempts: int = 10,
        is_random: bool = True,
        allow_repeat_structures: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            cif_folder: Path to the folder containing CIF files.
            action_classes: List of action classes to sample from.
            seed: Random seed.
            max_attempts: Max attempts to generate a valid sample per iteration.
            is_random: Whether to randomize the order of CIF files.
            allow_repeat_structures: If False, each CIF file is used at most once.
        """
        super().__init__(seed)
        self.cif_folder = cif_folder
        self.is_random = is_random
        self.allow_repeat_structures = allow_repeat_structures
        self.max_attempts = max_attempts

        if not os.path.exists(cif_folder):
            raise ValueError(f"CIF folder not found: {cif_folder}")

        self.cif_files = [
            os.path.join(cif_folder, f)
            for f in os.listdir(cif_folder)
            if f.endswith('.cif')
        ]

        if not self.cif_files:
            raise ValueError(f"No CIF files found in {cif_folder}")

        self.cif_indices = self._init_indices()
        self.action_classes = action_classes if action_classes else ready_actions

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
            return buffer.getvalue().decode('utf-8').replace('\r\n', '\n')

    def _generate_action_and_result(self, atoms: Atoms, action_cls: Type):
        """Create a random action instance and produce result atoms."""
        action, result_atoms = action_cls.apply_random(atoms, rng=self.rng, copy=True)
        return action, result_atoms

    def generate(
        self,
        num_samples: int = -1,
        action_cls: Optional[Type] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        count = 0
        idx = 0
        if num_samples < 0:
            num_samples = len(self.cif_indices)

        while count < num_samples:
            for _ in range(self.max_attempts):
                if idx >= len(self.cif_indices):
                    if not self.allow_repeat_structures:
                        print("No more unique structures available. Stopping generation.")
                        return
                    else:
                        self.cif_indices = self._init_indices()
                        idx = 0

                try:
                    cif_idx = self.cif_indices[idx]
                    atoms = read(self.cif_files[cif_idx])
                    atoms.wrap()

                    if action_cls:
                        selected_cls = action_cls
                    else:
                        selected_cls = self.rng.choice(self.action_classes)

                    action, result_atoms = self._generate_action_and_result(
                        atoms, selected_cls
                    )

                    if action is None:
                        idx += 1
                        continue

                    action_prompt = str(action)
                    input_cif = self._atoms_to_cif_string(atoms)
                    output_cif = self._atoms_to_cif_string(result_atoms)

                    yield {
                        "action_type": selected_cls.__name__,
                        "problem_id": str(uuid.uuid4()), # uuid for each sample
                        "mp_id": os.path.splitext(os.path.basename(self.cif_files[cif_idx]))[0], # source mpid from cif mp-xxxx.cif --> mp-xxxx
                        "action_prompt": action_prompt,
                        "input": input_cif,
                        "output": output_cif,
                    }

                    idx += 1
                    count += 1
                    break
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    print("File path: ", self.cif_files[self.cif_indices[idx]])
                    idx += 1
                    continue

    def generate_and_save_per_action(
        self, output_dir: str, num_samples_per_action: int
    ):
        """
        Generates samples for each action and saves them to separate JSON files.

        Args:
            output_dir: Directory to save the JSON files.
            num_samples_per_action: Number of samples to generate per action.
        """
        os.makedirs(output_dir, exist_ok=True)

        for action_cls in self.action_classes:
            action_name = action_cls.__name__
            samples = list(
                self.generate(num_samples_per_action, action_cls=action_cls)
            )

            if samples:
                out_file = os.path.join(output_dir, f"{action_name}.json")
                with open(out_file, 'w') as f:
                    json.dump(samples, f, indent=2)
                print(f"Saved {len(samples)} samples for {action_name} to {out_file}")
            else:
                print(f"No samples generated for {action_name}")


def generate_all(
    cif_folder: str,
    output_dir: str,
    action_classes: Optional[List[Type]] = None,
    num_samples_per_action: int = 1000,
    max_attempts: int = 10,
    is_random: bool = True,
    allow_repeat_structures: bool = False,
    seed: Optional[int] = 75,
):
    """
    Generate CIF-action pairs for all specified actions and save to output directory.
    """
    generator = CIFActionGenerator(
        cif_folder=cif_folder,
        action_classes=action_classes,
        seed=seed,
        max_attempts=max_attempts,
        is_random=is_random,
        allow_repeat_structures=allow_repeat_structures,
    )
    generator.generate_and_save_per_action(
        output_dir=output_dir,
        num_samples_per_action=num_samples_per_action,
    )


def main(args=None):
    from utils.args import get_generation_parser

    parser = get_generation_parser()
    args = parser.parse_args(args)

    generate_all(
        cif_folder=args.cif_folder,
        output_dir=args.output_dir,
        num_samples_per_action=args.num_samples,
        max_attempts=args.max_attempts,
        is_random=not args.no_random,
        allow_repeat_structures=args.allow_repeat,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
