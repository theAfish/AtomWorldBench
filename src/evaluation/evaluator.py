from models.base_model import BaseModel
import os
import pandas as pd
from utils.dataloader import load_data
from utils.extract_data import extract_from_string
from prompts.cif_action_prompt import cif_action_prompt
from evaluation.metrics import load_cif_file_from_string, check_atom_counts, match_structures
from tqdm import tqdm

class Evaluator:
    def __init__(
            self, 
            model: BaseModel,
            data_folder: str, 
            action_name: str = None,
            results_folder: str = "results"
    ):
        """
        Initialize the Evaluator with the data folder and optional action name.
        """
        self.model = model
        self.data_folder = data_folder
        self.action_name = action_name
        self.data = load_data(data_folder, action_name)
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

    def evaluate(self):
        """
        Evaluate the model on the provided data.
        """
        results = []
        num_unreadable_out = 0
        num_invalid_cif = 0
        for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Evaluating"):
        # for i, row in self.data.iterrows():
            input_cif = row['input_cif']
            action_prompt = row['action_prompt']
            output_cif = row['output_cif']

            prompt = cif_action_prompt(
                input_cif=input_cif,
                action_prompt=action_prompt,
                output_format="cif"
            )

            # Generate response from the model
            generated_output = self.model.generate(prompt)
            generated_output = extract_from_string(generated_output, format="cif")
            if generated_output is None:
                print(f"Invalid generated output for index {i}")
                num_unreadable_out += 1
                continue

            # Load structures from CIF strings
            input_structure = load_cif_file_from_string(input_cif)
            output_structure = load_cif_file_from_string(output_cif)
            generated_structure = load_cif_file_from_string(generated_output)

            if generated_structure is None:
                print(f"Invalid generated structure for index {i}")
                num_invalid_cif += 1
                continue

            # Check atom counts and structure match
            atom_counts_match = check_atom_counts(output_structure, generated_structure)
            if not atom_counts_match:
                print(f"Atom counts do not match for index {i}")
                num_invalid_cif += 1
                continue
            
            rmsd, max_diff = match_structures(output_structure, generated_structure)
            if rmsd == -1:
                print(f"Structures do not match for index {i}")
                num_invalid_cif += 1
                continue

            results.append({
                "input_cif": input_cif,
                "action_prompt": action_prompt,
                "generated_output": generated_output,
                "atom_counts_match": atom_counts_match,
                "rmsd": rmsd,
                "max_diff": max_diff,
            })
            print(f"RMSD: {rmsd}, Max Diff: {max_diff}")
            # for debug 
            if i > 10:
                break

        # Print summary of evaluation
        print(f"Evaluation completed. Total inputs: {len(self.data)}, ")
        print(f"Unreadable outputs: {num_unreadable_out}, Invalid CIFs: {num_invalid_cif}")

        # Save results to a DataFrame and then to a CSV file
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Evaluation results saved to {results_csv_path}")