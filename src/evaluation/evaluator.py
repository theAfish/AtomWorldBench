from models.base_model import BaseModel
import os
import pandas as pd
from utils.dataloader import load_data
from utils.extract_data import extract_from_string
from prompts.cif_action_prompt import cif_action_prompt
from evaluation.metrics import load_cif_file_from_string, check_atom_counts, match_structures

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
        for _, row in self.data.iterrows():
            input_cif = row['input_cif']
            action_prompt = row['action_prompt']
            output_cif = row['output_cif']

            # Generate response from the model
            generated_output = self.model.generate(action_prompt, input_cif=input_cif)
            generated_output = extract_from_string(generated_output, format="cif")
            if generated_output is None:
                print(f"Invalid generated output for input: {input_cif}")
                num_unreadable_out += 1
                continue

            # Load structures from CIF strings
            input_structure = load_cif_file_from_string(input_cif)
            output_structure = load_cif_file_from_string(output_cif)
            generated_structure = load_cif_file_from_string(generated_output)

            if generated_structure is None:
                print(f"Invalid generated structure for input: {input_cif}")
                num_invalid_cif += 1
                continue

            # Check atom counts and structure match
            atom_counts_match = check_atom_counts(input_structure, generated_structure)
            structures_match = match_structures(input_structure, generated_structure)

            results.append({
                "input_cif": input_cif,
                "action_prompt": action_prompt,
                "generated_output": generated_output,
                "atom_counts_match": atom_counts_match,
                "structures_match": structures_match
            })

        # Save results to a DataFrame and then to a CSV file
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Evaluation results saved to {results_csv_path}")