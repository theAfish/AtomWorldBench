from models.base_model import BaseModel
import os
import pandas as pd
from utils.dataloader import load_data
from utils.extract_data import extract_from_string
from prompts.cif_action_prompt import cif_action_prompt
from evaluation.metrics import load_cif_file_from_string, check_atom_counts, match_structures
from tqdm import tqdm
import logging
import time

logging.captureWarnings(True)
logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

    def evaluate(self, batch_size: int = 8, num_batch: int = -1):
        """
        Evaluate the model on the provided data in batches.
        """
        results = []
        wrongs = []
        num_unreadable_out = 0
        num_invalid_cif = 0

        prompts = []
        rows = []
        batch_count = 0
        for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="LLM Calling"):
            input_cif = row['input_cif']
            action_prompt = row['action_prompt']
            output_cif = row['output_cif']

            # check if the input_cif is empty
            if input_cif is None:
                raise ValueError(f"input_cif is None at row {i}")

            prompt = cif_action_prompt(
                input_cif=input_cif,
                action_prompt=action_prompt,
                output_format="cif"
            )
            prompts.append(prompt)
            rows.append(row)

            # Process in batches
            if len(prompts) == batch_size or i == len(self.data) - 1:
                # Generate responses in batch
                generated_outputs = self.model.generate_batch(prompts)
                for j, generated_output in enumerate(generated_outputs):
                    row = rows[j]
                    generated_cif = extract_from_string(generated_output, format="cif")
                    if generated_cif is None:
                        logging.info(f"Invalid generated output for index {i - len(prompts) + 1 + j}")
                        num_unreadable_out += 1
                        wrongs.append({
                            "input_cif": row['input_cif'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_output,
                            "target_cif": row['output_cif'],
                        })
                        continue

                    input_structure = load_cif_file_from_string(row['input_cif'])
                    output_structure = load_cif_file_from_string(row['output_cif'])
                    generated_structure = load_cif_file_from_string(generated_cif)

                    if generated_structure is None:
                        logging.info(f"Invalid generated structure for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif": row['input_cif'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_output,
                            "target_cif": row['output_cif'],
                        })
                        continue

                    atom_counts_match = check_atom_counts(output_structure, generated_structure)
                    if not atom_counts_match:
                        logging.info(f"Atom counts do not match for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif": row['input_cif'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_cif,
                            "target_cif": row['output_cif'],
                        })
                        continue

                    rmsd, max_diff = match_structures(output_structure, generated_structure)
                    if rmsd == -1:
                        logging.info(f"Structures do not match for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif": row['input_cif'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_output,
                            "target_cif": row['output_cif'],
                        })
                        continue

                    results.append({
                        "input_cif": row['input_cif'],
                        "action_prompt": row['action_prompt'],
                        "generated_cif": generated_cif,
                        "target_cif": row['output_cif'],
                        "rmsd": rmsd,
                        "max_diff": max_diff,
                    })
                    print(f"RMSD: {rmsd}, Max Diff: {max_diff}")

                prompts = []
                rows = []
                batch_count += 1

                # Stop if num_batch is reached
                if num_batch > 0 and batch_count >= num_batch:
                    break

        # Print summary of evaluation
        print(f"Evaluation completed. Total inputs: {len(self.data)}, ")
        print(f"Unreadable outputs: {num_unreadable_out}, Invalid CIFs: {num_invalid_cif}")

        # Save results to a DataFrame and then to a CSV file
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Evaluation results saved to {results_csv_path}")

        # Save wrongs to a DataFrame and then to a CSV file
        wrongs_df = pd.DataFrame(wrongs)
        wrongs_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_wrongs.csv")
        wrongs_df.to_csv(wrongs_csv_path, index=False)