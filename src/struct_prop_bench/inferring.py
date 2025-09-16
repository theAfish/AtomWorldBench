from models.base_model import BaseModel
import os
import pandas as pd
from utils.extract_data import extract_from_string
from evaluation.metrics import load_cif_file_from_string, check_partially_occupied_sites, check_atoms_too_close
from tqdm import tqdm
import logging
from pathlib import Path
from utils.load_model import load_model, load_config
import argparse
import datetime


CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent
RESULT_DIR = Path(__file__).parent.parent.parent / "results" / "StructPropBench"
os.makedirs(RESULT_DIR, exist_ok=True)

props_data_map = {
    "band_gap": "bandgap_nonmetal.csv",
    "bulk_modulus": "bulkmodulus_nonmetal.csv"
}


logging.captureWarnings(True)
logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PropertyActionInfer:
    def __init__(
            self, 
            model: BaseModel,
            data: pd.DataFrame, 
            results_folder: str = "results"
    ):
        """
        Initialize the Evaluator with the data folder and optional action name.
        """
        self.model = model
        self.data = data
        self.results_folder = results_folder

    def evaluate(self, batch_size: int = 8, num_batch: int = -1):
        """
        Evaluate the model on the provided data in batches.
        """
        results = []
        wrongs = []
        num_unreadable_out = 0
        num_invalid_cif = 0

        # Save results to a DataFrame and then to a CSV file
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.results_folder + "/cifs", exist_ok=True)

        prompts = []
        rows = []
        batch_count = 0
        for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="LLM Calling"):

            prompt = row['prompt']
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
                            "input_cif_name": row['input_cif_name'],
                            "generated_output": generated_output,
                            "wrong_type": "OutputFormatError"
                        })
                        continue

                    generated_structure = load_cif_file_from_string(generated_cif)

                    if generated_structure is None:
                        logging.info(f"Invalid generated structure for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif_name": row['input_cif_name'],
                            "generated_output": generated_output,
                            "wrong_type": "CIFParsingError"
                        })
                        continue

                    # check partially occupied 
                    if check_partially_occupied_sites(generated_structure):
                        logging.info(f"Generated structure has partially occupied sites for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif_name": row['input_cif_name'],
                            "generated_output": generated_output,
                            "wrong_type": "PartialOccupancyError"
                        })
                        continue

                    # check atoms too close
                    if check_atoms_too_close(generated_structure):
                        logging.info(f"Generated structure has atoms too close for index {i - len(prompts) + 1 + j}")
                        num_invalid_cif += 1
                        wrongs.append({
                            "input_cif_name": row['input_cif_name'],
                            "generated_output": generated_output,
                            "wrong_type": "AtomsTooCloseError"
                        })
                        continue

                    # save the structure to cif file
                    output_cif_path = os.path.join(self.results_folder, f"cifs/{row['input_cif_name']}_modified.cif")
                    generated_structure.to(filename=output_cif_path)

                    results.append({
                        "input_cif_name": row['input_cif_name'],
                        "generated_cif": generated_cif,
                        "generated_output": generated_output
                    })

                prompts = []
                rows = []
                batch_count += 1

                # Stop if num_batch is reached
                if num_batch > 0 and batch_count >= num_batch:
                    break

        # Print summary of evaluation
        print(f"Inference completed. Total inputs: {len(self.data)}, ")
        print(f"Unreadable outputs: {num_unreadable_out}, Invalid CIFs: {num_invalid_cif}")
        
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_folder, f"infer_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Inference results saved to {results_csv_path}")

        # Save wrongs to a DataFrame and then to a CSV file
        wrongs_df = pd.DataFrame(wrongs)
        wrongs_csv_path = os.path.join(self.results_folder, f"infer_wrongs.csv")
        wrongs_df.to_csv(wrongs_csv_path, index=False)




def run_inference(
        model_id: str,
        prop: str,
        batch_size: int,
        num_batch: int,
        config_name: str = "models",
        results_folder: str = None
    ):
    config = load_config(CONFIG_DIR / config_name)[model_id]
    model = load_model(config)
    data = pd.read_csv(DATA_DIR / props_data_map[prop])

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or RESULT_DIR}/{model_id}/{prop}/{timestamp}"
    

    evaluator = PropertyActionInfer(
        model=model,
        data=data,
        results_folder=results_folder
    )
    evaluator.evaluate(batch_size=batch_size, num_batch=num_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark with specified configuration.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek_chat",
        help="ID of the model to use (e.g., 'deepseek_chat', 'openai_gpt4')"
    )
    parser.add_argument(
        "-p",
        "--prop",
        type=str,
        default="band_gap",
        help="The property to modify (e.g., 'band_gap', 'bulk_modulus')"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="models",
        help=f"Name of config file (located under src/config)"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for the run. Default: 50"
    )
    parser.add_argument(
        "-n",
        "--num_batch",
        type=int,
        default=-1,
        help="Number of batches to use. Default: -1 for all data"
    )
    parser.add_argument(
        "-f",
        "--results_folder",
        type=str,
        default=None,
        help="Folder to save results. Default: 'results/struct_prop_bench/{model_id}/{prop}/{timestamp}'"
    )
    args = parser.parse_args()

    run_inference(
        model_id=args.model,
        prop=args.prop,
        batch_size=args.batch_size,
        num_batch=args.num_batch,
        config_name=args.config,
        results_folder=args.results_folder
    )