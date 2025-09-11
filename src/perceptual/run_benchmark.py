import datetime
import os
import argparse
import yaml
from pathlib import Path
from utils.load_model import load_model
from perceptual.evaluator import PerceptualEvaluator as Evaluator
from perceptual.dataloader import load_data

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent / "cif_modifications.csv"


def load_config(config_name: str) -> dict:
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist.")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_benchmark(
        model_id: str,
        batch_size: int,
        num_batch: int,
        config_name: str = "models",
        results_folder: str = None
    ):
    config = load_config(config_name)[model_id]

    # Initialize model
    model = load_model(config)


    data_path = DATA_DIR
    data = load_data(data_path)

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or "results/CifRepair"}/{model_id}/{timestamp}"
    

    evaluator = Evaluator(
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
        help="Folder to save results. Default: 'results/PointWorld/{model_id}/{action}/{timestamp}'"
    )
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model,
        batch_size=args.batch_size,
        num_batch=args.num_batch,
        config_name=args.config,
        results_folder=args.results_folder
    )