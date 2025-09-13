import datetime
import os
import argparse
import yaml
from pathlib import Path
from utils.load_model import load_model, load_config
from perceptual.evaluator.cif_repair_evaluator import CIFRepairEvaluator as Evaluator
from perceptual.utils.dataloader import load_data

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "cif_modifications.csv"


def run_benchmark(
        model_id: str,
        batch_size: int,
        num_batch: int,
        config_name: str = "models",
        results_folder: str = None
    ):
    config = load_config(CONFIG_DIR / config_name)[model_id]
    model = load_model(config)
    data = load_data(DATA_DIR)

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