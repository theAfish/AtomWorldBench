import datetime
import argparse
from pathlib import Path
from utils.load_model import load_model, load_config
from complementary.point_world.evaluator import PointWorldEvaluator as Evaluator

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"
RESULT_DIR = Path(__file__).parent.parent.parent.parent.parent / "results" / "PointWorld"
DATA_DIR = Path(__file__).parent.parent / "datasets"

action_names = [
    'move',
    'move_towards',
    'insert_between',
    'rotate_around'
]

def run_benchmark(
        model_id: str,
        action: str,
        batch_size: int,
        num_batch: int,
        config_name: str = "models",
        results_folder: str = None
    ):
    config = load_config(CONFIG_DIR / config_name)[model_id]
    model = load_model(config)

    if action not in action_names:
        raise ValueError(f"Invalid action '{action}'. Must be one of: {action_names}")

    data_path = DATA_DIR

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or RESULT_DIR}/{model_id}/{action}/{timestamp}"
    

    evaluator = Evaluator(
        model=model,
        data_folder=str(data_path),
        action_name=action,
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
        "-a",
        "--action",
        type=str,
        default=action_names[0],
        help=f"Name of the action for test. Default: {action_names[0]}"
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
        action=args.action,
        batch_size=args.batch_size,
        num_batch=args.num_batch,
        config_name=args.config,
        results_folder=args.results_folder
    )