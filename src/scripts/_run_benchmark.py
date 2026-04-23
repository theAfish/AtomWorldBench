import datetime
import os
from pathlib import Path
from utils.load_model import load_model, load_config
from atomworld.evaluation.evaluator import AtomWorldEvaluator as Evaluator

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results/AtomWorld"

print(f"Using config directory: {CONFIG_DIR}")
print(f"Using data directory: {DATA_DIR}")


action_names = [
    'add_atom_action',
    'change_atom_action',
    'delete_around_atom_action', 
    'delete_below_atom_action', 
    'insert_between_atoms_action',
    'move_around_atom_action',
    'move_atom_action',
    'move_selected_atoms_action',
    'move_towards_atom_action',
    'remove_atom_action',
    'rotate_around_atom_action',
    'swap_atoms_action'
]


def run_benchmark(
        model_id: str, 
        action: str, 
        batch_size: int,
        num_batch: int,
        config_name: str="models", 
        results_folder: str=None
    ):
    """
    Run the benchmark using the specified configuration.
    
    Args:
        config_name (str): Name of the configuration file (without .yaml extension).
    """
    
    config = load_config(CONFIG_DIR / config_name)[model_id]
    
    # Initialize model
    model = load_model(config)

    if action not in action_names:
        raise ValueError(f"Invalid action '{action}'. Must be one of: {action_names}")

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or RESULTS_DIR}/{model_id}/{action}/{timestamp}"
    
    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        data_folder=DATA_DIR,
        action_name=action,
        results_folder=results_folder
    )
    
    # Run evaluation
    evaluator.evaluate(batch_size=batch_size, num_batch=num_batch)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark with specified configuration.")
    parser.add_argument(
        "-m",
        "--model", 
        type=str, 
        default="deepseek_reasoner", 
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
        default=action_names[4], 
        help=f"Name of the action for test. Default: {action_names[4]}"
    )
    parser.add_argument(
        "-b",
        "--batch_size", 
        type=int, 
        default=50, 
        help=f"Batch size for the run. Default: 50"
    )
    parser.add_argument(
        "-n",
        "--num_batch", 
        type=int, 
        default=-1, 
        help=f"Number of batches to use. Default: -1 for all data"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        args.model, 
        action=args.action, 
        batch_size=args.batch_size, 
        num_batch=args.num_batch, 
        config_name=args.config
    )
