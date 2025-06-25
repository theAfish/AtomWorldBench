import os
import yaml
from pathlib import Path
from evaluation.evaluator import Evaluator
from models.openai_model import OpenAIModel

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"

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


def load_config(config_name: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_name (str): Name of the configuration file (without .yaml extension).
    
    Returns:
        dict: Configuration parameters.
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def run_benchmark(model_id: str, action: str, config_name: str="models", results_folder: str=None):
    """
    Run the benchmark using the specified configuration.
    
    Args:
        config_name (str): Name of the configuration file (without .yaml extension).
    """
    config = load_config(config_name)[model_id]

    api_key = os.path.expandvars(config.get("api_key", ""))

    # Initialize model
    model = OpenAIModel(
        model_name=config['model_name'],
        api_key=api_key,
        base_url=config.get('base_url'),
        temperature=config.get('temperature', 1)
    )

    # automatically set results folder if not provided
    if results_folder is None:
        results_folder = f"results/{model_id}/{action}"
    else:
        results_folder = f"{results_folder}/{model_id}/{action}"
    
    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        data_folder=DATA_DIR,
        action_name=action,
        results_folder=results_folder
    )
    
    # Run evaluation
    evaluator.evaluate(batch_size=1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark with specified configuration.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek_reasoner", 
        help="ID of the model to use (e.g., 'deepseek_chat', 'openai_gpt4')"
    )
    parser.add_argument(
        "--action", 
        type=str, 
        default=action_names[4], 
        help=f"Name of the action for test. Default: {action_names[4]}"
    )
    
    args = parser.parse_args()

    run_benchmark(args.model, action=args.action, config_name="models")
    
    # try:
    #     run_benchmark(args.model, action=args.action, config_name="models")
    # except Exception as e:
    #     print(f"Error running benchmark: {e}")