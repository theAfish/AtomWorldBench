import datetime
import os
import yaml
from pathlib import Path
from evaluation.evaluator import Evaluator
from models.openai_model import OpenAIModel
from models.azure_openai_model import AzureOpenAIModel
from models.huggingface_model import HuggingFaceModel
from models.vllm_model import vllmModel

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
    
    config = load_config(config_name)[model_id]
    
    # Initialize model
    model_class = config.get("class")
    if model_class == "OpenAIModel":
        api_key = os.path.expandvars(config.get("api_key", ""))
        model = OpenAIModel(
            model_name=config['model_name'],
            api_key=api_key,
            base_url=config.get('base_url'),
            temperature=config.get('temperature', 1)
        )
    elif model_class == "AzureOpenAIModel":
        model_name = os.path.expandvars(config.get("model_name", ""))
        api_key = os.path.expandvars(config.get("api_key", ""))
        api_version = os.path.expandvars(config.get("api_version", ""))
        azure_endpoint = os.path.expandvars(config.get("azure_endpoint", ""))
    
        model = AzureOpenAIModel(
            model_name=model_name,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            temperature=config.get('temperature', 1)
        )
    elif model_class == "HuggingFaceModel":
        model_name = config.get("model_name", None)
        device = config.get("device", "cpu")
        use_pipeline = config.get("use_pipeline", False)
        generation_params = {k: v for k, v in config.items() if k not in ["class", "model_name", "device", "use_pipeline"]}
        
        model = HuggingFaceModel(
            model_name=model_name,
            device=device,
            use_pipeline=use_pipeline,
            **generation_params
        )
    elif model_class == "vllmModel":
        model_name = config.get("model_name", None)
        generation_params = {k: v for k, v in config.items() if k not in ["class", "model_name", "device", "use_pipeline"]}
        
        model = vllmModel(
            model_name=model_name,
            **generation_params
        )
    else:
        raise ValueError(f"Unimplemented model_class '{model_class}'.")

    if action not in action_names:
        raise ValueError(f"Invalid action '{action}'. Must be one of: {action_names}")

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or "results"}/{timestamp}/{model_id}/{action}"
    
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
    # try:
    #     run_benchmark(
    #         args.model, 
    #         action=args.action, 
    #         batch_size=args.batch_size, 
    #         num_batch=args.num_batch, 
    #         config_name=args.config
    #     )
    # except Exception as e:
    #     print(f"Error running benchmark: {e}")