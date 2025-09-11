import datetime
import os
import argparse
import yaml
from pathlib import Path
from point_world.evaluator import Evaluator
from models.openai_model import OpenAIModel
from models.azure_openai_model import AzureOpenAIModel
from models.huggingface_model import HuggingFaceModel
from models.vllm_model import vllmModel

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "datasets"

action_names = [
    'move',
    'move_towards',
    'insert_between',
    'rotate_around'
]

def load_config(config_name: str) -> dict:
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist.")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_benchmark(
        model_id: str,
        action: str,
        batch_size: int,
        num_batch: int,
        config_name: str = "models",
        results_folder: str = None
    ):
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
        use_pipeline = config.get("use_pipeline", True)
        generation_params = {k: v for k, v in config.items() if
                             k not in ["class", "model_name", "device", "use_pipeline"]}

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

    data_path = DATA_DIR

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or "results/PointWorld"}/{model_id}/{action}/{timestamp}"
    

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