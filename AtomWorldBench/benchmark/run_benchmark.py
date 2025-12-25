import argparse
import os
import sys
import yaml
import re
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.inference.inferencer import AtomWorldInferencer
from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator
from models.openai_model import OpenAIModel
from models.azure_openai_model import AzureOpenAIModel
from utils.args import get_benchmark_parser

def load_model_from_config(config_path, model_key):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_key not in config:
        raise ValueError(f"Model {model_key} not found in config")
    
    model_config = config[model_key]
    model_class_name = model_config.pop('class')
    
    # Resolve env vars
    for k, v in model_config.items():
        if isinstance(v, str):
            # Handle ${VAR} syntax
            match = re.match(r'\$\{(.+)\}', v)
            if match:
                env_var = match.group(1)
                model_config[k] = os.environ.get(env_var)
            
    if model_class_name == 'OpenAIModel':
        return OpenAIModel(**model_config)
    elif model_class_name == 'AzureOpenAIModel':
        return AzureOpenAIModel(**model_config)
    else:
        raise ValueError(f"Unknown model class {model_class_name}")

def main():
    parser = get_benchmark_parser()
    
    args = parser.parse_args()

    # Setup folders
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    action_subfolder = args.action_name if args.action_name else "All_Actions"
    base_output_path = os.path.join(args.output_folder, "AtomWorld", action_subfolder, timestamp)
    
    inference_folder = os.path.join(base_output_path, "inference")
    evaluation_folder = os.path.join(base_output_path, "evaluation")

    inference_file = args.inference_file

    if not args.skip_inference:
        print(f"Loading model {args.model} from {args.config_path}...")
        try:
            model = load_model_from_config(args.config_path, args.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        print("Starting inference...")
        inferencer = AtomWorldInferencer(
            model=model,
            data_folder=args.data_folder,
            action_name=args.action_name,
            output_folder=inference_folder
        )
        
        inference_file = inferencer.infer(
            batch_size=args.batch_size,
            num_batch=args.num_batch,
            repeat=args.repeat
        )
    
    if inference_file:
        print(f"Starting evaluation on {inference_file}...")
        evaluator = AtomWorldEvaluator(
            action_name=args.action_name,
            results_folder=evaluation_folder
        )
        evaluator.evaluate(inference_file)

        if not args.keep_inference:
            print(f"Deleting inference file: {inference_file}")
            try:
                os.remove(inference_file)
            except OSError as e:
                print(f"Error deleting file {inference_file}: {e}")
    else:
        print("No inference file provided or generated.")

if __name__ == "__main__":
    main()
