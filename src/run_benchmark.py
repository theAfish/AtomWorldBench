#!/usr/bin/env python3
"""
Unified benchmark runner.

For AtomWorld benchmarks, uses the separated inference + evaluation pipeline.
For other benchmark types, falls back to the legacy runner.
"""

import os
import re
import sys
import yaml
import importlib
from datetime import datetime

from utils.args import get_benchmark_parser
from utils.visualization import plot_metrics_distribution
import models


def load_model_from_config(config_path, model_key):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if model_key not in config:
        available_models = list(config.keys())
        raise ValueError(
            f"Model {model_key} not found in config. Available models: {available_models}"
        )

    model_config = config[model_key]
    model_class_name = model_config.pop("class")

    # Resolve env vars
    for k, v in model_config.items():
        if isinstance(v, str):
            match = re.match(r"\$\{(.+)\}", v)
            if match:
                env_var = match.group(1)
                model_config[k] = os.environ.get(env_var)

    # Dynamic loading
    if hasattr(models, model_class_name):
        model_class = getattr(models, model_class_name)
    elif "." in model_class_name:
        module_name, class_name = model_class_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not import model class {model_class_name}: {e}"
            )
    else:
        raise ValueError(
            f"Unknown model class {model_class_name}. "
            f"Available models: {models.__all__}"
        )

    return model_class(**model_config)


def main(args=None):
    from benchmark.inference.inferencer import AtomWorldInferencer
    from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator

    parser = get_benchmark_parser()
    args = parser.parse_args(args)

    # Setup folders
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    action_subfolder = args.action_name if args.action_name else "All_Actions"
    base_output_path = os.path.join(
        args.output_folder, "AtomWorld", action_subfolder, args.model, timestamp
    )

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
            output_folder=inference_folder,
        )

        inference_file = inferencer.infer(
            batch_size=args.batch_size,
            num_batch=args.num_batch,
            restart_from_index=args.start_index,
            repeat=args.repeat,
        )

    if inference_file:
        print(f"Starting evaluation on {inference_file}...")
        evaluator = AtomWorldEvaluator(
            action_name=args.action_name,
            results_folder=evaluation_folder,
        )
        results = evaluator.evaluate(inference_file)

        # Plot metrics distribution
        try:
            print(f"Plotting metrics distribution to {evaluation_folder}...")
            plot_metrics_distribution(results, evaluation_folder)
        except Exception as e:
            print(f"Error plotting metrics distribution: {e}")

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