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


def _agent_cli_to_name(agent_cli: str) -> str:
    """Derive a short, filesystem-safe name from an agent CLI string.

    Examples::

        "python examples/my_agent/run.py"  →  "run"
        "my_agent"                          →  "my_agent"
    """
    # Take the last whitespace-separated token, strip path separators and extension
    last_token = agent_cli.strip().split()[-1]
    stem = os.path.splitext(os.path.basename(last_token))[0]
    # Replace any remaining non-alphanumeric characters with underscores
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "agent"


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
    from benchmark.inference.agent_inferencer import AgentInferencer
    from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator
    from atomworld.actions import get_action_category

    parser = get_benchmark_parser()
    args = parser.parse_args(args)

    # Setup folders
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    action_subfolder = args.action_name if args.action_name else "All_Actions"
    category = get_action_category(args.action_name or '', default='simple')

    # Use the agent name (derived from the CLI command) or the model name as the
    # run identifier so results land under a sensible subfolder.
    if args.agent_cli:
        run_id = _agent_cli_to_name(args.agent_cli)
    else:
        run_id = args.model

    base_output_path = os.path.join(
        args.output_folder, "AtomWorld", category, action_subfolder, run_id, timestamp
    )

    inference_folder = os.path.join(base_output_path, "inference")
    evaluation_folder = os.path.join(base_output_path, "evaluation")

    inference_file = args.inference_file

    if not args.skip_inference:
        if args.agent_cli:
            # ── Agent mode ──────────────────────────────────────────────────
            print(f"Starting agent inference with: {args.agent_cli}")
            inferencer = AgentInferencer(
                agent_cli=args.agent_cli,
                data_folder=args.data_folder,
                action_name=args.action_name,
                output_folder=inference_folder,
                timeout=args.timeout,
                batch_size=args.batch_size,
                keep_tmp_workspaces=args.keep_agent_tmp,
            )
            inference_file = inferencer.infer(
                num_batch=args.num_batch,
                restart_from_index=args.start_index,
                repeat=args.repeat,
            )
        else:
            # ── LLM mode ────────────────────────────────────────────────────
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
        evaluator_mode = None
        if not args.skip_inference:
            evaluator_mode = "agent" if args.agent_cli else "llm"

        evaluator = AtomWorldEvaluator(
            action_name=args.action_name,
            results_folder=evaluation_folder,
            inference_mode=evaluator_mode,
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