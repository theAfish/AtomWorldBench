import argparse


def get_benchmark_parser():
    parser = argparse.ArgumentParser(description="Run AtomWorld Benchmark")
    parser.add_argument(
        "-f",
        "--data_folder",
        type=str,
        default="data",
        help="Path to data folder",
    )
    parser.add_argument(
        "-a",
        "--action_name",
        type=str,
        default=None,
        help="Specific action to run",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek_chat",
        help="Model key in config file",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="src/config/models.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="results",
        help="Output folder",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=50, help="Batch size"
    )
    parser.add_argument(
        "-n", "--num_batch", type=int, default=-1, help="Number of batches"
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Start index for inference",
    )
    parser.add_argument(
        "-r", "--repeat", type=int, default=1, help="Repeat count"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference and only run evaluation",
    )
    parser.add_argument(
        "--inference_file",
        type=str,
        default=None,
        help="Path to inference results file (if skipping inference)",
    )
    parser.add_argument(
        "--keep_inference",
        action="store_true",
        help="Keep inference results after evaluation",
    )
    return parser


def get_visualization_parser():
    parser = argparse.ArgumentParser(
        description="Visualize AtomWorld Benchmark Results"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to evaluation_results.json file",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for plots. Defaults to the same folder as input file.",
    )
    return parser


def get_generation_parser():
    parser = argparse.ArgumentParser(
        description="Generate AtomWorld Benchmark Data"
    )
    parser.add_argument(
        "-c",
        "--cif_folder",
        type=str,
        required=True,
        help="Path to folder containing input CIF files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated JSON files",
    )
    parser.add_argument(
        "-a",
        "--action_names",
        type=str,
        nargs="+",
        default=None,
        help="List of action names to generate. If not provided, uses all ready actions.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per action",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=10,
        help="Max attempts to generate a valid sample",
    )
    parser.add_argument(
        "--seed", type=int, default=75, help="Random seed"
    )
    parser.add_argument(
        "--no_random",
        action="store_true",
        help="Disable random shuffling of structures",
    )
    parser.add_argument(
        "--allow_repeat",
        action="store_true",
        help="Allow repeating structures across samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Use the verbose (motif-based) action family instead of the "
            "default simple (index-based) actions."
        ),
    )
    return parser
