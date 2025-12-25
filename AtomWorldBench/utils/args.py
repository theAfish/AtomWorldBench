import argparse

def get_benchmark_parser():
    parser = argparse.ArgumentParser(description="Run AtomWorld Benchmark")
    parser.add_argument("-f", "--data_folder", type=str, default="AtomWorldBench/data", required=True, help="Path to data folder")
    parser.add_argument("-a", "--action_name", type=str, default=None, help="Specific action to run")
    parser.add_argument("-m", "--model", type=str, default="deepseek_chat", help="Model key in config file")
    parser.add_argument("-c", "--config_path", type=str, default="AtomWorldBench/config/llm_api_config.yaml", help="Path to config file")
    parser.add_argument("-o", "--output_folder", type=str, default="results", help="Output folder")
    parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("-n", "--num_batch", type=int, default=-1, help="Number of batches")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="Repeat count")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference and only run evaluation")
    parser.add_argument("--inference_file", type=str, default=None, help="Path to inference results file (if skipping inference)")
    parser.add_argument("--keep_inference", action="store_true", help="Keep inference results after evaluation")
    return parser
